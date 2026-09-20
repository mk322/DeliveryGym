"""Base ``WorldBundle``: the reusable map without a task's spawned assets.

design plan §5.1's required properties are enforced here rather than documented:

- stable IDs independent of array ordering — ``node_id``/``edge_id``/``entity_id``
  are required and uniqueness is validated;
- explicit units, axes, handedness, origin, and transforms — via
  ``CoordinateFrame`` and ``Transform`` from ``geometry``;
- semantic labels with provenance — ``LabelProvenance`` is mandatory, so a VLM
  suggestion can never be mistaken for authored metadata;
- edges carrying polyline, length, clearance, allowed modes, surface, and
  synthetic provenance;
- no absolute machine-specific paths — ``RelPath`` rejects them.

Two rules exist because of what M0 measured on Paris. Edges are explicitly
``certified`` or ``excluded`` with no third state, since the design plan M2 requires
every edge to be one or the other and excluded edges to be unavailable to
episode generation. And a synthetic edge may not be ``certified`` without
NavMesh evidence, because design plan §3.3 and 17 both single out the two long Paris
connectors (36.3 m and 45.3 m) as possibly crossing non-road space.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, Field, field_validator, model_validator

from embodiedbench.schemas.base import SchemaModel
from embodiedbench.schemas.geometry import CoordinateFrame, Transform, Vec3

_RELPATH_CHARS = re.compile(r"^[\w\-./ ]+$")


def _check_relative_path(value: str) -> str:
    """Reject anything that is not a portable, in-bundle relative path.

    design plan §5.1 requires no absolute machine-specific paths in portable
    artifacts. Written as a validator rather than a regex pattern because
    pydantic v2 compiles patterns with the Rust regex engine, which has no
    look-around.
    """
    if not value:
        raise ValueError("path must not be empty")
    if value.startswith("/") or value.startswith("\\"):
        raise ValueError(f"path must be relative to the bundle, got absolute {value!r}")
    if re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError(f"path must be relative to the bundle, got drive-absolute {value!r}")
    if ".." in value.split("/"):
        raise ValueError(f"path must not escape the bundle, got {value!r}")
    if not _RELPATH_CHARS.match(value):
        raise ValueError(f"path contains unsupported characters: {value!r}")
    return value


# A path inside a bundle. design plan §5.1: no absolute machine-specific paths.
RelPath = Annotated[str, AfterValidator(_check_relative_path)]
StableId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class LabelProvenance(str, Enum):
    """Where a semantic label came from (design plan §5.1, 6.1 P3)."""

    AUTHORED_METADATA = "authored_metadata"
    RULE = "rule"
    GEOMETRIC_HEURISTIC = "geometric_heuristic"
    VLM_SUGGESTION = "vlm_suggestion"
    HUMAN_OVERRIDE = "human_override"


class EdgeProvenance(str, Enum):
    """How a navigation edge came to exist."""

    AUTHORED_CENTERLINE = "authored_centerline"
    NAVMESH_DERIVED = "navmesh_derived"
    SYNTHETIC_CONNECTOR = "synthetic_connector"


class EdgeStatus(str, Enum):
    """the design plan M2: every edge is certified or excluded, with no third state."""

    CERTIFIED = "certified"
    EXCLUDED = "excluded"


class LocomotionMode(str, Enum):
    WALK = "walk"
    SCOOTER = "scooter"
    CAR = "car"
    BUS = "bus"


class NodeKind(str, Enum):
    """design plan §3.2 counts these separately, so the schema keeps them distinct."""

    JUNCTION = "junction"
    PASS_THROUGH = "pass_through"
    DEAD_END = "dead_end"
    DOCK = "dock"
    FREE_SPACE_SAMPLE = "free_space_sample"


class NavNode(SchemaModel):
    node_id: StableId
    kind: NodeKind
    position: Vec3
    road_name: str = ""
    address: str = ""


class NavMeshEvidence(SchemaModel):
    """Proof that a NavMesh path exists along an edge (the design plan M2)."""

    path_found: bool
    path_length_cm: float = Field(ge=0.0)
    waypoint_count: int = Field(ge=0)
    queried_at: str = ""
    backend: str = Field(default="spear_navigation_service")
    evidence_ref: RelPath | None = None


class NavEdge(SchemaModel):
    edge_id: StableId
    from_node: StableId
    to_node: StableId
    polyline: list[Vec3] = Field(min_length=2)
    length_cm: float = Field(gt=0.0)
    provenance: EdgeProvenance
    status: EdgeStatus
    bidirectional: bool = True
    slope_deg: float = 0.0
    width_cm: float | None = Field(default=None, gt=0.0)
    clearance_cm: float | None = Field(default=None, gt=0.0)
    surface: str = "road"
    allowed_modes: list[LocomotionMode] = Field(default_factory=lambda: [LocomotionMode.WALK])
    navmesh_evidence: NavMeshEvidence | None = None
    exclusion_reason: str = ""

    @model_validator(mode="after")
    def _consistency(self) -> "NavEdge":
        if self.from_node == self.to_node:
            raise ValueError(f"edge {self.edge_id} is a self-loop")
        if not self.allowed_modes:
            raise ValueError(f"edge {self.edge_id} allows no locomotion mode")
        if self.status is EdgeStatus.EXCLUDED and not self.exclusion_reason:
            raise ValueError(f"excluded edge {self.edge_id} must record why")
        if self.status is EdgeStatus.CERTIFIED:
            if self.provenance is EdgeProvenance.SYNTHETIC_CONNECTOR:
                evidence = self.navmesh_evidence
                if evidence is None or not evidence.path_found:
                    # design plan §6.1 P2: synthetic links are unsafe until a NavMesh
                    # path confirms them; design plan §17: they must not leak into eval.
                    raise ValueError(
                        f"synthetic connector {self.edge_id} cannot be certified without "
                        "NavMesh evidence of a path"
                    )
        return self


class NavGraph(SchemaModel):
    SCHEMA_ID = "embodiedbench/nav_graph"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    nodes: list[NavNode]
    edges: list[NavEdge]
    components: list[list[StableId]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _referential_integrity(self) -> "NavGraph":
        node_ids = [n.node_id for n in self.nodes]
        duplicates = {i for i in node_ids if node_ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate node ids: {sorted(duplicates)}")
        edge_ids = [e.edge_id for e in self.edges]
        duplicate_edges = {i for i in edge_ids if edge_ids.count(i) > 1}
        if duplicate_edges:
            raise ValueError(f"duplicate edge ids: {sorted(duplicate_edges)}")
        known = set(node_ids)
        for edge in self.edges:
            missing = {edge.from_node, edge.to_node} - known
            if missing:
                raise ValueError(f"edge {edge.edge_id} references unknown nodes {sorted(missing)}")
        for component in self.components:
            unknown = set(component) - known
            if unknown:
                raise ValueError(f"component references unknown nodes {sorted(unknown)}")
        return self

    def certified_edges(self) -> list[NavEdge]:
        return [e for e in self.edges if e.status is EdgeStatus.CERTIFIED]

    def certified_length_cm(self) -> float:
        return sum(e.length_cm for e in self.certified_edges())

    def total_length_cm(self) -> float:
        return sum(e.length_cm for e in self.edges)

    def certified_road_coverage(self) -> float:
        """Fraction of total edge length that is certified (the design plan M2, ADR-0004)."""
        total = self.total_length_cm()
        return self.certified_length_cm() / total if total > 0 else 0.0

    def neighbors(self, node_id: str, *, certified_only: bool = True) -> list[str]:
        out = []
        for edge in self.edges:
            if certified_only and edge.status is not EdgeStatus.CERTIFIED:
                continue
            if edge.from_node == node_id:
                out.append(edge.to_node)
            elif edge.bidirectional and edge.to_node == node_id:
                out.append(edge.from_node)
        return sorted(set(out))


class SemanticEntity(SchemaModel):
    """A labelled thing in the world, with where its label came from."""

    entity_id: StableId
    entity_type: str = Field(min_length=1)
    position: Vec3
    provenance: LabelProvenance
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    display_name: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _vlm_labels_carry_confidence(self) -> "SemanticEntity":
        # design plan §6.1 P3: VLM proposals are persisted with confidence for review.
        # An unscored suggestion cannot be triaged, so it is rejected at the door.
        if self.provenance is LabelProvenance.VLM_SUGGESTION and self.confidence is None:
            raise ValueError(f"VLM-suggested label for {self.entity_id} must carry a confidence")
        return self


class InteractionSite(SchemaModel):
    """A place an agent can act at, kept separate from semantic entities (P3)."""

    site_id: StableId
    site_type: str = Field(min_length=1)
    position: Vec3
    yaw_deg: float = 0.0
    nearest_node: StableId | None = None
    entity_id: StableId | None = None
    reachable_modes: list[LocomotionMode] = Field(default_factory=list)


class SourceProvenance(SchemaModel):
    """What the bundle was compiled from (design plan §5.1, 6.1 P0)."""

    ue_project: str = ""
    engine_version: str = ""
    level_package: str = ""
    source_content_sha256: Sha256 | None = None
    compiler_version: str = ""
    rules_version: str = ""
    manual_override_count: int = Field(default=0, ge=0)
    actor_count: int | None = Field(default=None, ge=0)


class WorldBundle(SchemaModel):
    """The immutable base map artifact (design plan §5.1)."""

    SCHEMA_ID = "embodiedbench/world_bundle"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    world_id: StableId
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    frames: list[CoordinateFrame] = Field(min_length=1)
    transforms: list[Transform] = Field(default_factory=list)
    nav_graph: NavGraph
    entities: list[SemanticEntity] = Field(default_factory=list)
    interaction_sites: list[InteractionSite] = Field(default_factory=list)
    source: SourceProvenance = Field(default_factory=SourceProvenance)
    previews: list[RelPath] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique_and_sites_resolve(self) -> "WorldBundle":
        entity_ids = [e.entity_id for e in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate entity ids")
        site_ids = [s.site_id for s in self.interaction_sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("duplicate interaction site ids")
        known_nodes = {n.node_id for n in self.nav_graph.nodes}
        known_entities = set(entity_ids)
        for site in self.interaction_sites:
            if site.nearest_node is not None and site.nearest_node not in known_nodes:
                raise ValueError(f"site {site.site_id} references unknown node {site.nearest_node}")
            if site.entity_id is not None and site.entity_id not in known_entities:
                raise ValueError(f"site {site.site_id} references unknown entity {site.entity_id}")
        frame_names = [f.name for f in self.frames]
        if len(frame_names) != len(set(frame_names)):
            raise ValueError("duplicate coordinate frame declarations")
        return self

    def sites_of_type(self, site_type: str) -> list[InteractionSite]:
        return [s for s in self.interaction_sites if s.site_type == site_type]
