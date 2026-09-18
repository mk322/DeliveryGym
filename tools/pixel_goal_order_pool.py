"""Validated, region-scoped order selection for the live Paris Pixel Goal run.

The stock DeliveryBench compiler derives address kerbs from a carriageway
graph.  A live pedestrian pawn cannot safely use those points.  This module
loads a versioned pool whose entrances, hand-over anchors, spawn points and
pedestrian edges were certified against one UE scene.  Random mode chooses a
seed-deterministic *pair of certified stops*; fixed mode names stops from the
same pool.  Neither mode accepts arbitrary coordinates.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from embodiedbench.compiler.road_network import Address, RoadNetwork, StreetNode


LEGACY_POOL_SCHEMA = "embodiedbench/validated-delivery-region/v1"
POOL_SCHEMA = "embodiedbench/trusted-pedestrian-delivery-region/v2"
SUPPORTED_POOL_SCHEMAS = frozenset((LEGACY_POOL_SCHEMA, POOL_SCHEMA))
ORDER_MODES = ("random", "fixed")
STOP_ROLES = frozenset(("pickup", "dropoff"))
EDGE_KINDS = frozenset(("sidewalk", "marked_crossing"))
AUDIT_SCHEMA = "embodiedbench/live-recast-surface-audit/v1"
CROSSWALK_ASSET_PREFIX = "PR_Crossswalk_"
AGENT_NAV_DATA_SOURCE = "GetNavDataForProps(ParisPocAgent)"
PEDESTRIAN_CONNECTIVITY_SOURCE = "ue_recast_navmesh_surface_audit"
CITYCORE_SEMANTIC_USE = "address_and_street_names_only"


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} keys are not exact: {actual!r}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and out < minimum:
        raise ValueError(f"{label} must be at least {minimum:g}")
    return out


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _vec(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) \
            or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class PoolRegion:
    name: str
    nav_bounds_center_cm: tuple[float, float, float]
    nav_bounds_extent_cm: tuple[float, float, float]


@dataclass(frozen=True)
class PoolValidation:
    source_path: str
    source_sha256: str
    min_door_anchor_cm: float
    max_door_anchor_cm: float
    max_navmesh_adjustment_cm: float
    max_entrance_connector_cm: float
    max_nav_path_deviation_cm: float
    max_nav_path_endpoint_error_cm: float
    max_spawn_displacement_cm: float


@dataclass(frozen=True)
class PedestrianGraphCertification:
    audit_path: str
    audit_sha256: str
    audit_schema: str
    stable_recast_topology_sha256: str
    polygon_count: int
    recast_adjacency_count: int
    visibility_edge_audit_count: int
    crosswalk_asset_count: int
    graph_method: str
    grid_spacing_cm: float
    grid_edge_sample_spacing_cm: float
    edge_sample_spacing_cm: float
    crosswalk_sample_spacing_cm: float
    max_crosswalk_road_transition_cm: float
    crosswalk_asset_prefix: str
    agent_nav_data_source: str
    connectivity_source: str
    semantic_source: str
    graph_sha256: str


@dataclass(frozen=True)
class PoolNode:
    id: str
    x_cm: float
    y_cm: float
    street_index: int
    role: str

    @property
    def position(self) -> tuple[float, float]:
        return (self.x_cm, self.y_cm)


@dataclass(frozen=True)
class PoolEdge:
    a: str
    b: str
    kind: str
    verified: bool
    source: str


@dataclass(frozen=True)
class PoolSpawn:
    id: str
    node_id: str
    z_cm: float
    yaw_deg: float
    verified: bool
    max_navmesh_adjustment_cm: float


@dataclass(frozen=True)
class PoolStop:
    id: str
    building_id: str
    street_index: int
    street_name: str
    number: int
    door_cm: tuple[float, float]
    handover_node_id: str
    poi_type: str
    roles: tuple[str, ...]
    entrance_source: str
    entrance_verified: bool
    surface: str
    navmesh_verified: bool
    max_navmesh_adjustment_cm: float
    entrance_asset_id: str | None = None
    entrance_face: str | None = None
    entrance_static_mesh_path: str | None = None

    @property
    def text(self) -> str:
        return f"{self.number} {self.street_name}"


@dataclass(frozen=True)
class EnginePathCertification:
    """What the engine itself said about the pool's ways: every directed leg
    the harness could plan (``candidate_legs``) judged from its start node by
    a resolve without a walk, and what the next pool version dropped on that
    verdict -- the nodes outside the one strongly connected component of the
    accepted legs, and the edges no accepted leg walks."""

    method: str
    certified_at: str
    source_pool_sha256: str
    verdicts_sha256: str
    legs: int
    accepted: int
    refused: int
    unjudged: int
    removed_edges: tuple[tuple[str, str], ...]
    removed_nodes: tuple[str, ...]
    #: Nodes the engine accepts no leg from (or to) but walks over inside
    #: accepted legs -- the middle of a marked crossing, say. They stay in
    #: the pool for the chains that pass over them; a pawn never stops on
    #: one and a pixel never means one.
    pass_through_nodes: tuple[str, ...]
    placement_tolerance_cm: float
    min_leg_cm: float
    navmesh_adjustment_cm: float
    leg_epsilon_cm: float
    max_leg_cm: float
    #: An accepted leg whose controller path is longer than its chord by
    #: more than this is not a straight walk and was not kept.
    max_path_excess_cm: float


@dataclass(frozen=True)
class ValidatedDeliveryPool:
    path: Path
    sha256: str
    schema: str
    pool_id: str
    version: int
    scene: str
    region: PoolRegion
    validation: PoolValidation
    pedestrian_graph: PedestrianGraphCertification | None
    spawns: tuple[PoolSpawn, ...]
    nodes: tuple[PoolNode, ...]
    edges: tuple[PoolEdge, ...]
    stops: tuple[PoolStop, ...]
    engine_certification: EnginePathCertification | None = None
    #: The directed legs the engine accepted, (start node, end node); the
    #: harness walks only these. ``None`` for a pool the engine has not
    #: judged, where the harness walks the straight parts of the edge route.
    certified_legs: frozenset[tuple[str, str]] | None = None

    @property
    def nodes_by_id(self) -> dict[str, PoolNode]:
        return {node.id: node for node in self.nodes}

    @property
    def pass_through_nodes(self) -> frozenset[str]:
        """Nodes a pawn walks over but never stops on (see
        ``EnginePathCertification.pass_through_nodes``); empty for a pool
        the engine has not judged."""
        if self.engine_certification is None:
            return frozenset()
        return frozenset(self.engine_certification.pass_through_nodes)

    @property
    def stops_by_id(self) -> dict[str, PoolStop]:
        return {stop.id: stop for stop in self.stops}

    @property
    def spawns_by_id(self) -> dict[str, PoolSpawn]:
        return {spawn.id: spawn for spawn in self.spawns}

    @property
    def profile(self) -> str:
        return f"{self.pool_id}/v{self.version}"


@dataclass(frozen=True)
class OrderConstraints:
    min_delivery_cm: float = 2_000.0
    max_delivery_cm: float = 15_000.0
    min_turns: int = 0
    # A same-street order can still require a real turn and zebra crossing
    # (for example, opposite sides of Rue Oberkampf).  Street identity is a
    # semantic filter, not a pedestrian-safety invariant, so callers opt in.
    require_different_streets: bool = False
    require_marked_crossing: bool = False

    def __post_init__(self) -> None:
        if (isinstance(self.min_delivery_cm, bool)
                or not isinstance(self.min_delivery_cm, (int, float))
                or not math.isfinite(self.min_delivery_cm)
                or self.min_delivery_cm < 0.0):
            raise ValueError("minimum delivery distance must be finite and non-negative")
        if (isinstance(self.max_delivery_cm, bool)
                or not isinstance(self.max_delivery_cm, (int, float))
                or not math.isfinite(self.max_delivery_cm) \
                or self.max_delivery_cm < self.min_delivery_cm):
            raise ValueError("maximum delivery distance must cover the minimum")
        if (isinstance(self.min_turns, bool)
                or not isinstance(self.min_turns, int)
                or self.min_turns < 0):
            raise ValueError("minimum turns must be a non-negative integer")
        if not isinstance(self.require_different_streets, bool):
            raise ValueError("require_different_streets must be boolean")
        if not isinstance(self.require_marked_crossing, bool):
            raise ValueError("require_marked_crossing must be boolean")

    def to_report(self) -> dict[str, Any]:
        return {
            "min_delivery_cm": self.min_delivery_cm,
            "max_delivery_cm": self.max_delivery_cm,
            "min_turns": self.min_turns,
            "require_different_streets": self.require_different_streets,
            "require_marked_crossing": self.require_marked_crossing,
        }


@dataclass(frozen=True)
class PoolPath:
    node_ids: tuple[str, ...]
    length_cm: float
    turns: int
    uses_marked_crossing: bool


@dataclass(frozen=True)
class ResolvedDeliveryScenario:
    mode: str
    seed: int
    scenario_id: str
    candidate_count: int
    spawn: PoolSpawn
    pickup: PoolStop
    dropoff: PoolStop
    approach: PoolPath
    delivery: PoolPath
    constraints: OrderConstraints
    requested_spawn_id: str | None
    requested_pickup_id: str | None
    requested_dropoff_id: str | None


POOL_KEYS_V1 = {
    "schema", "pool_id", "version", "scene", "region", "validation",
    "spawns", "nodes", "edges", "stops",
}
POOL_KEYS_V2 = POOL_KEYS_V1 | {"pedestrian_graph"}
#: A v3 pool carries the engine's own verdict on its ways: the block that
#: says how it was judged and the table of directed legs it accepted. The
#: two keys come together or not at all.
ENGINE_CERTIFICATION_KEY = "engine_certification"
CERTIFIED_LEGS_KEY = "certified_legs"
ENGINE_CERTIFICATION_KEYS = {
    "method", "certified_at", "source_pool_sha256", "verdicts_sha256",
    "legs", "accepted", "refused", "unjudged", "removed_edges", "removed_nodes",
    "pass_through_nodes",
    "placement_tolerance_cm", "min_leg_cm", "navmesh_adjustment_cm",
    "leg_epsilon_cm", "max_leg_cm", "max_path_excess_cm",
}
ENGINE_CERTIFICATION_METHOD = "engine_resolve_only_directed_legs_v1"
#: A harness leg is one straight walk between two pool nodes whose edge
#: chain keeps every node within this much of the chord (the Douglas-Peucker
#: tolerance the harness simplifies a route with). It is at least as long
#: as the picture's reach (the bottom edge of the front picture is about
#: 2.9 m from the camera, so a nearer point cannot be aimed at) and no
#: longer than this; a pawn stops short of a leg's end at any node on its
#: chain, which is how it reaches a node nearer than the picture's reach.
LEG_EPSILON_CM = 40.0
MIN_LEG_CM = 300.0
MAX_LEG_CM = 1000.0
REGION_KEYS = {"name", "nav_bounds_center_cm", "nav_bounds_extent_cm"}
VALIDATION_KEYS = {
    "source_path", "source_sha256", "min_door_anchor_cm",
    "max_door_anchor_cm", "max_navmesh_adjustment_cm",
    "max_entrance_connector_cm", "max_nav_path_deviation_cm",
    "max_nav_path_endpoint_error_cm", "max_spawn_displacement_cm",
}
SPAWN_KEYS = {
    "id", "node_id", "z_cm", "yaw_deg", "verified",
    "max_navmesh_adjustment_cm",
}
NODE_KEYS = {"id", "x_cm", "y_cm", "street_index", "role"}
EDGE_KEYS = {"a", "b", "kind", "verified", "source"}
STOP_KEYS_V1 = {
    "id", "building_id", "street_index", "street_name", "number",
    "door_cm", "handover_node_id", "poi_type", "roles",
    "entrance_source", "entrance_verified", "surface",
    "navmesh_verified", "max_navmesh_adjustment_cm",
}
STOP_KEYS_V2 = STOP_KEYS_V1 | {
    "entrance_asset_id", "entrance_face", "entrance_static_mesh_path",
}
PEDESTRIAN_GRAPH_KEYS = {
    "audit_path", "audit_sha256", "audit_schema",
    "stable_recast_topology_sha256", "polygon_count",
    "recast_adjacency_count", "visibility_edge_audit_count",
    "crosswalk_asset_count",
    "graph_method", "grid_spacing_cm", "grid_edge_sample_spacing_cm",
    "edge_sample_spacing_cm", "crosswalk_sample_spacing_cm",
    "max_crosswalk_road_transition_cm", "crosswalk_asset_prefix",
    "agent_nav_data_source", "connectivity_source", "semantic_source",
    "graph_sha256",
}


def _sha256(value: str, label: str) -> str:
    out = _text(value, label)
    if len(out) != 64 or any(ch not in "0123456789abcdef" for ch in out):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return out


def _graph_sha256(root: Mapping[str, Any]) -> str:
    payload = {
        key: root[key] for key in ("nodes", "edges", "spawns", "stops")
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_pedestrian_graph_certification(
    root: Mapping[str, Any], pool_path: Path, region: PoolRegion,
) -> tuple[PedestrianGraphCertification, Mapping[str, Any]]:
    row = _exact_keys(
        root["pedestrian_graph"], PEDESTRIAN_GRAPH_KEYS,
        "pedestrian graph certification",
    )
    certification = PedestrianGraphCertification(
        audit_path=_text(row["audit_path"], "pedestrian_graph.audit_path"),
        audit_sha256=_sha256(
            row["audit_sha256"], "pedestrian_graph.audit_sha256"),
        audit_schema=_text(
            row["audit_schema"], "pedestrian_graph.audit_schema"),
        stable_recast_topology_sha256=_sha256(
            row["stable_recast_topology_sha256"],
            "pedestrian_graph.stable_recast_topology_sha256",
        ),
        polygon_count=_integer(
            row["polygon_count"], "pedestrian_graph.polygon_count", minimum=1),
        recast_adjacency_count=_integer(
            row["recast_adjacency_count"],
            "pedestrian_graph.recast_adjacency_count", minimum=1,
        ),
        visibility_edge_audit_count=_integer(
            row["visibility_edge_audit_count"],
            "pedestrian_graph.visibility_edge_audit_count", minimum=0,
        ),
        crosswalk_asset_count=_integer(
            row["crosswalk_asset_count"],
            "pedestrian_graph.crosswalk_asset_count", minimum=1,
        ),
        graph_method=_text(
            row["graph_method"], "pedestrian_graph.graph_method"),
        grid_spacing_cm=_number(
            row["grid_spacing_cm"],
            "pedestrian_graph.grid_spacing_cm", minimum=0.001,
        ),
        grid_edge_sample_spacing_cm=_number(
            row["grid_edge_sample_spacing_cm"],
            "pedestrian_graph.grid_edge_sample_spacing_cm", minimum=0.001,
        ),
        edge_sample_spacing_cm=_number(
            row["edge_sample_spacing_cm"],
            "pedestrian_graph.edge_sample_spacing_cm", minimum=0.001,
        ),
        crosswalk_sample_spacing_cm=_number(
            row["crosswalk_sample_spacing_cm"],
            "pedestrian_graph.crosswalk_sample_spacing_cm", minimum=0.001,
        ),
        max_crosswalk_road_transition_cm=_number(
            row["max_crosswalk_road_transition_cm"],
            "pedestrian_graph.max_crosswalk_road_transition_cm", minimum=0.0,
        ),
        crosswalk_asset_prefix=_text(
            row["crosswalk_asset_prefix"],
            "pedestrian_graph.crosswalk_asset_prefix",
        ),
        agent_nav_data_source=_text(
            row["agent_nav_data_source"],
            "pedestrian_graph.agent_nav_data_source",
        ),
        connectivity_source=_text(
            row["connectivity_source"],
            "pedestrian_graph.connectivity_source",
        ),
        semantic_source=_text(
            row["semantic_source"], "pedestrian_graph.semantic_source"),
        graph_sha256=_sha256(
            row["graph_sha256"], "pedestrian_graph.graph_sha256"),
    )
    if certification.audit_schema != AUDIT_SCHEMA:
        raise ValueError("pedestrian graph audit schema is not supported")
    if certification.graph_method != "recast_projected_surface_grid_v2":
        raise ValueError("pedestrian graph method is not the trusted Recast grid")
    if certification.crosswalk_asset_prefix != CROSSWALK_ASSET_PREFIX:
        raise ValueError("pedestrian graph does not require exact PR_Crossswalk_* assets")
    if certification.agent_nav_data_source != AGENT_NAV_DATA_SOURCE:
        raise ValueError("pedestrian graph was not exported from the rollout agent NavData")
    if certification.connectivity_source != PEDESTRIAN_CONNECTIVITY_SOURCE:
        raise ValueError("pedestrian connectivity is not certified from UE Recast")
    if certification.semantic_source != CITYCORE_SEMANTIC_USE:
        raise ValueError("CityCore is not restricted to address/street semantics")
    if certification.graph_sha256 != _graph_sha256(root):
        raise ValueError("pedestrian graph SHA-256 does not match nodes/edges/stops")

    audit_path = Path(certification.audit_path)
    if not audit_path.is_absolute():
        audit_path = (pool_path.parent / audit_path).resolve()
    if not audit_path.is_file():
        raise ValueError(f"pedestrian audit source is unavailable: {audit_path}")
    audit_bytes = audit_path.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != certification.audit_sha256:
        raise ValueError("pedestrian audit source SHA-256 does not match the pool")
    try:
        audit = json.loads(audit_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pedestrian audit source is not valid UTF-8 JSON") from error
    if not isinstance(audit, Mapping) or audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("pedestrian audit source schema does not match the pool")
    if audit.get("scene") != root["scene"]:
        raise ValueError("pedestrian audit scene does not match the pool")
    if audit.get("polygon_count") != certification.polygon_count:
        raise ValueError("pedestrian audit polygon count does not match the pool")
    if audit.get("stable_recast_topology_sha256") \
            != certification.stable_recast_topology_sha256:
        raise ValueError("pedestrian audit stable Recast topology does not match")
    edge_rows = audit.get("edge_surface_audits")
    if not isinstance(edge_rows, list) \
            or len(edge_rows) != certification.recast_adjacency_count:
        raise ValueError("pedestrian audit Recast adjacency count does not match")
    grid = audit.get("pedestrian_recast_grid")
    grid_edges = audit.get("grid_edge_surface_audits")
    visibility_edges = audit.get("grid_visibility_surface_audits")
    if not isinstance(grid, Mapping) \
            or float(grid.get("spacing_cm", math.nan)) \
                != certification.grid_spacing_cm \
            or not isinstance(grid_edges, list) or not grid_edges \
            or any(
                not isinstance(item, Mapping)
                or float(item.get("sample_spacing_cm", math.nan))
                    != certification.grid_edge_sample_spacing_cm
                for item in grid_edges
            ):
        raise ValueError("pedestrian audit Recast surface grid does not match")
    if not isinstance(visibility_edges, list) \
            or len(visibility_edges) \
                != certification.visibility_edge_audit_count \
            or any(
                not isinstance(item, Mapping)
                or item.get("kind") != "recast_grid_visibility_connector"
                or float(item.get("sample_spacing_cm", math.nan))
                    != certification.grid_edge_sample_spacing_cm
                for item in visibility_edges
            ):
        raise ValueError("pedestrian audit visibility evidence does not match")
    catalog = audit.get("crosswalk_catalog")
    catalog_rows = (
        catalog.get("crosswalk_components")
        if isinstance(catalog, Mapping) else None
    )
    if not isinstance(catalog_rows, list) \
            or len(catalog_rows) != certification.crosswalk_asset_count \
            or any(
                not isinstance(item, Mapping)
                or not str(item.get("label", "")).startswith(
                    certification.crosswalk_asset_prefix)
                for item in catalog_rows
            ):
        raise ValueError("pedestrian audit exact crosswalk catalog does not match")
    audit_region = audit.get("region")
    if not isinstance(audit_region, Mapping) \
            or _vec(
                audit_region.get("nav_bounds_center_cm"), 3,
                "pedestrian audit region center",
            ) != region.nav_bounds_center_cm \
            or _vec(
                audit_region.get("nav_bounds_extent_cm"), 3,
                "pedestrian audit region extent",
            ) != region.nav_bounds_extent_cm:
        raise ValueError("pedestrian audit NavMesh bounds do not match the pool")
    if any(
        not isinstance(item, Mapping)
        or float(item.get("sample_spacing_cm", math.nan))
            != certification.edge_sample_spacing_cm
        for item in edge_rows
    ):
        raise ValueError("pedestrian audit edge sample spacing does not match")
    entrance_catalog = audit.get("building_entrance_catalog")
    entrance_rows = (
        entrance_catalog.get("entrance_components")
        if isinstance(entrance_catalog, Mapping) else None
    )
    if not isinstance(entrance_rows, list) or not entrance_rows \
            or entrance_catalog.get("required_mesh_token") != "Entrance" \
            or entrance_catalog.get("component_count") != len(entrance_rows):
        raise ValueError("pedestrian audit has no real UE entrance catalog")
    entrance_ids = [
        item.get("entrance_asset_id")
        for item in entrance_rows if isinstance(item, Mapping)
    ]
    if len(entrance_ids) != len(entrance_rows) \
            or any(not isinstance(value, str) or not value for value in entrance_ids) \
            or len(set(entrance_ids)) != len(entrance_ids) \
            or any(
                not isinstance(item.get("static_mesh_path"), str)
                or "entrance" not in item["static_mesh_path"].casefold()
                for item in entrance_rows if isinstance(item, Mapping)
            ):
        raise ValueError("pedestrian audit UE entrance catalog is malformed")
    building_source = audit.get("building_source")
    if not isinstance(building_source, Mapping) \
            or building_source.get("entrance_geometry_source") \
                != "ue_static_mesh_instance_containing_Entrance":
        raise ValueError("pedestrian audit does not use real UE entrance geometry")
    return certification, audit


def _load_engine_certification(
    root: Mapping[str, Any], *, version: int, edges: Sequence[PoolEdge],
    nodes: Sequence[PoolNode],
) -> EnginePathCertification:
    """The engine-verdict block of a v3 pool, checked against the pool it
    sits in: what it says was dropped is not there."""
    row = _exact_keys(root[ENGINE_CERTIFICATION_KEY], ENGINE_CERTIFICATION_KEYS,
                      "order pool engine certification")
    if version < 3:
        raise ValueError("engine certification needs pool version 3 or later")
    method = _text(row["method"], "engine_certification.method")
    if method != ENGINE_CERTIFICATION_METHOD:
        raise ValueError(f"engine certification method is not trusted: {method!r}")
    removed_edges_raw = row["removed_edges"]
    removed_raw = row["removed_nodes"]
    if not isinstance(removed_edges_raw, list) or not isinstance(removed_raw, list):
        raise ValueError("engine certification removed_edges/removed_nodes must be lists")
    removed_edges: list[tuple[str, str]] = []
    for item in removed_edges_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("engine certification removed edge must be [a, b]")
        removed_edges.append((_text(item[0], "removed edge a"), _text(item[1], "removed edge b")))
    removed = tuple(_text(item, "removed node") for item in removed_raw)
    present = {(edge.a, edge.b) for edge in edges} | {(edge.b, edge.a) for edge in edges}
    still = [pair for pair in removed_edges if pair in present]
    if still:
        raise ValueError(f"engine certification names removed edges the pool still has: {still[:5]}")
    node_ids = {node.id for node in nodes}
    kept = [node for node in removed if node in node_ids]
    if kept:
        raise ValueError(f"engine certification names removed nodes the pool still has: {kept[:5]}")
    pass_raw = row["pass_through_nodes"]
    if not isinstance(pass_raw, list):
        raise ValueError("engine certification pass_through_nodes must be a list")
    pass_through = tuple(_text(item, "pass-through node") for item in pass_raw)
    unknown = [node for node in pass_through if node not in node_ids]
    if unknown or len(set(pass_through)) != len(pass_through):
        raise ValueError(
            f"engine certification pass-through nodes must be distinct pool nodes: {unknown[:5]}")
    return EnginePathCertification(
        method=method,
        certified_at=_text(row["certified_at"], "engine_certification.certified_at"),
        source_pool_sha256=_sha256(row["source_pool_sha256"], "engine_certification.source_pool_sha256"),
        verdicts_sha256=_sha256(row["verdicts_sha256"], "engine_certification.verdicts_sha256"),
        legs=_integer(row["legs"], "engine_certification.legs", minimum=1),
        accepted=_integer(row["accepted"], "engine_certification.accepted", minimum=1),
        refused=_integer(row["refused"], "engine_certification.refused", minimum=0),
        unjudged=_integer(row["unjudged"], "engine_certification.unjudged", minimum=0),
        removed_edges=tuple(removed_edges),
        removed_nodes=removed,
        pass_through_nodes=pass_through,
        placement_tolerance_cm=_number(row["placement_tolerance_cm"], "engine_certification.placement_tolerance_cm", minimum=0.0),
        min_leg_cm=_number(row["min_leg_cm"], "engine_certification.min_leg_cm", minimum=0.0),
        navmesh_adjustment_cm=_number(row["navmesh_adjustment_cm"], "engine_certification.navmesh_adjustment_cm", minimum=0.0),
        leg_epsilon_cm=_number(row["leg_epsilon_cm"], "engine_certification.leg_epsilon_cm", minimum=0.0),
        max_leg_cm=_number(row["max_leg_cm"], "engine_certification.max_leg_cm", minimum=0.0),
        max_path_excess_cm=_number(row["max_path_excess_cm"], "engine_certification.max_path_excess_cm", minimum=0.0),
    )


def strongly_connected_components(
    node_ids: Sequence[str], arcs: Sequence[tuple[str, str]],
) -> list[set[str]]:
    """The strongly connected components of the directed graph, largest
    first (ties by smallest node id): the sets of nodes that all reach one
    another over the arcs. A pawn on a certified pool must be able to leave
    every node it can reach, so a pool's legs must form one component."""
    forward: dict[str, list[str]] = {node: [] for node in node_ids}
    backward: dict[str, list[str]] = {node: [] for node in node_ids}
    for a, b in arcs:
        forward[a].append(b)
        backward[b].append(a)
    # Kosaraju, iteratively: finish order on the forward graph, then sweep
    # the transpose in reverse finish order.
    order: list[str] = []
    seen: set[str] = set()
    for root in node_ids:
        if root in seen:
            continue
        seen.add(root)
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            node, index = stack[-1]
            if index < len(forward[node]):
                stack[-1] = (node, index + 1)
                neighbour = forward[node][index]
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append((neighbour, 0))
            else:
                order.append(node)
                stack.pop()
    assigned: set[str] = set()
    components: list[set[str]] = []
    for root in reversed(order):
        if root in assigned:
            continue
        component = {root}
        assigned.add(root)
        stack2 = [root]
        while stack2:
            node = stack2.pop()
            for neighbour in backward[node]:
                if neighbour not in assigned:
                    assigned.add(neighbour)
                    component.add(neighbour)
                    stack2.append(neighbour)
        components.append(component)
    components.sort(key=lambda component: (-len(component), min(component)))
    return components


def _load_certified_legs(
    root: Mapping[str, Any], *, nodes: Sequence[PoolNode],
    certification: EnginePathCertification,
) -> frozenset[tuple[str, str]]:
    """The leg table of a v3 pool: directed pairs of pool nodes, every
    node a pawn may stop on with a way in and a way out, all of them one
    strongly connected piece -- a pawn that can reach a node can leave it
    again -- and as many as the certification block says were accepted.
    A pass-through node starts and ends no leg."""
    raw = root[CERTIFIED_LEGS_KEY]
    if not isinstance(raw, list) or not raw:
        raise ValueError("certified_legs must be a nonempty list of [start, end]")
    node_ids = [node.id for node in nodes]
    known = set(node_ids)
    pass_through = set(certification.pass_through_nodes)
    legs: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"certified leg {index + 1} must be [start, end]")
        start = _text(item[0], f"certified leg {index + 1}.start")
        end = _text(item[1], f"certified leg {index + 1}.end")
        if start == end or start not in known or end not in known:
            raise ValueError(
                f"certified leg {index + 1} does not join two distinct pool nodes")
        if start in pass_through or end in pass_through:
            raise ValueError(
                f"certified leg {index + 1} starts or ends on a pass-through node")
        legs.append((start, end))
    if len(set(legs)) != len(legs):
        raise ValueError("certified_legs lists a leg twice")
    if len(legs) != certification.accepted:
        raise ValueError(
            f"certified_legs has {len(legs)} legs, the certification says "
            f"{certification.accepted} were accepted")
    stops = [node for node in node_ids if node not in pass_through]
    components = strongly_connected_components(stops, legs)
    if len(components) != 1:
        outside = sorted(set(stops) - components[0]) if components else stops
        raise ValueError(
            f"certified legs do not join every node both ways: {outside[:5]}")
    return frozenset(legs)


def load_validated_delivery_pool(path: str | Path) -> ValidatedDeliveryPool:
    """Load and fail-close validate one immutable region pool."""

    pool_path = Path(path).resolve()
    raw_bytes = pool_path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"order pool is not valid UTF-8 JSON: {pool_path}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("order pool must be a JSON object")
    schema = raw.get("schema")
    if schema not in SUPPORTED_POOL_SCHEMAS:
        raise ValueError(f"unsupported order pool schema: {schema!r}")
    expected_keys = POOL_KEYS_V2 if schema == POOL_SCHEMA else POOL_KEYS_V1
    if schema == POOL_SCHEMA and (
            ENGINE_CERTIFICATION_KEY in raw or CERTIFIED_LEGS_KEY in raw):
        expected_keys = expected_keys | {ENGINE_CERTIFICATION_KEY, CERTIFIED_LEGS_KEY}
    root = _exact_keys(raw, expected_keys, "order pool")

    region_raw = _exact_keys(root["region"], REGION_KEYS, "order pool region")
    region = PoolRegion(
        name=_text(region_raw["name"], "region.name"),
        nav_bounds_center_cm=_vec(
            region_raw["nav_bounds_center_cm"], 3, "region.nav_bounds_center_cm"),
        nav_bounds_extent_cm=_vec(
            region_raw["nav_bounds_extent_cm"], 3, "region.nav_bounds_extent_cm"),
    )
    if any(value <= 0.0 for value in region.nav_bounds_extent_cm):
        raise ValueError("region NavMesh extents must be positive")
    if schema == POOL_SCHEMA:
        pedestrian_graph, pedestrian_audit = \
            _load_pedestrian_graph_certification(root, pool_path, region)
    else:
        pedestrian_graph, pedestrian_audit = None, None

    validation_raw = _exact_keys(
        root["validation"], VALIDATION_KEYS, "order pool validation")
    source_sha = _sha256(
        validation_raw["source_sha256"], "validation.source_sha256")
    validation = PoolValidation(
        source_path=_text(validation_raw["source_path"], "validation.source_path"),
        source_sha256=source_sha,
        min_door_anchor_cm=_number(
            validation_raw["min_door_anchor_cm"],
            "validation.min_door_anchor_cm", minimum=0.0),
        max_door_anchor_cm=_number(
            validation_raw["max_door_anchor_cm"],
            "validation.max_door_anchor_cm", minimum=0.0),
        max_navmesh_adjustment_cm=_number(
            validation_raw["max_navmesh_adjustment_cm"],
            "validation.max_navmesh_adjustment_cm", minimum=0.0),
        max_entrance_connector_cm=_number(
            validation_raw["max_entrance_connector_cm"],
            "validation.max_entrance_connector_cm", minimum=0.0),
        max_nav_path_deviation_cm=_number(
            validation_raw["max_nav_path_deviation_cm"],
            "validation.max_nav_path_deviation_cm", minimum=0.0),
        max_nav_path_endpoint_error_cm=_number(
            validation_raw["max_nav_path_endpoint_error_cm"],
            "validation.max_nav_path_endpoint_error_cm", minimum=0.0),
        max_spawn_displacement_cm=_number(
            validation_raw["max_spawn_displacement_cm"],
            "validation.max_spawn_displacement_cm", minimum=0.0),
    )
    if validation.max_door_anchor_cm < validation.min_door_anchor_cm:
        raise ValueError("maximum door/anchor distance is below the minimum")
    source_path = Path(validation.source_path)
    if not source_path.is_absolute():
        source_path = (pool_path.parent / source_path).resolve()
    if not source_path.is_file():
        raise ValueError(
            f"certification source is unavailable: {source_path}")
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != validation.source_sha256:
        raise ValueError("certification source SHA-256 does not match the pool")
    try:
        source_root = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("certification source is not valid UTF-8 JSON") from error
    source_records = (
        source_root.get("buildings") if isinstance(source_root, Mapping) else None)
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("certification source has no CityCore building records")
    source_buildings: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(source_records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"certification building {index + 1} is not an object")
        building_id = _text(
            record.get("id"), f"certification building {index + 1}.id")
        if building_id in source_buildings:
            raise ValueError(
                f"duplicate building id in certification source: {building_id}")
        source_buildings[building_id] = record

    nodes: list[PoolNode] = []
    for index, item in enumerate(root["nodes"]):
        row = _exact_keys(item, NODE_KEYS, f"node {index + 1}")
        nodes.append(PoolNode(
            id=_text(row["id"], f"node {index + 1}.id"),
            x_cm=_number(row["x_cm"], f"node {index + 1}.x_cm"),
            y_cm=_number(row["y_cm"], f"node {index + 1}.y_cm"),
            street_index=_integer(
                row["street_index"], f"node {index + 1}.street_index", minimum=0),
            role=_text(row["role"], f"node {index + 1}.role"),
        ))
    node_ids = [node.id for node in nodes]
    if not nodes or len(node_ids) != len(set(node_ids)):
        raise ValueError("order pool nodes must be nonempty and uniquely identified")
    nodes_by_id = {node.id: node for node in nodes}

    edges: list[PoolEdge] = []
    seen_edges: set[tuple[str, str]] = set()
    for index, item in enumerate(root["edges"]):
        row = _exact_keys(item, EDGE_KEYS, f"edge {index + 1}")
        edge = PoolEdge(
            a=_text(row["a"], f"edge {index + 1}.a"),
            b=_text(row["b"], f"edge {index + 1}.b"),
            kind=_text(row["kind"], f"edge {index + 1}.kind"),
            verified=row["verified"] is True,
            source=_text(row["source"], f"edge {index + 1}.source"),
        )
        if edge.a == edge.b or edge.a not in nodes_by_id or edge.b not in nodes_by_id:
            raise ValueError(f"edge {index + 1} does not join two distinct pool nodes")
        if edge.kind not in EDGE_KINDS:
            raise ValueError(f"edge {index + 1} has unsafe kind {edge.kind!r}")
        if not edge.verified:
            raise ValueError(f"edge {index + 1} is not certified")
        if schema == POOL_SCHEMA:
            trusted_prefixes = (
                "ue_recast_adjacency:",
                "ue_recast_grid:",
                "ue_recast_visibility:",
                "ue_crosswalk_connector:",
                "ue_crosswalk:",
                "ue_entrance_connector:",
            )
            if not edge.source.startswith(trusted_prefixes):
                raise ValueError(
                    f"edge {index + 1} has no UE pedestrian evidence source")
            if edge.kind == "marked_crossing" and not edge.source.startswith(
                    f"ue_crosswalk:{CROSSWALK_ASSET_PREFIX}"):
                raise ValueError(
                    f"edge {index + 1} is not backed by an exact crosswalk asset")
            if edge.source.startswith(("ue_recast_grid:",
                                       "ue_recast_visibility:")) \
                    and (edge.kind != "sidewalk"
                         or nodes_by_id[edge.a].role != "recast_grid"
                         or nodes_by_id[edge.b].role != "recast_grid"):
                raise ValueError(
                    f"edge {index + 1} misuses pavement-grid evidence")
            crossing_roles = {
                nodes_by_id[edge.a].role, nodes_by_id[edge.b].role,
            }
            if "crossing" in crossing_roles \
                    and not edge.source.startswith((
                        f"ue_crosswalk:{CROSSWALK_ASSET_PREFIX}",
                        f"ue_crosswalk_connector:{CROSSWALK_ASSET_PREFIX}",
                    )):
                raise ValueError(
                    f"edge {index + 1} reaches a crossing without exact asset evidence")
        if "entrance_connector" in {
                nodes_by_id[edge.a].role, nodes_by_id[edge.b].role}:
            connector_cm = math.dist(
                nodes_by_id[edge.a].position, nodes_by_id[edge.b].position)
            if connector_cm > validation.max_entrance_connector_cm:
                raise ValueError(
                    f"edge {index + 1} entrance connector is {connector_cm:.1f} "
                    "cm, above the certified maximum")
        key = tuple(sorted((edge.a, edge.b)))
        if key in seen_edges:
            raise ValueError(f"duplicate undirected pool edge: {key}")
        seen_edges.add(key)
        edges.append(edge)
    if not edges:
        raise ValueError("order pool pedestrian graph is empty")

    spawns: list[PoolSpawn] = []
    for index, item in enumerate(root["spawns"]):
        row = _exact_keys(item, SPAWN_KEYS, f"spawn {index + 1}")
        spawn = PoolSpawn(
            id=_text(row["id"], f"spawn {index + 1}.id"),
            node_id=_text(row["node_id"], f"spawn {index + 1}.node_id"),
            z_cm=_number(row["z_cm"], f"spawn {index + 1}.z_cm"),
            yaw_deg=_number(row["yaw_deg"], f"spawn {index + 1}.yaw_deg"),
            verified=row["verified"] is True,
            max_navmesh_adjustment_cm=_number(
                row["max_navmesh_adjustment_cm"],
                f"spawn {index + 1}.max_navmesh_adjustment_cm", minimum=0.0),
        )
        if spawn.node_id not in nodes_by_id:
            raise ValueError(f"spawn {spawn.id!r} references an unknown node")
        if not spawn.verified:
            raise ValueError(f"spawn {spawn.id!r} is not certified")
        if spawn.max_navmesh_adjustment_cm \
                > validation.max_navmesh_adjustment_cm:
            raise ValueError(f"spawn {spawn.id!r} exceeds the NavMesh adjustment limit")
        spawns.append(spawn)
    spawn_ids = [spawn.id for spawn in spawns]
    if not spawns or len(spawn_ids) != len(set(spawn_ids)):
        raise ValueError("order pool spawns must be nonempty and uniquely identified")

    stops: list[PoolStop] = []
    address_texts: set[tuple[str, int]] = set()
    for index, item in enumerate(root["stops"]):
        row = _exact_keys(
            item,
            STOP_KEYS_V2 if schema == POOL_SCHEMA else STOP_KEYS_V1,
            f"stop {index + 1}",
        )
        roles_raw = row["roles"]
        if not isinstance(roles_raw, list) or not roles_raw:
            raise ValueError(f"stop {index + 1}.roles must be a nonempty list")
        roles = tuple(_text(role, f"stop {index + 1}.roles") for role in roles_raw)
        if len(roles) != len(set(roles)) or not set(roles) <= STOP_ROLES:
            raise ValueError(f"stop {index + 1}.roles contains duplicates or unknown roles")
        stop = PoolStop(
            id=_text(row["id"], f"stop {index + 1}.id"),
            building_id=_text(row["building_id"], f"stop {index + 1}.building_id"),
            street_index=_integer(
                row["street_index"], f"stop {index + 1}.street_index", minimum=0),
            street_name=_text(row["street_name"], f"stop {index + 1}.street_name"),
            number=_integer(row["number"], f"stop {index + 1}.number", minimum=1),
            door_cm=_vec(row["door_cm"], 2, f"stop {index + 1}.door_cm"),
            handover_node_id=_text(
                row["handover_node_id"], f"stop {index + 1}.handover_node_id"),
            poi_type=_text(row["poi_type"], f"stop {index + 1}.poi_type"),
            roles=roles,
            entrance_source=_text(
                row["entrance_source"], f"stop {index + 1}.entrance_source"),
            entrance_verified=row["entrance_verified"] is True,
            surface=_text(row["surface"], f"stop {index + 1}.surface"),
            navmesh_verified=row["navmesh_verified"] is True,
            max_navmesh_adjustment_cm=_number(
                row["max_navmesh_adjustment_cm"],
                f"stop {index + 1}.max_navmesh_adjustment_cm", minimum=0.0),
            entrance_asset_id=(
                _text(row["entrance_asset_id"],
                      f"stop {index + 1}.entrance_asset_id")
                if schema == POOL_SCHEMA else None
            ),
            entrance_face=(
                _text(row["entrance_face"],
                      f"stop {index + 1}.entrance_face")
                if schema == POOL_SCHEMA else None
            ),
            entrance_static_mesh_path=(
                _text(row["entrance_static_mesh_path"],
                      f"stop {index + 1}.entrance_static_mesh_path")
                if schema == POOL_SCHEMA else None
            ),
        )
        if stop.handover_node_id not in nodes_by_id:
            raise ValueError(f"stop {stop.id!r} references an unknown hand-over node")
        source_building = source_buildings.get(stop.building_id)
        if source_building is None:
            raise ValueError(
                f"stop {stop.id!r} is absent from the certified building source")
        if schema == POOL_SCHEMA:
            assert pedestrian_audit is not None
            entrance_catalog = pedestrian_audit["building_entrance_catalog"]
            entrance_by_id = {
                str(entry["entrance_asset_id"]): entry
                for entry in entrance_catalog["entrance_components"]
                if isinstance(entry, Mapping)
            }
            entrance = entrance_by_id.get(str(stop.entrance_asset_id))
            if entrance is None \
                    or entrance.get("actor_name") \
                        != source_building.get("source_actor") \
                    or entrance.get("actor_label") \
                        != source_building.get("source_label") \
                    or entrance.get("static_mesh_path") \
                        != stop.entrance_static_mesh_path \
                    or stop.entrance_face not in (
                        "local_y_min", "local_y_max"):
                raise ValueError(
                    f"stop {stop.id!r} does not match a real UE entrance asset")
            entrance_groups = pedestrian_audit.get("entrance_candidate_audits")
            matching_groups = [
                group for group in entrance_groups
                if isinstance(group, Mapping)
                and group.get("building_id") == stop.building_id
                and group.get("entrance_asset_id") == stop.entrance_asset_id
                and group.get("entrance_face") == stop.entrance_face
                and group.get("static_mesh_path")
                    == stop.entrance_static_mesh_path
                and isinstance(group.get("door_cm"), list)
                and len(group["door_cm"]) == 2
                and math.dist(
                    stop.door_cm,
                    (float(group["door_cm"][0]), float(group["door_cm"][1])),
                ) <= 0.1
            ] if isinstance(entrance_groups, list) else []
            if len(matching_groups) != 1:
                raise ValueError(
                    f"stop {stop.id!r} has no unique audited UE entrance face")
        else:
            centre = source_building.get("center_cm")
            bounds = source_building.get("bbox_cm")
            yaw = source_building.get("entrance_yaw_deg")
            if not isinstance(centre, Mapping) \
                    or not isinstance(bounds, Mapping) or yaw is None:
                raise ValueError(
                    f"stop {stop.id!r} has no source-authored entrance geometry")
            centre_xy = (
                _number(centre.get("x"), f"stop {stop.id}.source center x"),
                _number(centre.get("y"), f"stop {stop.id}.source center y"),
            )
            half_extent = (
                _number(bounds.get("x"), f"stop {stop.id}.source bbox x",
                        minimum=0.0) / 2.0,
                _number(bounds.get("y"), f"stop {stop.id}.source bbox y",
                        minimum=0.0) / 2.0,
            )
            yaw_radians = math.radians(
                _number(yaw, f"stop {stop.id}.source entrance yaw"))
            direction = (math.cos(yaw_radians), math.sin(yaw_radians))
            scale = min(
                half_extent[0] / abs(direction[0])
                if abs(direction[0]) > 1e-6 else float("inf"),
                half_extent[1] / abs(direction[1])
                if abs(direction[1]) > 1e-6 else float("inf"),
            )
            expected_door = (
                centre_xy[0] + direction[0] * scale,
                centre_xy[1] + direction[1] * scale,
            )
            if not math.isfinite(scale) \
                    or math.dist(stop.door_cm, expected_door) > 0.1:
                raise ValueError(
                    f"stop {stop.id!r} door does not match its source entrance")
        source_poi = str(source_building.get("poi_type") or "building")
        if stop.poi_type != source_poi:
            raise ValueError(
                f"stop {stop.id!r} POI type does not match its source building")
        if source_building.get("deliverybench_navigable") is not True:
            raise ValueError(
                f"stop {stop.id!r} is not navigable in its source building")
        node = nodes_by_id[stop.handover_node_id]
        if node.street_index != stop.street_index:
            raise ValueError(f"stop {stop.id!r} and its route node disagree on street")
        if schema == POOL_SCHEMA:
            assert pedestrian_audit is not None
            connector_rows = pedestrian_audit.get(
                "grid_connector_surface_audits")
            matching_connectors = [
                connector for connector in connector_rows
                if isinstance(connector, Mapping)
                and connector.get("kind") == "entrance_to_recast_grid"
                and connector.get("building_id") == stop.building_id
                and connector.get("entrance_asset_id")
                    == stop.entrance_asset_id
                and connector.get("entrance_face") == stop.entrance_face
                and connector.get("static_mesh_path")
                    == stop.entrance_static_mesh_path
                and isinstance(connector.get("anchor_cm"), list)
                and len(connector["anchor_cm"]) == 3
                and math.dist(
                    node.position,
                    (float(connector["anchor_cm"][0]),
                     float(connector["anchor_cm"][1])),
                ) <= 0.1
                and isinstance(connector.get("points"), list)
                and connector["points"]
                and all(
                    isinstance(point, Mapping)
                    and point.get("projected") is True
                    and point.get("surface_class") == "pavement"
                    and isinstance(point.get("stable_poly_id"), str)
                    for point in connector["points"]
                )
            ] if isinstance(connector_rows, list) else []
            expected_source_prefix = (
                f"ue_entrance_connector:{stop.building_id}:"
                f"{stop.entrance_asset_id}:{stop.entrance_face}:")
            matching_graph_edges = [
                edge for edge in edges
                if stop.handover_node_id in (edge.a, edge.b)
                and edge.source.startswith(expected_source_prefix)
            ]
            if not matching_connectors or len(matching_graph_edges) != 1:
                raise ValueError(
                    f"stop {stop.id!r} has no audited pavement connector")
        door_gap = math.dist(stop.door_cm, node.position)
        if not validation.min_door_anchor_cm <= door_gap <= validation.max_door_anchor_cm:
            raise ValueError(
                f"stop {stop.id!r} door/anchor gap {door_gap:.1f} cm is outside "
                "the certified range")
        if not stop.entrance_verified:
            raise ValueError(f"stop {stop.id!r} entrance is not certified")
        if stop.surface != "pavement":
            raise ValueError(f"stop {stop.id!r} is not certified as pavement")
        if not stop.navmesh_verified:
            raise ValueError(f"stop {stop.id!r} has no NavMesh certification")
        if stop.max_navmesh_adjustment_cm \
                > validation.max_navmesh_adjustment_cm:
            raise ValueError(f"stop {stop.id!r} exceeds the NavMesh adjustment limit")
        address_key = (stop.street_name.casefold(), stop.number)
        if address_key in address_texts:
            raise ValueError(f"duplicate address in order pool: {stop.text}")
        address_texts.add(address_key)
        stops.append(stop)
    stop_ids = [stop.id for stop in stops]
    if len(stops) < 2 or len(stop_ids) != len(set(stop_ids)):
        raise ValueError("order pool needs at least two uniquely identified stops")
    if not any("pickup" in stop.roles for stop in stops) \
            or not any("dropoff" in stop.roles for stop in stops):
        raise ValueError("order pool must contain pickup and drop-off stops")

    engine_certification = None
    certified_legs = None
    if ENGINE_CERTIFICATION_KEY in root:
        engine_certification = _load_engine_certification(
            root, version=_integer(root["version"], "version", minimum=1),
            edges=edges, nodes=nodes)
        certified_legs = _load_certified_legs(
            root, nodes=nodes, certification=engine_certification)

    pool = ValidatedDeliveryPool(
        path=pool_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema=schema,
        pool_id=_text(root["pool_id"], "pool_id"),
        version=_integer(root["version"], "version", minimum=1),
        scene=_text(root["scene"], "scene"),
        region=region,
        validation=validation,
        pedestrian_graph=pedestrian_graph,
        spawns=tuple(spawns),
        nodes=tuple(nodes),
        edges=tuple(edges),
        stops=tuple(stops),
        engine_certification=engine_certification,
        certified_legs=certified_legs,
    )
    # Connectivity is a certification property, not something the selector may
    # silently discover only for a lucky subset of seeds.
    reachable = _reachable_nodes(pool, pool.spawns[0].node_id)
    required = {spawn.node_id for spawn in pool.spawns} | {
        stop.handover_node_id for stop in pool.stops}
    if not required <= reachable:
        missing = sorted(required - reachable)
        raise ValueError(f"order pool has unreachable certified stops/spawns: {missing}")
    return pool


_ADJACENCY_CACHE: dict[str, dict[str, list[tuple[str, PoolEdge]]]] = {}


def _adjacency(pool: ValidatedDeliveryPool) -> dict[str, list[tuple[str, PoolEdge]]]:
    """Node -> sorted (neighbour, edge) rows; cached per pool contents and
    never mutated by a caller."""
    out = _ADJACENCY_CACHE.get(pool.sha256)
    if out is None:
        out = {node.id: [] for node in pool.nodes}
        for edge in pool.edges:
            out[edge.a].append((edge.b, edge))
            out[edge.b].append((edge.a, edge))
        for rows in out.values():
            rows.sort(key=lambda row: row[0])
        if len(_ADJACENCY_CACHE) > 64:
            _ADJACENCY_CACHE.clear()
        _ADJACENCY_CACHE[pool.sha256] = out
    return out


def _reachable_nodes(pool: ValidatedDeliveryPool, start: str) -> set[str]:
    adjacency = _adjacency(pool)
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour, _edge in adjacency[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator))
    projected = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.dist(point, projected)


def _simplify_route(
    points: Sequence[tuple[float, float]], *, epsilon_cm: float = 75.0,
) -> list[tuple[float, float]]:
    """Remove metre-grid stair-steps before counting meaningful turns."""

    if len(points) <= 2:
        return list(points)
    distance, split = max(
        (
            _point_to_segment_distance(point, points[0], points[-1]),
            index,
        )
        for index, point in enumerate(points[1:-1], 1)
    )
    if distance <= epsilon_cm:
        return [points[0], points[-1]]
    first = _simplify_route(points[:split + 1], epsilon_cm=epsilon_cm)
    second = _simplify_route(points[split:], epsilon_cm=epsilon_cm)
    return [*first[:-1], *second]


def _turn_count(points: Sequence[tuple[float, float]]) -> int:
    points = _simplify_route(points)
    count = 0
    for first, middle, last in zip(points, points[1:], points[2:]):
        incoming = math.degrees(math.atan2(
            middle[1] - first[1], middle[0] - first[0]))
        outgoing = math.degrees(math.atan2(
            last[1] - middle[1], last[0] - middle[0]))
        delta = abs((outgoing - incoming + 180.0) % 360.0 - 180.0)
        if delta >= 30.0:
            count += 1
    return count


def shortest_pool_path(
    pool: ValidatedDeliveryPool, start: str, end: str,
) -> PoolPath | None:
    """Deterministic Dijkstra route over only certified pedestrian edges."""

    nodes = pool.nodes_by_id
    if start not in nodes or end not in nodes:
        return None
    adjacency = _adjacency(pool)
    distance: dict[str, float] = {start: 0.0}
    previous: dict[str, tuple[str, PoolEdge]] = {}
    queue: list[tuple[float, str]] = [(0.0, start)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distance.get(current):
            continue
        if current == end:
            break
        for neighbour, edge in adjacency[current]:
            candidate = cost + math.dist(nodes[current].position, nodes[neighbour].position)
            existing = distance.get(neighbour)
            if existing is None or candidate < existing - 1e-9:
                distance[neighbour] = candidate
                previous[neighbour] = (current, edge)
                heapq.heappush(queue, (candidate, neighbour))
    if end not in distance:
        return None
    node_ids = [end]
    used_edges: list[PoolEdge] = []
    while node_ids[-1] != start:
        parent, edge = previous[node_ids[-1]]
        used_edges.append(edge)
        node_ids.append(parent)
    node_ids.reverse()
    used_edges.reverse()
    points = [nodes[node_id].position for node_id in node_ids]
    return PoolPath(
        node_ids=tuple(node_ids),
        length_cm=distance[end],
        turns=_turn_count(points),
        uses_marked_crossing=any(edge.kind == "marked_crossing" for edge in used_edges),
    )


def nearest_pool_node(
    pool: ValidatedDeliveryPool,
    point_cm: tuple[float, float],
    *,
    radius_cm: float,
    roles: tuple[str, ...] | None = None,
    stops_only: bool = False,
) -> tuple[PoolNode, float] | None:
    """The certified node closest to a point, if one lies within ``radius_cm``.

    This is how a resolved pixel becomes a destination the pawn is allowed to
    rest on: the engine says where the ray landed, the pool says which
    certified pedestrian point that is nearest to, and anything further from
    the certified graph than the radius is not a pedestrian destination at
    all. With ``stops_only`` a pass-through node (walked over, never stopped
    on) is not an answer. Ties break on node id so the answer is
    deterministic.
    """

    pass_through = pool.pass_through_nodes if stops_only else frozenset()
    best: tuple[float, str, PoolNode] | None = None
    for node in pool.nodes:
        if roles is not None and node.role not in roles:
            continue
        if node.id in pass_through:
            continue
        gap = math.dist(node.position, point_cm)
        if gap > radius_cm:
            continue
        key = (gap, node.id, node)
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None:
        return None
    return best[2], best[0]


_EDGE_PAIRS_CACHE: dict[str, dict[frozenset[str], str]] = {}


def _edge_pairs(pool: ValidatedDeliveryPool) -> dict[frozenset[str], str]:
    """Undirected edge -> kind, cached per pool contents."""
    pairs = _EDGE_PAIRS_CACHE.get(pool.sha256)
    if pairs is None:
        pairs = {frozenset((edge.a, edge.b)): edge.kind for edge in pool.edges}
        if len(_EDGE_PAIRS_CACHE) > 64:
            _EDGE_PAIRS_CACHE.clear()
        _EDGE_PAIRS_CACHE[pool.sha256] = pairs
    return pairs


def _chain_within(
    pool: ValidatedDeliveryPool, start_node_id: str, end_node_id: str,
    allowed: set[str],
) -> tuple[str, ...] | None:
    """The shortest edge path from start to end using only ``allowed``
    nodes, ties broken by node id; ``None`` when there is none."""
    nodes = pool.nodes_by_id
    adjacency = _adjacency(pool)
    distance: dict[str, float] = {start_node_id: 0.0}
    previous: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, start_node_id)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distance.get(current):
            continue
        if current == end_node_id:
            break
        for neighbour, _edge in adjacency[current]:
            if neighbour not in allowed:
                continue
            candidate = cost + math.dist(nodes[current].position, nodes[neighbour].position)
            existing = distance.get(neighbour)
            if existing is None or candidate < existing - 1e-9:
                distance[neighbour] = candidate
                previous[neighbour] = current
                heapq.heappush(queue, (candidate, neighbour))
    if end_node_id not in distance:
        return None
    chain = [end_node_id]
    while chain[-1] != start_node_id:
        chain.append(previous[chain[-1]])
    return tuple(reversed(chain))


def _nodes_along(
    pool: ValidatedDeliveryPool, a: tuple[float, float], b: tuple[float, float],
    epsilon_cm: float,
) -> set[str]:
    chord = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / chord, (b[1] - a[1]) / chord
    along: set[str] = set()
    for node in pool.nodes:
        dx, dy = node.x_cm - a[0], node.y_cm - a[1]
        t = dx * ux + dy * uy
        if -1e-6 <= t <= chord + 1e-6 and abs(dx * uy - dy * ux) <= epsilon_cm:
            along.add(node.id)
    return along


def leg_chain(
    pool: ValidatedDeliveryPool, start_node_id: str, end_node_id: str,
    *, epsilon_cm: float = LEG_EPSILON_CM, min_cm: float = MIN_LEG_CM,
    max_cm: float = MAX_LEG_CM,
) -> tuple[str, ...] | None:
    """The pool nodes a straight walk from one node to another passes over,
    or ``None`` when that walk is not a leg of this pool.

    A leg is one straight line between two nodes, between ``min_cm`` and
    ``max_cm`` long. Its chain is the shortest edge path between them that
    stays within ``epsilon_cm`` of that line (every node of the path between
    the ends and no further from the line than the tolerance the harness
    simplifies a route with), so the leg never leaves certified paving and
    the edges it walks are known. The rule is geometric: the same pair is
    the same leg however a route reached it.
    """

    nodes = pool.nodes_by_id
    if start_node_id == end_node_id or start_node_id not in nodes or end_node_id not in nodes:
        return None
    a, b = nodes[start_node_id].position, nodes[end_node_id].position
    chord = math.dist(a, b)
    if chord < min_cm or chord > max_cm:
        return None
    return _chain_within(pool, start_node_id, end_node_id,
                         _nodes_along(pool, a, b, epsilon_cm))


def candidate_legs(
    pool: ValidatedDeliveryPool, *, epsilon_cm: float = LEG_EPSILON_CM,
    min_cm: float = MIN_LEG_CM, max_cm: float = MAX_LEG_CM,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Every directed leg the pool's geometry offers (``leg_chain`` for every
    ordered pair of nodes, or for ``pairs`` only), keyed by (start, end):
    the set the engine is asked to judge, and the most the harness can ever
    walk."""

    import numpy as np

    ids = [node.id for node in pool.nodes]
    index = {node_id: i for i, node_id in enumerate(ids)}
    xy = np.array([[node.x_cm, node.y_cm] for node in pool.nodes], dtype=float)
    wanted: dict[int, list[int]] | None = None
    if pairs is not None:
        wanted = {}
        for start, end in pairs:
            wanted.setdefault(index[start], []).append(index[end])
    legs: dict[tuple[str, str], tuple[str, ...]] = {}
    for i, start in enumerate(ids):
        if wanted is not None and i not in wanted:
            continue
        d = xy - xy[i]
        chords = np.hypot(d[:, 0], d[:, 1])
        ends = (np.nonzero((chords >= min_cm) & (chords <= max_cm))[0]
                if wanted is None else np.array(wanted[i], dtype=int))
        for j in ends:
            chord = chords[j]
            if chord < min_cm or chord > max_cm:
                continue
            ux, uy = d[j, 0] / chord, d[j, 1] / chord
            t = d[:, 0] * ux + d[:, 1] * uy
            off = np.abs(d[:, 0] * uy - d[:, 1] * ux)
            picked = np.nonzero((t >= -1e-6) & (t <= chord + 1e-6) & (off <= epsilon_cm))[0]
            chain = _chain_within(pool, start, ids[j], {ids[k] for k in picked})
            if chain is not None:
                legs[(start, ids[j])] = chain
    return legs


_LEG_CHAINS_CACHE: dict[str, dict[tuple[str, str], tuple[str, ...]]] = {}


def certified_leg_chains(
    pool: ValidatedDeliveryPool,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """The chain of every certified leg of the pool, computed once per pool
    contents. A certified leg that is not a leg of the pool's geometry is an
    error in the pool file."""
    assert pool.certified_legs is not None
    key = f"{pool.sha256}:{hash(pool.certified_legs)}"
    chains = _LEG_CHAINS_CACHE.get(key)
    if chains is None:
        chains = candidate_legs(pool, pairs=sorted(pool.certified_legs))
        missing = sorted(set(pool.certified_legs) - set(chains))
        if missing:
            raise ValueError(
                f"certified legs are not legs of the pool's geometry: {missing[:5]}")
        if len(_LEG_CHAINS_CACHE) > 8:
            _LEG_CHAINS_CACHE.clear()
        _LEG_CHAINS_CACHE[key] = chains
    return chains


def certified_moves(
    pool: ValidatedDeliveryPool,
) -> dict[tuple[str, str], str]:
    """Every move the harness may make on a certified pool, (start, stop) ->
    the certified leg's end it aims at. A pawn aims at a leg's end and stops
    at any node on the leg's chain; a move to a leg's own end aims at it,
    a move to a node inside the chain aims at the end of the shortest
    certified leg through that node (the picture cannot aim nearer than the
    legs are long, so a near node is reached by aiming past it)."""
    nodes = pool.nodes_by_id
    pass_through = pool.pass_through_nodes
    moves: dict[tuple[str, str], tuple[float, str]] = {}
    for (start, end), chain in certified_leg_chains(pool).items():
        length = math.dist(nodes[start].position, nodes[end].position)
        for stop in chain[1:]:
            if stop in pass_through:
                continue                    # walked over, never stopped on
            key = (start, stop)
            rank = (0.0, end) if stop == end else (length, end)
            if key not in moves or rank < moves[key]:
                moves[key] = rank
    return {key: aim for key, (_rank, aim) in moves.items()}


@dataclass(frozen=True)
class PlannedLeg:
    """One move of a walk: stop at ``stop`` (a pool node) after aiming at
    ``aim`` (the end of the certified leg the move rides; the same node when
    the move is the whole leg)."""

    stop: str
    stop_cm: tuple[float, float]
    aim: str
    aim_cm: tuple[float, float]


#: A move costs its metres plus this, so a route prefers fewer, longer legs
#: when the metres tie (every leg is a capture and a resolve).
LEG_STEP_COST_CM = 50.0


def plan_pool_legs(
    pool: ValidatedDeliveryPool,
    start_node_id: str,
    end_node_id: str,
    *,
    avoid: Sequence[tuple[str, str]] = (),
) -> tuple[PoolPath, list[PlannedLeg]] | None:
    """A certified route from one node to another as straight moves, in
    walking order; an empty list when start and end coincide; ``None`` when
    the certified legs offer no way (or a node is not in the pool).

    The pool must carry the engine's leg table (``certified_legs``): the
    route is the cheapest sequence of moves over it -- metres walked plus a
    small cost per move, ties broken by node id -- never a move the engine
    refused, and never one whose stop follows a leg in ``avoid`` (a leg the
    engine refused just now, so the walk goes round it). The path returned
    lists every pool node the moves pass over, so its length, turns and
    crossing use read like the phone's route.
    """

    if pool.certified_legs is None:
        raise ValueError("the pool carries no certified legs; the harness walks only those")
    nodes = pool.nodes_by_id
    if start_node_id not in nodes or end_node_id not in nodes:
        return None
    if start_node_id == end_node_id:
        return PoolPath(node_ids=(start_node_id,), length_cm=0.0, turns=0,
                        uses_marked_crossing=False), []
    moves = certified_moves(pool)
    banned = set(avoid)
    adjacency: dict[str, list[tuple[str, str, float]]] = {node.id: [] for node in pool.nodes}
    for (start, stop), aim in moves.items():
        if (start, aim) in banned:
            continue
        adjacency[start].append(
            (stop, aim, math.dist(nodes[start].position, nodes[stop].position) + LEG_STEP_COST_CM))
    for rows in adjacency.values():
        rows.sort()
    distance: dict[str, float] = {start_node_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}
    queue: list[tuple[float, str]] = [(0.0, start_node_id)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != distance.get(current):
            continue
        if current == end_node_id:
            break
        for stop, aim, step in adjacency[current]:
            candidate = cost + step
            existing = distance.get(stop)
            if existing is None or candidate < existing - 1e-9:
                distance[stop] = candidate
                previous[stop] = (current, aim)
                heapq.heappush(queue, (candidate, stop))
    if end_node_id not in distance:
        return None
    route: list[tuple[str, str, str]] = []          # (start, stop, aim)
    current = end_node_id
    while current != start_node_id:
        start, aim = previous[current]
        route.append((start, current, aim))
        current = start
    route.reverse()
    chains = certified_leg_chains(pool)
    kinds = _edge_pairs(pool)
    node_ids: list[str] = [start_node_id]
    length_cm = 0.0
    crossing = False
    for start, stop, aim in route:
        chain = chains[(start, aim)]
        walked = chain[:chain.index(stop) + 1]
        node_ids.extend(walked[1:])
        length_cm += math.dist(nodes[start].position, nodes[stop].position)
        crossing = crossing or any(
            kinds[frozenset((x, y))] == "marked_crossing" for x, y in zip(walked, walked[1:]))
    corners = [nodes[start_node_id].position] + [nodes[stop].position for _s, stop, _a in route]
    path = PoolPath(node_ids=tuple(node_ids), length_cm=length_cm,
                    turns=_turn_count(corners), uses_marked_crossing=crossing)
    return path, [PlannedLeg(stop=stop, stop_cm=nodes[stop].position, aim=aim,
                             aim_cm=nodes[aim].position) for _start, stop, aim in route]


def _candidate_scenario(
    pool: ValidatedDeliveryPool,
    *,
    spawn: PoolSpawn,
    pickup: PoolStop,
    dropoff: PoolStop,
    constraints: OrderConstraints,
) -> tuple[PoolPath, PoolPath] | None:
    if pickup.id == dropoff.id:
        return None
    if "pickup" not in pickup.roles or "dropoff" not in dropoff.roles:
        return None
    if constraints.require_different_streets \
            and pickup.street_index == dropoff.street_index:
        return None
    approach = shortest_pool_path(pool, spawn.node_id, pickup.handover_node_id)
    delivery = shortest_pool_path(
        pool, pickup.handover_node_id, dropoff.handover_node_id)
    if approach is None or delivery is None:
        return None
    if not constraints.min_delivery_cm <= delivery.length_cm <= constraints.max_delivery_cm:
        return None
    if delivery.turns < constraints.min_turns:
        return None
    if constraints.require_marked_crossing and not delivery.uses_marked_crossing:
        return None
    return approach, delivery



#: A spawn stands on the pavement in front of its building's door, and the
#: pool records the yaw it was certified at. Where that yaw looks straight
#: into the door from within arm's reach, the front camera sees a door and
#: its glass and nothing of the street -- the seed-2 spawn of the Paris pool
#: started every run staring into a shopfront 0.8 m away. A courier who has
#: just stepped out of a door faces the street, so such a spawn is turned
#: round: the door goes into the rear view, the pavement into the front one.
#: Spawns whose certified yaw already looks along the pavement are left alone.
DOOR_FACING_MAX_DISTANCE_CM = 120.0
DOOR_FACING_MAX_ANGLE_DEG = 35.0


def face_away_from_own_door(pool: ValidatedDeliveryPool, spawn: PoolSpawn) -> PoolSpawn:
    """The spawn, turned round if it was certified staring into its own door."""
    stop = pool.stops_by_id.get("stop-" + spawn.id.removeprefix("spawn-near-"))
    node = pool.nodes_by_id.get(spawn.node_id)
    if stop is None or node is None:
        return spawn
    dx = stop.door_cm[0] - node.x_cm
    dy = stop.door_cm[1] - node.y_cm
    if math.hypot(dx, dy) > DOOR_FACING_MAX_DISTANCE_CM:
        return spawn
    bearing = math.degrees(math.atan2(dy, dx)) % 360.0
    off = abs((spawn.yaw_deg - bearing + 180.0) % 360.0 - 180.0)
    if off > DOOR_FACING_MAX_ANGLE_DEG:
        return spawn
    return replace(spawn, yaw_deg=(spawn.yaw_deg + 180.0) % 360.0)


#: The comparison protocol on the certified Paris pool: a 30-80 m delivery
#: with at least one turn, through a marked crossing. Under it the pool
#: has sixteen directed scenarios -- four orders (14 -> 11, 11 -> 14,
#: 11 -> 16 and 16 -> 11 Rue Oberkampf), each from the four certified
#: spawns -- and ``PROTOCOL_SEEDS`` are the smallest seeds that draw each of
#: them once in random mode (``resolve_delivery_scenario`` picks with
#: ``random.Random(seed)``, so the mapping is a property of the pool and the
#: constraints, not of any run).
#:
#: The split is at the scenario level and is fixed. ``dev`` is the six
#: scenarios that were looked at while the harness and the prompt were
#: built: every 4B run, every relay run and every oracle walk before the
#: split used one of them (seed 6 is the fixed-mode scenario, seed 9 the
#: one the harness was debugged on). ``heldout`` is the ten scenarios that
#: were never run, read or tuned against before the split; a reported
#: number is a number on ``heldout``. Four doors carry all sixteen
#: scenarios, so this does not hold out a door: three of the four orders
#: occur in ``dev`` (only 16 -> 11 does not), and every held-out scenario
#: shares its doors with a dev one. Holding out a street needs a second
#: certified region; until there is one, the held-out set is the fair
#: comparison this pool can offer, and its limit is stated with it.
PROTOCOL_CONSTRAINTS = OrderConstraints(
    min_delivery_cm=3000.0,
    max_delivery_cm=8000.0,
    min_turns=1,
    require_different_streets=False,
    require_marked_crossing=True,
)
PROTOCOL_SEEDS = (0, 1, 2, 3, 5, 6, 7, 9, 12, 14, 15, 16, 17, 18, 23, 31)
PROTOCOL_SPLITS: dict[str, tuple[int, ...]] = {
    "protocol": PROTOCOL_SEEDS,
    "dev": (0, 1, 2, 6, 7, 9),
    "heldout": (3, 5, 12, 14, 15, 16, 17, 18, 23, 31),
}


def protocol_seeds(split: str) -> tuple[int, ...]:
    """The seeds of a named split of the comparison protocol."""
    try:
        return PROTOCOL_SPLITS[split]
    except KeyError:
        raise ValueError(
            f"unknown protocol split {split!r}; one of {sorted(PROTOCOL_SPLITS)}"
        ) from None


def candidate_scenarios(
    pool: ValidatedDeliveryPool,
    *,
    constraints: OrderConstraints,
    spawn_id: str | None = None,
) -> list[tuple[PoolSpawn, PoolStop, PoolStop, PoolPath, PoolPath]]:
    """Every (spawn, pickup, drop-off) the constraints admit, in the order
    random mode draws from: spawns, pickups and drop-offs each sorted by id."""
    spawns = pool.spawns_by_id
    if spawn_id is not None and spawn_id not in spawns:
        raise ValueError(f"unknown certified spawn id: {spawn_id}")
    candidate_spawns = [spawns[spawn_id]] if spawn_id else list(pool.spawns)
    candidates = []
    for spawn in sorted(candidate_spawns, key=lambda row: row.id):
        for pickup in sorted(pool.stops, key=lambda row: row.id):
            for dropoff in sorted(pool.stops, key=lambda row: row.id):
                routed = _candidate_scenario(
                    pool, spawn=spawn, pickup=pickup, dropoff=dropoff,
                    constraints=constraints)
                if routed is not None:
                    candidates.append((spawn, pickup, dropoff, *routed))
    return candidates


def scenario_identity(
    pool: ValidatedDeliveryPool,
    *,
    spawn: PoolSpawn,
    pickup: PoolStop,
    dropoff: PoolStop,
    constraints: OrderConstraints,
) -> str:
    """The scenario id: the pool's bytes, the three certified ids and the
    constraints, hashed. It changes when the pool is re-pinned; the seed ->
    order mapping does not."""
    identity = "\0".join((
        pool.sha256, spawn.id, pickup.id, dropoff.id,
        f"{constraints.min_delivery_cm:.9f}",
        f"{constraints.max_delivery_cm:.9f}",
        str(constraints.min_turns),
        str(constraints.require_different_streets),
        str(constraints.require_marked_crossing),
    ))
    return "scenario-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def resolve_delivery_scenario(
    pool: ValidatedDeliveryPool,
    *,
    mode: str,
    seed: int,
    constraints: OrderConstraints,
    spawn_id: str | None = None,
    pickup_id: str | None = None,
    dropoff_id: str | None = None,
) -> ResolvedDeliveryScenario:
    """Resolve a random or explicitly named order through one validation path."""

    if mode not in ORDER_MODES:
        raise ValueError(f"order mode must be one of {ORDER_MODES}, not {mode!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("order seed must be an integer")
    spawns = pool.spawns_by_id
    stops = pool.stops_by_id
    if spawn_id is not None and spawn_id not in spawns:
        raise ValueError(f"unknown certified spawn id: {spawn_id}")

    if mode == "fixed":
        if not pickup_id or not dropoff_id:
            raise ValueError("fixed order mode requires pickup_id and dropoff_id")
        if pickup_id not in stops:
            raise ValueError(f"unknown certified pickup id: {pickup_id}")
        if dropoff_id not in stops:
            raise ValueError(f"unknown certified drop-off id: {dropoff_id}")
        spawn = spawns[spawn_id] if spawn_id else sorted(pool.spawns, key=lambda row: row.id)[0]
        pickup, dropoff = stops[pickup_id], stops[dropoff_id]
        routed = _candidate_scenario(
            pool, spawn=spawn, pickup=pickup, dropoff=dropoff,
            constraints=constraints)
        if routed is None:
            raise ValueError(
                "fixed order does not satisfy the certified route constraints")
        candidates = [(spawn, pickup, dropoff, *routed)]
    else:
        if pickup_id is not None or dropoff_id is not None:
            raise ValueError("random order mode does not accept fixed pickup/drop-off ids")
        candidates = candidate_scenarios(
            pool, constraints=constraints, spawn_id=spawn_id)
        if not candidates:
            raise ValueError("no certified order pair satisfies the requested constraints")

    selected = candidates[0] if mode == "fixed" else random.Random(seed).choice(candidates)
    spawn, pickup, dropoff, approach, delivery = selected
    spawn = face_away_from_own_door(pool, spawn)
    scenario_id = scenario_identity(
        pool, spawn=spawn, pickup=pickup, dropoff=dropoff, constraints=constraints)
    return ResolvedDeliveryScenario(
        mode=mode,
        seed=seed,
        scenario_id=scenario_id,
        candidate_count=len(candidates),
        spawn=spawn,
        pickup=pickup,
        dropoff=dropoff,
        approach=approach,
        delivery=delivery,
        constraints=constraints,
        requested_spawn_id=spawn_id,
        requested_pickup_id=pickup_id,
        requested_dropoff_id=dropoff_id,
    )


def build_validated_pool_network(
    network: RoadNetwork, pool: ValidatedDeliveryPool,
) -> RoadNetwork:
    """Build a pedestrian-only graph with CityCore as semantic metadata.

    The compiled CityCore node/edge graph follows carriageway centrelines.  It
    is useful for stable address numbers, street names and the phone's grey
    basemap, but it is never copied into the routing graph.  Every neighbour in
    the returned network therefore comes from the UE-certified pool.
    """

    compiled_by_building: dict[str, Address] = {}
    for address in network.addresses:
        if address.building_id in compiled_by_building:
            raise ValueError(
                "compiled city has duplicate addresses for building "
                f"{address.building_id!r}")
        compiled_by_building[address.building_id] = address
    for stop in pool.stops:
        compiled = compiled_by_building.get(stop.building_id)
        if compiled is None:
            raise ValueError(
                f"pool stop {stop.id!r} has no compiled city address")
        # V2 entrance geometry comes from the live UE *_Entrance* mesh catalog.
        # CityCore's ``Address.door`` is still the legacy bbox/yaw estimate and
        # is intentionally not an authority for physical geometry.  Retain the
        # equality check only for V1 pools, whose contract was authored from
        # that estimate; both versions continue to require exact address/street
        # semantics.
        compiled_door_mismatch = (
            pool.schema == LEGACY_POOL_SCHEMA
            and math.dist(compiled.door, stop.door_cm) > 0.1
        )
        if (compiled.street_index != stop.street_index
                or compiled.street_name != stop.street_name
                or compiled.number != stop.number
                or compiled.poi_type != stop.poi_type
                or compiled_door_mismatch):
            raise ValueError(
                f"pool stop {stop.id!r} does not match its compiled city address")
    out = RoadNetwork(
        map_name=network.map_name,
        streets=copy.deepcopy(network.streets),
        buildings=copy.deepcopy(network.buildings),
        notes=copy.deepcopy(network.notes),
    )
    for row in pool.nodes:
        if not 0 <= row.street_index < len(out.streets):
            raise ValueError(
                f"pool node {row.id!r} has unavailable street index {row.street_index}")
        out.nodes[row.id] = StreetNode(
            id=row.id,
            x_cm=row.x_cm,
            y_cm=row.y_cm,
            street_index=row.street_index,
            arc_cm=0.0,
        )
    for edge in pool.edges:
        out.nodes[edge.a].neighbours.add(edge.b)
        out.nodes[edge.b].neighbours.add(edge.a)
    addresses: list[Address] = []
    for stop in pool.stops:
        node = out.nodes[stop.handover_node_id]
        actual_street = out.streets[stop.street_index].name
        if actual_street != stop.street_name:
            raise ValueError(
                f"pool stop {stop.id!r} names {stop.street_name!r}, but street "
                f"index {stop.street_index} is {actual_street!r}")
        addresses.append(Address(
            building_id=stop.building_id,
            street_index=stop.street_index,
            street_name=stop.street_name,
            number=stop.number,
            arc_cm=node.arc_cm,
            offset_cm=math.dist(stop.door_cm, node.position),
            side="certified",
            door=stop.door_cm,
            nearest_node=stop.handover_node_id,
            poi_type=stop.poi_type,
            kerb_node=stop.handover_node_id,
            kerb=node.position,
        ))
    # The UE-backed episode may route only to stops certified for this physical
    # region.  Retaining the other 477 addresses would let navigate() target a
    # carriageway point outside the active NavMesh and would silently reintroduce
    # the problem this pool exists to remove.
    out.addresses = addresses
    out.notes.append(
        f"trusted pedestrian pool {pool.profile}: {len(pool.stops)} stops, "
        f"{len(pool.nodes)} UE-derived nodes, sha256 {pool.sha256}; CityCore "
        "connectivity discarded (address/street semantics only)")
    return out


def _stop_report(pool: ValidatedDeliveryPool, stop: PoolStop) -> dict[str, Any]:
    anchor = pool.nodes_by_id[stop.handover_node_id].position
    report = {
        "id": stop.id,
        "address": stop.text,
        "building_id": stop.building_id,
        "door_cm": list(stop.door_cm),
        "handover_cm": list(anchor),
        "door_anchor_gap_cm": math.dist(stop.door_cm, anchor),
        "route_node_id": stop.handover_node_id,
        "entrance_source": stop.entrance_source,
        "surface": stop.surface,
        "max_navmesh_adjustment_cm": stop.max_navmesh_adjustment_cm,
    }
    if stop.entrance_asset_id is not None:
        report.update({
            "entrance_asset_id": stop.entrance_asset_id,
            "entrance_face": stop.entrance_face,
            "entrance_static_mesh_path": stop.entrance_static_mesh_path,
        })
    return report


def scenario_report(
    pool: ValidatedDeliveryPool, scenario: ResolvedDeliveryScenario,
) -> dict[str, Any]:
    node = pool.nodes_by_id[scenario.spawn.node_id]
    pool_report: dict[str, Any] = {
        "schema": pool.schema,
        "id": pool.pool_id,
        "version": pool.version,
        "sha256": pool.sha256,
        "path": str(pool.path),
        "scene": pool.scene,
        "region": pool.region.name,
    }
    if pool.pedestrian_graph is not None:
        pool_report["pedestrian_graph"] = {
            "audit_sha256": pool.pedestrian_graph.audit_sha256,
            "stable_recast_topology_sha256":
                pool.pedestrian_graph.stable_recast_topology_sha256,
            "connectivity_source": pool.pedestrian_graph.connectivity_source,
            "semantic_source": pool.pedestrian_graph.semantic_source,
            "crosswalk_asset_prefix":
                pool.pedestrian_graph.crosswalk_asset_prefix,
        }
    return {
        "mode": scenario.mode,
        "seed": scenario.seed,
        "scenario_id": scenario.scenario_id,
        "candidate_count": scenario.candidate_count,
        "request": {
            "spawn_id": scenario.requested_spawn_id,
            "pickup_id": scenario.requested_pickup_id,
            "dropoff_id": scenario.requested_dropoff_id,
        },
        "pool": pool_report,
        "constraints": scenario.constraints.to_report(),
        "spawn": {
            "id": scenario.spawn.id,
            "route_node_id": scenario.spawn.node_id,
            "position_cm": [node.x_cm, node.y_cm, scenario.spawn.z_cm],
            "yaw_deg": scenario.spawn.yaw_deg,
            "max_navmesh_adjustment_cm":
                scenario.spawn.max_navmesh_adjustment_cm,
        },
        "pickup": _stop_report(pool, scenario.pickup),
        "dropoff": _stop_report(pool, scenario.dropoff),
    }


def selected_route_report(
    network: RoadNetwork,
    pool: ValidatedDeliveryPool,
    scenario: ResolvedDeliveryScenario,
) -> dict[str, Any]:
    node_ids = [*scenario.approach.node_ids, *scenario.delivery.node_ids[1:]]
    rows = []
    for node_id in node_ids:
        node = network.nodes[node_id]
        if node_id == scenario.spawn.node_id:
            role = "spawn"
        elif node_id == scenario.pickup.handover_node_id:
            role = "pickup"
        elif node_id == scenario.dropoff.handover_node_id:
            role = "dropoff"
        else:
            role = pool.nodes_by_id[node_id].role
        rows.append({
            "id": node_id,
            "x_cm": node.x_cm,
            "y_cm": node.y_cm,
            "street": network.streets[node.street_index].name,
            "role": role,
        })
    return {
        "profile": pool.profile,
        "waypoints": rows,
        "pickup_waypoint_id": scenario.pickup.handover_node_id,
        "dropoff_waypoint_id": scenario.dropoff.handover_node_id,
        "planned_total_cm": scenario.approach.length_cm + scenario.delivery.length_cm,
        "planned_delivery_cm": scenario.delivery.length_cm,
    }
