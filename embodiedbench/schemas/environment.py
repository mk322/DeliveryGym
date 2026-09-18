"""Task ``OverlaySpec`` and derived ``EnvironmentBundle`` (design plan §5.2, 5.3).

An overlay is a declarative patch, never an edited copy of the source map. Two
of design plan §5.2's requirements are enforced structurally: every placement records
its source asset, transform, rule, and seed, and every overlay must carry a
removal manifest so the source project stays unchanged.

``EnvironmentBundle`` declares compatibility rather than implying it (the design plan
5.3): a certificate is keyed by ``(task plugin, embodiment profile, navigation
mode, runtime)``, matching design plan §6.2's "not a single vague map grade".
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from embodiedbench.schemas.base import SchemaModel
from embodiedbench.schemas.geometry import CameraIntrinsics, Vec3
from embodiedbench.schemas.world import RelPath, Sha256, StableId


class CertificationGrade(str, Enum):
    """design plan §6.2's grades."""

    A = "A"  # eligible for public benchmark evaluation
    B = "B"  # training and development; limitations declared
    C = "C"  # visualization/research only
    FAIL = "fail"


class NavigationMode(str, Enum):
    """High-level navigation modes.

    ``nav_pixel_goal`` is an additive, live-UE-only experimental mode.  Unlike
    the two original point modes, its action semantics do not include metric
    depth or a pose lattice: UE resolves the selected visible pixel against the
    exact RGB camera snapshot and executes the continuous target internally.
    """

    NAV_WAYPOINT = "nav_waypoint"
    NAV_POINT_3D = "nav_point_3d"
    NAV_POINT_2D_DEPTH = "nav_point_2d_depth"
    NAV_PIXEL_GOAL = "nav_pixel_goal"


class PointExecutionVariant(str, Enum):
    """design plan §9.2's two declared execution variants."""

    SNAP = "3d_snap"
    STRICT = "3d_strict"


class AffordanceRequirement(SchemaModel):
    min: int = Field(ge=0)
    max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "AffordanceRequirement":
        if self.max is not None and self.max < self.min:
            raise ValueError("max must be at least min")
        return self


class OverlayPlacement(SchemaModel):
    """One materialized task asset, with the provenance design plan §5.2 requires."""

    placement_id: StableId
    affordance: str = Field(min_length=1)
    position: Vec3
    yaw_deg: float = 0.0
    # "Reuse valid authored assets where possible" (design plan §5.2.1) means a
    # placement may bind an existing entity instead of spawning; exactly one of
    # the two must be true, or the removal manifest cannot be built correctly.
    reused_entity_id: StableId | None = None
    spawned_asset_path: str = ""
    data_layer: str = ""
    rule: str = Field(min_length=1)
    seed: int
    nearest_node: StableId | None = None

    @model_validator(mode="after")
    def _reuse_or_spawn(self) -> "OverlayPlacement":
        reused = self.reused_entity_id is not None
        spawned = bool(self.spawned_asset_path)
        if reused == spawned:
            raise ValueError(
                f"placement {self.placement_id} must either reuse an entity or spawn an asset, "
                "not both and not neither"
            )
        if spawned and not self.data_layer:
            # design plan §5.2.3: spawn or modify only through a named Data Layer.
            raise ValueError(
                f"placement {self.placement_id} spawns an asset but names no UE Data Layer"
            )
        return self


class RemovalManifest(SchemaModel):
    """design plan §5.2.6: an inverse manifest so the source project is unchanged."""

    data_layers: list[str] = Field(default_factory=list)
    spawned_actor_ids: list[str] = Field(default_factory=list)
    source_umap_sha256: Sha256 | None = None
    verified_restores_source_hash: bool = False


class OverlaySpec(SchemaModel):
    """A declarative task patch over a base world (design plan §5.2)."""

    SCHEMA_ID = "embodiedbench/overlay_spec"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    overlay_id: StableId
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    task_plugin: str = Field(min_length=1)
    base_world_id: StableId
    base_world_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    requires_affordances: dict[str, AffordanceRequirement] = Field(default_factory=dict)
    requires_routes: dict[str, AffordanceRequirement] = Field(default_factory=dict)
    placement_policy: str = Field(default="reuse_then_spawn")
    seed: int = 0
    asset_paths: dict[str, str] = Field(default_factory=dict)
    placements: list[OverlayPlacement] = Field(default_factory=list)
    removal: RemovalManifest = Field(default_factory=RemovalManifest)

    @model_validator(mode="after")
    def _placements_satisfy_requirements(self) -> "OverlaySpec":
        ids = [p.placement_id for p in self.placements]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate placement ids")
        counts: dict[str, int] = {}
        for placement in self.placements:
            counts[placement.affordance] = counts.get(placement.affordance, 0) + 1
        for affordance, requirement in self.requires_affordances.items():
            found = counts.get(affordance, 0)
            if found < requirement.min:
                raise ValueError(
                    f"overlay {self.overlay_id} requires at least {requirement.min} "
                    f"{affordance!r} but places {found}"
                )
            if requirement.max is not None and found > requirement.max:
                raise ValueError(
                    f"overlay {self.overlay_id} allows at most {requirement.max} "
                    f"{affordance!r} but places {found}"
                )
        return self

    def deficit(self, available: dict[str, int]) -> dict[str, int]:
        """How many of each affordance are still missing (design plan §6.1 P4)."""
        out = {}
        for affordance, requirement in self.requires_affordances.items():
            short = requirement.min - available.get(affordance, 0)
            if short > 0:
                out[affordance] = short
        return out


class ObservationManifestEntry(SchemaModel):
    """One baked observation row (design plan §6.1 P5)."""

    key: str = Field(min_length=1)
    node_id: StableId | None = None
    position: Vec3
    yaw_deg: float
    rgb_path: RelPath
    depth_path: RelPath | None = None
    semantic_path: RelPath | None = None
    rgb_sha256: Sha256
    depth_sha256: Sha256 | None = None
    condition: str = "default"
    status: str = Field(default="ok", pattern="^(ok|failed|skipped)$")
    asset_hash: Sha256 | None = None
    overlay_hash: Sha256 | None = None


class Certificate(SchemaModel):
    """A grade for one exact combination, not for a map (design plan §6.2)."""

    SCHEMA_ID = "embodiedbench/certificate"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    environment_id: StableId
    environment_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    task_plugin: str
    task_plugin_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    embodiment_profile: str
    navigation_mode: NavigationMode
    runtime: str = Field(pattern="^(text|cached|live)$")
    grade: CertificationGrade
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    certified_road_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    issued_at: str = ""
    reviewer: str | None = None

    @model_validator(mode="after")
    def _grade_matches_evidence(self) -> "Certificate":
        if self.grade is CertificationGrade.A:
            if self.checks_failed:
                raise ValueError(
                    "grade A requires all checks to pass; failed: " + ", ".join(self.checks_failed)
                )
            if self.limitations:
                raise ValueError("grade A cannot declare limitations")
        if self.grade is CertificationGrade.B and not self.limitations:
            # design plan §6.2: grade B means "known limitations are declared". A
            # grade B with none declared is either an A or an unreviewed guess.
            raise ValueError("grade B must declare its known limitations")
        return self


class SensorSpec(SchemaModel):
    """What a runtime can observe here (design plan §7.1 'declared sensor capability')."""

    channels: list[str] = Field(default_factory=lambda: ["rgb"])
    intrinsics: CameraIntrinsics | None = None
    pose_lattice_spacing_cm: float | None = Field(default=None, gt=0.0)
    pose_lattice_headings: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _lattice_is_all_or_nothing(self) -> "SensorSpec":
        # design plan §9.5 versions lattice resolution and headings in
        # environment.json; half a lattice spec cannot be honoured by a runtime.
        if (self.pose_lattice_spacing_cm is None) != (self.pose_lattice_headings is None):
            raise ValueError("pose lattice needs both a spacing and a heading count")
        return self


class EnvironmentBundle(SchemaModel):
    """Base world + overlay + sensors + certificates (design plan §5.3)."""

    SCHEMA_ID = "embodiedbench/environment_bundle"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    environment_id: StableId
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    base_world_id: StableId
    base_world_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    base_world_sha256: Sha256 | None = None
    overlay_id: StableId | None = None
    overlay_version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    supported_embodiment_profiles: list[str] = Field(default_factory=list)
    supported_navigation_modes: list[NavigationMode] = Field(default_factory=list)
    supported_runtimes: list[str] = Field(default_factory=lambda: ["text"])
    sensors: SensorSpec = Field(default_factory=SensorSpec)
    certificates: list[Certificate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _declares_rather_than_implies(self) -> "EnvironmentBundle":
        for runtime in self.supported_runtimes:
            if runtime not in ("text", "cached", "live"):
                raise ValueError(f"unknown runtime {runtime!r}")
        # A point mode needs a pose lattice in every runtime, because the design plan
        # 9.5 forbids snapping in cached mode only: that would make two
        # different transition systems.
        point_modes = {NavigationMode.NAV_POINT_3D, NavigationMode.NAV_POINT_2D_DEPTH}
        if point_modes & set(self.supported_navigation_modes):
            if self.sensors.pose_lattice_spacing_cm is None:
                raise ValueError(
                    "point navigation modes require a declared pose lattice "
                    "(design plan §9.5); otherwise cached and live runtimes would differ"
                )
        if NavigationMode.NAV_PIXEL_GOAL in self.supported_navigation_modes:
            non_live = [runtime for runtime in self.supported_runtimes if runtime != "live"]
            if non_live or "live" not in self.supported_runtimes:
                raise ValueError(
                    "nav_pixel_goal requires an environment bundle dedicated to the live runtime"
                )
        for certificate in self.certificates:
            if certificate.navigation_mode not in self.supported_navigation_modes:
                raise ValueError(
                    f"certificate for {certificate.navigation_mode.value} on an environment that "
                    "does not declare that navigation mode"
                )
            if certificate.runtime not in self.supported_runtimes:
                raise ValueError(
                    f"certificate for runtime {certificate.runtime!r} which is not supported"
                )
            if (
                certificate.navigation_mode is NavigationMode.NAV_PIXEL_GOAL
                and certificate.runtime != "live"
            ):
                raise ValueError("nav_pixel_goal certificates require the live runtime")
        return self

    def certificate_for(
        self, *, task_plugin: str, embodiment_profile: str, navigation_mode: NavigationMode, runtime: str
    ) -> Certificate | None:
        for certificate in self.certificates:
            if (
                certificate.task_plugin == task_plugin
                and certificate.embodiment_profile == embodiment_profile
                and certificate.navigation_mode is navigation_mode
                and certificate.runtime == runtime
            ):
                return certificate
        return None
