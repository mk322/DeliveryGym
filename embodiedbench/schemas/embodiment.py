"""Embodiment boundary (design plan §9.6).

The high-level agent is embodiment-independent: it emits a
``NavigationRequest``, and an adapter turns that into locomotion. design plan §26
frames the point — a future robot, avatar, vehicle, or controller model should
attach without retraining the high-level harness.

``ControllerResult`` carries everything design plan §9.6 lists, and design plan §9.4's
audit requirement drives the rest: every accepted request stores its original
image coordinates, the source of its distance, the continuous targets in camera,
agent, and UE frames, the projected target, the quantized target when one
applies, and the controller trajectory. That is what makes all three navigation
modes convertible to and auditable in UE.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from embodiedbench.schemas.base import CapabilityError, SchemaModel
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.geometry import Pose, Vec3
from embodiedbench.schemas.runtime import ControllerOutcomeCode, ImagePoint
from embodiedbench.schemas.world import LocomotionMode, RelPath, StableId

ControllerOutcome = ControllerOutcomeCode


class DistanceSource(str, Enum):
    """Where a navigation request's range came from (design plan §9.2, 9.3)."""

    MODEL_PREDICTION = "model_prediction"
    EXTERNAL_DEPTH = "external_depth"
    GRAPH_EDGE = "graph_edge"
    UE_GEOMETRY_TRACE = "ue_geometry_trace"


class EmbodimentCapabilities(SchemaModel):
    """What a controller can do."""

    SCHEMA_ID = "embodiedbench/embodiment_capabilities"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    profile_id: str = Field(min_length=1)
    locomotion_modes: list[LocomotionMode] = Field(min_length=1)
    navigation_modes: list[NavigationMode] = Field(min_length=1)
    max_range_m: float = Field(gt=0.0)
    max_speed_m_s: float = Field(gt=0.0)
    radius_cm: float = Field(gt=0.0)
    height_cm: float = Field(gt=0.0)
    max_slope_deg: float = Field(default=30.0, ge=0.0, lt=90.0)
    supports_emergency_stop: bool = True
    # design plan §9.6 reserves manipulation behind a separate capability rather than
    # letting fake pick/place primitives into the navigation interface.
    supports_skill_requests: bool = False

    def require(self, mode: NavigationMode) -> None:
        if mode not in self.navigation_modes:
            raise CapabilityError(
                f"embodiment {self.profile_id!r} does not support {mode.value}"
            )


class EmbodimentProfile(SchemaModel):
    """Robot/avatar dimensions, control modes, and sensors (design plan §2.1.4).

    Distinct from a courier profile, which is task configuration: shift,
    equipment, battery, and carrying capacity belong to the Delivery task.
    """

    SCHEMA_ID = "embodiedbench/embodiment_profile"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    profile_id: str = Field(min_length=1)
    version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    capabilities: EmbodimentCapabilities
    sensor_channels: list[str] = Field(default_factory=lambda: ["rgb"])
    description: str = ""

    @model_validator(mode="after")
    def _ids_agree(self) -> "EmbodimentProfile":
        if self.capabilities.profile_id != self.profile_id:
            raise ValueError("profile id and capability profile id must match")
        return self


class NavigationRequest(SchemaModel):
    """A resolved, executable navigation target (design plan §9)."""

    SCHEMA_ID = "embodiedbench/navigation_request"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    request_id: StableId
    mode: NavigationMode
    distance_source: DistanceSource
    # The original model output, kept verbatim for audit (design plan §9.4).
    source_image_point: ImagePoint | None = None
    target_node: StableId | None = None
    # The continuous chain, one entry per frame it passed through.
    target_camera: Vec3 | None = None
    target_agent: Vec3 | None = None
    target_world: Vec3 | None = None
    projected_target: Vec3 | None = None
    quantized_target: Pose | None = None
    # Live Pixel Goal audit fields.  The two IDs bind the action to the exact
    # RGB capture; the two positions distinguish grounding from NavMesh
    # validation without exposing either value to the policy.
    camera_snapshot_id: StableId | None = None
    camera_intrinsics_id: StableId | None = None
    raw_world_hit: Vec3 | None = None
    validated_navigation_target: Vec3 | None = None
    navmesh_adjustment_cm: float | None = Field(default=None, ge=0.0)
    # The exact UE controller polyline planned for the selected visible point.
    # These fields stay audit-only: neither coordinates nor detour metrics are
    # returned to the policy.
    controller_path_points: list[Vec3] = Field(default_factory=list)
    controller_path_length_cm: float | None = Field(default=None, ge=0.0)
    controller_path_direct_cm: float | None = Field(default=None, ge=0.0)
    controller_path_stretch_ratio: float | None = Field(default=None, ge=0.0)
    max_range_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _mode_matches_inputs(self) -> "NavigationRequest":
        if self.mode is NavigationMode.NAV_WAYPOINT:
            if self.target_node is None:
                raise ValueError("nav_waypoint requests must resolve to a stable node id")
            if self.distance_source is not DistanceSource.GRAPH_EDGE:
                raise ValueError("nav_waypoint range comes from the graph edge")
        else:
            if self.source_image_point is None:
                raise ValueError(f"{self.mode.value} requests must record the source image point")
            if self.distance_source is DistanceSource.GRAPH_EDGE:
                raise ValueError(f"{self.mode.value} range cannot come from a graph edge")
        if self.mode is NavigationMode.NAV_POINT_3D:
            if self.distance_source is not DistanceSource.MODEL_PREDICTION:
                raise ValueError("nav_point_3d range is the model's prediction")
        if self.mode is NavigationMode.NAV_POINT_2D_DEPTH:
            if self.distance_source is not DistanceSource.EXTERNAL_DEPTH:
                raise ValueError("nav_point_2d_depth range comes from external metric depth")
        if self.mode is NavigationMode.NAV_PIXEL_GOAL:
            if self.distance_source is not DistanceSource.UE_GEOMETRY_TRACE:
                raise ValueError("nav_pixel_goal range comes from an internal UE geometry trace")
            if self.source_image_point is None or self.source_image_point.distance_m is not None:
                raise ValueError("nav_pixel_goal must record only its normalized source image point")
            if not self.camera_snapshot_id or not self.camera_intrinsics_id:
                raise ValueError("nav_pixel_goal requires an exact camera snapshot and intrinsics id")
            if self.raw_world_hit is None or self.target_world is None:
                raise ValueError("nav_pixel_goal must record its raw world hit")
            if self.validated_navigation_target is None or self.projected_target is None:
                raise ValueError("nav_pixel_goal must record its validated navigation target")
            if self.navmesh_adjustment_cm is None:
                raise ValueError("nav_pixel_goal must record its NavMesh adjustment")
            path_metrics = (
                self.controller_path_length_cm,
                self.controller_path_direct_cm,
                self.controller_path_stretch_ratio,
            )
            if self.controller_path_points and any(
                    value is None for value in path_metrics):
                raise ValueError(
                    "nav_pixel_goal controller path points require all metrics")
            if not self.controller_path_points and any(
                    value is not None for value in path_metrics):
                raise ValueError(
                    "nav_pixel_goal controller path metrics require path points")
            if self.quantized_target is not None:
                raise ValueError("nav_pixel_goal targets must not be quantized")
        return self


class PoseEstimate(SchemaModel):
    pose: Pose
    covariance_trace: float | None = Field(default=None, ge=0.0)
    source: str = "simulator_ground_truth"


class ControllerResult(SchemaModel):
    """design plan §9.6's controller result."""

    SCHEMA_ID = "embodiedbench/controller_result"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    request_id: StableId
    outcome: ControllerOutcome
    requested_target: Vec3 | None = None
    accepted_target: Vec3 | None = None
    final_pose: Pose | None = None
    # Feet, rather than capsule centre, are the metric endpoint for Pixel Goal.
    final_feet_position: Vec3 | None = None
    controller_result: str | None = None
    execution_error_planar_m: float | None = Field(default=None, ge=0.0)
    execution_error_3d_m: float | None = Field(default=None, ge=0.0)
    trajectory_ref: RelPath | None = None
    elapsed_sim_s: float = Field(default=0.0, ge=0.0)
    distance_travelled_cm: float = Field(default=0.0, ge=0.0)
    energy_used: float = Field(default=0.0, ge=0.0)
    collisions: int = Field(default=0, ge=0)
    violations: list[str] = Field(default_factory=list)
    failure_reason: str = ""
    # design plan §9.2: always log this for nav_point_3d, even when execution uses
    # the model distance -- otherwise a model emitting a constant distance looks
    # competent after snapping.
    distance_model_m: float | None = Field(default=None, gt=0.0)
    distance_external_m: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _outcome_is_coherent(self) -> "ControllerResult":
        if self.outcome is ControllerOutcome.ACCEPTED:
            if self.final_pose is None:
                raise ValueError("an accepted controller result must report a final pose")
            if self.failure_reason:
                raise ValueError("an accepted controller result cannot carry a failure reason")
        elif not self.failure_reason:
            raise ValueError(f"{self.outcome.value} must explain itself in failure_reason")
        return self

    def distance_error_m(self) -> float | None:
        """abs(model - external) when both are known (design plan §9.2)."""
        if self.distance_model_m is None or self.distance_external_m is None:
            return None
        return abs(self.distance_model_m - self.distance_external_m)
