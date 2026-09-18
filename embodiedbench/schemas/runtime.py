"""Runtime contract: observations, actions, events, and step results.

design plan §7 fixes the shape. Three things are load-bearing and enforced here.

**``terminated`` and ``truncated`` are separate, with separate causes.**
design plan §7 asks for Gymnasium semantics "including the distinction between task
termination and budget truncation", and the design plan M1 requires them to have
separately tested causes. A ``StepResult`` cannot be both, and whichever is set
must carry a reason drawn from the matching half of ``TerminationReason``.

**Privileged state never reaches the agent.** ``StepResult`` carries only a
``privileged_state_ref`` (design plan §7), and ``Observation`` has no field that can
hold it. design plan §12.6 and 13 both make leak tests a release gate, so the type
system is the first line rather than the only one.

**Action outcomes are typed.** design plan §9.4 enumerates the outcomes a point
request can have, and distinguishes ``no_ground_hit`` from hitting a wall. Those
are enum members, not strings, so a runtime cannot invent a synonym that an
evaluator then fails to recognize.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import Field, model_validator

from embodiedbench.schemas.base import CapabilityError, SchemaModel
from embodiedbench.schemas.environment import NavigationMode, PointExecutionVariant
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.world import RelPath, Sha256, StableId


class RuntimeMode(str, Enum):
    """design plan §7.1: there are three runtime modes, not two."""

    TEXT = "text"
    CACHED = "cached"
    LIVE = "live"


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────


class NavWaypointAction(SchemaModel):
    """design plan §9.1, the production path for DeliveryBench v1."""

    type: Literal["nav_waypoint"] = "nav_waypoint"
    target_node: StableId | None = None
    # Set-of-Marks: a visual policy answers with a mark id, which the runtime
    # resolves to a stable node id *before recording the action* (design plan §9.1).
    target_mark: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_target(self) -> "NavWaypointAction":
        if (self.target_node is None) == (self.target_mark is None):
            raise ValueError("give exactly one of target_node or target_mark")
        return self


class ImagePoint(SchemaModel):
    u_norm: float = Field(ge=0.0, le=1.0)
    v_norm: float = Field(ge=0.0, le=1.0)
    distance_m: float | None = Field(default=None, gt=0.0)


class NavPoint3DAction(SchemaModel):
    """design plan §9.2: the model predicts the image point *and* the metric range."""

    type: Literal["nav_point_3d"] = "nav_point_3d"
    frame: FrameName = FrameName.CAMERA
    target: ImagePoint
    variant: PointExecutionVariant = PointExecutionVariant.SNAP

    @model_validator(mode="after")
    def _distance_required(self) -> "NavPoint3DAction":
        if self.target.distance_m is None:
            raise ValueError("nav_point_3d requires a model-predicted distance_m")
        if self.frame is not FrameName.CAMERA:
            raise ValueError("nav_point_3d targets are expressed in the camera frame")
        return self


class NavPoint2DDepthAction(SchemaModel):
    """design plan §9.3: the model picks where; an external depth provider picks how far."""

    type: Literal["nav_point_2d_depth"] = "nav_point_2d_depth"
    frame: FrameName = FrameName.CAMERA
    target: ImagePoint

    @model_validator(mode="after")
    def _no_model_distance(self) -> "NavPoint2DDepthAction":
        if self.target.distance_m is not None:
            # Accepting a distance here would silently turn this into
            # nav_point_3d and invalidate the comparison design plan §9.7.6 wants.
            raise ValueError(
                "nav_point_2d_depth must not carry a model distance; range comes from the "
                "external metric-depth provider"
            )
        if self.frame is not FrameName.CAMERA:
            raise ValueError("nav_point_2d_depth targets are expressed in the camera frame")
        return self


class NavPixelGoalAction(SchemaModel):
    """A visible walkable-ground point selected in the current RGB frame.

    The policy supplies only normalized image coordinates.  The live runtime
    binds them to the exact camera snapshot associated with that observation;
    depth, UE geometry, NavMesh state, and world coordinates are deliberately
    absent from the policy-facing action.
    """

    type: Literal["nav_pixel_goal"] = "nav_pixel_goal"
    frame: FrameName = FrameName.CAMERA
    target: ImagePoint

    @model_validator(mode="after")
    def _normalized_point_only(self) -> "NavPixelGoalAction":
        if self.target.distance_m is not None:
            raise ValueError("nav_pixel_goal must not carry distance_m")
        if self.frame is not FrameName.CAMERA:
            raise ValueError("nav_pixel_goal targets are expressed in the camera frame")
        return self


class TaskAction(SchemaModel):
    """A non-navigation task action, e.g. ACCEPT_ORDER or PICKUP."""

    type: Literal["task"] = "task"
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


NavigationAction = Annotated[
    Union[NavWaypointAction, NavPoint3DAction, NavPoint2DDepthAction, NavPixelGoalAction],
    Field(discriminator="type"),
]
AnyAction = Annotated[
    Union[
        NavWaypointAction,
        NavPoint3DAction,
        NavPoint2DDepthAction,
        NavPixelGoalAction,
        TaskAction,
    ],
    Field(discriminator="type"),
]

_NAV_MODE_FOR_ACTION = {
    "nav_waypoint": NavigationMode.NAV_WAYPOINT,
    "nav_point_3d": NavigationMode.NAV_POINT_3D,
    "nav_point_2d_depth": NavigationMode.NAV_POINT_2D_DEPTH,
    "nav_pixel_goal": NavigationMode.NAV_PIXEL_GOAL,
}


class ActionEnvelope(SchemaModel):
    """One agent decision, as submitted to the runtime.

    ``subgoal`` exists for interpretability, and design plan §10.2 requires that task
    execution depend only on the validated action field — so nothing in the
    runtime reads it.
    """

    SCHEMA_ID = "embodiedbench/action_envelope"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    episode_id: StableId
    step_index: int = Field(ge=0)
    action: AnyAction
    subgoal: str | None = None
    idempotency_key: str | None = None

    def navigation_mode(self) -> NavigationMode | None:
        return _NAV_MODE_FOR_ACTION.get(self.action.type)


class ActionStatus(str, Enum):
    """design plan §7: accepted / rejected / failed, plus controller status."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class ControllerOutcomeCode(str, Enum):
    """design plan §9.4's typed outcomes."""

    ACCEPTED = "accepted"
    CLIPPED = "clipped"
    NO_GROUND_HIT = "no_ground_hit"
    NOT_TRAVERSABLE = "not_traversable"
    OUT_OF_RANGE = "out_of_range"
    NO_CONTROLLER_PATH = "no_controller_path"
    LOW_DEPTH_CONFIDENCE = "low_depth_confidence"
    POSE_LATTICE_UNAVAILABLE = "pose_lattice_unavailable"
    CAMERA_SNAPSHOT_UNAVAILABLE = "camera_snapshot_unavailable"
    HIT_NOT_WALKABLE_GROUND = "hit_not_walkable_ground"
    NAVMESH_ADJUSTMENT_EXCEEDED = "navmesh_adjustment_exceeded"
    CONTROLLER_FAILED = "controller_failed"
    EXECUTION_TIMEOUT = "execution_timeout"


class ActionResult(SchemaModel):
    """What the runtime did with an action."""

    status: ActionStatus
    error_code: str | None = None
    message: str = ""
    controller_outcome: ControllerOutcomeCode | None = None
    resolved_target_node: StableId | None = None

    @model_validator(mode="after")
    def _failures_explain_themselves(self) -> "ActionResult":
        if self.status is not ActionStatus.ACCEPTED and not self.error_code:
            # the design plan M6 requires every invalid action to receive typed feedback
            # on the next turn; an unexplained rejection cannot produce that.
            raise ValueError(f"{self.status.value} action result must carry an error_code")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Observations
# ─────────────────────────────────────────────────────────────────────────────


class MediaRef(SchemaModel):
    """A reference to an image, by relative path and content hash (design plan §10.4)."""

    channel: str = Field(min_length=1)
    path: RelPath
    sha256: Sha256
    width_px: int | None = Field(default=None, gt=0)
    height_px: int | None = Field(default=None, gt=0)


class MarkCandidate(SchemaModel):
    """One numbered Set-of-Marks candidate offered to a visual policy."""

    mark: int = Field(ge=0)
    node_id: StableId
    bearing_deg: float
    distance_m: float = Field(ge=0.0)


class Observation(SchemaModel):
    """What the agent sees. Contains no privileged state, by construction."""

    SCHEMA_ID = "embodiedbench/observation"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    episode_id: StableId
    step_index: int = Field(ge=0)
    text: str = ""
    media: list[MediaRef] = Field(default_factory=list)
    marks: list[MarkCandidate] = Field(default_factory=list)
    # The agent's own pose is observable; the world's privileged state is not.
    agent_pose: Pose | None = None
    available_actions: list[str] = Field(default_factory=list)
    budgets_remaining: dict[str, float] = Field(default_factory=dict)
    last_action_result: ActionResult | None = None

    @model_validator(mode="after")
    def _marks_are_dense_and_unique(self) -> "Observation":
        marks = [m.mark for m in self.marks]
        if len(marks) != len(set(marks)):
            raise ValueError("duplicate Set-of-Marks ids in one observation")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Events and step results
# ─────────────────────────────────────────────────────────────────────────────


class Event(SchemaModel):
    """A structured environment event (design plan §7, 13.2)."""

    SCHEMA_ID = "embodiedbench/event"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    event_id: StableId
    episode_id: StableId
    step_index: int = Field(ge=0)
    kind: str = Field(min_length=1)
    sim_time_s: float = Field(ge=0.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    # design plan §12.6 requires leak tests; marking an event privileged lets the
    # public projection filter without guessing from the kind string.
    privileged: bool = False


class TerminationReason(str, Enum):
    """Causes, split by which flag they set.

    the design plan M1 requires terminated and truncated to have separately tested
    causes, so the reason itself declares which it belongs to and a validator
    rejects a mismatch.
    """

    # terminated: the task itself ended
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    UNRECOVERABLE_STATE = "unrecoverable_state"
    # truncated: a budget or the harness stopped it
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    SIM_TIME_BUDGET_EXHAUSTED = "sim_time_budget_exhausted"
    TOOL_CALL_BUDGET_EXHAUSTED = "tool_call_budget_exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"

    @property
    def is_termination(self) -> bool:
        return self in {
            TerminationReason.TASK_SUCCESS,
            TerminationReason.TASK_FAILURE,
            TerminationReason.UNRECOVERABLE_STATE,
        }


class ResetInfo(SchemaModel):
    SCHEMA_ID = "embodiedbench/reset_info"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    episode_id: StableId
    seed: int
    environment_id: StableId
    environment_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    runtime_mode: RuntimeMode
    spawn_pose: Pose | None = None
    privileged_state_ref: str | None = None


class StepResult(SchemaModel):
    """design plan §7's step result."""

    SCHEMA_ID = "embodiedbench/step_result"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    observation: Observation
    reward: float = 0.0
    reward_components: dict[str, float] = Field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    termination_reason: TerminationReason | None = None
    events: list[Event] = Field(default_factory=list)
    action_result: ActionResult
    metrics_delta: dict[str, float] = Field(default_factory=dict)
    privileged_state_ref: str | None = None

    @model_validator(mode="after")
    def _termination_is_coherent(self) -> "StepResult":
        if self.terminated and self.truncated:
            raise ValueError("an episode cannot both terminate and truncate on one step")
        if (self.terminated or self.truncated) and self.termination_reason is None:
            raise ValueError("an ended episode must state why")
        if self.termination_reason is not None:
            if not (self.terminated or self.truncated):
                raise ValueError("a termination reason was given but the episode did not end")
            if self.terminated and not self.termination_reason.is_termination:
                raise ValueError(
                    f"{self.termination_reason.value} is a truncation cause, but terminated=True"
                )
            if self.truncated and self.termination_reason.is_termination:
                raise ValueError(
                    f"{self.termination_reason.value} is a termination cause, but truncated=True"
                )
        if self.reward_components:
            total = sum(self.reward_components.values())
            if abs(total - self.reward) > 1e-6:
                raise ValueError(
                    f"reward {self.reward} does not equal its components' sum {total}"
                )
        return self


class RuntimeCapabilities(SchemaModel):
    """What a runtime advertises (design plan §7)."""

    SCHEMA_ID = "embodiedbench/runtime_capabilities"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    mode: RuntimeMode
    observation_channels: list[str] = Field(default_factory=lambda: ["text"])
    navigation_modes: list[NavigationMode] = Field(default_factory=list)
    # design plan §7: an optional extension, evaluator/harness-only in every v1
    # track, never agent-visible.
    supports_snapshot: bool = False
    max_range_m: float | None = Field(default=None, gt=0.0)
    protocol_version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")

    @model_validator(mode="after")
    def _point_modes_declare_range(self) -> "RuntimeCapabilities":
        # design plan §9.4: point modes must declare max_range_m.
        point_modes = {NavigationMode.NAV_POINT_3D, NavigationMode.NAV_POINT_2D_DEPTH}
        if point_modes & set(self.navigation_modes) and self.max_range_m is None:
            raise ValueError("a runtime offering point navigation must declare max_range_m")
        if (
            NavigationMode.NAV_PIXEL_GOAL in self.navigation_modes
            and self.mode is not RuntimeMode.LIVE
        ):
            raise ValueError("nav_pixel_goal is available only from a live runtime")
        return self

    def require_action(self, envelope: ActionEnvelope) -> None:
        """Raise ``CapabilityError`` if this runtime cannot execute the action."""
        mode = envelope.navigation_mode()
        if mode is not None and mode not in self.navigation_modes:
            raise CapabilityError(
                f"runtime {self.mode.value} does not offer {mode.value}; "
                f"it offers {[m.value for m in self.navigation_modes]}"
            )
