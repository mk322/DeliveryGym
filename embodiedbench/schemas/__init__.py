"""Versioned protocol schemas (the design plan M1).

Every contract the design plan names lives here: WorldBundle, OverlaySpec,
EnvironmentBundle, observations, actions, events, runtime results, embodiment
capabilities, EpisodeSpec, trajectory, and score report.

All schemas are ``v0.x`` with no backward-compatibility promise before M12
(the design plan M1); consumers pin exact revisions. ``SCHEMA_REGISTRY`` maps every
schema id to its model so a fixture suite can round-trip all of them without
an ad-hoc list that drifts.
"""

from embodiedbench.schemas.base import (
    CapabilityError,
    FrameError,
    SchemaError,
    SchemaModel,
    SchemaVersion,
    VersionError,
    schema_registry,
)
from embodiedbench.schemas.embodiment import (
    ControllerOutcome,
    ControllerResult,
    EmbodimentCapabilities,
    EmbodimentProfile,
    NavigationRequest,
    PoseEstimate,
)
from embodiedbench.schemas.environment import (
    Certificate,
    CertificationGrade,
    EnvironmentBundle,
    ObservationManifestEntry,
    OverlayPlacement,
    OverlaySpec,
)
from embodiedbench.schemas.episode import Budgets, EpisodeSpec, TaskConfigRef
from embodiedbench.schemas.geometry import (
    CoordinateFrame,
    FrameName,
    Pose,
    Transform,
    Vec3,
)
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    ActionResult,
    ActionStatus,
    Event,
    NavPixelGoalAction,
    NavPoint2DDepthAction,
    NavPoint3DAction,
    NavWaypointAction,
    Observation,
    ResetInfo,
    RuntimeCapabilities,
    RuntimeMode,
    StepResult,
    TaskAction,
    TerminationReason,
)
from embodiedbench.schemas.trajectory import (
    ScoreReport,
    TokenAccounting,
    Trajectory,
    TrajectoryTurn,
)
from embodiedbench.schemas.world import (
    NavEdge,
    NavGraph,
    NavNode,
    SemanticEntity,
    WorldBundle,
)

SCHEMA_REGISTRY = schema_registry

__all__ = [
    "ActionEnvelope",
    "ActionResult",
    "ActionStatus",
    "Budgets",
    "CapabilityError",
    "Certificate",
    "CertificationGrade",
    "ControllerOutcome",
    "ControllerResult",
    "CoordinateFrame",
    "EmbodimentCapabilities",
    "EmbodimentProfile",
    "EnvironmentBundle",
    "EpisodeSpec",
    "Event",
    "FrameError",
    "FrameName",
    "NavEdge",
    "NavGraph",
    "NavNode",
    "NavPixelGoalAction",
    "NavPoint2DDepthAction",
    "NavPoint3DAction",
    "NavWaypointAction",
    "NavigationRequest",
    "Observation",
    "ObservationManifestEntry",
    "OverlayPlacement",
    "OverlaySpec",
    "Pose",
    "PoseEstimate",
    "ResetInfo",
    "RuntimeCapabilities",
    "RuntimeMode",
    "SCHEMA_REGISTRY",
    "SchemaError",
    "SchemaModel",
    "SchemaVersion",
    "ScoreReport",
    "SemanticEntity",
    "StepResult",
    "TaskAction",
    "TaskConfigRef",
    "TerminationReason",
    "TokenAccounting",
    "Trajectory",
    "TrajectoryTurn",
    "Transform",
    "Vec3",
    "VersionError",
    "WorldBundle",
]
