"""Point-based navigation: the shared geometry chain (design plan §9.2-9.5).

``nav_waypoint`` is the production path and is defined on every map, because a
one-hop step to a named neighbour needs no geometry at all. The two point modes
do need geometry, and design plan §9.4 fixes the order:

    image point + distance
        -> unproject with camera intrinsics
        -> camera-to-agent-to-world transform
        -> range and ground-hit validation
        -> project to embodiment-traversable space
        -> quantise to the declared pose lattice
        -> request an executable controller trajectory

Both modes run the *same* chain and differ only in where the distance comes
from, which is the point of writing it once:

``nav_point_3d``        the model predicts ``distance_m`` itself
``nav_point_2d_depth``  an external metric-depth provider supplies the range

design plan §9.5 is emphatic that quantisation is part of the *action semantics*, not
a cached-runtime convenience: "snapping only in cached mode is forbidden because
it would create different transition systems". So the lattice lives here, where
every runtime shares it, rather than in the cached runtime.

On the depth provider, design plan §9.3 says plainly that on flat ground with a known
camera pose a ground-plane raycast can supply the range, and that where that is
true the mode "may be close to point selection plus environment geometry rather
than a distinct metric-reasoning problem" and should be reported accordingly.
That is exactly the situation on Paris, and ``GroundPlaneDepth`` says so in its
own docstring rather than letting a reader assume otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from embodiedbench.schemas.embodiment import DistanceSource, NavigationRequest
from embodiedbench.schemas.environment import NavigationMode, PointExecutionVariant
from embodiedbench.schemas.geometry import CameraIntrinsics, Pose, Vec3
from embodiedbench.schemas.runtime import ControllerOutcomeCode

# design plan §9.4: start the Paris pilot near 18 m, about its median graph edge, so
# one point action and one waypoint action cover comparable ground.
DEFAULT_MAX_RANGE_M = 18.0
# design plan §9.5's pilot design point.
DEFAULT_LATTICE_SPACING_CM = 200.0
DEFAULT_LATTICE_HEADINGS = 8


class PointNavRejected(Exception):
    """A point request that cannot be executed, carrying a typed outcome."""

    def __init__(self, outcome: ControllerOutcomeCode, detail: str = ""):
        self.outcome = outcome
        self.detail = detail
        super().__init__(f"{outcome.value}: {detail}" if detail else outcome.value)


# ─────────────────────────────────────────────────────────────────────────────
# Pose lattice
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoseLattice:
    """The finite set of poses a point action may terminate at.

    design plan §9.5: a continuous end pose has no cached observation, so either
    point modes run live only, or quantisation becomes part of the action
    semantics and every runtime applies the same rule. This is that rule.

    Snapping to graph nodes instead was considered and rejected in the plan --
    Paris's median edge is about 18.5 m, far too coarse to represent where a
    point action actually lands.
    """

    spacing_cm: float = DEFAULT_LATTICE_SPACING_CM
    headings: int = DEFAULT_LATTICE_HEADINGS

    def __post_init__(self) -> None:
        if self.spacing_cm <= 0:
            raise ValueError("lattice spacing must be positive")
        if self.headings <= 0:
            raise ValueError("a lattice needs at least one heading")

    @staticmethod
    def _round_half_up(value: float) -> float:
        """Round halves consistently upward.

        Python's ``round`` is banker's rounding, so a coordinate landing exactly
        on a cell boundary snaps up or down depending on the parity of the
        neighbouring cell. Both rules are deterministic, but only this one is
        predictable by a reader checking a trajectory by hand, and boundary hits
        are common because graph nodes sit on round coordinates.
        """
        return math.floor(value + 0.5)

    def quantise_position(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        return (
            self._round_half_up(x_cm / self.spacing_cm) * self.spacing_cm,
            self._round_half_up(y_cm / self.spacing_cm) * self.spacing_cm,
        )

    def quantise_heading(self, yaw_deg: float) -> float:
        step = 360.0 / self.headings
        return (self._round_half_up((yaw_deg % 360.0) / step) * step) % 360.0

    def quantise(self, pose: Pose) -> Pose:
        x, y = self.quantise_position(pose.position.x_cm, pose.position.y_cm)
        return Pose(
            frame=pose.frame,
            position=Vec3(x_cm=x, y_cm=y, z_cm=pose.position.z_cm),
            yaw_deg=self.quantise_heading(pose.yaw_deg),
        )

    def max_snap_error_cm(self) -> float:
        """Worst planar error quantisation can introduce (half the diagonal)."""
        return self.spacing_cm * math.sqrt(2.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Depth providers
# ─────────────────────────────────────────────────────────────────────────────


class DepthProvider(Protocol):
    """Supplies metric range for an image point (design plan §9.3)."""

    name: str

    def distance_m(self, u_norm: float, v_norm: float, camera: Pose,
                   intrinsics: CameraIntrinsics) -> float: ...


@dataclass
class GroundPlaneDepth:
    """Range from intersecting the view ray with the ground plane.

    design plan §9.3 sanctions this and simultaneously warns about what it means: on
    flat ground with a known camera pose this makes ``nav_point_2d_depth`` close
    to "point selection plus environment geometry", not an independent
    metric-depth problem. Any result obtained with this provider must be
    reported as such rather than as evidence of depth reasoning.

    A ray at or above the horizon never meets the ground, which is a
    ``no_ground_hit`` -- distinct, as design plan §9.4 requires, from hitting a wall.

    The returned range is the *slant* range from camera to ground hit, because
    that is what ``CameraIntrinsics.unproject`` consumes and what a model
    predicting ``distance_m`` for ``nav_point_3d`` means. Returning the ground
    distance instead is a plausible reading that silently makes the two point
    modes land in different places for the same image point -- which is exactly
    what the shared chain exists to prevent -- so the convention is stated here
    and asserted by ``test_both_modes_share_one_chain``.
    """

    name: str = "ground_plane_raycast"
    ground_z_cm: float = 0.0

    def distance_m(
        self, u_norm: float, v_norm: float, camera: Pose, intrinsics: CameraIntrinsics
    ) -> float:
        # A unit-length ray, so scaling it to reach the ground gives the slant
        # range directly rather than a horizontal distance needing conversion.
        ray = intrinsics.unproject(u_norm, v_norm, 1.0)
        # Camera space: x right, y down, z forward. A point below the optical
        # axis (y > 0) is toward the ground.
        if ray.y_cm <= 1e-6:
            raise PointNavRejected(
                ControllerOutcomeCode.NO_GROUND_HIT,
                "the selected point is at or above the horizon",
            )
        if ray.z_cm <= 0:
            raise PointNavRejected(
                ControllerOutcomeCode.NO_GROUND_HIT, "ground intersection is behind the camera"
            )
        height_cm = camera.position.z_cm - self.ground_z_cm
        if height_cm <= 0:
            raise PointNavRejected(
                ControllerOutcomeCode.NO_GROUND_HIT, "camera is at or below the ground plane"
            )
        # Similar triangles: scale the unit ray until it drops by the camera
        # height. The scale factor *is* the slant range, since the ray is unit.
        return (height_cm / ray.y_cm) / 100.0


@dataclass
class ModelPredictedDepth:
    """The model's own ``distance_m`` (design plan §9.2)."""

    name: str = "model_prediction"
    distance: float = 0.0

    def distance_m(self, *_args: Any, **_kwargs: Any) -> float:
        return self.distance


# ─────────────────────────────────────────────────────────────────────────────
# Traversability
# ─────────────────────────────────────────────────────────────────────────────


class TraversableSpace(Protocol):
    """Whether the embodiment can stand at a world position."""

    def project(self, x_cm: float, y_cm: float) -> tuple[float, float] | None: ...


@dataclass
class GraphProximityTraversable:
    """Traversable where close enough to the navigation graph.

    A stand-in for a NavMesh projection until Paris carries certified NavMesh
    geometry. It is deliberately conservative -- a point far from any known edge
    is rejected rather than assumed walkable -- and it is named so that no
    reader mistakes it for a NavMesh query.
    """

    points: list[tuple[float, float]]
    tolerance_cm: float = 600.0

    def project(self, x_cm: float, y_cm: float) -> tuple[float, float] | None:
        best = None
        best_distance = float("inf")
        for px, py in self.points:
            distance = math.hypot(px - x_cm, py - y_cm)
            if distance < best_distance:
                best, best_distance = (px, py), distance
        if best is None or best_distance > self.tolerance_cm:
            return None
        return best


# ─────────────────────────────────────────────────────────────────────────────
# The chain
# ─────────────────────────────────────────────────────────────────────────────


def camera_to_world(camera: Pose, camera_point: Vec3) -> Vec3:
    """Camera-space point into world space.

    Camera convention: +z forward, +x right, +y down, no pitch or roll, so the
    transform is a yaw rotation plus the camera translation. Height is taken
    from the camera and reduced by the downward component.
    """
    radians = math.radians(camera.yaw_deg)
    forward_x, forward_y = math.cos(radians), math.sin(radians)
    right_x, right_y = math.sin(radians), -math.cos(radians)
    world_x = camera.position.x_cm + forward_x * camera_point.z_cm + right_x * camera_point.x_cm
    world_y = camera.position.y_cm + forward_y * camera_point.z_cm + right_y * camera_point.x_cm
    world_z = camera.position.z_cm - camera_point.y_cm
    return Vec3(x_cm=world_x, y_cm=world_y, z_cm=world_z)


@dataclass
class PointNavResolution:
    """A resolved point action, with the typed outcome design plan §9.4 requires.

    The outcome cannot be recovered from the ``NavigationRequest`` alone: a
    clipped request and an in-range one are structurally identical once the
    range has been capped. An earlier version stashed the flag on the pydantic
    model with ``setattr``, which happened to work and would have gone silent
    the moment the model gained ``model_config = {"frozen": True}``.
    """

    request: NavigationRequest
    outcome: ControllerOutcomeCode
    distance_m: float
    clipped: bool

    @property
    def accepted(self) -> bool:
        return self.outcome in (
            ControllerOutcomeCode.ACCEPTED,
            ControllerOutcomeCode.CLIPPED,
        )


def resolve_point_action(
    *,
    action: Any,
    camera: Pose,
    intrinsics: CameraIntrinsics,
    traversable: TraversableSpace,
    lattice: PoseLattice,
    depth: DepthProvider | None = None,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    request_id: str = "req",
) -> PointNavResolution:
    """Run design plan §9.4's chain, raising a typed rejection at the first failure."""
    target = action.target
    mode = (
        NavigationMode.NAV_POINT_3D
        if action.type == "nav_point_3d"
        else NavigationMode.NAV_POINT_2D_DEPTH
    )

    # 1. range, from whichever source this mode declares
    if mode is NavigationMode.NAV_POINT_3D:
        if target.distance_m is None:
            raise PointNavRejected(
                ControllerOutcomeCode.LOW_DEPTH_CONFIDENCE, "no model distance supplied"
            )
        distance_m = float(target.distance_m)
        source = DistanceSource.MODEL_PREDICTION
    else:
        if depth is None:
            raise PointNavRejected(
                ControllerOutcomeCode.LOW_DEPTH_CONFIDENCE,
                "nav_point_2d_depth requires an external depth provider",
            )
        distance_m = depth.distance_m(target.u_norm, target.v_norm, camera, intrinsics)
        source = DistanceSource.EXTERNAL_DEPTH

    if not math.isfinite(distance_m) or distance_m <= 0:
        raise PointNavRejected(
            ControllerOutcomeCode.LOW_DEPTH_CONFIDENCE, f"non-positive range {distance_m}"
        )

    # 2. range cap. design plan §9.2's strict variant treats the distance as
    # load-bearing and refuses rather than silently repairing it along the ray.
    clipped = False
    if distance_m > max_range_m:
        variant = getattr(action, "variant", None)
        if mode is NavigationMode.NAV_POINT_3D and variant is PointExecutionVariant.STRICT:
            raise PointNavRejected(
                ControllerOutcomeCode.OUT_OF_RANGE,
                f"{distance_m:.2f} m exceeds the {max_range_m:.2f} m cap",
            )
        distance_m = max_range_m
        clipped = True

    # 3. unproject and transform
    camera_point = intrinsics.unproject(target.u_norm, target.v_norm, distance_m * 100.0)
    world_point = camera_to_world(camera, camera_point)

    # 4. project onto traversable space
    projected = traversable.project(world_point.x_cm, world_point.y_cm)
    if projected is None:
        raise PointNavRejected(
            ControllerOutcomeCode.NOT_TRAVERSABLE,
            "no traversable surface near the unprojected target",
        )
    projected_vec = Vec3(x_cm=projected[0], y_cm=projected[1], z_cm=0.0)

    # 5. quantise, in every runtime alike (design plan §9.5)
    heading = math.degrees(
        math.atan2(projected_vec.y_cm - camera.position.y_cm,
                   projected_vec.x_cm - camera.position.x_cm)
    ) % 360.0
    quantised = lattice.quantise(
        Pose(frame=camera.frame, position=projected_vec, yaw_deg=heading)
    )

    request = NavigationRequest(
        request_id=request_id,
        mode=mode,
        distance_source=source,
        source_image_point=target,
        target_camera=camera_point,
        target_agent=camera_point,
        target_world=world_point,
        projected_target=projected_vec,
        quantized_target=quantised,
        max_range_m=max_range_m,
    )
    return PointNavResolution(
        request=request,
        # Clipping is reported, not hidden: the 3d_snap/3d_strict comparison in
        # design plan §9.7.6 is meaningless if a repaired distance looks accepted.
        outcome=(
            ControllerOutcomeCode.CLIPPED if clipped else ControllerOutcomeCode.ACCEPTED
        ),
        distance_m=distance_m,
        clipped=clipped,
    )


def distance_error_m(model_distance_m: float, external_distance_m: float) -> float:
    """design plan §9.2: always log |model - external|, even when snapping repairs it.

    Without this a policy emitting a constant distance looks competent after the
    snap, because the projection hides the error.
    """
    return abs(float(model_distance_m) - float(external_distance_m))
