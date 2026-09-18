"""design plan §9.7.2's oracle: prove the point chain before any policy uses it.

The plan puts this second in the research sequence, before either point mode is
implemented for training, and the reason is worth restating: a coordinate chain
can be wrong in a way that no single-direction test detects. Unproject with a
flipped axis and re-project with the same flip and every round trip closes. The
first Paris capture pointed the camera at the sky for a whole render because
``unreal.Rotator`` is positionally ``(roll, pitch, yaw)`` and nothing in the
pipeline disagreed with itself.

So the oracle does not test the chain against itself. It starts from a *known
world position* -- a real graph node, at a real camera pose the album was baked
at -- and asks:

    world node -> world-to-camera -> project to image -> [the chain] -> world

If the chain is right, the recovered position is the node it started from, to
within the lattice's stated snap error. If any step has a sign, axis, or units
error, the recovered position lands somewhere else and the residual says how
far. Nothing here consults the chain to decide what the right answer is.

``world_to_camera`` is written out separately rather than reusing
``camera_to_world`` inverted, because an inverse derived from the thing under
test cannot falsify it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.embodiment.point_nav import (
    GraphProximityTraversable,
    GroundPlaneDepth,
    PointNavRejected,
    PoseLattice,
    resolve_point_action,
)
from embodiedbench.schemas.geometry import CameraIntrinsics, FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import NavPoint2DDepthAction, NavPoint3DAction

# The album's plain view: 640x480 at a 90-degree horizontal field of view, from
# a camera 160 cm above the ground.
ALBUM_INTRINSICS = CameraIntrinsics(
    width_px=640, height_px=480, fx_px=320.0, fy_px=320.0,
    cx_px=320.0, cy_px=240.0, near_cm=10.0, far_cm=1.0e5,
)
ALBUM_CAMERA_Z_CM = 160.0
ALBUM_YAWS = (0.0, 90.0, 180.0, 270.0)


def world_to_camera(camera: Pose, world_point: Vec3) -> Vec3:
    """World point into the camera frame.

    Derived independently of ``camera_to_world`` on purpose: an inverse obtained
    by algebraically inverting the function under test agrees with it even when
    both are wrong.

    Camera convention, stated so it can be checked rather than inferred: +z runs
    along the camera's heading, +x to its right, +y downward.
    """
    dx = world_point.x_cm - camera.position.x_cm
    dy = world_point.y_cm - camera.position.y_cm
    dz = world_point.z_cm - camera.position.z_cm
    radians = math.radians(camera.yaw_deg)
    cos_yaw, sin_yaw = math.cos(radians), math.sin(radians)
    forward = dx * cos_yaw + dy * sin_yaw
    right = dx * sin_yaw - dy * cos_yaw
    return Vec3(x_cm=right, y_cm=-dz, z_cm=forward)


@dataclass
class OracleCase:
    """One round trip, and what it recovered."""

    camera_x_cm: float
    camera_y_cm: float
    camera_yaw_deg: float
    target_x_cm: float
    target_y_cm: float
    mode: str
    visible: bool
    accepted: bool
    rejection: str | None = None
    clipped: bool = False
    u_norm: float | None = None
    v_norm: float | None = None
    true_distance_m: float | None = None
    external_distance_m: float | None = None
    recovered_x_cm: float | None = None
    recovered_y_cm: float | None = None
    # Before projection to the graph. This is the chain's own accuracy.
    unprojected_residual_cm: float | None = None
    # After projection. Useful, but it hides any error smaller than the spacing
    # between nodes, so it cannot be the only thing checked -- a 2% scale error
    # passed a whole oracle run on this measure alone.
    residual_cm: float | None = None
    quantised_residual_cm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
        }


@dataclass
class OracleReport:
    """The verdict, with the evidence that produced it."""

    map_name: str
    cases: int = 0
    visible: int = 0
    accepted: int = 0
    clipped: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    max_unprojected_residual_cm: float = 0.0
    mean_unprojected_residual_cm: float = 0.0
    max_residual_cm: float = 0.0
    mean_residual_cm: float = 0.0
    max_quantised_residual_cm: float = 0.0
    lattice_snap_bound_cm: float = 0.0
    max_depth_error_m: float = 0.0
    tolerance_cm: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # A run with no visible targets proves nothing, so it is not a pass.
        return not self.failures and self.visible > 0 and self.accepted > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "cases": self.cases,
            "visible": self.visible,
            "accepted": self.accepted,
            "clipped": self.clipped,
            "rejected": self.rejected,
            "max_unprojected_residual_cm": round(self.max_unprojected_residual_cm, 3),
            "mean_unprojected_residual_cm": round(self.mean_unprojected_residual_cm, 3),
            "max_residual_cm": round(self.max_residual_cm, 3),
            "mean_residual_cm": round(self.mean_residual_cm, 3),
            "max_quantised_residual_cm": round(self.max_quantised_residual_cm, 3),
            "lattice_snap_bound_cm": round(self.lattice_snap_bound_cm, 3),
            "max_depth_error_m": round(self.max_depth_error_m, 4),
            "tolerance_cm": round(self.tolerance_cm, 3),
            "failures": self.failures[:40],
            "failure_count": len(self.failures),
            "notes": self.notes,
            "status": "pass" if self.passed else "fail",
        }


def round_trip(
    *,
    camera: Pose,
    target: Vec3,
    intrinsics: CameraIntrinsics,
    traversable: GraphProximityTraversable,
    lattice: PoseLattice,
    mode: str,
    max_range_m: float,
) -> OracleCase:
    """Project a known world point into the image and drive it back out."""
    case = OracleCase(
        camera_x_cm=camera.position.x_cm,
        camera_y_cm=camera.position.y_cm,
        camera_yaw_deg=camera.yaw_deg,
        target_x_cm=target.x_cm,
        target_y_cm=target.y_cm,
        mode=mode,
        visible=False,
        accepted=False,
    )

    camera_point = world_to_camera(camera, target)
    projected = intrinsics.project(camera_point)
    if projected is None:
        # Behind the camera or outside the sensor. Not a failure: an album frame
        # simply does not show every neighbour.
        return case
    u_norm, v_norm, distance_cm = projected
    case.visible = True
    case.u_norm, case.v_norm = u_norm, v_norm
    case.true_distance_m = distance_cm / 100.0

    if mode == "nav_point_3d":
        # The oracle plays a *perfect* distance predictor, which is the point:
        # with a correct range, any residual is the chain's, not the policy's.
        action: Any = NavPoint3DAction(
            target={"u_norm": u_norm, "v_norm": v_norm, "distance_m": distance_cm / 100.0}
        )
        depth = None
    else:
        action = NavPoint2DDepthAction(target={"u_norm": u_norm, "v_norm": v_norm})
        depth = GroundPlaneDepth()
        try:
            case.external_distance_m = depth.distance_m(u_norm, v_norm, camera, intrinsics)
        except PointNavRejected:
            case.external_distance_m = None

    try:
        resolution = resolve_point_action(
            action=action, camera=camera, intrinsics=intrinsics,
            traversable=traversable, lattice=lattice, depth=depth,
            max_range_m=max_range_m,
        )
    except PointNavRejected as rejected:
        case.rejection = rejected.outcome.value
        return case

    request = resolution.request
    case.accepted = True
    case.clipped = resolution.clipped
    case.recovered_x_cm = request.projected_target.x_cm
    case.recovered_y_cm = request.projected_target.y_cm
    case.unprojected_residual_cm = math.hypot(
        request.target_world.x_cm - target.x_cm,
        request.target_world.y_cm - target.y_cm,
    )
    case.residual_cm = math.hypot(
        request.projected_target.x_cm - target.x_cm,
        request.projected_target.y_cm - target.y_cm,
    )
    case.quantised_residual_cm = math.hypot(
        request.quantized_target.position.x_cm - target.x_cm,
        request.quantized_target.position.y_cm - target.y_cm,
    )
    return case


def _rejection_is_legitimate(case: OracleCase, max_range_m: float) -> bool:
    """Whether the chain was entitled to refuse this known-good target.

    Only one reason survives scrutiny. The oracle's targets are real graph nodes
    at ground level, in view, so:

    out_of_range        legitimate exactly when the true slant range really does
                        exceed the cap. Neighbours are filtered by *horizontal*
                        distance, and the camera sits 160 cm up, so a target at
                        the edge of the band can be a few centimetres beyond it.
    not_traversable     never legitimate: the target is a graph node.
    no_ground_hit       never legitimate: the target is on the ground and below
                        the horizon, or it would not have projected into frame.
    low_depth_confidence never legitimate: the oracle supplies an exact range.
    """
    if case.rejection == "out_of_range":
        return (case.true_distance_m or 0.0) > max_range_m
    return False


def run_oracle(
    *,
    map_name: str,
    node_positions: list[tuple[str, float, float]],
    max_range_m: float,
    lattice: PoseLattice | None = None,
    intrinsics: CameraIntrinsics | None = None,
    tolerance_cm: float = 1.0,
    max_cameras: int = 200,
    camera_z_cm: float = ALBUM_CAMERA_Z_CM,
) -> OracleReport:
    """Round-trip every visible near neighbour from a sample of album poses.

    Cameras are drawn at a fixed stride rather than at random, so the sample is
    the same on every run and a regression is attributable.
    """
    lattice = lattice or PoseLattice()
    intrinsics = intrinsics or ALBUM_INTRINSICS
    report = OracleReport(
        map_name=map_name,
        tolerance_cm=tolerance_cm,
        lattice_snap_bound_cm=lattice.max_snap_error_cm(),
    )

    if not node_positions:
        report.notes.append("no graph nodes: nothing to validate")
        return report

    ordered = sorted(node_positions, key=lambda n: (n[1], n[2], n[0]))
    points = [(x, y) for _, x, y in ordered]
    # The traversable set is the graph itself, and the tolerance is tight: the
    # oracle aims at a node, so a correct chain lands on that node, and a loose
    # tolerance would let a wrong answer snap to a different node and pass.
    traversable = GraphProximityTraversable(points=points, tolerance_cm=200.0)

    stride = max(1, len(ordered) // max_cameras)
    residuals: list[float] = []
    raw_residuals: list[float] = []

    for index in range(0, len(ordered), stride):
        _, cx, cy = ordered[index]
        neighbours = [
            (x, y)
            for x, y in points
            if 0 < math.hypot(x - cx, y - cy) <= max_range_m * 100.0
        ]
        if not neighbours:
            continue
        for yaw in ALBUM_YAWS:
            camera = Pose(
                frame=FrameName.BUNDLE_WORLD,
                position=Vec3(x_cm=cx, y_cm=cy, z_cm=camera_z_cm),
                yaw_deg=yaw,
            )
            for tx, ty in neighbours[:8]:
                for mode in ("nav_point_3d", "nav_point_2d_depth"):
                    case = round_trip(
                        camera=camera,
                        target=Vec3(x_cm=tx, y_cm=ty, z_cm=0.0),
                        intrinsics=intrinsics,
                        traversable=traversable,
                        lattice=lattice,
                        mode=mode,
                        max_range_m=max_range_m,
                    )
                    report.cases += 1
                    if not case.visible:
                        continue
                    report.visible += 1
                    if not case.accepted:
                        reason = case.rejection or "unknown"
                        report.rejected[reason] = report.rejected.get(reason, 0) + 1
                        # A rejection is itself a result that has to be judged.
                        # The oracle aims at a real graph node, on the ground, in
                        # front of the camera and inside the range cap, so the
                        # chain has no legitimate reason to refuse it. Counting
                        # rejections without judging them is how an earlier draft
                        # of this file gave a deliberately sign-flipped transform
                        # a clean pass: the flip pushed 90% of targets off the
                        # road, they were rejected as not_traversable, and only
                        # the handful the flip happened not to move were scored.
                        if not _rejection_is_legitimate(case, max_range_m):
                            failure = case.to_dict()
                            failure["check"] = "known-good target was rejected"
                            report.failures.append(failure)
                        continue
                    report.accepted += 1
                    if not case.clipped:
                        residuals.append(case.residual_cm or 0.0)
                        raw_residuals.append(case.unprojected_residual_cm or 0.0)
                        report.max_residual_cm = max(
                            report.max_residual_cm, case.residual_cm or 0.0
                        )
                        report.max_unprojected_residual_cm = max(
                            report.max_unprojected_residual_cm,
                            case.unprojected_residual_cm or 0.0,
                        )
                        report.max_quantised_residual_cm = max(
                            report.max_quantised_residual_cm, case.quantised_residual_cm or 0.0
                        )
                        if case.true_distance_m and case.external_distance_m:
                            report.max_depth_error_m = max(
                                report.max_depth_error_m,
                                abs(case.true_distance_m - case.external_distance_m),
                            )
                    if case.clipped:
                        # A clipped target legitimately lands short, so its
                        # residual says nothing about the chain. What must hold
                        # is that the clip only fired because the range really
                        # was over the cap -- neighbours are filtered by
                        # *horizontal* distance and the camera sits 160 cm up,
                        # so the slant range can exceed the cap by a few
                        # centimetres at the edge of the band.
                        report.clipped += 1
                        if (case.true_distance_m or 0.0) <= max_range_m:
                            failure = case.to_dict()
                            failure["check"] = "clipped a target that was inside the range cap"
                            report.failures.append(failure)
                        continue

                    # Check the raw chain, not just the snapped result. The
                    # graph projection swallows any error smaller than the
                    # internode spacing: a deliberate 2% scale bug produced a
                    # clean report until this check existed.
                    if (case.unprojected_residual_cm or 0.0) > tolerance_cm:
                        failure = case.to_dict()
                        failure["check"] = "unprojected target does not match the true position"
                        report.failures.append(failure)
                    elif (case.residual_cm or 0.0) > tolerance_cm:
                        failure = case.to_dict()
                        failure["check"] = "projected target landed on the wrong node"
                        report.failures.append(failure)

    report.mean_residual_cm = (sum(residuals) / len(residuals)) if residuals else 0.0
    report.mean_unprojected_residual_cm = (
        (sum(raw_residuals) / len(raw_residuals)) if raw_residuals else 0.0
    )

    if report.visible == 0:
        report.notes.append(
            "no target was visible from any sampled pose; the oracle proved nothing"
        )
    if report.max_quantised_residual_cm > lattice.max_snap_error_cm() + tolerance_cm:
        report.failures.append({
            "check": "quantised_residual_within_lattice_bound",
            "observed_cm": round(report.max_quantised_residual_cm, 3),
            "bound_cm": round(lattice.max_snap_error_cm(), 3),
            "detail": (
                "quantisation moved a target further than the lattice's own stated "
                "bound, so the published snap error understates what the runtime does"
            ),
        })
    # A ground-plane raycast aimed at a point that *is* on the ground must
    # recover the true range. Any gap is a bug in the provider, not depth noise.
    if report.max_depth_error_m > 0.05:
        report.failures.append({
            "check": "ground_plane_depth_matches_true_range",
            "observed_m": round(report.max_depth_error_m, 4),
            "detail": (
                "the ground-plane provider disagreed with the true range to a "
                "ground-level target, so the two point modes do not share a range"
            ),
        })
    return report
