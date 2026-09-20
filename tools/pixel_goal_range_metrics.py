"""Pure measurements and acceptance checks for Paris Pixel Goal experiments."""

from __future__ import annotations

import math


CALIBRATION_U_NORM = 0.5
CALIBRATION_V_VALUES = (0.64, 0.68, 0.70, 0.74, 0.78, 0.82, 0.86, 0.92, 0.98)


def _vector(values) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError("expected a finite three-dimensional vector")
    return vector


def _distance_3d_cm(a, b) -> float:
    return math.dist(_vector(a), _vector(b))


def _planar_distance_m(a, b) -> float:
    a_x, a_y, _ = _vector(a)
    b_x, b_y, _ = _vector(b)
    return math.hypot(a_x - b_x, a_y - b_y) / 100.0


def calculate_range_metrics(
    initial_feet_cm,
    raw_hit_cm,
    validated_target_cm,
    camera_origin_cm,
):
    return {
        "initial_feet_to_raw_hit_planar_m": (
            _planar_distance_m(initial_feet_cm, raw_hit_cm)
            if raw_hit_cm is not None
            else None
        ),
        "initial_feet_to_validated_target_planar_m": (
            _planar_distance_m(initial_feet_cm, validated_target_cm)
            if validated_target_cm is not None
            else None
        ),
        "raw_hit_to_validated_target_planar_m": (
            _planar_distance_m(raw_hit_cm, validated_target_cm)
            if raw_hit_cm is not None and validated_target_cm is not None
            else None
        ),
        "camera_origin_to_raw_hit_3d_m": (
            _distance_3d_cm(camera_origin_cm, raw_hit_cm) / 100.0
            if raw_hit_cm is not None
            else None
        ),
    }


def assert_fixed_pose(
    reference,
    observed,
    *,
    feet_tolerance_cm: float = 1.0,
    camera_tolerance_cm: float = 1.0,
    rotation_tolerance_deg: float = 0.1,
):
    feet_drift_cm = _distance_3d_cm(reference["feet_cm"], observed["feet_cm"])
    camera_origin_drift_cm = _distance_3d_cm(
        reference["camera_origin_cm"], observed["camera_origin_cm"]
    )
    camera_rotation_drift_degrees = [
        abs((float(observed_axis) - float(reference_axis) + 180.0) % 360.0 - 180.0)
        for reference_axis, observed_axis in zip(
            reference["camera_rotation_degrees"],
            observed["camera_rotation_degrees"],
            strict=True,
        )
    ]
    if feet_drift_cm > feet_tolerance_cm:
        raise ValueError(
            f"reset-feet drift {feet_drift_cm:.6f} cm exceeds "
            f"{feet_tolerance_cm:.6f} cm"
        )
    if camera_origin_drift_cm > camera_tolerance_cm:
        raise ValueError(
            f"camera-origin drift {camera_origin_drift_cm:.6f} cm exceeds "
            f"{camera_tolerance_cm:.6f} cm"
        )
    if any(
        drift > rotation_tolerance_deg
        for drift in camera_rotation_drift_degrees
    ):
        raise ValueError(
            "camera-rotation drift "
            f"{camera_rotation_drift_degrees} exceeds "
            f"{rotation_tolerance_deg:.6f} degrees"
        )
    return {
        "passed": True,
        "feet_drift_cm": feet_drift_cm,
        "camera_origin_drift_cm": camera_origin_drift_cm,
        "camera_rotation_drift_degrees": camera_rotation_drift_degrees,
    }


def validate_calibration_report(report):
    if report.get("navigation_mode") != "nav_pixel_goal":
        raise ValueError("calibration must use nav_pixel_goal")
    if float(report.get("sample_u_norm", -1.0)) != CALIBRATION_U_NORM:
        raise ValueError("calibration must use u=0.5")
    configured_v_values = [float(value) for value in report.get("sample_v_values", [])]
    if configured_v_values != list(CALIBRATION_V_VALUES):
        raise ValueError("calibration v grid does not match the fixed protocol")
    samples = report.get("samples", [])
    if len(samples) != len(CALIBRATION_V_VALUES):
        raise ValueError("calibration must contain exactly nine samples")
    snapshots = []
    qualifying_distances = []
    for index, (sample, expected_v) in enumerate(
        zip(samples, CALIBRATION_V_VALUES, strict=True), start=1
    ):
        if int(sample.get("index", -1)) != index:
            raise ValueError("calibration sample order is invalid")
        if float(sample.get("u_norm", -1.0)) != CALIBRATION_U_NORM:
            raise ValueError("calibration sample u does not equal 0.5")
        if float(sample.get("v_norm", -1.0)) != expected_v:
            raise ValueError("calibration sample v order is invalid")
        snapshot_id = sample.get("camera_snapshot_id")
        intrinsics_id = sample.get("camera_intrinsics_id")
        if not snapshot_id or not intrinsics_id:
            raise ValueError("every calibration sample needs exact camera IDs")
        snapshots.append(snapshot_id)
        if sample.get("pose_consistency", {}).get("passed") is not True:
            raise ValueError("every calibration sample needs fixed-pose proof")
        if sample.get("reset", {}).get("success") is not True:
            raise ValueError("every calibration sample must end with reset")
        valid = sample.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("every calibration sample needs explicit validity")
        if valid:
            if sample.get("cancel", {}).get("success") is not True:
                raise ValueError("accepted calibration sample must be cancelled")
            distance = sample.get("metrics", {}).get(
                "initial_feet_to_validated_target_planar_m"
            )
            if distance is None:
                raise ValueError("accepted sample needs validated-target distance")
            if 5.0 <= float(distance) <= 10.0:
                qualifying_distances.append(float(distance))
        elif not sample.get("rejection_reason"):
            raise ValueError("rejected calibration sample needs explicit reason")
    if len(set(snapshots)) != len(snapshots):
        raise ValueError("calibration samples must use unique camera snapshots")
    if not qualifying_distances:
        raise ValueError("no valid calibration target distance lies in [5, 10] m")
    return {
        "passed": True,
        "qualifying_target_distances_m": qualifying_distances,
    }


def validate_adaptive_report(report, expected_moves: int = 6):
    if report.get("navigation_mode") != "nav_pixel_goal":
        raise ValueError("adaptive demo must use nav_pixel_goal")
    config = report.get("config", {})
    if float(config.get("max_navmesh_adjustment_cm", -1.0)) != 10.0:
        raise ValueError("maximum NavMesh adjustment must remain 10 cm")
    if float(config.get("acceptance_radius_cm", -1.0)) != 15.0:
        raise ValueError("MoveTo acceptance radius must remain 15 cm")
    summary = report.get("summary", {})
    if int(summary.get("pre_action_reset_calls", -1)) != 1:
        raise ValueError("adaptive demo must contain exactly one pre-action reset")
    if int(summary.get("action_phase_reset_calls", -1)) != 0:
        raise ValueError("adaptive demo must contain zero action-phase resets")
    continuity_errors = [
        float(value) for value in summary.get("continuity_errors_cm", [])
    ]
    if max(continuity_errors, default=0.0) > 1.0:
        raise ValueError("adaptive trajectory continuity exceeds 1 cm")
    steps = report.get("successful_steps", [])
    if len(steps) != expected_moves:
        raise ValueError(f"adaptive demo needs exactly {expected_moves} successes")
    long_distances = []
    short_distances = []
    for index, step in enumerate(steps):
        if step.get("navigation_mode") != "nav_pixel_goal":
            raise ValueError("every adaptive step must use nav_pixel_goal")
        if step.get("controller_result") != "success":
            raise ValueError("every adaptive step needs controller success")
        if float(step.get("execution_error_planar_m", math.inf)) >= 0.25:
            raise ValueError("every adaptive step needs planar error below 0.25 m")
        if not step.get("camera_snapshot_id") or not step.get(
            "camera_intrinsics_id"
        ):
            raise ValueError("every adaptive step needs exact input camera IDs")
        if not step.get("post_move_camera_snapshot_id") or not step.get(
            "post_move_camera_intrinsics_id"
        ):
            raise ValueError("every adaptive step needs exact post-move camera IDs")
        distance = float(
            step.get("initial_feet_to_validated_target_planar_m", math.nan)
        )
        if not math.isfinite(distance):
            raise ValueError("every adaptive step needs measured target distance")
        context = step.get("visual_context")
        if context == "clear_straight_long" and 5.0 <= distance <= 10.0:
            long_distances.append(distance)
        if context == "near_obstacle_short" and distance < 5.0:
            short_distances.append(distance)
        if index + 1 < len(steps):
            following = steps[index + 1]
            if step.get("after_frame") != following.get("before_frame"):
                raise ValueError("adaptive frame chain is broken")
            if step.get("post_move_camera_snapshot_id") != following.get(
                "camera_snapshot_id"
            ):
                raise ValueError("adaptive camera snapshot chain is broken")
            if step.get("post_move_camera_intrinsics_id") != following.get(
                "camera_intrinsics_id"
            ):
                raise ValueError("adaptive camera intrinsics chain is broken")
    if len(long_distances) < 2:
        raise ValueError("adaptive long-range quota requires two [5, 10] m steps")
    if len(short_distances) < 2:
        raise ValueError("adaptive short-range quota requires two sub-5 m steps")
    return {
        "passed": True,
        "successful_move_count": len(steps),
        "long_range_count": len(long_distances),
        "short_range_count": len(short_distances),
        "rejected_attempt_count": len(report.get("rejected_attempts", [])),
    }
