"""Acceptance contract for the live CityCore_Paris Pixel Goal PoC.

This module deliberately has no Unreal or SPEAR dependency.  It validates the
evidence emitted by the live harness and keeps the milestone threshold separate
from the controller's configurable acceptance radius.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


PARIS_POC_SCENE = "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
PARIS_POC_MAX_ERROR_M = 0.25
PARIS_POC_INITIAL_ACCEPTANCE_RADIUS_CM = 15.0
PARIS_POC_INITIAL_NAVMESH_ADJUSTMENT_CM = 10.0
PARIS_POC_MAX_HORIZONTAL_NAV_EXTENT_CM = 5_000.0


@dataclass(frozen=True)
class ParisPocRegion:
    """Internal deterministic setup for one real Paris sidewalk."""

    name: str
    agent_spawn_cm: tuple[float, float, float]
    agent_yaw_deg: float
    nav_bounds_center_cm: tuple[float, float, float]
    nav_bounds_extent_cm: tuple[float, float, float]

    def to_report(self) -> dict[str, object]:
        report = asdict(self)
        for key in (
            "agent_spawn_cm",
            "nav_bounds_center_cm",
            "nav_bounds_extent_cm",
        ):
            report[key] = list(report[key])
        return report


RUE_DE_RIVOLI_SIDEWALK = ParisPocRegion(
    name="rue_de_rivoli_sidewalk",
    agent_spawn_cm=(-26_805.8209, 9_437.5276, 90.0),
    agent_yaw_deg=165.0056,
    nav_bounds_center_cm=(-27_385.3767, 9_592.8117, 100.0),
    nav_bounds_extent_cm=(1_800.0, 700.0, 300.0),
)


RUE_DE_RIVOLI_QWEN_POST_BOLLARDS = ParisPocRegion(
    name="rue_de_rivoli_qwen_post_bollards",
    agent_spawn_cm=(-30_283.2448, 10_368.9367, 90.0),
    agent_yaw_deg=165.0055935,
    nav_bounds_center_cm=(-32_698.1226, 11_015.7486, 100.0),
    nav_bounds_extent_cm=(4_500.0, 2_000.0, 300.0),
)


PARIS_CENTRAL_NORTHBOUND_SIDEWALK = ParisPocRegion(
    name="paris_central_northbound_sidewalk",
    agent_spawn_cm=(-5_712.6473, -10_000.0, 90.0),
    agent_yaw_deg=90.0,
    nav_bounds_center_cm=(-5_712.6473, -6_000.0, 100.0),
    nav_bounds_extent_cm=(300.0, 5_000.0, 300.0),
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vec3(value: object, name: str) -> tuple[float, float, float]:
    values = _list(value, name)
    if len(values) != 3:
        raise ValueError(f"{name} must have three coordinates")
    return tuple(_number(item, name) for item in values)  # type: ignore[return-value]


def _uv(value: object, name: str) -> tuple[float, float]:
    values = _list(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must have two normalized coordinates")
    uv = (_number(values[0], name), _number(values[1], name))
    if not all(0.0 <= item <= 1.0 for item in uv):
        raise ValueError(f"{name} must be normalized to [0, 1]")
    return uv


def validate_paris_poc_report(report: Mapping[str, Any]) -> dict[str, object]:
    """Validate and summarize the narrow Milestone 1B-PoC evidence."""

    scene = report.get("scene")
    if scene != PARIS_POC_SCENE or report.get("real_paris_content") is not True:
        raise ValueError("report must identify live CityCore_Paris content")
    if report.get("navigation_mode") != "nav_pixel_goal":
        raise ValueError("report must use nav_pixel_goal")

    trial_protocol = _mapping(report.get("trial_protocol"), "trial_protocol")
    if (
        trial_protocol.get("independent_trials") is not True
        or trial_protocol.get("reset_between_valid_goals") is not True
    ):
        raise ValueError("Paris PoC requires independent reset calibration trials")
    if trial_protocol.get("reset_is_policy_action") is not False:
        raise ValueError("trial reset must remain a live fixture operation")

    region = _mapping(report.get("region"), "region")
    if not region.get("name"):
        raise ValueError("region must have a stable name")
    _vec3(region.get("agent_spawn_cm"), "region.agent_spawn_cm")
    _number(region.get("agent_yaw_deg"), "region.agent_yaw_deg")
    _vec3(region.get("nav_bounds_center_cm"), "region.nav_bounds_center_cm")
    extents = _vec3(
        region.get("nav_bounds_extent_cm"), "region.nav_bounds_extent_cm"
    )
    if any(extent <= 0.0 for extent in extents):
        raise ValueError("local NavMesh bounds extents must be positive")
    if max(extents[:2]) > PARIS_POC_MAX_HORIZONTAL_NAV_EXTENT_CM:
        raise ValueError("Paris PoC requires a local NavMesh, not city-wide bounds")

    config = _mapping(report.get("config"), "config")
    radius = _number(config.get("acceptance_radius_cm"), "acceptance radius")
    if not math.isclose(
        radius, PARIS_POC_INITIAL_ACCEPTANCE_RADIUS_CM, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("initial Paris PoC comparison requires a 15 cm radius")
    adjustment = _number(
        config.get("max_navmesh_adjustment_cm"), "maximum NavMesh adjustment"
    )
    if not math.isclose(
        adjustment,
        PARIS_POC_INITIAL_NAVMESH_ADJUSTMENT_CM,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("initial Paris PoC comparison requires a 10 cm adjustment")

    poc_status = _mapping(report.get("poc_status"), "poc_status")
    if (
        poc_status.get("dynamic_runtime_generation") is not True
        or poc_status.get("supports_runtime_generation") is not True
        or poc_status.get("navmesh_has_valid_data") is not True
    ):
        raise ValueError("Paris PoC requires a verified dynamic runtime NavMesh")
    active_tiles = _number(
        poc_status.get("active_navmesh_tiles"), "active NavMesh tiles"
    )
    if active_tiles < 1:
        raise ValueError("Paris PoC requires populated NavMesh tiles")

    strict = _mapping(report.get("strict_execution"), "strict_execution")
    forbidden = {
        "project_goal_location": "destination projection",
        "allow_partial_path": "partial path",
        "straight_line_fallback": "straight-line fallback",
        "graph_projection": "graph projection",
        "pose_lattice_quantization": "PoseLattice quantization",
    }
    for key, label in forbidden.items():
        if strict.get(key) is not False:
            raise ValueError(f"Paris Pixel Goal forbids {label}")

    goals = _list(report.get("manual_goals"), "manual_goals")
    if len(goals) != 5:
        raise ValueError("Paris PoC requires exactly five valid goals")
    actions: set[tuple[float, float]] = set()
    errors: list[float] = []
    for index, raw_goal in enumerate(goals, start=1):
        goal = _mapping(raw_goal, f"manual_goals[{index}]")
        actions.add(_uv(goal.get("requested_uv"), f"manual_goals[{index}].uv"))
        if not goal.get("camera_snapshot_id") or not goal.get(
            "camera_intrinsics_id"
        ):
            raise ValueError("every goal must identify its exact camera snapshot")
        for key in (
            "raw_world_hit_cm",
            "validated_navigation_target_cm",
            "initial_feet_position_cm",
            "final_feet_position_cm",
        ):
            _vec3(goal.get(key), f"manual_goals[{index}].{key}")
        if goal.get("controller_request_result") != "request_successful":
            raise ValueError("every valid goal must start a controller request")
        if goal.get("controller_result") != "success":
            raise ValueError("every valid goal must finish successfully")
        error = _number(
            goal.get("execution_error_planar_m"),
            f"manual_goals[{index}].execution_error_planar_m",
        )
        if error >= PARIS_POC_MAX_ERROR_M:
            raise ValueError("every planar execution error must be below 0.25 m")
        errors.append(error)
        samples = _list(
            goal.get("feet_position_samples_cm"),
            f"manual_goals[{index}].feet_position_samples_cm",
        )
        distinct_samples = {
            _vec3(sample, f"manual_goals[{index}].feet sample") for sample in samples
        }
        if len(distinct_samples) < 3:
            raise ValueError("every successful move needs three distinct feet samples")
        trial_reset = _mapping(
            goal.get("trial_reset"), f"manual_goals[{index}].trial_reset"
        )
        if (
            trial_reset.get("success") is not True
            or trial_reset.get("controller_stopped") is not True
            or trial_reset.get("camera_snapshots_invalidated") is not True
            or trial_reset.get("poc_ready") is not True
        ):
            raise ValueError("every goal requires a verified independent trial reset")
    if len(actions) != 5:
        raise ValueError("Paris PoC requires five distinct normalized actions")

    invalids = _list(report.get("invalid_cases"), "invalid_cases")
    if len(invalids) != 2:
        raise ValueError("Paris PoC requires exactly two invalid goals")
    for index, raw_invalid in enumerate(invalids, start=1):
        invalid = _mapping(raw_invalid, f"invalid_cases[{index}]")
        _uv(invalid.get("requested_uv"), f"invalid_cases[{index}].uv")
        if invalid.get("accepted") is not False:
            raise ValueError("every invalid goal must be explicitly rejected")
        if not invalid.get("rejection_reason"):
            raise ValueError("every invalid goal must record a rejection reason")
        raw_hit = invalid.get("raw_world_hit_cm")
        if raw_hit is not None:
            _vec3(raw_hit, f"invalid_cases[{index}].raw_world_hit_cm")

    return {
        "passed": True,
        "valid_goal_count": len(goals),
        "invalid_goal_count": len(invalids),
        "maximum_execution_error_planar_m": max(errors),
        "acceptance_radius_cm": radius,
        "max_navmesh_adjustment_cm": adjustment,
    }
