from __future__ import annotations

from copy import deepcopy

import pytest

from embodiedbench.runtime.pixel_goal_paris_poc import validate_paris_poc_report


def _goal(index: int, u: float, v: float, error_m: float = 0.12) -> dict[str, object]:
    return {
        "index": index,
        "requested_uv": [u, v],
        "camera_snapshot_id": f"paris-snapshot-{index}",
        "camera_intrinsics_id": "perspective-640x360-hfov90",
        "raw_world_hit_cm": [-27000.0 - index * 100.0, 9500.0, 0.0],
        "validated_navigation_target_cm": [
            -27000.0 - index * 100.0,
            9500.0,
            0.0,
        ],
        "initial_feet_position_cm": [-26800.0, 9437.0, 0.0],
        "final_feet_position_cm": [-27000.0 - index * 100.0, 9512.0, 0.0],
        "controller_request_result": "request_successful",
        "controller_result": "success",
        "execution_error_planar_m": error_m,
        "feet_position_samples_cm": [
            [-26800.0, 9437.0, 0.0],
            [-26900.0, 9460.0, 0.0],
            [-27000.0 - index * 100.0, 9512.0, 0.0],
        ],
        "trial_reset": {
            "success": True,
            "controller_stopped": True,
            "camera_snapshots_invalidated": True,
            "poc_ready": True,
        },
    }


def _passing_report() -> dict[str, object]:
    return {
        "milestone": "1B-PoC",
        "navigation_mode": "nav_pixel_goal",
        "scene": "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints",
        "real_paris_content": True,
        "trial_protocol": {
            "independent_trials": True,
            "reset_between_valid_goals": True,
            "reset_is_policy_action": False,
        },
        "region": {
            "name": "rue_de_rivoli_sidewalk",
            "agent_spawn_cm": [-26805.82, 9437.53, 90.0],
            "agent_yaw_deg": 165.0056,
            "nav_bounds_center_cm": [-27385.38, 9592.82, 100.0],
            "nav_bounds_extent_cm": [1400.0, 450.0, 300.0],
        },
        "config": {
            "acceptance_radius_cm": 15.0,
            "max_navmesh_adjustment_cm": 10.0,
        },
        "poc_status": {
            "dynamic_runtime_generation": True,
            "supports_runtime_generation": True,
            "navmesh_has_valid_data": True,
            "active_navmesh_tiles": 6,
        },
        "strict_execution": {
            "project_goal_location": False,
            "allow_partial_path": False,
            "straight_line_fallback": False,
            "graph_projection": False,
            "pose_lattice_quantization": False,
        },
        "manual_goals": [
            _goal(1, 0.50, 0.82),
            _goal(2, 0.36, 0.78),
            _goal(3, 0.64, 0.74),
            _goal(4, 0.44, 0.90),
            _goal(5, 0.58, 0.70),
        ],
        "invalid_cases": [
            {
                "name": "facade",
                "requested_uv": [0.95, 0.45],
                "accepted": False,
                "rejection_reason": "hit_not_walkable_ground",
                "raw_world_hit_cm": [-26900.0, 9200.0, 180.0],
            },
            {
                "name": "sky",
                "requested_uv": [0.5, 0.05],
                "accepted": False,
                "rejection_reason": "no_geometry_hit",
                "raw_world_hit_cm": None,
            },
        ],
    }


def test_passing_paris_poc_report_meets_the_narrow_acceptance_contract():
    result = validate_paris_poc_report(_passing_report())

    assert result["passed"] is True
    assert result["valid_goal_count"] == 5
    assert result["invalid_goal_count"] == 2
    assert result["maximum_execution_error_planar_m"] == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.update(scene="/Game/EmptyLevel"),
            "CityCore_Paris",
        ),
        (
            lambda report: report["config"].update(acceptance_radius_cm=25.0),
            "15 cm",
        ),
        (
            lambda report: report["strict_execution"].update(allow_partial_path=True),
            "partial path",
        ),
        (
            lambda report: report["strict_execution"].update(graph_projection=True),
            "graph projection",
        ),
        (
            lambda report: report["poc_status"].update(
                dynamic_runtime_generation=False
            ),
            "dynamic runtime NavMesh",
        ),
        (
            lambda report: report["poc_status"].update(active_navmesh_tiles=0),
            "populated NavMesh tiles",
        ),
        (
            lambda report: report["manual_goals"][0].update(
                execution_error_planar_m=0.25
            ),
            "below 0.25 m",
        ),
        (
            lambda report: report["manual_goals"][0].update(
                feet_position_samples_cm=[[-26800.0, 9437.0, 0.0]]
            ),
            "three distinct",
        ),
        (
            lambda report: report["manual_goals"][0]["trial_reset"].update(
                success=False
            ),
            "independent trial reset",
        ),
        (
            lambda report: report["trial_protocol"].update(
                reset_is_policy_action=True
            ),
            "fixture operation",
        ),
        (
            lambda report: report["invalid_cases"][0].update(accepted=True),
            "invalid goal",
        ),
    ],
)
def test_report_rejects_non_poc_or_privileged_execution(mutate, message):
    report = deepcopy(_passing_report())
    mutate(report)

    with pytest.raises(ValueError, match=message):
        validate_paris_poc_report(report)


def test_report_rejects_non_local_navmesh_bounds():
    report = _passing_report()
    report["region"]["nav_bounds_extent_cm"] = [40_000.0, 40_000.0, 1_000.0]

    with pytest.raises(ValueError, match="local NavMesh"):
        validate_paris_poc_report(report)


def test_report_requires_five_distinct_normalized_actions_and_two_invalids():
    report = _passing_report()
    report["manual_goals"][1]["requested_uv"] = report["manual_goals"][0][
        "requested_uv"
    ]

    with pytest.raises(ValueError, match="five distinct"):
        validate_paris_poc_report(report)

    report = _passing_report()
    report["invalid_cases"].pop()
    with pytest.raises(ValueError, match="exactly two"):
        validate_paris_poc_report(report)
