"""Acceptance checks for a report produced by the live Milestone 1A runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageStat


@pytest.fixture(scope="module")
def live_report() -> dict[str, object]:
    report_path = os.environ.get("PIXEL_GOAL_M1A_REPORT")
    if not report_path:
        pytest.skip("PIXEL_GOAL_M1A_REPORT does not point to a live UE report")
    path = Path(report_path)
    if not path.is_file():
        pytest.fail(f"Pixel Goal live report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_live_report_has_five_distinct_successful_manual_goals(live_report):
    goals = live_report["manual_goals"]

    assert len(goals) == 5
    assert len({tuple(goal["requested_uv"]) for goal in goals}) == 5
    assert all(goal["controller_result"] == "success" for goal in goals)
    assert all(goal["execution_error_planar_m"] <= 0.15 for goal in goals)


def test_live_report_proves_strict_invalid_rejection(live_report):
    invalid_cases = live_report["invalid_cases"]
    reasons = {case["rejection_reason"] for case in invalid_cases}

    assert "hit_not_walkable_ground" in reasons
    assert "no_geometry_hit" in reasons
    assert reasons & {"navmesh_projection_failed", "navmesh_adjustment_exceeded"}
    assert all(case["accepted"] is False for case in invalid_cases)


def test_live_report_proves_continuous_motion_and_exact_frame_binding(live_report):
    for goal in live_report["manual_goals"]:
        samples = goal["feet_position_samples_cm"]
        assert len(samples) >= 3
        assert samples[0] != samples[-1]
        assert goal["camera_snapshot_id"]
        assert goal["camera_intrinsics_id"]
        assert goal["raw_world_hit_cm"]
        assert goal["validated_navigation_target_cm"]
        assert goal["final_feet_position_cm"]


def test_live_report_keeps_v0_tolerances_configurable(live_report):
    config = live_report["config"]

    assert config["max_navmesh_adjustment_cm"] == pytest.approx(10.0)
    assert config["acceptance_radius_cm"] == pytest.approx(15.0)
    assert live_report["navigation_mode"] == "nav_pixel_goal"


def test_live_report_proves_timeout_aborts_controller(live_report):
    cancellation = live_report["timeout_cancellation"]

    assert cancellation["outcome"] == "execution_timeout"
    assert cancellation["controller_result"] == "aborted"
    assert cancellation["failure_reason"] == "execution_timeout"
    assert cancellation["post_cancel_drift_cm"] <= 1.0


def test_live_report_rgb_frames_show_visible_scene_geometry(live_report):
    report_path = Path(os.environ["PIXEL_GOAL_M1A_REPORT"])
    for goal in live_report["manual_goals"]:
        image = Image.open(report_path.parent / goal["input_frame"]).convert("L")
        assert ImageStat.Stat(image).mean[0] >= 10.0
