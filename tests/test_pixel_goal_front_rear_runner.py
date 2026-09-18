"""Deterministic contracts for the dual-view probe and delivery runners.

No test in this module opens a real model endpoint or starts Unreal Engine.
Fakes exercise the model-turn boundary; pure fixtures exercise report
validation and probe geometry; compiled stand-ins exercise launcher identity.
"""

from __future__ import annotations

import argparse
import copy
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
from threading import Thread
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import tools.run_pixel_goal_front_rear_delivery as delivery_runner
import tools.run_pixel_goal_front_rear_probe as probe_runner
from embodiedbench.agent.courier.loop import FormatError
from embodiedbench.agent.courier.session import Frame, Observation
from embodiedbench.agent.courier.tools import TOOLS_BY_NAME, WALK_TO_PIXEL_FRONT_REAR
from embodiedbench.compiler.road_network import RoadNetwork, Street
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
    EmbodiedCourierEnv,
)
from embodiedbench.runtime.live.protocol import Pose
from tools.pixel_goal_courier_backend import (
    TURNING_DROPOFF_NODE_ID,
    TURNING_PICKUP_NODE_ID,
    TURNING_OVERSHOOT_CONTEXT_WAYPOINTS,
    TURNING_REAR_CONTEXT_WAYPOINT,
    TURNING_ROUTE_PROFILE,
    TURNING_ROUTE_WAYPOINTS,
    build_turning_delivery_network,
    turning_route_report,
)
from tools.run_pixel_goal_front_rear_delivery import (
    TURNING_DELIVERY_REGION,
    build_front_rear_setup_request as build_delivery_setup_request,
    run_model_turns,
    validate_front_rear_delivery_report,
)
from tools.run_pixel_goal_front_rear_probe import (
    bearing_degrees,
    build_front_rear_setup_request as build_probe_setup_request,
    planar_target_dot,
    signed_bearing_delta_degrees,
    validate_front_rear_probe_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
# The launcher-identity tests copy the real SimWorldEditor binary out of the
# build the launcher pins; without that build present they cannot run.
_REAL_EDITOR = REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor"


def _require_real_build() -> None:
    if not _REAL_EDITOR.exists():
        pytest.skip("needs the SimWorld UE editor build at .simworld-ue")
SYSTEM_PROMPT = "system mechanics only; no private bindings"
FRAME_KEYS = {
    "label", "path", "kind", "svg", "view_id", "capture_group_id",
    "camera_snapshot_id", "camera_intrinsics_id", "camera_yaw_deg",
    "width", "height", "sha256", "capture_pose", "capture_timing",
    "pair_timing", "required_group",
}


def report_frame(view: str, group: str, turn: int) -> dict[str, object]:
    yaw = 90.0 if view == "front" else 270.0
    return {
        "label": f"[{view}, north] policy-visible caption",
        "path": f"/opaque/{turn:02d}-{0 if view == 'front' else 1:02d}.png",
        "kind": "photograph",
        "svg": "",
        "view_id": view,
        "capture_group_id": group,
        "camera_snapshot_id": f"private-snapshot-{view}-{turn}",
        "camera_intrinsics_id": "private-intrinsics-640x360",
        "camera_yaw_deg": yaw,
        "width": 640,
        "height": 360,
        "sha256": hashlib.sha256(report_image_bytes(view, turn)).hexdigest(),
        "capture_pose": {
            "x_cm": float(turn), "y_cm": 2.0, "z_cm": 90.0,
            "yaw_deg": 90.0,
        },
        "capture_timing": {
            "capture_read_ms": 1.0, "encode_ms": 2.0, "wall_ms": 3.0,
        },
        "pair_timing": {
            "capture_read_ms": 2.0, "encode_ms": 4.0, "wall_ms": 8.0,
        },
        "required_group": True,
    }


def report_image_bytes(view: str, turn: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + f"audited-{view}-{turn}".encode()


def report_image_url(view: str, turn: int) -> str:
    encoded = base64.b64encode(report_image_bytes(view, turn)).decode()
    return f"data:image/png;base64,{encoded}"


def user_message(text: str, turn: int) -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {
                "url": report_image_url("front", turn)}},
            {"type": "image_url", "image_url": {
                "url": report_image_url("rear", turn)}},
        ],
    }


def latency(count: int, median: float, p95: float) -> dict[str, object]:
    return {"count": count, "median_s": median, "p95_s": p95}


def passing_delivery_report() -> dict[str, object]:
    long_reply = (
        "THOUGHT: inspect the simultaneous pair without exposing private ids. "
        + "reasoning stays complete; " * 30
        + '\n```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```'
    )
    second_reply = "THOUGHT: pause\n```\nwait()\n```"
    first_text = "turn one observation; choose visible ground"
    second_text = "turn two observation; fresh simultaneous pair"
    first_frames = [
        report_frame("front", "view-pair-1", 1),
        report_frame("rear", "view-pair-1", 1),
    ]
    second_frames = [
        report_frame("front", "view-pair-2", 2),
        report_frame("rear", "view-pair-2", 2),
    ]
    report = {
        "experiment": "Qwen front/rear pixel-goal delivery (real SPEAR engine)",
        "action_space": "pixel_goal_front_rear",
        "camera_view": "front_rear",
        "region": {"name": "corridor"},
        "pickup_offset_cm": 1_000.0,
        "dropoff_offset_cm": 3_500.0,
        "dropoff_tolerance_cm": 300.0,
        "model": {
            "id": "qwen3-vl-8b",
            "endpoint": "http://127.0.0.1:30001/v1/chat/completions",
            "max_tokens": 400,
            "max_images_per_prompt": 3,
            "history_turns": 8,
            "system_prompt": SYSTEM_PROMPT,
        },
        "setup_request": {
            "scene": "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints",
            "enable_rear_camera": True,
        },
        "setup": {"success": True, "rear_camera_enabled": True},
        "config": {
            "max_navmesh_adjustment_cm": 20.0,
            "acceptance_radius_cm": 15.0,
            "execution_timeout_s": 45.0,
        },
        "turns": 2,
        "transcript": [
            {
                "turn": 1,
                "input": {
                    "system_prompt": SYSTEM_PROMPT,
                    "text": first_text,
                    "frames": first_frames,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        user_message(first_text, 1),
                    ],
                },
                "output": {
                    "raw_reply": long_reply,
                    "parsed_action":
                        'walk_to_pixel(view="rear", u=0.5, v=0.8)',
                    "status": "accepted",
                    "error": "",
                    "feedback": "You walk 7 m toward where you pointed.",
                    "rejected_replies": [],
                    "model_attempts": [],
                },
                "timing": {
                    "model_latency_s": 0.25,
                    "turn_wall_latency_s": 0.5,
                },
                "movement": {
                    "kind": "pixel_goal",
                    "selected_view": "rear",
                    "pixel_uv": [0.5, 0.8],
                    "capture_group_id": "view-pair-1",
                    "camera_snapshot_id": "private-snapshot-rear-1",
                    "feedback": "You walk 7 m toward where you pointed.",
                    "start_pose": {
                        "x_cm": 1.0, "y_cm": 2.0, "z_cm": 90.0,
                        "yaw_deg": 90.0,
                    },
                    "end_pose": {
                        "x_cm": -699.0, "y_cm": 2.0, "z_cm": 90.0,
                        "yaw_deg": 270.0,
                    },
                    "resolution_outcome": "resolved",
                    "controller_outcome": "arrived",
                    "walked_cm": 700.0,
                    "raw": {
                        "kind": "pixel_goal",
                        "selected_view": "rear",
                        "pixel_uv": [0.5, 0.8],
                        "capture_group_id": "view-pair-1",
                        "camera_snapshot_id": "private-snapshot-rear-1",
                        "walked_cm": 700.0,
                        "end_pose": {
                            "x_cm": -699.0, "y_cm": 2.0, "z_cm": 90.0,
                            "yaw_deg": 270.0,
                        },
                        "outcome": "arrived",
                        "controller_status": "arrived",
                    },
                },
            },
            {
                "turn": 2,
                "input": {
                    "system_prompt": None,
                    "text": second_text,
                    "frames": second_frames,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": first_text},
                        {"role": "assistant", "content": long_reply},
                        user_message(second_text, 2),
                    ],
                },
                "output": {
                    "raw_reply": second_reply,
                    "parsed_action": "wait()",
                    "status": "accepted",
                    "error": "",
                    "feedback": "You wait.",
                    "rejected_replies": [],
                    "model_attempts": [],
                },
                "timing": {
                    "model_latency_s": 0.3,
                    "turn_wall_latency_s": 0.6,
                },
                "movement": None,
            },
        ],
        "summary": {
            "delivered": False,
            "latency": {
                "view_pair_capture": latency(2, 0.15, 0.195),
                "pixel_action_execute": latency(1, 0.2, 0.2),
                "model_inference": latency(2, 0.275, 0.2975),
                "turn_wall": latency(2, 0.55, 0.595),
            },
        },
        "termination": "out_of_turns",
        "model_transport_stats": {
            "model_calls": 2,
            "requeries": 0,
            "truncations": 0,
            "budget_clamps": 0,
            "history_drops": 0,
            "transport_retries": 0,
            "timeouts": 0,
            "unparseable_turns": 0,
        },
        "backend_events": [
            {
                "kind": "view_pair_capture",
                "capture_group_id": "view-pair-1",
                "wall_latency_s": 0.1,
                "ue_timing": {"wall_ms": 8.0},
                "views": {
                    "front": {"wall_ms": 3.0},
                    "rear": {"wall_ms": 3.0},
                },
            },
            {
                "kind": "pixel_action_execute",
                "capture_group_id": "view-pair-1",
                "view_id": "rear",
                "camera_snapshot_id": "private-snapshot-rear-1",
                "pixel_uv": [0.5, 0.8],
                "wall_latency_s": 0.2,
                "outcome": "accepted",
            },
            {
                "kind": "view_pair_capture",
                "capture_group_id": "view-pair-2",
                "wall_latency_s": 0.2,
                "ue_timing": {"wall_ms": 8.0},
                "views": {
                    "front": {"wall_ms": 3.0},
                    "rear": {"wall_ms": 3.0},
                },
            },
        ],
    }
    for turn in report["transcript"]:
        reply = turn["output"]["raw_reply"]
        turn["output"]["model_attempts"] = [{
            "payload": {
                "model": "qwen3-vl-8b",
                "messages": copy.deepcopy(turn["input"]["messages"]),
                "max_tokens": 400,
                "temperature": 0.0,
            },
            "result": {
                "choices": [{
                    "message": {"content": reply},
                    "finish_reason": "stop",
                }],
            },
            "error": None,
            "parse_error": None,
        }]
    return report


def test_delivery_report_validator_accepts_complete_serializable_contract():
    report = passing_delivery_report()

    accepted = validate_front_rear_delivery_report(report)

    assert accepted == {"passed": True, "turn_count": 2, "walk_turn_count": 1}
    assert set(report) == {
        "experiment", "action_space", "camera_view", "region",
        "pickup_offset_cm", "dropoff_offset_cm", "dropoff_tolerance_cm",
        "model", "setup_request", "setup", "config", "turns",
        "transcript", "summary", "termination", "model_transport_stats",
        "backend_events",
    }
    assert set(report["transcript"][0]["input"]["frames"][0]) == FRAME_KEYS
    json.dumps(report, allow_nan=False)


def pooled_delivery_report() -> dict[str, object]:
    # This test replays the legacy V1 report contract.  V2 has a dedicated
    # trusted-pedestrian-graph test suite and intentionally contains no
    # CityCore carriageway connectivity.
    pool = delivery_runner.load_validated_delivery_pool(
        REPO_ROOT / "configs/pixel_goal/paris_validated_delivery_pool_v1.json")
    constraints = delivery_runner.OrderConstraints(
        min_delivery_cm=2_000.0,
        max_delivery_cm=15_000.0,
        require_different_streets=True,
    )
    scenario = delivery_runner.resolve_delivery_scenario(
        pool,
        mode="random",
        seed=7,
        constraints=constraints,
    )
    network = delivery_runner.build_validated_pool_network(
        delivery_runner.build_road_network(
            delivery_runner.MAPS, map_name="citycore-paris"),
        pool,
    )
    spawn_node = pool.nodes_by_id[scenario.spawn.node_id]
    region = {
        "name": pool.region.name,
        "agent_spawn_cm": [
            spawn_node.x_cm, spawn_node.y_cm, scenario.spawn.z_cm],
        "agent_yaw_deg": scenario.spawn.yaw_deg,
        "nav_bounds_center_cm": list(pool.region.nav_bounds_center_cm),
        "nav_bounds_extent_cm": list(pool.region.nav_bounds_extent_cm),
    }
    report = passing_delivery_report()
    report.pop("pickup_offset_cm")
    report.pop("dropoff_offset_cm")
    report.update({
        "region": region,
        "pickup_tolerance_cm": 300.0,
        "scenario": delivery_runner.scenario_report(pool, scenario),
        "route": delivery_runner.selected_route_report(network, pool, scenario),
    })
    report["setup_request"].update({
        "agent_spawn_cm": region["agent_spawn_cm"],
        "agent_yaw_deg": region["agent_yaw_deg"],
        "nav_bounds_center_cm": region["nav_bounds_center_cm"],
        "nav_bounds_extent_cm": region["nav_bounds_extent_cm"],
    })
    return report


def test_delivery_report_validator_replays_the_random_pool_request_exactly():
    report = pooled_delivery_report()

    assert validate_front_rear_delivery_report(report) == {
        "passed": True, "turn_count": 2, "walk_turn_count": 1,
    }

    changed_request = copy.deepcopy(report)
    changed_request["scenario"]["request"]["pickup_id"] = "paix-3"
    with pytest.raises(ValueError):
        validate_front_rear_delivery_report(changed_request)

    weaker_runtime = copy.deepcopy(report)
    weaker_runtime["config"]["max_navmesh_adjustment_cm"] = 25.1
    with pytest.raises(ValueError, match="certified limit"):
        validate_front_rear_delivery_report(weaker_runtime)


def test_delivery_report_validator_can_audit_an_explicit_interactive_model_id():
    report = passing_delivery_report()
    model_id = "gpt-5.6-codex-interactive"
    report["model"]["id"] = model_id
    for turn in report["transcript"]:
        for attempt in turn["output"]["model_attempts"]:
            attempt["payload"]["model"] = model_id

    accepted = validate_front_rear_delivery_report(
        report, expected_model_id=model_id)

    assert accepted == {"passed": True, "turn_count": 2, "walk_turn_count": 1}
    with pytest.raises(ValueError, match="report model id is not qwen3-vl-8b"):
        validate_front_rear_delivery_report(report)


def _network_for_turning_route() -> RoadNetwork:
    return RoadNetwork(
        map_name="test-paris",
        streets=[
            Street(
                index=index,
                name=f"Street {index}",
                source=f"source-{index}",
                width_cm=600.0,
                polyline=[(0.0, 0.0), (100.0, 0.0)],
            )
            for index in range(22)
        ],
    )


def test_turning_delivery_overlay_is_an_exact_l_shaped_chain():
    original = _network_for_turning_route()

    overlaid = build_turning_delivery_network(original)

    assert original.nodes == {}
    context_id = TURNING_REAR_CONTEXT_WAYPOINT[0]
    overshoot_ids = [row[0] for row in TURNING_OVERSHOOT_CONTEXT_WAYPOINTS]
    assert set(overlaid.nodes) == {
        *(row[0] for row in TURNING_ROUTE_WAYPOINTS), context_id, *overshoot_ids}
    ids = [row[0] for row in TURNING_ROUTE_WAYPOINTS]
    for index, node_id in enumerate(ids):
        expected = set(ids[max(0, index - 1):index]) | set(ids[index + 1:index + 2])
        if index == 0:
            expected.add(context_id)
        if TURNING_ROUTE_WAYPOINTS[index][4] == "corner":
            expected.add(overshoot_ids[0])
        assert overlaid.nodes[node_id].neighbours == expected
    assert overlaid.nodes[context_id].neighbours == {ids[0]}
    assert len(overlaid.nodes[ids[0]].neighbours) == 2
    corner_id = next(row[0] for row in TURNING_ROUTE_WAYPOINTS if row[4] == "corner")
    for index, overshoot_id in enumerate(overshoot_ids):
        expected = {
            corner_id if index == 0 else overshoot_ids[index - 1],
        }
        if index + 1 < len(overshoot_ids):
            expected.add(overshoot_ids[index + 1])
        assert overlaid.nodes[overshoot_id].neighbours == expected
    assert overlaid.nodes[TURNING_PICKUP_NODE_ID].position == (-5934.8349, -8700.0)
    assert overlaid.nodes[TURNING_DROPOFF_NODE_ID].position == (
        -4692.1603348614335, -4114.651577036972)


def test_long_turn_builds_navmesh_far_enough_to_include_the_far_sidewalk():
    straight_region = delivery_runner.DELIVERY_REGION
    assert TURNING_DELIVERY_REGION.agent_spawn_cm == straight_region.agent_spawn_cm
    assert TURNING_DELIVERY_REGION.nav_bounds_center_cm == (
        straight_region.nav_bounds_center_cm)
    assert straight_region.nav_bounds_extent_cm[0] == 1_000.0
    assert TURNING_DELIVERY_REGION.nav_bounds_extent_cm == (
        2_000.0, 5_000.0, 300.0)
    center_x = TURNING_DELIVERY_REGION.nav_bounds_center_cm[0]
    extent_x = TURNING_DELIVERY_REGION.nav_bounds_extent_cm[0]
    assert center_x + extent_x > -4_300.0


def test_long_turn_waypoints_follow_the_scene_authored_pedestrian_crossing():
    rows = {row[0]: row for row in TURNING_ROUTE_WAYPOINTS}

    assert TURNING_ROUTE_PROFILE == "paris-poc-long-l-turn-v4-pedestrian"
    assert rows["pixelgoal-turn-corner"][1:3] == (-5950.0, -3200.0)
    crossing = [rows[node_id] for node_id in (
        "pixelgoal-turn-crossing-entry",
        "pixelgoal-turn-crossing-centre",
        "pixelgoal-turn-crossing-exit",
    )]
    assert [row[1] for row in crossing] == pytest.approx([
        -5704.827975050328,
        -5406.915006050115,
        -5109.002037049902,
    ])
    assert [row[2] for row in crossing] == pytest.approx([
        -2935.666851566335,
        -2935.666851566335,
        -2935.666851566335,
    ])
    assert [row[4] for row in crossing] == [
        "crossing_entry", "crossing", "crossing_exit"]
    # PR_SidewalkEdgeSeparation sits at x≈-4980. The remaining bend points
    # stay on its building/pedestrian side before reaching the validated door.
    assert all(rows[node_id][1] > -4979.0 for node_id in (
        "pixelgoal-turn-sidewalk-bend-1",
        "pixelgoal-turn-sidewalk-bend-2",
        "pixelgoal-turn-sidewalk-1",
        "pixelgoal-turn-sidewalk-2",
        TURNING_DROPOFF_NODE_ID,
    ))


def test_a_missed_turn_keeps_a_route_back_through_the_corner():
    network = build_turning_delivery_network(_network_for_turning_route())
    env = object.__new__(CourierEnv)
    env.network = network
    overshoot_ids = [row[0] for row in TURNING_OVERSHOOT_CONTEXT_WAYPOINTS]
    corner_id = next(
        row[0] for row in TURNING_ROUTE_WAYPOINTS if row[4] == "corner")

    route = env.route_nodes(overshoot_ids[-1], TURNING_DROPOFF_NODE_ID)

    planned_ids = [row[0] for row in TURNING_ROUTE_WAYPOINTS]
    corner_index = planned_ids.index(corner_id)
    assert route == [*reversed(overshoot_ids), *planned_ids[corner_index:]]


def test_the_false_turn_11_pose_stays_east_until_the_real_zebra_crossing():
    network = build_turning_delivery_network(_network_for_turning_route())
    env = object.__new__(EmbodiedCourierEnv)
    env.network = network
    env.streets = network.streets
    env.action_space = ACTION_SPACE_PIXEL_GOAL_FRONT_REAR
    env.arrived_from = None
    env._last_pose_match = None
    env._walked_bearing = 90.81623840332031

    # Exact Turn 11 pose from the failed v1 rollout.  FPV showed a continuous
    # eastbound pavement here, but v1 named this coordinate the corner and the
    # phone banner falsely said north-east.
    env.node_id = "pixelgoal-turn-straight-2"
    env.ue_pose = Pose(
        -5953.862937666888,
        -5670.207299596036,
        99.25283479690552,
        90.81623840332031,
    )
    selected, _gap = env._match_pose_node(env._here_cm())
    env.node_id = selected
    route = env.route_nodes(selected, TURNING_DROPOFF_NODE_ID)
    instruction = env._next_instruction(env._map_route_points(route))

    assert selected == "pixelgoal-turn-straight-2"
    assert instruction == {"next_street": "Street 21", "next_heading": "east"}

    # Exact folded-route landing from the v3 rollout. V4 has no copied
    # post-corner controller point here, so it remains on the approach.
    approach_start = (-5972.826182160346, -4339.169017188201)
    approach_landing = (-5853.167057123334, -3666.4636815694575)
    env.node_id = "pixelgoal-turn-straight-4"
    env.ue_pose = Pose(*approach_landing, 99.25283479690552, 75.31627012056468)
    env._walked_bearing = bearing_degrees(approach_start, approach_landing)
    selected, _gap = env._match_pose_node(
        env._here_cm(), movement_cm=math.dist(approach_start, approach_landing))
    env.node_id = selected
    route = env.route_nodes(selected, TURNING_DROPOFF_NODE_ID)
    instruction = env._next_instruction(env._map_route_points(route))

    assert selected == "pixelgoal-turn-pre-corner"
    assert instruction == {"next_street": "Street 21", "next_heading": "east"}

    # Exact Turn 18 pose from the v2 rollout: the crossing is plainly visible
    # and the body has not yet followed the curved edge. V4 must switch the
    # phone here to the real north-east crossing entry, not v3's north-west
    # roadway chord.
    env.node_id = "pixelgoal-turn-pre-corner"
    env.ue_pose = Pose(
        -5878.435858882756,
        -3363.040958218099,
        99.25283479690552,
        85.02955339703576,
    )
    env._walked_bearing = 85.02955339703576
    previous_pose = (-5917.275265360121, -3809.630703365772)
    selected, _gap = env._match_pose_node(
        env._here_cm(),
        movement_cm=math.dist(previous_pose, env._here_cm()),
    )
    env.node_id = selected
    route = env.route_nodes(selected, TURNING_DROPOFF_NODE_ID)
    route_points = env._map_route_points(route)
    instruction = env._next_instruction(route_points)
    next_row = next(
        row for row in env._raw_candidates()
        if row["node"] == "pixelgoal-turn-crossing-entry")

    assert selected == "pixelgoal-turn-corner"
    assert instruction == {
        "next_street": "Street 2", "next_heading": "east"}
    assert route_points[0] == (
        -5878.435858882756, -3363.040958218099)
    assert route_points[1] == pytest.approx(
        (-5704.827975050328, -2935.666851566335))
    expected_bearing = bearing_degrees(
        env._here_cm(), (-5704.827975050328, -2935.666851566335))
    assert next_row["bearing"] == pytest.approx(expected_bearing)
    assert next_row["relative"] == "straight ahead"
    assert abs(signed_bearing_delta_degrees(
        env.facing(), next_row["bearing"])) < 45.0


def test_delivery_report_validator_accepts_audited_long_turn_route():
    report = passing_delivery_report()
    report.pop("pickup_offset_cm")
    report.pop("dropoff_offset_cm")
    report["route"] = turning_route_report(
        build_turning_delivery_network(_network_for_turning_route()))

    accepted = validate_front_rear_delivery_report(report)

    assert accepted == {"passed": True, "turn_count": 2, "walk_turn_count": 1}
    assert report["route"]["profile"] == TURNING_ROUTE_PROFILE
    assert report["route"]["planned_total_cm"] > 5_500.0
    assert report["route"]["planned_delivery_cm"] > 4_500.0


def test_delivery_report_validator_rejects_tampered_long_turn_geometry():
    report = passing_delivery_report()
    report.pop("pickup_offset_cm")
    report.pop("dropoff_offset_cm")
    report["route"] = turning_route_report(
        build_turning_delivery_network(_network_for_turning_route()))
    report["route"]["waypoints"][-1]["x_cm"] += 1.0

    with pytest.raises(ValueError, match="waypoint is not exact"):
        validate_front_rear_delivery_report(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(action_space="pixel_goal"), "action_space"),
        (lambda r: r.update(camera_view="forward"), "camera_view"),
        (lambda r: r.update(dropoff_tolerance_cm=800.0), "dropoff tolerance"),
        (lambda r: r["model"].update(id="qwen3-vl-4b"), "model id"),
        (lambda r: r["setup_request"].update(enable_rear_camera=False),
         "rear camera"),
        (lambda r: r["setup"].update(rear_camera_enabled=False),
         "setup response"),
        (lambda r: r["transcript"][0]["input"]["frames"].reverse(),
         "front, rear"),
        (lambda r: r["transcript"][0]["input"]["frames"][1].update(
            capture_group_id="other-group"), "capture group"),
        (lambda r: r["transcript"][0]["input"]["frames"][1].update(
            camera_snapshot_id="private-snapshot-front-1"), "distinct snapshots"),
        (lambda r: r["transcript"][0]["output"].update(raw_reply=""),
         "raw reply"),
        (lambda r: r["transcript"][0]["timing"].pop("model_latency_s"),
         "timing"),
        (lambda r: r["transcript"][0]["input"]["messages"][-1][
            "content"].pop(), "message images"),
        (lambda r: r["transcript"][0]["input"]["messages"][-1][
            "content"][2]["image_url"].update(
                url=report_image_url("front", 1)), "image hash"),
        (lambda r: (
            r["transcript"][0]["input"]["frames"].append(copy.deepcopy(
                r["transcript"][0]["input"]["frames"][0])),
            r["transcript"][0]["input"]["messages"][-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": report_image_url("front", 1)},
            }),
        ), "frame cardinality"),
        (lambda r: r["transcript"][0]["input"]["messages"][-1][
            "content"][0].update(text="different text"), "message text"),
        (lambda r: r["transcript"][1]["input"]["messages"].__setitem__(
            slice(1, 3), list(reversed(
                r["transcript"][1]["input"]["messages"][1:3]))),
         "retained history"),
        (lambda r: r["transcript"][1]["input"]["messages"].pop(1),
         "retained history"),
        (lambda r: r["transcript"][0]["movement"].update(
            selected_view="front"), "selected view"),
        (lambda r: r["transcript"][0]["movement"].update(pixel_uv=[0.4, 0.8]),
         "pixel"),
        (lambda r: r["transcript"][0]["movement"]["raw"].update(
            capture_group_id="different"), "raw movement"),
        (lambda r: r["transcript"][0]["movement"]["raw"].update(
            outcome="timeout"), "raw movement outcome"),
        (lambda r: r["transcript"][0]["movement"]["raw"]["end_pose"].update(
            x_cm=-650.0), "raw movement end pose"),
        (lambda r: r["transcript"][0]["movement"]["raw"].update(
            walked_cm=701.0), "raw movement distance"),
        (lambda r: (
            r["transcript"][0]["movement"].update(
                controller_outcome="success"),
            r["transcript"][0]["movement"]["raw"].update(
                outcome="success", controller_status="success"),
        ), "controller outcome"),
        (lambda r: r["transcript"][0]["input"]["frames"][0][
            "capture_pose"].update(x_cm=2.0), "start pose"),
        (lambda r: r["transcript"][0]["movement"].pop("start_pose"),
         "movement contract"),
        (lambda r: r["transcript"][0]["movement"].update(
            controller_outcome=""), "controller outcome"),
        (lambda r: (
            r["transcript"][0]["input"].update(
                text="leak private-snapshot-rear-1"),
            r["transcript"][0]["input"]["messages"][-1]["content"][0].update(
                text="leak private-snapshot-rear-1"),
        ), "private frame identity"),
        (lambda r: r["summary"]["latency"].pop("turn_wall"),
         "summary latency"),
        (lambda r: r["summary"]["latency"]["model_inference"].update(
            count=1), "latency count"),
        (lambda r: r["summary"]["latency"]["turn_wall"].update(
            p95_s=0.1), "p95"),
        (lambda r: r["summary"]["latency"]["model_inference"].update(
            median_s=0.28, p95_s=0.3), "model_inference latency"),
        (lambda r: r["transcript"][0]["output"].update(
            raw_reply=r["transcript"][0]["output"]["raw_reply"][:400]),
         "complete raw reply"),
        (lambda r: r["summary"].update(unserializable={"bad"}),
         "JSON serializable"),
        (lambda r: r["transcript"][0]["timing"].update(
            model_latency_s=math.nan), "finite"),
    ],
)
def test_delivery_report_validator_fails_closed(mutate, message):
    report = copy.deepcopy(passing_delivery_report())
    mutate(report)

    with pytest.raises(ValueError, match=message):
        validate_front_rear_delivery_report(report)


def test_delivery_report_accepts_audited_exhausted_format_error_turn():
    report = passing_delivery_report()
    turn = report["transcript"][1]
    turn["output"].update({
        "raw_reply": "I cannot format the requested call.",
        "parsed_action": None,
        "status": "format_error",
        "error": "reply contains no fenced action",
        "feedback": "That reply could not be read as an action.",
        "rejected_replies": ["I cannot format the requested call."],
    })
    turn["output"]["model_attempts"][0]["result"]["choices"][0][
        "message"]["content"] = "I cannot format the requested call."
    turn["movement"] = None
    report["model_transport_stats"]["unparseable_turns"] = 1

    accepted = validate_front_rear_delivery_report(report)

    assert accepted["turn_count"] == 2


def test_delivery_report_accepts_a_run_containing_only_audited_format_error():
    report = passing_delivery_report()
    turn = copy.deepcopy(report["transcript"][1])
    turn["turn"] = 1
    turn["input"]["system_prompt"] = SYSTEM_PROMPT
    turn["input"]["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_message(turn["input"]["text"], 2),
    ]
    turn["output"].update({
        "raw_reply": "still no fenced action",
        "parsed_action": None,
        "status": "format_error",
        "error": "reply contains no fenced action",
        "feedback": "That reply could not be read as an action.",
        "rejected_replies": ["still no fenced action"],
    })
    attempt = turn["output"]["model_attempts"][0]
    attempt["payload"]["messages"] = copy.deepcopy(turn["input"]["messages"])
    attempt["result"]["choices"][0]["message"]["content"] = (
        "still no fenced action")
    report["turns"] = 1
    report["transcript"] = [turn]
    report["backend_events"] = [report["backend_events"][2]]
    report["summary"]["latency"] = {
        "view_pair_capture": latency(1, 0.2, 0.2),
        "pixel_action_execute": latency(0, None, None),
        "model_inference": latency(1, 0.3, 0.3),
        "turn_wall": latency(1, 0.6, 0.6),
    }
    report["model_transport_stats"].update(
        model_calls=1, unparseable_turns=1)

    assert validate_front_rear_delivery_report(report) == {
        "passed": True, "turn_count": 1, "walk_turn_count": 0,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda output: output.update(status="accepted", error="not empty"),
        lambda output: output.update(status="format_error", error=""),
        lambda output: output.update(status="invented", error="bad"),
    ],
)
def test_delivery_report_rejects_inconsistent_status_and_error(mutate):
    report = passing_delivery_report()
    mutate(report["transcript"][1]["output"])

    with pytest.raises(ValueError, match="status|error"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_recomputes_backend_latency_from_retained_events():
    report = passing_delivery_report()
    report["backend_events"][0]["wall_latency_s"] = 0.3

    with pytest.raises(ValueError, match="view_pair_capture latency"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_recomputes_action_latency_from_retained_events():
    report = passing_delivery_report()
    report["backend_events"][1]["wall_latency_s"] = 0.25

    with pytest.raises(ValueError, match="pixel_action_execute latency"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_cross_checks_controller_outcome_with_turn_status():
    report = passing_delivery_report()
    movement = report["transcript"][0]["movement"]
    movement["controller_outcome"] = "timeout"
    movement["raw"].update(outcome="timeout", controller_status="timeout")
    report["backend_events"][1]["outcome"] = "execution_timeout"

    with pytest.raises(ValueError, match="movement outcome.*turn status"):
        validate_front_rear_delivery_report(report)


def refused_delivery_report_with_end_yaw(end_yaw: float) -> dict[str, object]:
    report = passing_delivery_report()
    turn = report["transcript"][0]
    start_pose = {
        "x_cm": 1.0,
        "y_cm": 2.0,
        "z_cm": 90.0,
        "yaw_deg": -101.79339666864348,
    }
    end_pose = {**start_pose, "yaw_deg": end_yaw}
    for frame in turn["input"]["frames"]:
        frame["capture_pose"] = copy.deepcopy(start_pose)
    feedback = "That did not work: selected ground is not walkable."
    turn["output"].update({
        "status": "rejected",
        "error": "unwalkable_pixel",
        "feedback": feedback,
    })
    turn["movement"] = {
        "kind": "pixel_refused",
        "selected_view": "rear",
        "pixel_uv": [0.5, 0.8],
        "capture_group_id": "view-pair-1",
        "camera_snapshot_id": "private-snapshot-rear-1",
        "feedback": feedback,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "resolution_outcome": "rejected",
        "controller_outcome": "not_started",
        "walked_cm": 0.0,
        "raw": {
            "kind": "pixel_refused",
            "selected_view": "rear",
            "pixel_uv": [0.5, 0.8],
            "capture_group_id": "view-pair-1",
            "camera_snapshot_id": "private-snapshot-rear-1",
            "code": "unwalkable_pixel",
        },
    }
    report["backend_events"][1]["outcome"] = "resolution_rejected"
    return report


def test_delivery_report_treats_wrapped_yaw_as_the_same_refused_pose():
    report = refused_delivery_report_with_end_yaw(258.20660400390625)

    accepted = validate_front_rear_delivery_report(report)

    assert accepted["passed"] is True
    assert accepted["walk_turn_count"] == 1


def test_delivery_report_still_rejects_a_real_refused_yaw_change():
    report = refused_delivery_report_with_end_yaw(259.20660400390625)

    with pytest.raises(ValueError, match="refused movement changed the pose"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_cross_checks_model_attempt_statistics():
    report = passing_delivery_report()
    report["transcript"][1]["output"]["model_attempts"].clear()

    with pytest.raises(ValueError, match="model attempt"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_cross_checks_internal_history_drop_statistics():
    report = passing_delivery_report()
    report["model_transport_stats"]["history_drops"] = 1

    with pytest.raises(ValueError, match="model attempt"):
        validate_front_rear_delivery_report(report)


@pytest.mark.parametrize("field", ["timeouts", "transport_retries"])
def test_delivery_report_cross_checks_transport_retry_statistics(field):
    report = passing_delivery_report()
    report["model_transport_stats"][field] = 1

    with pytest.raises(ValueError, match="model attempt"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_allows_budget_truncation_after_a_parseable_model_reply():
    report = passing_delivery_report()
    output = report["transcript"][1]["output"]
    output.update({
        "parsed_action": None,
        "status": "truncated",
        "error": "wall_seconds",
        "feedback": "The shift budget was exhausted.",
    })

    assert validate_front_rear_delivery_report(report)["passed"] is True


def test_delivery_report_records_and_validates_a_final_turn_truncation_attempt():
    report = passing_delivery_report()
    final_output = report["transcript"][-1]["output"]
    truncated = copy.deepcopy(final_output["model_attempts"][0])
    truncated["payload"]["max_tokens"] = 200
    truncated["result"]["choices"][0] = {
        "message": {"content": "unfinished reasoning"},
        "finish_reason": "length",
    }
    final_output["model_attempts"].insert(0, truncated)
    report["model_transport_stats"].update(model_calls=3, truncations=1)

    assert validate_front_rear_delivery_report(report)["passed"] is True

    final_output["model_attempts"].pop(0)
    with pytest.raises(ValueError, match="model attempt"):
        validate_front_rear_delivery_report(report)


def test_delivery_report_accepts_one_positionally_final_phone_map_frame():
    report = passing_delivery_report()
    turn = report["transcript"][0]
    map_frame = {
        key: None for key in FRAME_KEYS
    }
    map_frame.update({
        "label": "[map] the map on your phone",
        "path": "",
        "kind": "map",
        "svg": "<svg xmlns='http://www.w3.org/2000/svg'/>",
        "required_group": False,
    })
    turn["input"]["frames"].append(map_frame)
    map_item = {
        "type": "image_url",
        "image_url": {"url": report_image_url("map", 1)},
    }
    turn["input"]["messages"][-1]["content"].append(map_item)
    turn["output"]["model_attempts"][0]["payload"]["messages"] = copy.deepcopy(
        turn["input"]["messages"])

    assert validate_front_rear_delivery_report(report)["passed"] is True


def observation(tmp_path: Path, turn: int) -> Observation:
    group = f"view-pair-{turn}"
    frames: list[Frame] = []
    for index, view in enumerate(("front", "rear")):
        path = tmp_path / f"opaque-{turn}-{index}.png"
        path.write_bytes(f"frame-{turn}-{view}".encode())
        frames.append(Frame(
            label=f"[{view}] visible caption",
            path=str(path),
            view_id=view,
            capture_group_id=group,
            camera_snapshot_id=f"secret-snapshot-{turn}-{view}",
            camera_intrinsics_id="secret-intrinsics",
            camera_yaw_deg=90.0 if view == "front" else 270.0,
            width=640,
            height=360,
            sha256=("a" if view == "front" else "b") * 64,
            capture_pose={
                "x_cm": float(turn - 1) * 700.0,
                "y_cm": 0.0,
                "z_cm": 90.0,
                "yaw_deg": 90.0,
            },
            capture_timing={
                "capture_read_ms": 1.0, "encode_ms": 2.0, "wall_ms": 3.0,
            },
            pair_timing={
                "capture_read_ms": 2.0, "encode_ms": 4.0, "wall_ms": 8.0,
            },
            required_group=True,
        ))
    return Observation(
        text=f"observation {turn}; no private snapshot or group value", frames=frames)


class FakePose:
    def __init__(self, x: float, yaw: float = 90.0) -> None:
        self.x_cm = x
        self.y_cm = 0.0
        self.z_cm = 90.0
        self.yaw_deg = yaw

    def to_dict(self) -> dict[str, float]:
        return {
            "x_cm": self.x_cm, "y_cm": self.y_cm, "z_cm": self.z_cm,
            "yaw_deg": self.yaw_deg,
        }


class FakeCourierSession:
    allowed = ["walk_to_pixel", "wait"]

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = observations
        self.index = 0
        self.pending: Observation | None = None
        self.finished = False
        self.feedback = ""
        self.env = SimpleNamespace(embodied_log=[], ue_pose=FakePose(0.0))
        self.replies: list[str] = []

    def observe(self) -> Observation:
        if self.pending is None:
            self.pending = self.observations[self.index]
        return self.pending

    def step(self, reply: str):
        assert self.pending is self.observations[self.index]
        self.pending = None
        self.replies.append(reply)
        if self.index == 0:
            self.env.ue_pose = FakePose(-700.0, yaw=270.0)
            self.env.embodied_log.append({
                "kind": "pixel_goal",
                "selected_view": "rear",
                "pixel_uv": (0.5, 0.8),
                "capture_group_id": "view-pair-1",
                "camera_snapshot_id": "secret-snapshot-1-rear",
                "outcome": "arrived",
                "walked_cm": 700.0,
                "end_pose": self.env.ue_pose.to_dict(),
            })
            self.feedback = "You walk 7 m toward where you pointed."
            log = SimpleNamespace(
                action='walk_to_pixel(view="rear", u=0.5, v=0.8)',
                status="accepted", error="",
            )
        else:
            self.feedback = "You wait."
            log = SimpleNamespace(action="wait()", status="accepted", error="")
            self.finished = True
        self.index += 1
        return log


class DuplicateMovementSession(FakeCourierSession):
    def step(self, reply: str):
        log = super().step(reply)
        self.env.embodied_log.append(copy.deepcopy(self.env.embodied_log[-1]))
        return log


class FormatErrorCourierSession(FakeCourierSession):
    def step(self, reply: str):
        assert self.pending is self.observations[self.index]
        self.pending = None
        self.replies.append(reply)
        self.feedback = "That reply could not be read as an action."
        self.finished = True
        self.index += 1
        return SimpleNamespace(
            action="", status="format_error",
            error="reply contains no fenced action",
        )


class ManyTurnSession(FakeCourierSession):
    def step(self, reply: str):
        assert self.pending is self.observations[self.index]
        self.pending = None
        self.replies.append(reply)
        self.feedback = "You wait."
        self.index += 1
        if self.index == len(self.observations):
            self.finished = True
        return SimpleNamespace(action="wait()", status="accepted", error="")


class FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.messages: list[list[dict[str, object]]] = []
        self.stats = SimpleNamespace(as_dict=lambda: {"model_calls": 2})

    def act(self, messages, parse):
        self.messages.append(copy.deepcopy(messages))
        reply = self.replies[len(self.messages) - 1]
        return reply, parse(reply), []


class ParserCheckingModel:
    def act(self, _messages, parse):
        malformed = (
            "THOUGHT: wrong view type\n"
            "```\nwalk_to_pixel(view=7, u=0.5, v=0.8)\n```"
        )
        with pytest.raises(FormatError, match="view"):
            parse(malformed)
        valid = (
            "THOUGHT: corrected\n"
            '```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```'
        )
        return valid, parse(valid), [malformed]


def test_model_requery_parser_uses_the_active_dual_view_tool_metadata(tmp_path):
    """Using the legacy global tool would silently accept numeric ``view``."""
    session = FakeCourierSession([observation(tmp_path, 1)])
    session._tools_by_name = {
        "walk_to_pixel": WALK_TO_PIXEL_FRONT_REAR,
        "wait": TOOLS_BY_NAME["wait"],
    }
    ticks = iter((0.0, 0.1, 0.2, 0.3))

    transcript = run_model_turns(
        session,
        ParserCheckingModel(),
        system_prompt=SYSTEM_PROMPT,
        max_turns=1,
        history_turns=0,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: (
            "data:image/png;base64," + Path(frame.path).name),
    )

    assert transcript[0]["output"]["parsed_action"] == (
        'walk_to_pixel(view="rear", u=0.5, v=0.8)')


def test_model_loop_requires_one_unique_movement_event_for_the_exact_binding(
        tmp_path):
    session = DuplicateMovementSession([observation(tmp_path, 1)])
    model = FakeModel([
        'THOUGHT: reverse\n```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```',
    ])
    ticks = iter((0.0, 0.1, 0.2, 0.3))

    with pytest.raises(RuntimeError, match="unique matching movement"):
        run_model_turns(
            session,
            model,
            system_prompt=SYSTEM_PROMPT,
            max_turns=1,
            history_turns=0,
            perf_counter_fn=lambda: next(ticks),
            image_encoder=lambda frame, _index: report_image_url(
                str(frame.view_id), 1),
        )


def test_real_model_client_requery_attempts_retain_exact_payloads_and_results(
        tmp_path):
    malformed = "not a callable response"
    valid = 'THOUGHT: reverse\n```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```'

    class ScriptedClient(delivery_runner.AuditedModelClient):
        def __init__(self):
            super().__init__(
                "http://unused.invalid", "qwen3-vl-8b", max_requeries=1)
            self.responses = [
                {"choices": [{"message": {"content": malformed},
                               "finish_reason": "stop"}]},
                {"choices": [{"message": {"content": valid},
                               "finish_reason": "stop"}]},
            ]

        def _raw_post(self, _payload):
            return self.responses.pop(0)

    session = FakeCourierSession([observation(tmp_path, 1)])
    ticks = iter((0.0, 0.1, 0.2, 0.3))
    transcript = run_model_turns(
        session,
        ScriptedClient(),
        system_prompt=SYSTEM_PROMPT,
        max_turns=1,
        history_turns=0,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: report_image_url(
            str(frame.view_id), 1),
    )

    output = transcript[0]["output"]
    attempts = output["model_attempts"]
    assert output["rejected_replies"] == [malformed]
    assert [attempt["result"]["choices"][0]["message"]["content"]
            for attempt in attempts] == [malformed, valid]
    assert attempts[0]["payload"]["messages"] == transcript[0]["input"]["messages"]
    assert attempts[1]["payload"]["messages"][-2:] == [
        {"role": "assistant", "content": malformed},
        {"role": "user", "content": (
            "That reply could not be read as an action: No action found. End your "
            "reply with a fenced block containing exactly one call. The block "
            "holds the call and nothing else.\n"
            "Reply again, in the required shape, with exactly one call."
        )},
    ]


def test_real_model_transport_retry_records_the_failed_http_attempt(
        tmp_path, monkeypatch):
    valid = 'THOUGHT: reverse\n```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```'
    responses = [
        TimeoutError("queue timed out"),
        {"choices": [{"message": {"content": valid},
                       "finish_reason": "stop"}]},
    ]

    def scripted_post(_client, _payload):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(delivery_runner.ModelClient, "_post", scripted_post)
    monkeypatch.setattr(
        delivery_runner.time, "sleep", lambda _seconds: None, raising=False)
    session = FakeCourierSession([observation(tmp_path, 1)])
    model = delivery_runner.AuditedModelClient(
        "http://unused.invalid", "qwen3-vl-8b", max_timeout_retries=1)
    ticks = iter((0.0, 0.1, 0.2, 0.3))

    output = run_model_turns(
        session,
        model,
        system_prompt=SYSTEM_PROMPT,
        max_turns=1,
        history_turns=0,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: report_image_url(
            str(frame.view_id), 1),
    )[0]["output"]

    assert len(output["model_attempts"]) == 2
    assert output["model_attempts"][0]["result"] is None
    assert output["model_attempts"][0]["error"] == (
        "TimeoutError: queue timed out")
    assert output["model_attempts"][1]["result"]["choices"][0][
        "message"]["content"] == valid
    assert model.stats.timeouts == 1


def test_real_model_client_exhaustion_keeps_every_reply_and_format_turn(
        tmp_path):
    replies = ["still malformed one", "still malformed two"]

    class ExhaustedClient(delivery_runner.AuditedModelClient):
        def __init__(self):
            super().__init__(
                "http://unused.invalid", "qwen3-vl-8b", max_requeries=1)
            self.responses = [
                {"choices": [{"message": {"content": reply},
                               "finish_reason": "stop"}]}
                for reply in replies
            ]

        def _raw_post(self, _payload):
            return self.responses.pop(0)

    session = FormatErrorCourierSession([observation(tmp_path, 1)])
    ticks = iter((0.0, 0.1, 0.2, 0.3))
    transcript = run_model_turns(
        session,
        ExhaustedClient(),
        system_prompt=SYSTEM_PROMPT,
        max_turns=1,
        history_turns=0,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: report_image_url(
            str(frame.view_id), 1),
    )

    turn = transcript[0]
    assert turn["output"]["raw_reply"] == replies[-1]
    assert turn["output"]["rejected_replies"] == replies
    assert turn["output"]["parsed_action"] is None
    assert turn["output"]["status"] == "format_error"
    assert turn["movement"] is None
    assert len(turn["output"]["model_attempts"]) == 2


def run_final_truncation_turn(tmp_path):
    truncated = "unfinished reasoning and no complete call"

    class TruncatedClient(delivery_runner.AuditedModelClient):
        def __init__(self):
            super().__init__(
                "http://unused.invalid", "qwen3-vl-8b",
                max_requeries=0, max_tokens=100, max_tokens_ceiling=200)

        def _raw_post(self, _payload):
            return {"choices": [{
                "message": {"content": truncated},
                "finish_reason": "length",
            }]}

    current = observation(tmp_path, 1)
    current.frames = [
        replace(
            frame,
            sha256=hashlib.sha256(
                report_image_bytes(str(frame.view_id), 1)).hexdigest(),
        )
        for frame in current.frames
    ]
    session = FormatErrorCourierSession([current])
    model = TruncatedClient()
    ticks = iter((0.0, 0.1, 0.2, 0.3))
    turn = run_model_turns(
        session,
        model,
        system_prompt=SYSTEM_PROMPT,
        max_turns=1,
        history_turns=0,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: report_image_url(
            str(frame.view_id), 1),
    )[0]

    return truncated, turn, model


def test_real_model_client_final_truncation_is_not_lost(tmp_path):
    truncated, turn, _model = run_final_truncation_turn(tmp_path)

    assert turn["output"]["raw_reply"] == truncated
    assert turn["output"]["parsed_action"] is None
    assert turn["output"]["model_attempts"][0]["result"][
        "choices"][0]["finish_reason"] == "length"


def test_real_runner_exhausted_final_truncation_validates_as_audited_turn(
        tmp_path):
    truncated, turn, model = run_final_truncation_turn(tmp_path)
    report = passing_delivery_report()
    report["model"].update(max_tokens=100, history_turns=0)
    report["turns"] = 1
    report["transcript"] = [turn]
    report["summary"]["latency"] = {
        "view_pair_capture": latency(1, 0.1, 0.1),
        "pixel_action_execute": latency(0, None, None),
        "model_inference": latency(1, 0.1, 0.1),
        "turn_wall": latency(1, 0.3, 0.3),
    }
    report["model_transport_stats"] = model.stats.as_dict()
    report["backend_events"] = [report["backend_events"][0]]

    assert turn["output"]["raw_reply"] == truncated
    assert turn["output"]["rejected_replies"] == []
    assert validate_front_rear_delivery_report(report) == {
        "passed": True, "turn_count": 1, "walk_turn_count": 0,
    }

    not_truncated = copy.deepcopy(report)
    not_truncated["transcript"][0]["output"]["model_attempts"][-1][
        "result"]["choices"][0]["finish_reason"] = "stop"
    with pytest.raises(ValueError, match="rejected replies"):
        validate_front_rear_delivery_report(not_truncated)


def test_real_model_client_history_drop_keeps_the_failed_and_retried_payloads(
        tmp_path):
    first = 'THOUGHT: reverse\n```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```'
    second = "THOUGHT: wait\n```\nwait()\n```"

    class HistoryDropClient(delivery_runner.AuditedModelClient):
        def __init__(self):
            super().__init__(
                "http://unused.invalid", "qwen3-vl-8b", max_requeries=1)
            self.responses = [
                {"choices": [{"message": {"content": first},
                               "finish_reason": "stop"}]},
                RuntimeError(
                    "HTTP 400: Input length (10384) exceeds model's maximum "
                    "context length (10240)."),
                {"choices": [{"message": {"content": second},
                               "finish_reason": "stop"}]},
            ]

        def _raw_post(self, _payload):
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    session = FakeCourierSession([
        observation(tmp_path, 1), observation(tmp_path, 2),
    ])
    model = HistoryDropClient()
    ticks = iter(float(index) / 10.0 for index in range(8))
    transcript = run_model_turns(
        session,
        model,
        system_prompt=SYSTEM_PROMPT,
        max_turns=2,
        history_turns=8,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: report_image_url(
            str(frame.view_id), int(frame.capture_group_id.rsplit("-", 1)[1])),
    )

    attempts = transcript[1]["output"]["model_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["result"] is None
    assert "maximum context length" in attempts[0]["error"]
    # The retry sheds the oldest user/assistant exchange and KEEPS the system
    # message in front of it -- dropping messages[0] took the tool manual
    # with it, and the rest of the episode ran without a rulebook.
    first = attempts[0]["payload"]["messages"]
    assert first[0]["role"] == "system"
    assert attempts[1]["payload"]["messages"] == first[:1] + first[3:]
    assert model.stats.history_drops == 1


def test_two_turn_model_loop_preserves_exact_inputs_raw_reply_and_binding(
        tmp_path):
    first = (
        "THOUGHT: " + "keep all of this model output. " * 30
        + '\n```\nwalk_to_pixel(view="rear", u=0.5, v=0.8)\n```'
    )
    second = "THOUGHT: wait\n```\nwait()\n```"
    session = FakeCourierSession([observation(tmp_path, 1), observation(tmp_path, 2)])
    model = FakeModel([first, second])
    ticks = iter((0.0, 0.1, 0.35, 0.5, 1.0, 1.1, 1.4, 1.6))

    transcript = run_model_turns(
        session,
        model,
        system_prompt=SYSTEM_PROMPT,
        max_turns=2,
        history_turns=8,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: (
            "data:image/png;base64," + Path(frame.path).name),
    )

    assert session.replies == [first, second]
    assert transcript[0]["output"]["raw_reply"] == first
    assert len(transcript[0]["output"]["raw_reply"]) > 400
    assert transcript[0]["input"]["system_prompt"] == SYSTEM_PROMPT
    assert transcript[1]["input"]["system_prompt"] is None
    assert transcript[0]["input"]["text"] == session.observations[0].text
    assert transcript[0]["input"]["frames"] == [
        frame.to_dict() for frame in session.observations[0].frames]
    assert transcript[0]["input"]["messages"] == model.messages[0]
    assert transcript[1]["input"]["messages"] == model.messages[1]
    assert "secret-snapshot" not in transcript[0]["input"]["text"]
    assert "secret-snapshot" not in json.dumps(model.messages)
    assert transcript[0]["input"]["frames"][1][
        "camera_snapshot_id"] == "secret-snapshot-1-rear"
    assert transcript[0]["movement"] == {
        "kind": "pixel_goal",
        "selected_view": "rear",
        "pixel_uv": [0.5, 0.8],
        "capture_group_id": "view-pair-1",
        "camera_snapshot_id": "secret-snapshot-1-rear",
        "feedback": "You walk 7 m toward where you pointed.",
        "start_pose": {
            "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0, "yaw_deg": 90.0,
        },
        "end_pose": {
            "x_cm": -700.0, "y_cm": 0.0, "z_cm": 90.0,
            "yaw_deg": 270.0,
        },
        "resolution_outcome": "resolved",
        "controller_outcome": "arrived",
        "walked_cm": 700.0,
        "raw": {
            "kind": "pixel_goal",
            "selected_view": "rear",
            "pixel_uv": (0.5, 0.8),
            "capture_group_id": "view-pair-1",
            "camera_snapshot_id": "secret-snapshot-1-rear",
            "outcome": "arrived",
            "walked_cm": 700.0,
            "end_pose": {
                "x_cm": -700.0, "y_cm": 0.0, "z_cm": 90.0,
                "yaw_deg": 270.0,
            },
        },
    }
    assert transcript[0]["timing"] == {
        "model_latency_s": pytest.approx(0.25),
        "turn_wall_latency_s": pytest.approx(0.5),
    }
    assert transcript[1]["timing"] == {
        "model_latency_s": pytest.approx(0.3),
        "turn_wall_latency_s": pytest.approx(0.6),
    }


def test_model_loop_retains_complete_bounded_history_through_the_final_turn(
        tmp_path):
    observations = [observation(tmp_path, turn) for turn in range(1, 11)]
    replies = [f"THOUGHT: {turn}\n```\nwait()\n```" for turn in range(1, 11)]
    session = ManyTurnSession(observations)
    model = FakeModel(replies)
    ticks = iter(float(index) / 10.0 for index in range(40))

    transcript = run_model_turns(
        session,
        model,
        system_prompt=SYSTEM_PROMPT,
        max_turns=10,
        history_turns=8,
        perf_counter_fn=lambda: next(ticks),
        image_encoder=lambda frame, _index: report_image_url(
            str(frame.view_id), int(frame.capture_group_id.rsplit("-", 1)[1])),
    )

    final_messages = transcript[-1]["input"]["messages"]
    assert len(final_messages) == 18
    assert [message["role"] for message in final_messages] == [
        "system",
        *[role for _turn in range(8) for role in ("user", "assistant")],
        "user",
    ]
    assert final_messages[1] == {
        "role": "user", "content": observations[1].text,
    }
    assert final_messages[-2] == {
        "role": "assistant", "content": replies[8],
    }


def probe_latency_benchmark() -> dict[str, object]:
    single_samples = []
    dual_samples = []
    for cycle in range(1, 21):
        single_samples.append({
            "cycle": cycle,
            "sequence": (cycle - 1) * 2,
            "capture_timing": {
                "total_ms": 3.0 * cycle,
                "lookup_ms": 0.1,
                "init_ms": 0.2,
                "capture_read_ms": float(cycle),
                "encode_ms": 2.0 * cycle,
            },
            "client_wall_s": 0.25,
            "camera_location_cm": [10.0, 20.0, 160.0],
            "camera_rotation_degrees": [-10.0, 0.0, 0.0],
        })
        dual_samples.append({
            "cycle": cycle,
            "sequence": (cycle - 1) * 2 + 1,
            "pose": [0.0, 0.0, 90.0, 0.0],
            "views": [
                {
                    "view_id": "front",
                    "camera_location_cm": [10.0, 20.0, 160.0],
                    "camera_rotation_degrees": [-10.0, 0.0, 0.0],
                },
                {
                    "view_id": "rear",
                    "camera_location_cm": [10.0, 20.0, 160.0],
                    "camera_rotation_degrees": [-10.0, 180.0, 0.0],
                },
            ],
            "capture_timing": {
                "capture_read_ms": 2.0 * cycle,
                "encode_ms": 4.0 * cycle,
                "wall_ms": 6.0 * cycle,
            },
            "client_wall_s": 0.5,
        })
    return {
        "cycles": 20,
        "width_px": 640,
        "height_px": 360,
        "fov_degrees": 90.0,
        "order": [condition for _cycle in range(20)
                  for condition in ("single", "dual")],
        "before_status": {
            "success": True,
            "poc_ready": True,
            "agent_position_cm": [0.0, 0.0, 90.0],
        },
        "after_status": {
            "success": True,
            "poc_ready": True,
            "agent_position_cm": [0.0, 0.0, 90.0],
        },
        "single_samples": single_samples,
        "dual_samples": dual_samples,
        "statistics": {
            "single": {
                "capture_read_ms": {"count": 20, "median": 10.5, "p95": 19.05},
                "encode_ms": {"count": 20, "median": 21.0, "p95": 38.1},
                "ue_wall_ms": {
                    "count": 20, "median": 31.5, "p95": 57.150000000000006,
                },
                "client_wall_s": {"count": 20, "median": 0.25, "p95": 0.25},
            },
            "dual": {
                "capture_read_ms": {"count": 20, "median": 21.0, "p95": 38.1},
                "encode_ms": {"count": 20, "median": 42.0, "p95": 76.2},
                "ue_wall_ms": {
                    "count": 20, "median": 63.0, "p95": 114.30000000000001,
                },
                "client_wall_s": {"count": 20, "median": 0.5, "p95": 0.5},
            },
        },
        "dual_over_single_ratio": {
            metric: {"median": 2.0, "p95": 2.0}
            for metric in (
                "capture_read_ms", "encode_ms", "ue_wall_ms", "client_wall_s")
        },
    }


def probe_png_bytes(luma: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (640, 360), (luma, luma, luma)).save(
        output, format="PNG")
    return output.getvalue()


def probe_image_record(index: int, *, luma: int | None = None) -> dict[str, object]:
    pair_index, view_index = divmod(index, 2)
    view_id = ("front", "rear")[view_index]
    level = index + 10 if luma is None else luma
    data = probe_png_bytes(level)
    return {
        "pair_index": pair_index,
        "view_id": view_id,
        "artifact": f"pair-{pair_index:02d}-{view_index:02d}.png",
        "bytes_base64": base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_size": len(data),
        "width_px": 640,
        "height_px": 360,
        "luma": {"min": level, "mean": float(level), "max": level},
    }


def passing_probe_report() -> dict[str, object]:
    return {
        "experiment": "front/rear scripted Pixel Goal probe",
        "action_space": "pixel_goal_front_rear",
        "camera_view": "front_rear",
        "setup_request": {"enable_rear_camera": True},
        "setup": {"success": True, "rear_camera_enabled": True},
        "latency_benchmark": probe_latency_benchmark(),
        "image_evidence": [probe_image_record(index) for index in range(12)],
        "rear_reversal": {
            "selected_view": "rear",
            "pixel_uv": [0.5, 0.8],
            "capture_group_id": "view-pair-1",
            "camera_snapshot_id": "rear-snapshot-1",
            "start_pose": {
                "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0, "yaw_deg": 0.0,
            },
            "accepted_target_cm": [-800.0, 0.0, 0.0],
            "end_pose": {
                "x_cm": -700.0, "y_cm": 0.0, "z_cm": 90.0,
                "yaw_deg": 180.0,
            },
            "target_dot_body_forward_cm": -800.0,
            "controller_outcome": "accepted",
            "displacement_cm": 700.0,
            "movement_bearing_deg": 180.0,
            "post_front_yaw_deg": 175.0,
            "alignment_error_deg": 5.0,
            "trial_reset": {"success": True},
        },
        "steering": {
            "front_left": {
                "selected_view": "front", "pixel_uv": [0.28, 0.8],
                "start_pose": {
                    "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0,
                    "yaw_deg": 0.0,
                },
                "end_pose": {
                    "x_cm": 100.0, "y_cm": -100.0, "z_cm": 90.0,
                    "yaw_deg": -45.0,
                },
                "movement_bearing_deg": 315.0,
                "signed_bearing_delta_deg": -45.0,
                "controller_outcome": "accepted",
                "displacement_cm": math.sqrt(20_000.0),
                "trial_reset": {"success": True},
            },
            "front_right": {
                "selected_view": "front", "pixel_uv": [0.60, 0.8],
                "start_pose": {
                    "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0,
                    "yaw_deg": 0.0,
                },
                "end_pose": {
                    "x_cm": 100.0, "y_cm": 100.0, "z_cm": 90.0,
                    "yaw_deg": 45.0,
                },
                "movement_bearing_deg": 45.0,
                "signed_bearing_delta_deg": 45.0,
                "controller_outcome": "accepted",
                "displacement_cm": math.sqrt(20_000.0),
                "trial_reset": {"success": True},
            },
        },
        "artifacts": {
            "images": [
                f"pair-{pair:02d}-{view:02d}.png"
                for pair in range(6) for view in range(2)
            ],
            "events": "events.jsonl",
            "trajectory": "trajectory.csv",
            "ue_log": "ue.log",
        },
    }


def test_probe_geometry_and_report_accept_reversal_with_opposite_steering():
    assert bearing_degrees((0.0, 0.0), (-10.0, 0.0)) == pytest.approx(180.0)
    assert planar_target_dot(
        (0.0, 0.0, 90.0, 0.0), (-800.0, 0.0)) == pytest.approx(-800.0)
    assert signed_bearing_delta_degrees(350.0, 10.0) == pytest.approx(20.0)
    assert signed_bearing_delta_degrees(10.0, 350.0) == pytest.approx(-20.0)

    assert validate_front_rear_probe_report(passing_probe_report()) == {
        "passed": True,
        "rear_reversal": True,
        "limited_steering": True,
    }


class BenchmarkEndpoint:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []


class BenchmarkRuntime:
    def __init__(self, endpoint: BenchmarkEndpoint, trace: list[object]) -> None:
        self.endpoint = endpoint
        self.trace = trace
        self.config = SimpleNamespace(fov_degrees=90.0)
        self.single_cycle = 0
        self.dual_cycle = 0

    def capture_frame(self, *, agent_tag, width_px, height_px):
        self.single_cycle += 1
        cycle = self.single_cycle
        self.trace.append(("single", agent_tag, width_px, height_px))
        response = {
            "success": True,
            "camera_snapshot_id": f"single-{cycle}",
            "camera_intrinsics_id": "intrinsics",
            "camera_location_cm": [10.0, 20.0, 160.0],
            "camera_rotation_degrees": [-10.0, 0.0, 0.0],
            "capture_timing": {
                "total_ms": 3.0 * cycle,
                "lookup_ms": 0.1,
                "init_ms": 0.2,
                "capture_read_ms": float(cycle),
                "encode_ms": 2.0 * cycle,
            },
        }
        self.endpoint.calls.append(("PixelGoal_CaptureFrameJson", {}, response))
        return SimpleNamespace(
            camera_snapshot_id=f"single-{cycle}",
            camera_intrinsics_id="intrinsics",
        )

    def capture_view_pair(
            self, *, agent_tag, width_px, height_px, fov_degrees):
        self.dual_cycle += 1
        cycle = self.dual_cycle
        self.trace.append(
            ("dual", agent_tag, width_px, height_px, fov_degrees))
        response = {
            "success": True,
            "capture_group_id": f"view-pair-{cycle}",
            "pose": {
                "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0,
                "yaw_deg": 0.0,
            },
            "views": [
                {
                    "view_id": "front",
                    "camera_snapshot_id": f"front-{cycle}",
                    "camera_location_cm": [10.0, 20.0, 160.0],
                    "camera_rotation_degrees": [-10.0, 0.0, 0.0],
                },
                {
                    "view_id": "rear",
                    "camera_snapshot_id": f"rear-{cycle}",
                    "camera_location_cm": [10.0, 20.0, 160.0],
                    "camera_rotation_degrees": [-10.0, 180.0, 0.0],
                },
            ],
        }
        self.endpoint.calls.append(
            ("PixelGoal_CaptureViewPairJson", {}, response))
        return SimpleNamespace(
            capture_group_id=f"view-pair-{cycle}",
            pose=(0.0, 0.0, 90.0, 0.0),
            front=SimpleNamespace(camera_snapshot_id=f"front-{cycle}"),
            rear=SimpleNamespace(camera_snapshot_id=f"rear-{cycle}"),
            capture_timing={
                "capture_read_ms": 2.0 * cycle,
                "encode_ms": 4.0 * cycle,
                "wall_ms": 6.0 * cycle,
            },
        )


def benchmark_ticks():
    values = []
    clock = 0.0
    for _cycle in range(1, 21):
        values.extend((clock, clock + 0.25))
        clock += 1.0
        values.extend((clock, clock + 0.5))
        clock += 1.0
    return iter(values)


def test_capture_latency_benchmark_alternates_20_same_pose_raw_samples():
    trace: list[object] = []
    endpoint = BenchmarkEndpoint()
    runtime = BenchmarkRuntime(endpoint, trace)
    benchmark_ticks_values = benchmark_ticks()

    def status():
        trace.append("status")
        return {
            "success": True,
            "poc_ready": True,
            "agent_position_cm": [0.0, 0.0, 90.0],
        }

    actual = probe_runner.run_capture_latency_benchmark(
        runtime,
        endpoint,
        status_fn=status,
        cycles=20,
        agent_tag="PixelGoalParisPocAgent",
        width_px=640,
        height_px=360,
        fov_degrees=90.0,
        perf_counter_fn=lambda: next(benchmark_ticks_values),
    )

    assert actual == probe_latency_benchmark()
    expected_trace: list[object] = ["status"]
    for _cycle in range(20):
        expected_trace.extend((
            ("single", "PixelGoalParisPocAgent", 640, 360),
            ("dual", "PixelGoalParisPocAgent", 640, 360, 90.0),
        ))
    expected_trace.append("status")
    assert trace == expected_trace


@pytest.mark.parametrize("cycles", [19, 21])
def test_capture_latency_benchmark_requires_exactly_20_cycles(cycles):
    trace: list[object] = []
    endpoint = BenchmarkEndpoint()
    runtime = BenchmarkRuntime(endpoint, trace)

    with pytest.raises(ValueError, match="exactly 20"):
        probe_runner.run_capture_latency_benchmark(
            runtime,
            endpoint,
            status_fn=lambda: {},
            cycles=cycles,
            agent_tag="PixelGoalParisPocAgent",
            width_px=640,
            height_px=360,
            fov_degrees=90.0,
        )

    assert trace == []


def test_probe_cli_accepts_only_20_latency_cycles():
    parser = probe_runner._build_argument_parser()

    assert parser.parse_args(["--latency-cycles", "20"]).latency_cycles == 20
    with pytest.raises(SystemExit):
        parser.parse_args(["--latency-cycles", "21"])


def test_probe_cli_defaults_to_the_inward_right_ground_fixture():
    args = probe_runner._build_argument_parser().parse_args([])

    assert args.front_right_pixel == "front,0.60,0.80"
    assert probe_runner.parse_selection(args.front_right_pixel) == (
        "front", 0.60, 0.80)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b["single_samples"][0].pop("camera_location_cm"),
         "single sample"),
        (lambda b: b["single_samples"][0].update(
            camera_location_cm=[0.0, 0.0, 90.0]), "same fixed camera"),
        (lambda b: b["single_samples"][9]["camera_location_cm"].__setitem__(
            0, 10.0000001), "same fixed camera"),
        (lambda b: b["single_samples"][11][
            "camera_rotation_degrees"].__setitem__(1, 1.0),
         "same fixed camera"),
        (lambda b: b["dual_samples"][7]["views"][1][
            "camera_location_cm"].__setitem__(2, 160.5),
         "same optical location"),
        (lambda b: b["dual_samples"][3]["views"][1][
            "camera_rotation_degrees"].__setitem__(1, 179.9999999),
         "front/rear yaw"),
    ],
)
def test_probe_report_rejects_missing_synthetic_or_drifted_sample_pose(
        mutate, message):
    report = passing_probe_report()
    mutate(report["latency_benchmark"])

    with pytest.raises(ValueError, match=message):
        validate_front_rear_probe_report(report)


def test_capture_latency_benchmark_rejects_zero_ratio_denominators():
    benchmark = probe_latency_benchmark()
    for sample in benchmark["single_samples"]:
        sample["capture_timing"]["capture_read_ms"] = 0.0
    benchmark["statistics"]["single"]["capture_read_ms"].update(
        median=0.0, p95=0.0)
    benchmark["dual_over_single_ratio"]["capture_read_ms"].update(
        median=0.0, p95=0.0)

    with pytest.raises(ValueError, match="zero denominator"):
        probe_runner.validate_capture_latency_benchmark(benchmark)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b["single_samples"][0]["capture_timing"].pop(
            "encode_ms"), "capture timing"),
        (lambda b: b["single_samples"][0].update(client_wall_s=True),
         "finite"),
        (lambda b: b["dual_samples"][0]["capture_timing"].update(
            wall_ms=math.inf), "finite"),
        (lambda b: b["statistics"]["single"]["client_wall_s"].update(
            median=None), "statistics"),
        (lambda b: b["dual_over_single_ratio"]["client_wall_s"].update(
            p95=None), "ratio"),
    ],
)
def test_capture_latency_benchmark_rejects_missing_boolean_or_nonfinite_values(
        mutate, message):
    benchmark = probe_latency_benchmark()
    mutate(benchmark)

    with pytest.raises(ValueError, match=message):
        probe_runner.validate_capture_latency_benchmark(benchmark)


def test_probe_runs_fixed_reset_benchmark_before_mechanism_movement(
        tmp_path, monkeypatch):
    trace: list[str] = []

    class StopAfterFirstMovement(RuntimeError):
        pass

    class FakeSession:
        def __init__(self, _config):
            pass

        def begin_play(self):
            pass

    class FakeEndpoint:
        def __init__(self, _session, _event_log):
            pass

    class FakeRuntime:
        def __init__(self, _endpoint, _config):
            pass

        def capture_view_pair(self, **_kwargs):
            trace.append("mechanism_capture")
            return SimpleNamespace(
                pose=(0.0, 0.0, 90.0, 0.0),
                capture_group_id="mechanism-0",
                front=SimpleNamespace(camera_snapshot_id="front-0"),
                rear=SimpleNamespace(camera_snapshot_id="rear-0"),
            )

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FakeSession)
    monkeypatch.setattr(probe_runner, "SpearPixelGoalEndpoint", FakeEndpoint)
    monkeypatch.setattr(
        probe_runner, "_prepare_paris_loop",
        lambda *_args, **_kwargs: {
            "setup": {"success": True, "rear_camera_enabled": True},
        })
    monkeypatch.setattr(probe_runner, "LivePixelGoalRuntime", FakeRuntime)
    monkeypatch.setattr(
        probe_runner, "_reset_paris_poc_trial",
        lambda *_args: trace.append("reset") or {"success": True})
    monkeypatch.setattr(
        probe_runner, "run_capture_latency_benchmark",
        lambda *_args, **_kwargs: trace.append("benchmark")
        or probe_latency_benchmark(),
        raising=False,
    )
    monkeypatch.setattr(
        probe_runner, "_save_pair",
        lambda _pair, _output, pair_index: [
            f"pair-{pair_index:02d}-00.png",
            f"pair-{pair_index:02d}-01.png",
        ])

    def stop_on_movement(*_args):
        trace.append("movement")
        raise StopAfterFirstMovement

    monkeypatch.setattr(probe_runner, "_execute", stop_on_movement)
    monkeypatch.setattr(probe_runner, "_cleanup_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", lambda *_args: None)
    args = SimpleNamespace(
        rear_pixel="rear,0.50,0.80",
        front_left_pixel="front,0.28,0.80",
        front_right_pixel="front,0.60,0.80",
        simworld_root=str(tmp_path / "simworld"),
        citycore_content=str(tmp_path / "citycore"),
        output=str(tmp_path / "output"),
        spear_config="fake.yaml",
        launch_mode="attach",
        shutdown_attached_editor=True,
        max_navmesh_adjustment_cm=20.0,
        acceptance_radius_cm=15.0,
        execution_timeout_s=45.0,
        poll_interval_s=0.05,
        navmesh_timeout_s=120.0,
        capture_warmup_s=6.0,
        capture_width=640,
        capture_height=360,
        latency_cycles=20,
    )

    with pytest.raises(StopAfterFirstMovement):
        probe_runner.run_probe(args)

    assert trace == ["reset", "benchmark", "mechanism_capture", "movement"]


def _offline_probe_args(tmp_path: Path) -> argparse.Namespace:
    return probe_runner._build_argument_parser().parse_args([
        "--simworld-root", str(tmp_path / "simworld"),
        "--citycore-content", str(tmp_path / "citycore"),
        "--spear-config", "fake.yaml",
        "--output", str(tmp_path / "output"),
        "--shutdown-attached-editor",
    ])


def _patch_probe_outer_lifecycle(monkeypatch, session_type) -> None:
    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", session_type)
    monkeypatch.setattr(probe_runner, "_cleanup_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", lambda *_args: None)


def _patch_probe_success_path(tmp_path: Path, monkeypatch) -> list[str]:
    """Replace only the live mechanism, leaving finalizers to each test."""

    trace: list[str] = []

    class FakeSession:
        def __init__(self, _config):
            trace.append("session")

        def begin_play(self):
            trace.append("begin_play")

    class FakeEndpoint:
        def __init__(self, _session, _event_log):
            trace.append("endpoint")

        def call(self, _name, _payload):
            return {"success": True, "poc_ready": True}

    class FakePair:
        def __init__(self, index):
            self.capture_group_id = f"view-pair-{index}"
            self.pose = (0.0, 0.0, 90.0, 0.0)
            self.front = SimpleNamespace(
                camera_snapshot_id=f"front-snapshot-{index}",
                camera_yaw_deg=180.0 if index == 2 else 0.0,
            )
            self.rear = SimpleNamespace(
                camera_snapshot_id=f"rear-snapshot-{index}",
                camera_yaw_deg=180.0,
            )

        def frame(self, view):
            return self.front if view == "front" else self.rear

    class FakeRuntime:
        def __init__(self, _endpoint, _config):
            self.capture_count = 0

        def capture_view_pair(self, **_kwargs):
            self.capture_count += 1
            trace.append(f"capture-{self.capture_count}")
            return FakePair(self.capture_count)

    def save_pair(_pair, output_dir, pair_index):
        names = []
        for view_index in range(2):
            name = f"pair-{pair_index:02d}-{view_index:02d}.png"
            (output_dir / name).write_bytes(
                probe_png_bytes(10 + pair_index * 2 + view_index))
            names.append(name)
        return names

    def execute(_runtime, pair, selection):
        view, u, v = selection
        if view == "rear":
            target = SimpleNamespace(x_cm=-20.0, y_cm=0.0, z_cm=0.0)
            end = (-10.0, 0.0)
            bearing = 180.0
        elif u < 0.5:
            target = SimpleNamespace(x_cm=20.0, y_cm=-20.0, z_cm=0.0)
            end = (10.0, -10.0)
            bearing = 315.0
        else:
            target = SimpleNamespace(x_cm=20.0, y_cm=20.0, z_cm=0.0)
            end = (10.0, 10.0)
            bearing = 45.0
        record = {
            "selected_view": view,
            "pixel_uv": [u, v],
            "capture_group_id": pair.capture_group_id,
            "camera_snapshot_id": pair.frame(view).camera_snapshot_id,
            "start_pose": {
                "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0,
                "yaw_deg": 0.0,
            },
            "accepted_target_cm": [target.x_cm, target.y_cm, target.z_cm],
            "controller_outcome": "accepted",
            "end_pose": {
                "x_cm": end[0], "y_cm": end[1], "z_cm": 90.0,
                "yaw_deg": bearing,
            },
            "displacement_cm": math.dist((0.0, 0.0), end),
            "movement_bearing_deg": bearing,
        }
        return (
            SimpleNamespace(request=SimpleNamespace(projected_target=target)),
            SimpleNamespace(),
            record,
        )

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FakeSession)
    monkeypatch.setattr(probe_runner, "SpearPixelGoalEndpoint", FakeEndpoint)
    monkeypatch.setattr(
        probe_runner,
        "_prepare_paris_loop",
        lambda *_args, **_kwargs: {
            "setup": {"success": True, "rear_camera_enabled": True},
            "status": {"success": True, "poc_ready": True},
            "capture_warmup_ms": 6000.0,
        },
    )
    monkeypatch.setattr(probe_runner, "LivePixelGoalRuntime", FakeRuntime)
    monkeypatch.setattr(
        probe_runner,
        "_reset_paris_poc_trial",
        lambda *_args: {"success": True},
    )
    monkeypatch.setattr(
        probe_runner,
        "run_capture_latency_benchmark",
        lambda *_args, **_kwargs: probe_latency_benchmark(),
    )
    monkeypatch.setattr(probe_runner, "_save_pair", save_pair)
    monkeypatch.setattr(probe_runner, "_execute", execute)
    return trace


def _probe_traceback_chain(error: BaseException) -> list[tuple[str, str, int]]:
    """Return the complete code/file/line traceback chain for exact comparison."""

    chain: list[tuple[str, str, int]] = []
    traceback = BaseException.__getattribute__(error, "__traceback__")
    while traceback is not None:
        code = traceback.tb_frame.f_code
        chain.append((code.co_name, code.co_filename, traceback.tb_lineno))
        traceback = traceback.tb_next
    return chain


def _capture_probe_exception(
    args: argparse.Namespace,
) -> tuple[BaseException, list[tuple[str, str, int]]]:
    """Catch run_probe at one fixed call site so traceback baselines are stable."""

    try:
        probe_runner.run_probe(args)
    except BaseException as error:
        return error, _probe_traceback_chain(error)
    raise AssertionError("probe unexpectedly succeeded")


def _probe_args_with_output(
    tmp_path: Path, output_name: str,
) -> argparse.Namespace:
    args = _offline_probe_args(tmp_path)
    args.output = str(tmp_path / output_name)
    return args


@pytest.mark.parametrize(
    "primary_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
    ids=("exception", "keyboard-interrupt", "system-exit"),
)
def test_fix3_primary_same_object_finalizer_preserves_complete_traceback(
        tmp_path, monkeypatch, primary_type):
    current_error: list[BaseException] = [primary_type("primary failure")]
    cleanup_rethrows = False
    trace: list[str] = []

    class FailingSession:
        def __init__(self, _config):
            trace.append("session")

        def begin_play(self):
            trace.append("begin_play")
            raise current_error[0]

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        if cleanup_rethrows:
            raise current_error[0]

    def copy_log(*_args):
        trace.append("copy")

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FailingSession)
    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    baseline_primary = current_error[0]
    baseline_error, baseline_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "baseline-output"))
    assert baseline_error is baseline_primary
    assert baseline_chain[-1][0] == "begin_play"

    current_error[0] = primary_type("primary failure")
    hostile_primary = current_error[0]
    cleanup_rethrows = True
    hostile_error, hostile_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "same-object-output"))

    assert hostile_error is hostile_primary
    assert hostile_chain == baseline_chain
    assert trace[-4:] == ["session", "begin_play", "cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "same-object-output/probe_failure.json").read_text(
            encoding="utf-8"))
    expected_text = f"{primary_type.__name__}: primary failure"
    assert failure["error"] == expected_text
    assert failure["finalization_errors"] == [expected_text]
    assert not (tmp_path / "same-object-output/probe_report.json").exists()


@pytest.mark.parametrize(
    "formatting_error_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
    ids=("exception", "keyboard-interrupt", "system-exit"),
)
def test_fix3_hostile_primary_formatting_never_masks_distinct_finalizers(
        tmp_path, monkeypatch, formatting_error_type):
    formatting_error = formatting_error_type("formatting must stay secondary")

    class HostilePrimary(RuntimeError):
        def __str__(self):
            raise formatting_error

    class CleanupFailure(RuntimeError):
        pass

    primary = HostilePrimary("hidden primary text")
    trace: list[str] = []

    class FailingSession:
        def __init__(self, _config):
            trace.append("session")

        def begin_play(self):
            trace.append("begin_play")
            raise primary

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        raise CleanupFailure("distinct cleanup failure")

    def copy_log(*_args):
        trace.append("copy")

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FailingSession)
    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    escaped, chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "output"))

    assert escaped is primary
    assert chain[-1][0] == "begin_play"
    assert trace == ["session", "begin_play", "cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == (
        "HostilePrimary: <exception text unavailable: "
        f"{formatting_error_type.__name__}>")
    assert failure["finalization_errors"] == [
        "CleanupFailure: distinct cleanup failure"]
    assert not (tmp_path / "output/probe_report.json").exists()


def test_fix3_hostile_secondary_evidence_formatting_cannot_replace_primary(
        tmp_path, monkeypatch):
    class PrimaryFailure(RuntimeError):
        pass

    class EvidenceFormattingFailure(RuntimeError):
        pass

    formatting_error = EvidenceFormattingFailure("do not stringify recursively")

    class HostileEvidenceFailure(ValueError):
        def __str__(self):
            raise formatting_error

    primary = PrimaryFailure("primary lifecycle failure")
    finalizer_trace: list[str] = []

    class FailingSession:
        def __init__(self, _config):
            pass

        def begin_play(self):
            raise primary

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FailingSession)
    monkeypatch.setattr(
        probe_runner,
        "analyze_partial_probe_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HostileEvidenceFailure("hidden evidence text")),
    )
    monkeypatch.setattr(
        probe_runner, "_cleanup_session",
        lambda *_args, **_kwargs: finalizer_trace.append("cleanup"),
    )
    monkeypatch.setattr(
        probe_runner, "_copy_latest_ue_log",
        lambda *_args: finalizer_trace.append("copy"),
    )

    escaped, chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "output"))

    assert escaped is primary
    assert chain[-1][0] == "begin_play"
    assert finalizer_trace == ["cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == "PrimaryFailure: primary lifecycle failure"
    assert failure["secondary_evidence_error"] == (
        "HostileEvidenceFailure: <exception text unavailable: "
        "EvidenceFormattingFailure>")


@pytest.mark.parametrize(
    "formatting_error_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
    ids=("exception", "keyboard-interrupt", "system-exit"),
)
def test_fix3_no_primary_hostile_finalizer_keeps_object_traceback_and_text(
        tmp_path, monkeypatch, formatting_error_type):
    trace = _patch_probe_success_path(tmp_path, monkeypatch)
    current_error: list[BaseException] = [RuntimeError("baseline finalizer")]
    copy_fails = False

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        raise current_error[0]

    class CopyFailure(RuntimeError):
        pass

    def copy_log(*_args):
        trace.append("copy")
        if copy_fails:
            raise CopyFailure("copy also failed")

    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    baseline = current_error[0]
    baseline_error, baseline_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "baseline-output"))
    assert baseline_error is baseline

    formatting_error = formatting_error_type("formatting must stay secondary")

    class HostileFinalizer(RuntimeError):
        def __str__(self):
            raise formatting_error

    current_error[0] = HostileFinalizer("hidden finalizer text")
    hostile = current_error[0]
    copy_fails = True
    escaped, hostile_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "hostile-output"))

    assert escaped is hostile
    assert hostile_chain == baseline_chain
    assert trace[-2:] == ["cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "hostile-output/probe_failure.json").read_text(
            encoding="utf-8"))
    safe_text = (
        "HostileFinalizer: <exception text unavailable: "
        f"{formatting_error_type.__name__}>")
    assert failure["error"] == safe_text
    assert failure["finalization_errors"] == [
        safe_text, "CopyFailure: copy also failed"]
    assert not (tmp_path / "hostile-output/probe_report.json").exists()


def test_fix4_hostile_str_subclasses_are_normalized_to_builtin_text():
    formatting_error = RuntimeError("hostile format must not escape")

    class HostileText(str):
        def __format__(self, _format_spec):
            raise formatting_error

        def __str__(self):
            raise formatting_error

    class HostileFailure(RuntimeError):
        def __str__(self):
            return HostileText("hidden failure text")

    HostileFailure.__name__ = HostileText("HostileFailure")

    formatted = probe_runner._typed_error(HostileFailure())

    assert type(formatted) is str
    assert formatted == "HostileFailure: hidden failure text"


def test_fix4_no_primary_hostile_str_subclass_preserves_first_and_finalizes(
        tmp_path, monkeypatch):
    trace = _patch_probe_success_path(tmp_path, monkeypatch)
    current_error: list[BaseException] = [RuntimeError("baseline finalizer")]
    copy_fails = False

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        raise current_error[0]

    class CopyFailure(RuntimeError):
        pass

    def copy_log(*_args):
        trace.append("copy")
        if copy_fails:
            raise CopyFailure("copy also failed")

    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    baseline = current_error[0]
    baseline_error, baseline_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "baseline-output"))
    assert baseline_error is baseline

    formatting_error = RuntimeError("hostile format must stay secondary")

    class HostileText(str):
        def __format__(self, _format_spec):
            raise formatting_error

        def __str__(self):
            raise formatting_error

    class HostileFinalizer(RuntimeError):
        def __str__(self):
            return HostileText("hidden finalizer text")

    HostileFinalizer.__name__ = HostileText("HostileFinalizer")
    hostile = HostileFinalizer()
    current_error[0] = hostile
    copy_fails = True

    escaped, hostile_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "hostile-output"))

    assert escaped is hostile
    assert hostile_chain == baseline_chain
    assert trace[-2:] == ["cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "hostile-output/probe_failure.json").read_text(
            encoding="utf-8"))
    assert failure["error"] == "HostileFinalizer: hidden finalizer text"
    assert failure["finalization_errors"] == [
        "HostileFinalizer: hidden finalizer text",
        "CopyFailure: copy also failed",
    ]
    assert not (tmp_path / "hostile-output/probe_report.json").exists()


def test_fix3_no_primary_rejected_error_attachments_are_best_effort(
        tmp_path, monkeypatch):
    trace = _patch_probe_success_path(tmp_path, monkeypatch)
    current_error: list[BaseException] = [RuntimeError("baseline finalizer")]
    attachment_attempts: list[str] = []

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        raise current_error[0]

    def copy_log(*_args):
        trace.append("copy")

    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    baseline = current_error[0]
    baseline_error, baseline_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "baseline-output"))
    assert baseline_error is baseline

    class AttachmentFailure(RuntimeError):
        pass

    class NoteFailure(RuntimeError):
        pass

    class RejectingFinalizer(RuntimeError):
        def __setattr__(self, name, value):
            if name == "finalization_errors":
                attachment_attempts.append("attribute")
                raise AttachmentFailure("attribute rejected")
            super().__setattr__(name, value)

        def add_note(self, note):
            attachment_attempts.append("note")
            raise NoteFailure("note rejected")

    current_error[0] = RejectingFinalizer("cleanup failed")
    hostile = current_error[0]
    escaped, hostile_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "hostile-output"))

    assert escaped is hostile
    assert hostile_chain == baseline_chain
    assert attachment_attempts == ["attribute", "note"]
    assert trace[-2:] == ["cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "hostile-output/probe_failure.json").read_text(
            encoding="utf-8"))
    assert failure["error"] == "RejectingFinalizer: cleanup failed"
    assert failure["finalization_errors"] == [
        "RejectingFinalizer: cleanup failed"]
    assert not (tmp_path / "hostile-output/probe_report.json").exists()


def test_fix3_no_primary_same_object_from_both_finalizers_uses_first_traceback(
        tmp_path, monkeypatch):
    trace = _patch_probe_success_path(tmp_path, monkeypatch)
    shared = RuntimeError("shared finalizer failure")
    copy_rethrows = False

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        raise shared

    def copy_log(*_args):
        trace.append("copy")
        if copy_rethrows:
            raise shared

    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    baseline_error, baseline_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "baseline-output"))
    assert baseline_error is shared
    BaseException.__setattr__(shared, "__traceback__", None)
    copy_rethrows = True
    escaped, same_object_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "same-object-output"))

    assert escaped is shared
    assert same_object_chain == baseline_chain
    assert "_raise_first_finalization_error" not in {
        name for name, _filename, _line in same_object_chain}
    assert trace[-2:] == ["cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "same-object-output/probe_failure.json").read_text(
            encoding="utf-8"))
    assert failure["finalization_errors"] == [
        "RuntimeError: shared finalizer failure",
        "RuntimeError: shared finalizer failure",
    ]
    assert not (tmp_path / "same-object-output/probe_report.json").exists()


def test_fix3_report_write_error_survives_hostile_format_and_same_object_evidence(
        tmp_path, monkeypatch):
    trace = _patch_probe_success_path(tmp_path, monkeypatch)
    monkeypatch.setattr(
        probe_runner, "_cleanup_session",
        lambda *_args, **_kwargs: trace.append("cleanup"),
    )
    monkeypatch.setattr(
        probe_runner, "_copy_latest_ue_log",
        lambda *_args: trace.append("copy"),
    )
    real_write = probe_runner._write_json_atomic
    current_error: list[BaseException] = [RuntimeError("baseline report write")]
    rethrow_same_during_evidence = False
    write_trace: list[str] = []

    class EvidenceWriteFailure(OSError):
        pass

    def hostile_write(path, payload):
        if path.name == "probe_report.json":
            write_trace.append("report")
            raise current_error[0]
        if path.name == "probe_failure.json":
            write_trace.append("failure")
            if rethrow_same_during_evidence:
                raise current_error[0]
            raise EvidenceWriteFailure("failure evidence write failed")
        return real_write(path, payload)

    monkeypatch.setattr(probe_runner, "_write_json_atomic", hostile_write)
    baseline = current_error[0]
    baseline_error, baseline_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "baseline-output"))
    assert baseline_error is baseline

    class ReportFormattingFailure(RuntimeError):
        pass

    formatting_error = ReportFormattingFailure("do not replace report error")

    class HostileReportWrite(OSError):
        def __str__(self):
            raise formatting_error

    current_error[0] = HostileReportWrite("hidden report-write text")
    hostile = current_error[0]
    rethrow_same_during_evidence = True
    escaped, hostile_chain = _capture_probe_exception(
        _probe_args_with_output(tmp_path, "hostile-output"))

    assert escaped is hostile
    assert hostile_chain == baseline_chain
    assert write_trace == ["report", "failure", "report", "failure"]
    assert trace[-2:] == ["cleanup", "copy"]
    assert not (tmp_path / "hostile-output/probe_report.json").exists()
    assert not (tmp_path / "hostile-output/probe_failure.json").exists()


@pytest.mark.parametrize(
    ("artifact", "old_bytes"),
    [
        ("probe_report.json", b"old success report"),
        ("probe_failure.json", b"old failure report"),
        ("events.jsonl", b"old events"),
        ("pair-00-00.png", b"old image"),
        ("trajectory.csv", b"old trajectory"),
        ("unknown.bin", b"unknown prior artifact"),
    ],
)
def test_probe_rejects_each_nonempty_output_before_lifecycle_without_mutation(
        tmp_path, monkeypatch, artifact, old_bytes):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stale_path = output_dir / artifact
    stale_path.write_bytes(old_bytes)
    lifecycle: list[str] = []

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)

    class ForbiddenSession:
        def __init__(self, _config):
            lifecycle.append("session")

    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", ForbiddenSession)
    monkeypatch.setattr(
        probe_runner, "_cleanup_session",
        lambda *_args, **_kwargs: lifecycle.append("cleanup"),
    )
    monkeypatch.setattr(
        probe_runner, "_copy_latest_ue_log",
        lambda *_args: lifecycle.append("copy"),
    )
    before = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*") if path.is_file()
    }

    with pytest.raises(FileExistsError, match="non-empty"):
        probe_runner.run_probe(_offline_probe_args(tmp_path))

    after = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*") if path.is_file()
    }
    assert after == before == {Path(artifact): old_bytes}
    assert lifecycle == []


@pytest.mark.parametrize("precreate", [False, True], ids=("fresh", "empty"))
def test_probe_accepts_fresh_or_existing_empty_output_directory(
        tmp_path, monkeypatch, precreate):
    output_dir = tmp_path / "output"
    if precreate:
        output_dir.mkdir()
    trace = _patch_probe_success_path(tmp_path, monkeypatch)
    monkeypatch.setattr(
        probe_runner, "_cleanup_session",
        lambda *_args, **_kwargs: trace.append("cleanup"),
    )
    monkeypatch.setattr(
        probe_runner, "_copy_latest_ue_log",
        lambda *_args: trace.append("copy"),
    )

    report = probe_runner.run_probe(_offline_probe_args(tmp_path))

    assert report["steering"]["front_right"]["pixel_uv"] == [0.60, 0.80]
    assert output_dir.is_dir()
    assert (output_dir / "probe_report.json").is_file()
    assert not (output_dir / "probe_failure.json").exists()
    assert trace[-2:] == ["cleanup", "copy"]


@pytest.mark.parametrize(
    ("cleanup_fails", "copy_fails", "expected_errors"),
    [
        (True, False, ["CleanupFailure: cleanup failed"]),
        (False, True, ["CopyFailure: copy failed"]),
        (True, True, [
            "CleanupFailure: cleanup failed",
            "CopyFailure: copy failed",
        ]),
    ],
    ids=("cleanup", "copy", "cleanup-and-copy"),
)
def test_probe_primary_exception_survives_independent_finalizer_failures(
        tmp_path, monkeypatch, cleanup_fails, copy_fails, expected_errors):
    class PrimaryFailure(RuntimeError):
        pass

    class CleanupFailure(RuntimeError):
        pass

    class CopyFailure(RuntimeError):
        pass

    primary = PrimaryFailure("primary lifecycle failure")
    trace: list[str] = []

    class FailingSession:
        def __init__(self, _config):
            trace.append("session")

        def begin_play(self):
            trace.append("begin_play")
            raise primary

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        if cleanup_fails:
            raise CleanupFailure("cleanup failed")

    def copy_log(*_args):
        trace.append("copy")
        if copy_fails:
            raise CopyFailure("copy failed")

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FailingSession)
    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    with pytest.raises(PrimaryFailure) as raised:
        probe_runner.run_probe(_offline_probe_args(tmp_path))

    assert raised.value is primary
    assert raised.traceback[-1].name == "begin_play"
    assert trace == ["session", "begin_play", "cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == "PrimaryFailure: primary lifecycle failure"
    assert failure["finalization_errors"] == expected_errors
    assert not (tmp_path / "output/probe_report.json").exists()


def test_probe_failure_report_rewrite_failure_preserves_primary_exception(
        tmp_path, monkeypatch, caplog):
    class PrimaryFailure(RuntimeError):
        pass

    class CleanupFailure(RuntimeError):
        pass

    class RewriteFailure(OSError):
        pass

    primary = PrimaryFailure("primary lifecycle failure")

    class FailingSession:
        def __init__(self, _config):
            pass

        def begin_play(self):
            raise primary

    monkeypatch.setattr(probe_runner, "validate_mount_inputs", lambda *_args: None)
    monkeypatch.setattr(probe_runner, "AttachedParisGameSession", FailingSession)
    monkeypatch.setattr(
        probe_runner, "_cleanup_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CleanupFailure("cleanup failed")),
    )
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", lambda *_args: None)
    real_write = probe_runner._write_json_atomic
    write_calls = 0

    def fail_rewrite(path, payload):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise RewriteFailure("failure report rewrite failed")
        return real_write(path, payload)

    monkeypatch.setattr(probe_runner, "_write_json_atomic", fail_rewrite)

    with caplog.at_level(logging.ERROR), pytest.raises(PrimaryFailure) as raised:
        probe_runner.run_probe(_offline_probe_args(tmp_path))

    assert raised.value is primary
    assert write_calls == 2
    persisted = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert persisted["error"] == "PrimaryFailure: primary lifecycle failure"
    assert persisted["finalization_errors"] == []
    assert "failed to rewrite probe failure evidence" in caplog.text


@pytest.mark.parametrize(
    ("cleanup_fails", "copy_fails", "first_type", "expected_errors"),
    [
        (True, False, "CleanupFailure", ["CleanupFailure: cleanup failed"]),
        (False, True, "CopyFailure", ["CopyFailure: copy failed"]),
        (True, True, "CleanupFailure", [
            "CleanupFailure: cleanup failed",
            "CopyFailure: copy failed",
        ]),
    ],
    ids=("cleanup", "copy", "cleanup-and-copy"),
)
def test_probe_success_path_raises_first_finalizer_and_records_all_failures(
        tmp_path, monkeypatch, cleanup_fails, copy_fails, first_type,
        expected_errors):
    class CleanupFailure(RuntimeError):
        pass

    class CopyFailure(RuntimeError):
        pass

    cleanup_error = CleanupFailure("cleanup failed")
    copy_error = CopyFailure("copy failed")
    trace = _patch_probe_success_path(tmp_path, monkeypatch)

    def cleanup(*_args, **_kwargs):
        trace.append("cleanup")
        if cleanup_fails:
            raise cleanup_error

    def copy_log(*_args):
        trace.append("copy")
        if copy_fails:
            raise copy_error

    monkeypatch.setattr(probe_runner, "_cleanup_session", cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", copy_log)

    with pytest.raises(RuntimeError) as raised:
        probe_runner.run_probe(_offline_probe_args(tmp_path))

    expected_first = cleanup_error if cleanup_fails else copy_error
    assert type(raised.value).__name__ == first_type
    assert raised.value is expected_first
    assert raised.value.finalization_errors == tuple(expected_errors)
    assert trace[-2:] == ["cleanup", "copy"]
    failure = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == expected_errors[0]
    assert failure["failure_stage"] == "finalization"
    assert failure["finalization_errors"] == expected_errors
    assert failure["latency_benchmark"] == probe_latency_benchmark()
    assert len(failure["images"]) == 12
    assert len(failure["image_evidence"]) == 12
    assert len(failure["trajectory"]) == 3
    assert not (tmp_path / "output/probe_report.json").exists()


def test_probe_failure_persists_completed_benchmark_and_ten_image_prefix(
        tmp_path, monkeypatch):
    class RightProbeRejected(RuntimeError):
        pass

    class CleanupFailure(RuntimeError):
        pass

    class CopyFailure(RuntimeError):
        pass

    class FakeSession:
        def __init__(self, _config):
            pass

        def begin_play(self):
            pass

    class FakeEndpoint:
        def __init__(self, _session, _event_log):
            pass

        def call(self, _name, _payload):
            return {"success": True, "poc_ready": True}

    class FakePair:
        def __init__(self, index):
            self.capture_group_id = f"view-pair-{index}"
            self.pose = (0.0, 0.0, 90.0, 0.0)
            self.front = SimpleNamespace(
                camera_snapshot_id=f"front-snapshot-{index}",
                camera_yaw_deg=180.0 if index == 2 else 0.0,
            )
            self.rear = SimpleNamespace(
                camera_snapshot_id=f"rear-snapshot-{index}",
                camera_yaw_deg=180.0,
            )

        def frame(self, view):
            return self.front if view == "front" else self.rear

    capture_calls: list[FakePair] = []

    class FakeRuntime:
        def __init__(self, _endpoint, _config):
            pass

        def capture_view_pair(self, **_kwargs):
            pair = FakePair(len(capture_calls) + 1)
            capture_calls.append(pair)
            return pair

    preparation = {
        "setup": {
            "success": True,
            "rear_camera_enabled": True,
            "active_navmesh_tiles": 33,
        },
        "status": {"success": True, "poc_ready": True},
        "capture_warmup_ms": 6000.0,
    }
    benchmark = probe_latency_benchmark()
    attempts: list[tuple[str, float, float]] = []

    def save_pair(_pair, output_dir, pair_index):
        names = []
        for view_index in range(2):
            name = f"pair-{pair_index:02d}-{view_index:02d}.png"
            (output_dir / name).write_bytes(
                probe_png_bytes(10 + pair_index * 2 + view_index))
            names.append(name)
        return names

    def execute(_runtime, pair, selection):
        attempts.append(selection)
        if selection[0] == "front" and selection[1] > 0.5:
            raise RightProbeRejected("navmesh_projection_failed")
        if selection[0] == "rear":
            target = SimpleNamespace(x_cm=-20.0, y_cm=0.0, z_cm=0.0)
            end = (-10.0, 0.0)
            bearing = 180.0
        else:
            target = SimpleNamespace(x_cm=20.0, y_cm=-20.0, z_cm=0.0)
            end = (10.0, -10.0)
            bearing = 315.0
        record = {
            "selected_view": selection[0],
            "pixel_uv": [selection[1], selection[2]],
            "capture_group_id": pair.capture_group_id,
            "camera_snapshot_id": pair.frame(selection[0]).camera_snapshot_id,
            "start_pose": {
                "x_cm": 0.0, "y_cm": 0.0, "z_cm": 90.0,
                "yaw_deg": 0.0,
            },
            "accepted_target_cm": [target.x_cm, target.y_cm, target.z_cm],
            "controller_outcome": "accepted",
            "end_pose": {
                "x_cm": end[0], "y_cm": end[1], "z_cm": 90.0,
                "yaw_deg": bearing,
            },
            "displacement_cm": math.dist((0.0, 0.0), end),
            "movement_bearing_deg": bearing,
        }
        return (
            SimpleNamespace(request=SimpleNamespace(projected_target=target)),
            SimpleNamespace(),
            record,
        )

    _patch_probe_outer_lifecycle(monkeypatch, FakeSession)
    finalizer_trace: list[str] = []

    def fail_cleanup(*_args, **_kwargs):
        finalizer_trace.append("cleanup")
        raise CleanupFailure("cleanup retained as secondary")

    def fail_copy(*_args):
        finalizer_trace.append("copy")
        raise CopyFailure("copy retained as secondary")

    monkeypatch.setattr(probe_runner, "_cleanup_session", fail_cleanup)
    monkeypatch.setattr(probe_runner, "_copy_latest_ue_log", fail_copy)
    monkeypatch.setattr(probe_runner, "SpearPixelGoalEndpoint", FakeEndpoint)
    monkeypatch.setattr(
        probe_runner, "_prepare_paris_loop",
        lambda *_args, **_kwargs: preparation,
    )
    monkeypatch.setattr(probe_runner, "LivePixelGoalRuntime", FakeRuntime)
    reset_count = 0

    def reset(*_args):
        nonlocal reset_count
        reset_count += 1
        return {"success": True, "reset_index": reset_count}

    monkeypatch.setattr(probe_runner, "_reset_paris_poc_trial", reset)
    monkeypatch.setattr(
        probe_runner, "run_capture_latency_benchmark",
        lambda *_args, **_kwargs: benchmark,
    )
    monkeypatch.setattr(probe_runner, "_save_pair", save_pair)
    monkeypatch.setattr(probe_runner, "_execute", execute)
    args = _offline_probe_args(tmp_path)

    with pytest.raises(RightProbeRejected, match="navmesh_projection_failed"):
        probe_runner.run_probe(args)

    failure_path = tmp_path / "output/probe_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    expected_images = [
        f"pair-{index // 2:02d}-{index % 2:02d}.png"
        for index in range(10)
    ]
    assert args.front_right_pixel == "front,0.60,0.80"
    assert attempts == [
        ("rear", 0.50, 0.80),
        ("front", 0.28, 0.80),
        ("front", 0.60, 0.80),
    ]
    assert len(capture_calls) == 5
    assert failure["error"] == (
        "RightProbeRejected: navmesh_projection_failed")
    assert failure["failure_stage"] == "front_right.execute"
    assert failure["attempted_action"] == {
        "trial_name": "front_right",
        "selected_view": "front",
        "pixel_uv": [0.60, 0.80],
        "capture_group_id": "view-pair-5",
        "camera_snapshot_id": "front-snapshot-5",
    }
    assert failure["latency_benchmark"] == benchmark
    assert failure["latency_benchmark"]["statistics"]["single"][
        "client_wall_s"] == {"count": 20, "median": 0.25, "p95": 0.25}
    assert failure["latency_benchmark"]["dual_over_single_ratio"][
        "client_wall_s"] == {"median": 2.0, "p95": 2.0}
    assert failure["setup_request"]["enable_rear_camera"] is True
    assert failure["preparation"] == preparation
    assert failure["config"] == {
        "max_navmesh_adjustment_cm": 20.0,
        "acceptance_radius_cm": 15.0,
        "execution_timeout_s": 45.0,
    }
    assert failure["images"] == expected_images
    assert [row["name"] for row in failure["trajectory"]] == [
        "rear_reversal", "front_left"]
    assert len(failure["image_evidence"]) == 10
    probe_runner.validate_partial_probe_image_evidence(
        failure["image_evidence"],
        image_names=failure["images"],
        width_px=640,
        height_px=360,
    )
    assert failure["secondary_evidence_error"] is None
    assert failure["finalization_errors"] == [
        "CleanupFailure: cleanup retained as secondary",
        "CopyFailure: copy retained as secondary",
    ]
    assert finalizer_trace == ["cleanup", "copy"]
    assert not (tmp_path / "output/probe_report.json").exists()
    with pytest.raises(ValueError):
        validate_front_rear_probe_report(failure)


def test_probe_early_failure_persists_null_benchmark_without_unbound_local(
        tmp_path, monkeypatch):
    class PreBenchmarkFailure(RuntimeError):
        pass

    class FailingSession:
        def __init__(self, _config):
            raise PreBenchmarkFailure("session unavailable")

    _patch_probe_outer_lifecycle(monkeypatch, FailingSession)
    args = _offline_probe_args(tmp_path)

    with pytest.raises(PreBenchmarkFailure, match="session unavailable"):
        probe_runner.run_probe(args)

    failure = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == "PreBenchmarkFailure: session unavailable"
    assert failure["failure_stage"] == "session_initialization"
    assert failure["attempted_action"] is None
    assert failure["latency_benchmark"] is None
    assert failure["preparation"] is None
    assert failure["images"] == []
    assert failure["image_evidence"] == []
    assert failure["trajectory"] == []
    assert failure["secondary_evidence_error"] is None
    assert failure["finalization_errors"] == []
    assert not (tmp_path / "output/probe_report.json").exists()


def test_probe_secondary_evidence_error_does_not_mask_original_failure(
        tmp_path, monkeypatch):
    class OriginalFailure(RuntimeError):
        pass

    class SecondaryEvidenceFailure(ValueError):
        pass

    class FailingSession:
        def __init__(self, _config):
            raise OriginalFailure("original lifecycle failure")

    _patch_probe_outer_lifecycle(monkeypatch, FailingSession)
    monkeypatch.setattr(
        probe_runner,
        "analyze_partial_probe_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SecondaryEvidenceFailure("could not decode partial evidence")),
        raising=False,
    )
    args = _offline_probe_args(tmp_path)

    with pytest.raises(OriginalFailure, match="original lifecycle failure"):
        probe_runner.run_probe(args)

    failure = json.loads(
        (tmp_path / "output/probe_failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == "OriginalFailure: original lifecycle failure"
    assert failure["secondary_evidence_error"] == (
        "SecondaryEvidenceFailure: could not decode partial evidence")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b["order"].__setitem__(0, "dual"), "alternating"),
        (lambda b: b["single_samples"].pop(), "20 single"),
        (lambda b: b["single_samples"][0]["capture_timing"].update(
            capture_read_ms=-1.0), "non-negative"),
        (lambda b: b["dual_samples"][7]["pose"].__setitem__(0, 1.0),
         "same spawn"),
        (lambda b: b["after_status"]["agent_position_cm"].__setitem__(1, 1.0),
         "same spawn"),
        (lambda b: b["statistics"]["single"]["capture_read_ms"].update(
            median=10.6, p95=19.1), "statistics"),
        (lambda b: b["dual_over_single_ratio"]["encode_ms"].update(
            median=1.9), "ratio"),
        (lambda b: b.update(width_px=320), "640x360"),
        (lambda b: b.update(fov_degrees=80.0), "configured FOV"),
    ],
)
def test_probe_report_recomputes_latency_benchmark(mutate, message):
    report = passing_probe_report()
    mutate(report["latency_benchmark"])

    with pytest.raises(ValueError, match=message):
        validate_front_rear_probe_report(report)


def test_saved_image_evidence_is_computed_from_all_12_real_files(tmp_path):
    expected = [probe_image_record(index) for index in range(12)]
    image_names = []
    for record in expected:
        image_names.append(record["artifact"])
        (tmp_path / record["artifact"]).write_bytes(
            base64.b64decode(record["bytes_base64"]))

    actual = probe_runner.analyze_probe_images(
        tmp_path, image_names, width_px=640, height_px=360)

    assert actual == expected


def _write_probe_image_prefix(tmp_path: Path, length: int = 10) -> list[str]:
    names: list[str] = []
    for index in range(length):
        record = probe_image_record(index)
        name = str(record["artifact"])
        (tmp_path / name).write_bytes(base64.b64decode(record["bytes_base64"]))
        names.append(name)
    return names


def test_partial_probe_image_evidence_accepts_only_a_complete_ten_image_prefix(
        tmp_path):
    image_names = _write_probe_image_prefix(tmp_path)

    evidence = probe_runner.analyze_partial_probe_images(
        tmp_path, image_names, width_px=640, height_px=360)

    assert evidence == [probe_image_record(index) for index in range(10)]
    probe_runner.validate_partial_probe_image_evidence(
        evidence,
        image_names=image_names,
        width_px=640,
        height_px=360,
    )
    with pytest.raises(ValueError, match="12 positionally ordered"):
        probe_runner.analyze_probe_images(
            tmp_path, image_names, width_px=640, height_px=360)


@pytest.mark.parametrize(
    ("image_names", "message"),
    [
        ([f"pair-{index // 2:02d}-{index % 2:02d}.png"
          for index in range(9)], "even-length prefix"),
        ([
            *[f"pair-{index // 2:02d}-{index % 2:02d}.png"
              for index in range(8)],
            "pair-05-00.png", "pair-05-01.png",
        ], "ordered prefix"),
        ([
            "pair-00-01.png", "pair-00-00.png",
            *[f"pair-{index // 2:02d}-{index % 2:02d}.png"
              for index in range(2, 10)],
        ], "ordered prefix"),
        ([f"pair-{index // 2:02d}-{index % 2:02d}.png"
          for index in range(14)], "at most 12"),
    ],
)
def test_partial_probe_image_evidence_rejects_odd_missing_reordered_or_overlong(
        tmp_path, image_names, message):
    with pytest.raises(ValueError, match=message):
        probe_runner.analyze_partial_probe_images(
            tmp_path, image_names, width_px=640, height_px=360)


def test_partial_probe_image_evidence_rejects_corrupt_image_bytes(tmp_path):
    image_names = _write_probe_image_prefix(tmp_path)
    (tmp_path / image_names[-1]).write_bytes(b"not an image")

    with pytest.raises(ValueError, match="decode"):
        probe_runner.analyze_partial_probe_images(
            tmp_path, image_names, width_px=640, height_px=360)


def test_partial_probe_image_evidence_rejects_duplicate_image_bytes(tmp_path):
    image_names = _write_probe_image_prefix(tmp_path)
    (tmp_path / image_names[1]).write_bytes(
        (tmp_path / image_names[0]).read_bytes())

    with pytest.raises(ValueError, match="duplicate"):
        probe_runner.analyze_partial_probe_images(
            tmp_path, image_names, width_px=640, height_px=360)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda evidence: evidence[0].update(
            bytes_base64=evidence[1]["bytes_base64"]), "SHA-256"),
        (lambda evidence: evidence.__setitem__(
            1, copy.deepcopy(evidence[0])), "position"),
        (lambda evidence: evidence.pop(), "12 image"),
        (lambda evidence: evidence.__setitem__(
            slice(0, 2), list(reversed(evidence[0:2]))), "position"),
        (lambda evidence: evidence.__setitem__(
            0, probe_image_record(0, luma=0)), "black"),
        (lambda evidence: evidence[0].update(
            bytes_base64=base64.b64encode(b"not an image").decode("ascii"),
            sha256=hashlib.sha256(b"not an image").hexdigest(),
            byte_size=len(b"not an image")), "decode"),
    ],
)
def test_probe_report_recomputes_positional_image_evidence(mutate, message):
    report = passing_probe_report()
    mutate(report["image_evidence"])

    with pytest.raises(ValueError, match=message):
        validate_front_rear_probe_report(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["setup_request"].update(enable_rear_camera=False),
         "rear camera"),
        (lambda r: r["setup"].update(success=False), "setup response"),
        (lambda r: r["setup"].update(rear_camera_enabled=False),
         "setup response"),
        (lambda r: r["rear_reversal"].update(selected_view="front"),
         "rear"),
        (lambda r: r["rear_reversal"].update(
            target_dot_body_forward_cm=1.0), "derived target dot"),
        (lambda r: r["rear_reversal"].update(
            accepted_target_cm=[800.0, 0.0, 0.0]), "derived target dot"),
        (lambda r: r["rear_reversal"].update(controller_outcome="failed"),
         "terminal success"),
        (lambda r: r["rear_reversal"].update(displacement_cm=0.0),
         "derived rear displacement"),
        (lambda r: r["rear_reversal"].update(alignment_error_deg=25.1),
         "derived alignment"),
        (lambda r: (
            r["rear_reversal"]["end_pose"].update(y_cm=700.0),
            r["rear_reversal"].update(displacement_cm=math.hypot(700.0, 700.0)),
        ),
         "derived movement bearing"),
        (lambda r: r["rear_reversal"].update(post_front_yaw_deg=90.0),
         "derived alignment"),
        (lambda r: (
            r["steering"]["front_left"]["end_pose"].update(y_cm=100.0),
            r["steering"]["front_left"].update(movement_bearing_deg=45.0),
            r["steering"]["front_left"].update(
                signed_bearing_delta_deg=45.0),
            r["steering"]["front_right"]["end_pose"].update(y_cm=-100.0),
            r["steering"]["front_right"].update(movement_bearing_deg=315.0),
            r["steering"]["front_right"].update(
                signed_bearing_delta_deg=-45.0),
        ), "left-negative/right-positive"),
        (lambda r: r["steering"]["front_left"].pop("trial_reset"),
         "independent reset"),
        (lambda r: (
            r["steering"]["front_right"]["start_pose"].update(x_cm=1.0),
            r["steering"]["front_right"]["end_pose"].update(x_cm=101.0),
        ), "same spawn"),
        (lambda r: r["steering"]["front_left"]["end_pose"].update(
            y_cm=-50.0), "derived steering"),
        (lambda r: r["steering"]["front_right"].update(
            controller_outcome="failed"), "steering terminal success"),
        (lambda r: r["artifacts"]["images"].pop(), "six ordered pairs"),
        (lambda r: r["artifacts"]["images"].__setitem__(
            1, r["artifacts"]["images"][0]), "six ordered pairs"),
    ],
)
def test_probe_report_fails_closed(mutate, message):
    report = copy.deepcopy(passing_probe_report())
    mutate(report)

    with pytest.raises(ValueError, match=message):
        validate_front_rear_probe_report(report)


def test_setup_requests_enable_rear_camera_without_changing_base_region():
    region = SimpleNamespace(name="region")
    base_builder = lambda value: {"region_name": value.name, "sentinel": 7}

    assert build_delivery_setup_request(region, base_builder=base_builder) == {
        "region_name": "region", "sentinel": 7, "enable_rear_camera": True,
    }
    assert build_probe_setup_request(region, base_builder=base_builder) == {
        "region_name": "region", "sentinel": 7, "enable_rear_camera": True,
    }


def _compile_no_rpc_module(path: Path) -> None:
    source = path.with_suffix(".c")
    source.write_text("void unrelated_symbol(void) {}\n", encoding="utf-8")
    subprocess.run([
        "cc", "-shared", "-fPIC",
        "-Wl,-soname,libSimWorldEditor-SimWorld.so",
        str(source), "-o", str(path),
    ], check=True)


def _compile_foreign_rpc_module(path: Path, *, include_native: bool) -> None:
    binary_dir = path.parent
    dependency_source = binary_dir / "module-identity-dependency.c"
    dependency_source.write_text(
        "void module_identity_dependency(void) {}\n", encoding="utf-8")
    dependency_names = (
        "libSimWorldEditor-Core.so",
        "libSimWorldEditor-CoreUObject.so",
        "libSimWorldEditor-SpCore.so",
    )
    for name in dependency_names:
        subprocess.run([
            "cc", "-shared", "-fPIC", f"-Wl,-soname,{name}",
            str(dependency_source), "-o", str(binary_dir / name),
        ], check=True)
    source = path.with_suffix(".c")
    source_text = (
        "void reflected_rpc(void) __asm__(\"_ZN21USpPixelGoalSubsystem33"
        "execPixelGoal_CaptureViewPairJsonEP7UObjectR6FFramePv\");\n"
        "void reflected_rpc(void) {}\n"
    )
    if include_native:
        source_text += (
            "void native_rpc(void) __asm__(\"_ZN21USpPixelGoalSubsystem29"
            "PixelGoal_CaptureViewPairJsonERK7FString\");\n"
            "void native_rpc(void) {}\n"
        )
    source.write_text(source_text, encoding="utf-8")
    subprocess.run([
        "cc", "-shared", "-fPIC",
        "-Wl,-soname,libSimWorldEditor-SimWorld.so",
        "-Wl,-rpath,$ORIGIN", "-Wl,--no-as-needed", f"-L{binary_dir}",
        str(source), *(f"-l:{name}" for name in dependency_names),
        "-o", str(path),
    ], check=True)
    subprocess.run([
        "objcopy", "--only-keep-debug", str(path),
        str(binary_dir / "libSimWorldEditor-SimWorld.debug"),
    ], check=True)


def _compile_process_switching_editor(path: Path, launch_marker: Path) -> None:
    binary_dir = path.parent
    dependency_source = binary_dir / "identity-dependency.c"
    dependency_source.write_text(
        "void launcher_identity_dependency(void) {}\n", encoding="utf-8")
    dependency_names = (
        "libSimWorldEditor-Core.so",
        "libSimWorldEditor-CoreUObject.so",
        "libSimWorldEditor-Engine.so",
    )
    for name in dependency_names:
        subprocess.run([
            "cc", "-shared", "-fPIC", f"-Wl,-soname,{name}",
            str(dependency_source), "-o", str(binary_dir / name),
        ], check=True)
    source = path.with_suffix(".c")
    source.write_text(
        "#include <stdio.h>\n"
        "#include <unistd.h>\n"
        "int main(void) {\n"
        f"  FILE *marker = fopen({json.dumps(str(launch_marker))}, \"w\");\n"
        "  if (marker) { fprintf(marker, \"%d\\n\", getpid()); fclose(marker); }\n"
        "  execl(\"/usr/bin/sleep\", \"sleep\", \"2\", (char *)0);\n"
        "  return 70;\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run([
        "cc", "-no-pie", str(source), "-o", str(path),
        "-Wl,-rpath,$ORIGIN", "-Wl,--no-as-needed", f"-L{binary_dir}",
        *(f"-l:{name}" for name in dependency_names),
    ], check=True)
    subprocess.run([
        "objcopy", "--only-keep-debug", str(path), f"{path}.debug",
    ], check=True)


def _compile_readiness_then_switching_editor(
        path: Path,
        launch_marker: Path,
        readiness_marker: Path,
        switch_marker: Path,
) -> None:
    """Build one exact-PID editor stand-in that switches only after readiness."""
    binary_dir = path.parent
    dependency_source = binary_dir / "readiness-identity-dependency.c"
    dependency_source.write_text(
        "void launcher_readiness_identity_dependency(void) {}\n",
        encoding="utf-8",
    )
    dependency_names = (
        "libSimWorldEditor-Core.so",
        "libSimWorldEditor-CoreUObject.so",
        "libSimWorldEditor-Engine.so",
    )
    for name in dependency_names:
        subprocess.run([
            "cc", "-shared", "-fPIC", f"-Wl,-soname,{name}",
            str(dependency_source), "-o", str(binary_dir / name),
        ], check=True)
    switch_target = binary_dir / "post-readiness-editor-image"
    switch_source = switch_target.with_suffix(".c")
    switch_source.write_text(
        "#include <stdio.h>\n"
        "#include <unistd.h>\n"
        "int main(void) {\n"
        f"  FILE *marker = fopen({json.dumps(str(switch_marker))}, \"w\");\n"
        "  if (marker) { fprintf(marker, \"%d\\n\", getpid()); fclose(marker); }\n"
        "  sleep(30);\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run([
        "cc", "-no-pie", str(switch_source), "-o", str(switch_target),
    ], check=True)
    source = path.with_suffix(".c")
    source.write_text(
        "#include <arpa/inet.h>\n"
        "#include <errno.h>\n"
        "#include <netinet/in.h>\n"
        "#include <signal.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "#include <sys/socket.h>\n"
        "#include <unistd.h>\n"
        "static volatile sig_atomic_t switch_requested = 0;\n"
        "static void request_switch(int signal_number) {\n"
        "  (void)signal_number; switch_requested = 1;\n"
        "}\n"
        "static void write_pid(const char *path) {\n"
        "  FILE *marker = fopen(path, \"w\");\n"
        "  if (marker) { fprintf(marker, \"%d\\n\", getpid()); fclose(marker); }\n"
        "}\n"
        "int main(int argc, char **argv) {\n"
        "  signal(SIGUSR1, request_switch);\n"
        f"  write_pid({json.dumps(str(launch_marker))});\n"
        "  usleep(200000);\n"
        "  const char *run_id = NULL;\n"
        "  const char *log_path = NULL;\n"
        "  for (int index = 1; index < argc; ++index) {\n"
        "    if (strncmp(argv[index], \"-PixelGoalRunId=\", 16) == 0)\n"
        "      run_id = argv[index] + 16;\n"
        "    if (strncmp(argv[index], \"-AbsLog=\", 8) == 0)\n"
        "      log_path = argv[index] + 8;\n"
        "  }\n"
        "  const char *port_text = getenv(\"SIMWORLD_RPC_PORT\");\n"
        "  if (!run_id || !log_path || !port_text) return 71;\n"
        "  int server = socket(AF_INET, SOCK_STREAM, 0);\n"
        "  int reuse = 1;\n"
        "  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));\n"
        "  struct sockaddr_in address = {0};\n"
        "  address.sin_family = AF_INET;\n"
        "  address.sin_port = htons((unsigned short)atoi(port_text));\n"
        "  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);\n"
        "  if (server < 0 || bind(server, (struct sockaddr *)&address,\n"
        "      sizeof(address)) != 0 || listen(server, 1) != 0) return 72;\n"
        "  FILE *log = fopen(log_path, \"w\");\n"
        "  if (!log) return 73;\n"
        "  fprintf(log, \"%s\\nLoad map complete "
        "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints\\n\", run_id);\n"
        "  fclose(log);\n"
        "  int client;\n"
        "  do { client = accept(server, NULL, NULL); }\n"
        "  while (client < 0 && errno == EINTR);\n"
        "  if (client < 0) return 74;\n"
        f"  write_pid({json.dumps(str(readiness_marker))});\n"
        "  close(client); close(server);\n"
        "  for (int index = 0; index < 500 && !switch_requested; ++index)\n"
        "    usleep(10000);\n"
        "  if (!switch_requested) { pause(); return 75; }\n"
        f"  execl({json.dumps(str(switch_target))}, "
        "\"post-readiness-editor-image\", (char *)0);\n"
        "  return 76;\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run([
        "cc", "-no-pie", str(source), "-o", str(path),
        "-Wl,-rpath,$ORIGIN", "-Wl,--no-as-needed", f"-L{binary_dir}",
        *(f"-l:{name}" for name in dependency_names),
    ], check=True)
    subprocess.run([
        "objcopy", "--only-keep-debug", str(path), f"{path}.debug",
    ], check=True)


def run_controlled_probe_launcher(
        tmp_path: Path,
        *,
        module_library: str | None,
        has_pair_rpc: bool,
        editor_kind: str = "project",
        manifest_build_id: str | None = None,
        version_build_id: str | None = None,
        selected_via_symlink: bool = False,
        gpu_used_mib: int = 8192,
        mismatched_debug: str | None = None,
        reflected_only_rpc: bool = False,
        foreign_rpc_module: bool = False,
        target_build_id: str | None = None,
        bypass_static_identity_for_process_test: bool = False,
        launcher_kind: str = "probe",
        has_cairosvg: bool = True,
        cairosvg_mode: str = "valid",
):
    _require_real_build()
    if launcher_kind not in {"probe", "delivery"}:
        raise AssertionError(f"unknown launcher kind {launcher_kind}")
    supported_cairosvg_modes = {
        "valid",
        "import_system_exit_zero",
        "import_keyboard_interrupt",
        "render_system_exit_zero",
        "render_runtime_error",
        "decode_invalid",
        "decode_system_exit_zero",
    }
    if cairosvg_mode not in supported_cairosvg_modes:
        raise AssertionError(f"unknown cairosvg mode {cairosvg_mode}")
    if not has_cairosvg and cairosvg_mode != "valid":
        raise AssertionError("a missing cairosvg module cannot select a mode")
    launcher_name = f"run_pixel_goal_front_rear_{launcher_kind}.sh"
    runner_name = f"run_pixel_goal_front_rear_{launcher_kind}.py"
    fake_repo = tmp_path / "repo"
    tools_dir = fake_repo / "tools"
    binary_dir = fake_repo / ".simworld-ue/Binaries/Linux"
    plugin_dir = fake_repo / ".simworld-ue/Plugins"
    tools_dir.mkdir(parents=True)
    binary_dir.mkdir(parents=True)
    plugin_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "tools" / launcher_name,
        tools_dir / launcher_name,
    )
    identity_helper = REPO_ROOT / "tools/pixel_goal_launcher_identity.py"
    if identity_helper.is_file():
        shutil.copy2(identity_helper, tools_dir / identity_helper.name)
        if bypass_static_identity_for_process_test:
            real_helper = tools_dir / "pixel_goal_launcher_identity_real.py"
            shutil.copy2(identity_helper, real_helper)
            (tools_dir / identity_helper.name).write_text(
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n"
                "if sys.argv[1] == 'validate-bundle':\n"
                "    flag = sys.argv.index('--expected-editor')\n"
                "    print(Path(sys.argv[flag + 1]).resolve())\n"
                "else:\n"
                "    helper = Path(__file__).with_name(\n"
                "        'pixel_goal_launcher_identity_real.py')\n"
                "    raise SystemExit(subprocess.call(\n"
                "        [sys.executable, str(helper), *sys.argv[1:]]))\n",
                encoding="utf-8",
            )
    (fake_repo / ".simworld-ue/SimWorld.uproject").write_text(
        "{}", encoding="utf-8")
    launch_marker = tmp_path / "editor-was-launched"
    readiness_marker = tmp_path / "editor-accepted-readiness-rpc"
    switch_marker = tmp_path / "editor-switched-executable"
    editor = binary_dir / "SimWorldEditor"
    if editor_kind == "project":
        shutil.copy2(
            REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor", editor)
        shutil.copy2(
            REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor.debug",
            binary_dir / "SimWorldEditor.debug",
        )
    elif editor_kind == "script":
        editor.write_text(
            f"#!/usr/bin/env bash\ntouch {launch_marker}\n", encoding="utf-8")
        editor.chmod(0o755)
    elif editor_kind == "process_mismatch":
        _compile_process_switching_editor(editor, launch_marker)
    elif editor_kind == "post_readiness_process_mismatch":
        _compile_readiness_then_switching_editor(
            editor, launch_marker, readiness_marker, switch_marker)
    else:
        raise AssertionError(f"unknown editor kind {editor_kind}")
    target_path = binary_dir / "SimWorldEditor.target"
    shutil.copy2(
        REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor.target",
        target_path,
    )
    if target_build_id is not None:
        target = json.loads(target_path.read_text(encoding="utf-8"))
        target["Version"]["BuildId"] = target_build_id
        target_path.write_text(json.dumps(target), encoding="utf-8")
    version_path = binary_dir / "SimWorldEditor.version"
    shutil.copy2(
        REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor.version",
        version_path,
    )
    if version_build_id is not None:
        version = json.loads(version_path.read_text(encoding="utf-8"))
        version["BuildId"] = version_build_id
        version_path.write_text(json.dumps(version), encoding="utf-8")
    if module_library is not None:
        manifest_path = binary_dir / "SimWorldEditor.modules"
        shutil.copy2(
            REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor.modules",
            manifest_path,
        )
        if (module_library != "libSimWorldEditor-SimWorld.so"
                or manifest_build_id is not None):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["Modules"]["SimWorld"] = module_library
            if manifest_build_id is not None:
                manifest["BuildId"] = manifest_build_id
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if has_pair_rpc:
            shutil.copy2(
                REPO_ROOT
                / ".simworld-ue/Binaries/Linux/libSimWorldEditor-SimWorld.so",
                binary_dir / module_library,
            )
            shutil.copy2(
                REPO_ROOT
                / ".simworld-ue/Binaries/Linux/libSimWorldEditor-SimWorld.debug",
                binary_dir / f"{module_library.removesuffix('.so')}.debug",
            )
        elif reflected_only_rpc:
            _compile_foreign_rpc_module(
                binary_dir / module_library, include_native=False)
        elif foreign_rpc_module:
            _compile_foreign_rpc_module(
                binary_dir / module_library, include_native=True)
        elif module_library == "libSimWorldEditor-SimWorld.so":
            _compile_no_rpc_module(binary_dir / module_library)
        else:
            (binary_dir / module_library).write_bytes(b"stale module")
    if mismatched_debug == "editor":
        mismatch_source = binary_dir / "mismatched-editor.c"
        mismatch_binary = binary_dir / "mismatched-editor"
        mismatch_source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        subprocess.run([
            "cc", "-no-pie", str(mismatch_source), "-o", str(mismatch_binary),
        ], check=True)
        subprocess.run([
            "objcopy", "--only-keep-debug", str(mismatch_binary),
            str(binary_dir / "SimWorldEditor.debug"),
        ], check=True)
    elif mismatched_debug == "module":
        shutil.copy2(
            REPO_ROOT / ".simworld-ue/Binaries/Linux/libSimWorldEditor-SpCore.debug",
            binary_dir / "libSimWorldEditor-SimWorld.debug",
        )

    citycore = tmp_path / "citycore"
    (citycore / "Scenes").mkdir(parents=True)
    (citycore / "Scenes/ParisCity_FinalBlueprints.umap").write_bytes(b"map")
    citycore.chmod(0o555)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        f"#!/usr/bin/env bash\nprintf '5, {gpu_used_mib}\\n'\n",
        encoding="utf-8")
    nvidia_smi.chmod(0o755)
    (fake_repo / "spear.py").write_text(
        "from types import SimpleNamespace\n"
        "class Config:\n"
        "    def __init__(self):\n"
        "        self.SP_SERVICES = SimpleNamespace(\n"
        "            RPC_SERVICE=SimpleNamespace(RPC_SERVER_PORT=0))\n"
        "    def defrost(self): pass\n"
        "    def freeze(self): pass\n"
        "    def dump(self, stream, default_flow_style=False):\n"
        "        stream.write('{}\\n')\n"
        "def get_config(user_config_files): return Config()\n",
        encoding="utf-8",
    )
    if has_cairosvg:
        fake_png_buffer = BytesIO()
        Image.new("RGB", (2, 2), (128, 128, 128)).save(
            fake_png_buffer, format="PNG")
        fake_png = fake_png_buffer.getvalue()
        cairosvg_source = {
            "valid": (
                f"PNG = {fake_png!r}\n"
                "def svg2png(*, bytestring):\n"
                "    assert bytestring.startswith(b'<svg')\n"
                "    return PNG\n"
            ),
            "import_system_exit_zero": "raise SystemExit(0)\n",
            "import_keyboard_interrupt": "raise KeyboardInterrupt\n",
            "render_system_exit_zero": (
                "def svg2png(*, bytestring):\n"
                "    raise SystemExit(0)\n"
            ),
            "render_runtime_error": (
                "def svg2png(*, bytestring):\n"
                "    raise RuntimeError('renderer failed')\n"
            ),
            "decode_invalid": (
                "def svg2png(*, bytestring):\n"
                "    return b'not a png'\n"
            ),
            "decode_system_exit_zero": (
                f"PNG = {fake_png!r}\n"
                "def svg2png(*, bytestring):\n"
                "    return PNG\n"
            ),
        }[cairosvg_mode]
        (fake_repo / "cairosvg.py").write_text(
            cairosvg_source, encoding="utf-8")
        if cairosvg_mode == "decode_system_exit_zero":
            fake_pil = fake_repo / "PIL"
            fake_pil.mkdir()
            (fake_pil / "__init__.py").write_text(
                "from . import Image\n", encoding="utf-8")
            (fake_pil / "Image.py").write_text(
                "def open(*args, **kwargs):\n"
                "    raise SystemExit(0)\n",
                encoding="utf-8",
            )
    if not has_cairosvg:
        # Shadow whatever cairosvg the test interpreter has: the launcher's
        # gate must trip on ABSENCE, under any python the suite runs with.
        (fake_repo / "cairosvg.py").write_text(
            "raise ModuleNotFoundError(\"No module named 'cairosvg'\")\n",
            encoding="utf-8")
    runner_marker = tmp_path / f"{launcher_kind}-runner-was-invoked"
    (tools_dir / runner_name).write_text(
        "from pathlib import Path\n"
        f"Path({str(runner_marker)!r}).write_text('invoked\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        rpc_port = port_probe.getsockname()[1]
    env = os.environ.copy()
    env.update({
        "CITYCORE_PARIS_CONTENT": str(citycore),
        "PIXEL_GOAL_PYTHON": sys.executable,
        "PIXEL_GOAL_GPU": "5",
        "SIMWORLD_RPC_PORT": str(rpc_port),
        "PATH": f"{fake_bin}:{env['PATH']}",
    })
    if launcher_kind == "delivery":
        env["PIXEL_GOAL_UNREAL_EDITOR"] = str(editor)
    if selected_via_symlink:
        selected_dir = tmp_path / "selected"
        selected_dir.mkdir()
        selected = selected_dir / "SimWorldEditor"
        selected.symlink_to(editor)
        env["PIXEL_GOAL_UNREAL_EDITOR"] = str(selected)
    if editor_kind == "post_readiness_process_mismatch":
        env["PIXEL_GOAL_LAUNCHER_TEST_SIGNAL_AFTER_READINESS"] = "1"
        env["PIXEL_GOAL_LAUNCHER_TEST_SWITCH_ACK"] = str(switch_marker)
    model_server = None
    model_thread = None
    if launcher_kind == "delivery":
        model_payload = json.dumps({
            "data": [{"id": "qwen3-vl-8b"}],
        }).encode("utf-8")

        model_marker = tmp_path / "model-endpoint-was-queried"

        class ModelHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                model_marker.write_text(self.path, encoding="utf-8")
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(model_payload)))
                self.end_headers()
                self.wfile.write(model_payload)

            def log_message(self, _format, *_args):
                pass

        model_server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
        model_thread = Thread(target=model_server.serve_forever, daemon=True)
        model_thread.start()
        env["QWEN_ENDPOINT"] = (
            f"http://127.0.0.1:{model_server.server_port}"
            "/v1/chat/completions")
    try:
        result = subprocess.run(
            ["bash", str(tools_dir / launcher_name)],
            cwd=fake_repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if model_server is not None:
            model_server.shutdown()
            model_server.server_close()
        if model_thread is not None:
            model_thread.join(timeout=5.0)
        citycore.chmod(0o755)
    return result, launch_marker


@pytest.mark.parametrize(
    ("module_library", "has_pair_rpc", "message"),
    [
        (None, False, "SimWorldEditor.modules"),
        ("libUnrealEditor-SimWorld.so", True, "matching SimWorldEditor module"),
        ("libSimWorldEditor-SimWorld.so", False,
         "PixelGoal_CaptureViewPairJson"),
    ],
)
def test_probe_launcher_fails_before_launch_for_missing_or_stale_project_module(
        tmp_path, module_library, has_pair_rpc, message):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library=module_library,
        has_pair_rpc=has_pair_rpc,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_accepts_fresh_project_module_before_gpu_gate(tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
    )

    assert result.returncode == 2
    assert "using 8192 MiB" in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_rejects_reflected_wrapper_without_native_rpc(tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=False,
        reflected_only_rpc=True,
    )

    assert result.returncode == 2
    assert "both the native and reflected" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_accepts_only_a_symlink_resolving_to_exact_project_target(
        tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        selected_via_symlink=True,
    )

    assert result.returncode == 2
    assert "using 8192 MiB" in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_rejects_arbitrary_script_at_project_target_boundary(
        tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        editor_kind="script",
    )

    assert result.returncode == 2
    assert "ELF" in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_rejects_foreign_editor_elf_before_gpu_or_launch(
        tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        editor_kind="process_mismatch",
    )

    assert result.returncode == 2
    assert "approved project build" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_rejects_foreign_module_with_exact_rpc_exports(
        tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=False,
        foreign_rpc_module=True,
    )

    assert result.returncode == 2
    assert "approved project build" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


@pytest.mark.parametrize("sidecar", ["manifest", "version"])
def test_probe_launcher_rejects_mixed_build_identity_before_gpu_or_launch(
        tmp_path, sidecar):
    mismatched = "00000000-1111-2222-3333-444444444444"
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        manifest_build_id=mismatched if sidecar == "manifest" else None,
        version_build_id=mismatched if sidecar == "version" else None,
    )

    assert result.returncode == 2
    assert "BuildId" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


def test_probe_launcher_rejects_consistently_forged_metadata_build_id(
        tmp_path):
    mismatched = "00000000-1111-2222-3333-444444444444"
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        manifest_build_id=mismatched,
        target_build_id=mismatched,
        version_build_id=mismatched,
    )

    assert result.returncode == 2
    assert "approved project build" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


@pytest.mark.parametrize("artifact", ["editor", "module"])
def test_probe_launcher_rejects_mismatched_elf_build_companion(
        tmp_path, artifact):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        mismatched_debug=artifact,
    )

    assert result.returncode == 2
    assert "ELF Build ID" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


def run_launcher_identity_helper(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/pixel_goal_launcher_identity.py"),
            *(str(arg) for arg in args),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_identity_helper_accepts_exact_project_target_and_symlink(
        tmp_path):
    _require_real_build()
    expected = REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor"
    selected = tmp_path / "SimWorldEditor"
    selected.symlink_to(expected)

    exact = run_launcher_identity_helper(
        "validate-bundle", "--expected-editor", expected,
        "--selected-editor", expected,
    )
    linked = run_launcher_identity_helper(
        "validate-bundle", "--expected-editor", expected,
        "--selected-editor", selected,
    )

    assert exact.returncode == 0, exact.stderr
    assert Path(exact.stdout.strip()) == expected.resolve()
    assert linked.returncode == 0, linked.stderr
    assert Path(linked.stdout.strip()) == expected.resolve()


def test_launcher_identity_helper_rejects_renamed_same_basename_copy(tmp_path):
    _require_real_build()
    expected = REPO_ROOT / ".simworld-ue/Binaries/Linux/SimWorldEditor"
    selected_dir = tmp_path / "renamed"
    selected_dir.mkdir()
    selected = selected_dir / "SimWorldEditor"
    shutil.copy2(expected, selected)

    result = run_launcher_identity_helper(
        "validate-bundle", "--expected-editor", expected,
        "--selected-editor", selected,
    )

    assert result.returncode == 2
    assert "exact resolved project SimWorldEditor" in result.stderr


def test_launcher_process_identity_helper_fails_closed_for_different_executable(
        tmp_path):
    expected = tmp_path / "expected-editor"
    selected = tmp_path / "selected-editor"
    shutil.copy2("/usr/bin/sleep", expected)
    shutil.copy2("/usr/bin/sleep", selected)
    process = subprocess.Popen([str(selected), "30"])
    try:
        result = run_launcher_identity_helper(
            "verify-process", "--pid", process.pid,
            "--expected-editor", expected,
            "--timeout-s", 0.1,
        )

        assert result.returncode == 2
        assert "process executable identity mismatch" in result.stderr
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5.0)


def test_launcher_process_identity_helper_accepts_the_exact_started_executable(
        tmp_path):
    expected = tmp_path / "SimWorldEditor"
    shutil.copy2("/usr/bin/sleep", expected)
    process = subprocess.Popen([str(expected), "30"])
    try:
        result = run_launcher_identity_helper(
            "verify-process", "--pid", process.pid,
            "--expected-editor", expected,
            "--timeout-s", 0.5,
        )

        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == expected.resolve()
    finally:
        process.terminate()
        process.wait(timeout=5.0)


def test_probe_launcher_process_mismatch_fails_before_readiness_and_cleans_pid(
        tmp_path):
    unrelated = subprocess.Popen(["/usr/bin/sleep", "30"])
    try:
        result, launch_marker = run_controlled_probe_launcher(
            tmp_path,
            module_library="libSimWorldEditor-SimWorld.so",
            has_pair_rpc=True,
            editor_kind="process_mismatch",
            gpu_used_mib=0,
            bypass_static_identity_for_process_test=True,
        )

        assert result.returncode == 2
        assert "process executable identity mismatch" in result.stderr
        assert "before readiness" in result.stderr
        assert launch_marker.is_file()
        launched_pid = int(launch_marker.read_text(encoding="utf-8"))
        assert not Path(f"/proc/{launched_pid}").exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5.0)


def test_probe_launcher_rechecks_same_pid_after_real_readiness_before_runner(
        tmp_path):
    unrelated = subprocess.Popen(["/usr/bin/sleep", "30"])
    launch_marker = tmp_path / "editor-was-launched"
    readiness_marker = tmp_path / "editor-accepted-readiness-rpc"
    switch_marker = tmp_path / "editor-switched-executable"
    runner_marker = tmp_path / "probe-runner-was-invoked"
    runner_path = tmp_path / "repo/tools/run_pixel_goal_front_rear_probe.py"
    editor_log = (
        tmp_path / "repo/.simworld-ue/Saved/Logs/PixelGoalFrontRearProbe.log")
    try:
        result, returned_launch_marker = run_controlled_probe_launcher(
            tmp_path,
            module_library="libSimWorldEditor-SimWorld.so",
            has_pair_rpc=True,
            editor_kind="post_readiness_process_mismatch",
            gpu_used_mib=0,
            bypass_static_identity_for_process_test=True,
        )

        assert returned_launch_marker == launch_marker
        assert result.returncode == 2
        assert "process executable identity mismatch" in result.stderr
        assert "post-readiness-editor-image" in result.stderr
        assert "before runner dispatch" in result.stderr
        assert launch_marker.is_file()
        assert readiness_marker.is_file()
        assert switch_marker.is_file()
        log_text = editor_log.read_text(encoding="utf-8")
        run_id, ready_line = log_text.splitlines()
        assert run_id.startswith("pixel-goal-front-rear-probe-")
        assert ready_line == (
            "Load map complete "
            "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints")
        launched_pid = int(launch_marker.read_text(encoding="utf-8"))
        assert int(readiness_marker.read_text(encoding="utf-8")) == launched_pid
        assert int(switch_marker.read_text(encoding="utf-8")) == launched_pid
        assert runner_path.is_file()
        assert not runner_marker.exists()
        assert not Path(f"/proc/{launched_pid}").exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5.0)


def test_task12_delivery_launcher_authenticates_project_editor_twice():
    delivery = (
        REPO_ROOT / "tools/run_pixel_goal_front_rear_delivery.sh"
    ).read_text(encoding="utf-8")

    assert "$simworld_root/Binaries/Linux/SimWorldEditor" in delivery
    assert "Engine/Binaries/Linux/UnrealEditor" not in delivery
    assert delivery.count("validate-bundle") == 1
    assert delivery.count("verify-process") == 2
    assert "cairosvg.svg2png" in delivery
    assert "Launched editor process identity failed before readiness" in delivery
    assert "Launched editor process identity failed before runner dispatch" in delivery
    assert delivery.index("validate-bundle") < delivery.index("cairosvg.svg2png")
    assert delivery.index("cairosvg.svg2png") < delivery.index("/v1/models")
    assert delivery.index("/v1/models") < delivery.index('"$editor_bin" "$simworld_root/SimWorld.uproject"')
    assert delivery.rindex("verify-process") < delivery.index(
        '"$repo_root/tools/run_pixel_goal_front_rear_delivery.py"')


def test_task12_delivery_launcher_rejects_stale_module_before_gpu_or_launch(
        tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libUnrealEditor-SimWorld.so",
        has_pair_rpc=True,
        launcher_kind="delivery",
    )

    assert result.returncode == 2
    assert "matching SimWorldEditor module" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not launch_marker.exists()


def test_task12_delivery_launcher_checks_cairosvg_before_endpoint_or_launch(
        tmp_path):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        launcher_kind="delivery",
        has_cairosvg=False,
    )

    assert result.returncode == 2
    assert "functional cairosvg phone-map rasterizer" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not (tmp_path / "model-endpoint-was-queried").exists()
    assert not launch_marker.exists()


@pytest.mark.parametrize("cairosvg_mode", [
    "import_system_exit_zero",
    "import_keyboard_interrupt",
    "render_system_exit_zero",
    "render_runtime_error",
    "decode_invalid",
    "decode_system_exit_zero",
])
def test_task12_delivery_launcher_rejects_cairosvg_baseexceptions_before_endpoint(
        tmp_path, cairosvg_mode):
    result, launch_marker = run_controlled_probe_launcher(
        tmp_path,
        module_library="libSimWorldEditor-SimWorld.so",
        has_pair_rpc=True,
        launcher_kind="delivery",
        cairosvg_mode=cairosvg_mode,
    )

    assert result.returncode == 2
    assert "functional cairosvg phone-map rasterizer" in result.stderr
    assert "using 8192 MiB" not in result.stderr
    assert not (tmp_path / "model-endpoint-was-queried").exists()
    assert not launch_marker.exists()


def test_task12_delivery_launcher_process_mismatch_fails_before_readiness(
        tmp_path):
    unrelated = subprocess.Popen(["/usr/bin/sleep", "30"])
    try:
        result, launch_marker = run_controlled_probe_launcher(
            tmp_path,
            module_library="libSimWorldEditor-SimWorld.so",
            has_pair_rpc=True,
            editor_kind="process_mismatch",
            gpu_used_mib=0,
            bypass_static_identity_for_process_test=True,
            launcher_kind="delivery",
        )

        assert result.returncode == 2
        assert "process executable identity mismatch" in result.stderr
        assert "before readiness" in result.stderr
        assert launch_marker.is_file()
        launched_pid = int(launch_marker.read_text(encoding="utf-8"))
        assert not Path(f"/proc/{launched_pid}").exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5.0)


def test_task12_delivery_launcher_rechecks_same_pid_before_model_runner(
        tmp_path):
    unrelated = subprocess.Popen(["/usr/bin/sleep", "30"])
    launch_marker = tmp_path / "editor-was-launched"
    readiness_marker = tmp_path / "editor-accepted-readiness-rpc"
    switch_marker = tmp_path / "editor-switched-executable"
    runner_marker = tmp_path / "delivery-runner-was-invoked"
    editor_log_dir = (
        tmp_path / "repo/.simworld-ue/Saved/Logs")
    try:
        result, returned_launch_marker = run_controlled_probe_launcher(
            tmp_path,
            module_library="libSimWorldEditor-SimWorld.so",
            has_pair_rpc=True,
            editor_kind="post_readiness_process_mismatch",
            gpu_used_mib=0,
            bypass_static_identity_for_process_test=True,
            launcher_kind="delivery",
        )

        assert returned_launch_marker == launch_marker
        assert result.returncode == 2
        assert "process executable identity mismatch" in result.stderr
        assert "post-readiness-editor-image" in result.stderr
        assert "before runner dispatch" in result.stderr
        assert launch_marker.is_file()
        assert readiness_marker.is_file()
        assert switch_marker.is_file()
        editor_logs = [
            path for path in editor_log_dir.glob(
                "PixelGoalFrontRearDelivery-pixel-goal-front-rear-delivery-*.log")
            if not path.name.endswith(".stdout.log")
        ]
        assert len(editor_logs) == 1
        run_id, ready_line = editor_logs[0].read_text(
            encoding="utf-8").splitlines()
        assert run_id.startswith("pixel-goal-front-rear-delivery-")
        assert ready_line == (
            "Load map complete "
            "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints")
        launched_pid = int(launch_marker.read_text(encoding="utf-8"))
        assert int(readiness_marker.read_text(encoding="utf-8")) == launched_pid
        assert int(switch_marker.read_text(encoding="utf-8")) == launched_pid
        assert not runner_marker.exists()
        assert not Path(f"/proc/{launched_pid}").exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5.0)


def test_launchers_are_syntactically_valid_and_do_not_own_qwen_processes():
    delivery_path = REPO_ROOT / "tools/run_pixel_goal_front_rear_delivery.sh"
    probe_path = REPO_ROOT / "tools/run_pixel_goal_front_rear_probe.sh"
    delivery = delivery_path.read_text(encoding="utf-8")
    probe = probe_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(delivery_path)], check=True)
    subprocess.run(["bash", "-n", str(probe_path)], check=True)

    assert "run_pixel_goal_front_rear_delivery.py" in delivery
    assert "run_pixel_goal_front_rear_probe.py" in probe
    assert "$simworld_root/Binaries/Linux/SimWorldEditor" in probe
    assert "Engine/Binaries/Linux/UnrealEditor" not in probe
    assert "QWEN_ENDPOINT" in delivery
    assert "http://127.0.0.1:30001/v1/chat/completions" in delivery
    assert "QWEN_MODEL_NAME" in delivery
    assert "qwen3-vl-8b" in delivery
    assert "/v1/models" in delivery
    assert delivery.index("/v1/models") < delivery.index(
        '"$editor_bin" "$simworld_root/SimWorld.uproject"')
    assert "QWEN_MAX_IMAGES_PER_PROMPT" in delivery
    assert "enable_rear_camera" not in delivery  # Python owns setup JSON.
    for forbidden in (
        "qwen_pid", "vllm.entrypoints", "QWEN_MODEL_PATH", "pkill",
        'kill "$qwen_pid"',
    ):
        assert forbidden not in delivery
    assert "QWEN_ENDPOINT" not in probe
    assert "qwen3-vl" not in probe.lower()
