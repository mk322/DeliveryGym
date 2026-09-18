#!/usr/bin/env python3
"""Run and fully record a Qwen3-VL-8B front/rear Pixel Goal delivery.

The model server is externally owned.  This module only calls the configured
OpenAI-compatible endpoint; the companion launcher verifies that endpoint
before it starts the separately owned Unreal process.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from embodiedbench.agent.courier.frame_alias import FrameAliases
from embodiedbench.agent.courier.loop import parse_reply
from embodiedbench.agent.courier.model_io import ModelClient, split_reasoning
from embodiedbench.agent.courier.session import CourierSession, Frame
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
    CAMERA_VIEW_FRONT_REAR,
)
from embodiedbench.runtime.live.protocol import CameraSpec, Pose
from embodiedbench.runtime.pixel_goal import LivePixelGoalRuntime, PixelGoalConfig
from embodiedbench.schemas.runtime import ControllerOutcomeCode
from embodiedbench.runtime.live.protocol import PIXEL_VIEWS, PIXEL_VIEWS_QUAD
from tools.pixel_goal_courier_backend import (
    make_pawn_mover,
    make_pawn_turner,
    CORRIDOR_DROPOFF_TOLERANCE_CM,
    DEFAULT_DROPOFF_OFFSET_CM,
    DEFAULT_PICKUP_OFFSET_CM,
    TURNING_DROPOFF_NODE_ID,
    TURNING_PICKUP_NODE_ID,
    TURNING_ROUTE_PROFILE,
    TURNING_ROUTE_WAYPOINTS,
    VALIDATED_POOL_PICKUP_TOLERANCE_CM,
    CorridorDeliveryEnv,
    SpearTrackBClient,
    TurningDeliveryEnv,
    ValidatedPoolDeliveryEnv,
    build_turning_delivery_network,
    latency_statistics,
    turning_route_report,
)
from tools.pixel_goal_order_pool import (
    OrderConstraints,
    SUPPORTED_POOL_SCHEMAS,
    build_validated_pool_network,
    load_validated_delivery_pool,
    resolve_delivery_scenario,
    scenario_report,
    selected_route_report,
)
from tools.pixel_goal_capture_preflight import (
    CAPTURE_READINESS_VERSION,
    CaptureReadinessGate,
    VulkanOomMonitor,
)
from tools.run_pixel_goal_1b_closed_loop import _prepare_paris_loop, _write_json_atomic
from tools.run_pixel_goal_1b_poc import (
    AttachedParisGameSession,
    _reset_paris_poc_trial,
    _wait_for_paris_poc,
    _warm_up_paris_capture,
    build_paris_setup_request,
    validate_mount_inputs,
)
from tools.run_pixel_goal_full_delivery import (
    AGENT_TAG,
    DELIVERY_REGION,
    MAPS,
)
from tools.run_pixel_goal_m1a import (
    SpearPixelGoalEndpoint,
    _cleanup_session,
    _copy_latest_ue_log,
)


# The straight corridor needs only a narrow strip around its sidewalk.  A
# real turn at the zebra crossing reaches the pavement on the far side at
# x≈-4300 cm; the corridor box ended at x=-4600 and rejected that genuine
# sidewalk hit before the controller could move.  Keep the larger local
# NavMesh build exclusive to the long-turn profile.
TURNING_DELIVERY_REGION = replace(
    DELIVERY_REGION,
    name="paris_pixel_goal_front_rear_long_turn",
    nav_bounds_extent_cm=(2_000.0, 5_000.0, 300.0),
)

DEFAULT_VALIDATED_ORDER_POOL = (
    REPO_ROOT / "configs/pixel_goal/paris_trusted_pedestrian_pool_v3.json"
)

# The model ids a report may carry. A run names its model in the artifact
# and the validator reloads it, so an id has to be listed here before a
# number under it can exist -- the point is that no unlabelled model ever
# produces a report.
AUDITED_MODEL_LABELS = {
    "qwen3-vl-4b": "Qwen3-VL-4B",
    "qwen3-vl-8b": "Qwen3-VL-8B",
    "gpt-5.6-codex-interactive": "GPT-5.6 Codex interactive",
    # A person or an interactive model answering through
    # tools/relay_policy_server.py: the same prompt, images and rules as a
    # served model, one file per turn.
    "claude-fable-5.1-interactive": "Claude Fable 5.1 interactive (relay)",
    "human-interactive": "Human (relay)",
    # The harness's own check, not a policy: a script behind the relay that
    # projects the certified route's next point into the photographs. Its
    # report proves the harness can carry a courier to both doors; it is
    # never a benchmark number.
    "harness-oracle": "Harness oracle (route projection, not a policy)",
}


REPORT_KEYS = {
    "experiment", "action_space", "camera_view", "region",
    "pickup_offset_cm", "dropoff_offset_cm", "dropoff_tolerance_cm",
    "model", "setup_request", "setup", "config", "turns", "transcript",
    "summary", "termination", "model_transport_stats", "backend_events",
}
TURNING_REPORT_KEYS = (
    REPORT_KEYS - {"pickup_offset_cm", "dropoff_offset_cm"} | {"route"}
)
POOL_REPORT_KEYS = (
    REPORT_KEYS - {"pickup_offset_cm", "dropoff_offset_cm"}
    | {"pickup_tolerance_cm", "route", "scenario"}
)
# The four-view harness reports which views it showed and whether the walk
# followed certified pedestrian ways.
QUAD_REPORT_KEYS = POOL_REPORT_KEYS | {"pixel_views", "pedestrian_routing"}
ROUTE_KEYS = {
    "profile", "waypoints", "pickup_waypoint_id", "dropoff_waypoint_id",
    "planned_total_cm", "planned_delivery_cm",
}
ROUTE_WAYPOINT_KEYS = {"id", "x_cm", "y_cm", "street", "role"}
MODEL_KEYS = {
    "id", "endpoint", "max_tokens", "max_images_per_prompt", "history_turns",
    "system_prompt",
}
TURN_KEYS = {"turn", "input", "output", "timing", "movement"}
INPUT_KEYS = {"system_prompt", "text", "frames", "messages"}
OUTPUT_KEYS = {
    "raw_reply", "parsed_action", "status", "error", "feedback",
    "rejected_replies", "model_attempts",
}
TIMING_KEYS = {"model_latency_s", "turn_wall_latency_s"}
FRAME_KEYS = {
    "label", "path", "kind", "svg", "view_id", "capture_group_id",
    "camera_snapshot_id", "camera_intrinsics_id", "camera_yaw_deg", "width",
    "height", "sha256", "capture_pose", "capture_timing", "pair_timing",
    "required_group",
}
MOVEMENT_KEYS = {
    "kind", "selected_view", "pixel_uv", "capture_group_id",
    "camera_snapshot_id", "feedback", "start_pose", "end_pose",
    "resolution_outcome", "controller_outcome", "walked_cm", "raw",
}
LATENCY_KEYS = {
    "view_pair_capture", "pixel_action_execute", "model_inference", "turn_wall",
}
LATENCY_STAT_KEYS = {"count", "median_s", "p95_s"}
POSE_KEYS = {"x_cm", "y_cm", "z_cm", "yaw_deg"}
MODEL_ATTEMPT_KEYS = {"payload", "result", "error", "parse_error"}
MODEL_PAYLOAD_KEYS = {"model", "messages", "max_tokens", "temperature"}
TRANSPORT_STATS_KEYS = {
    "model_calls", "requeries", "truncations", "budget_clamps",
    "history_drops", "transport_retries", "timeouts", "unparseable_turns",
}
CAPTURE_EVENT_KEYS = {
    "kind", "capture_group_id", "wall_latency_s", "ue_timing", "views",
}
QUAD_CAPTURE_EVENT_KEYS = CAPTURE_EVENT_KEYS | {"capture_group_ids", "gate_applied"}
NUDGE_EVENT_KEYS = {"kind", "from_cm", "to_cm", "landed_cm", "distance_cm", "residual_cm"}
LEG_EVENT_KEYS = {
    "kind", "outcome", "target_cm", "pixel_uv", "wall_latency_s",
}
ACTION_EVENT_KEYS = {
    "kind", "capture_group_id", "view_id", "camera_snapshot_id", "pixel_uv",
    "wall_latency_s", "outcome",
}
NON_ACTION_STATUSES = {"format_error", "truncated_reply", "truncated"}
CONTROLLER_OUTCOMES = {"arrived", "stuck", "timeout"}


class AuditedModelClient(ModelClient):
    """ModelClient with an exact, turn-scoped transport audit trail."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._turn_attempts: list[dict[str, Any]] = []

    def begin_audit_turn(self) -> None:
        self._turn_attempts = []

    def record_requery(self, reply: str, error: str) -> None:
        if not self._turn_attempts:
            raise RuntimeError("model requery has no recorded transport attempt")
        attempt = self._turn_attempts[-1]
        if _attempt_reply(attempt) != reply:
            raise RuntimeError("model requery reply does not match its transport result")
        attempt["parse_error"] = error

    def finish_audit_turn(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._turn_attempts)

    def _raw_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return super()._post(payload)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = {
            "payload": copy.deepcopy(payload),
            "result": None,
            "error": None,
            "parse_error": None,
        }
        try:
            result = self._raw_post(payload)
        except Exception as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
            self._turn_attempts.append(attempt)
            raise
        attempt["result"] = copy.deepcopy(result)
        self._turn_attempts.append(attempt)
        return result


def build_front_rear_setup_request(
    region: Any,
    *,
    base_builder: Callable[[Any], dict[str, object]] = build_paris_setup_request,
) -> dict[str, object]:
    request = dict(base_builder(region))
    request["enable_rear_camera"] = True
    return request


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{label} does not match the exact report contract: {actual!r}")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if nonnegative and number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _validate_pose(value: Any, label: str) -> None:
    pose = _exact_keys(value, POSE_KEYS, label)
    for key in POSE_KEYS:
        _finite_number(pose[key], f"{label}.{key}")


def _literal(value: ast.AST) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError("walk_to_pixel action contains a non-literal argument") from error


def _walk_binding(action: str) -> tuple[str, float, float] | None:
    if not isinstance(action, str) or not action:
        raise ValueError("parsed action must be a non-empty string")
    try:
        expression = ast.parse(action, mode="eval").body
    except SyntaxError as error:
        raise ValueError("parsed action is not a single call") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("parsed action is not a single named call")
    if expression.func.id != "walk_to_pixel":
        return None
    if expression.args and expression.keywords:
        raise ValueError("walk_to_pixel may not mix positional and keyword arguments")
    if expression.args:
        if len(expression.args) != 3:
            raise ValueError("walk_to_pixel needs view and one pixel")
        view, u, v = (_literal(item) for item in expression.args)
    else:
        if any(keyword.arg is None for keyword in expression.keywords):
            raise ValueError("walk_to_pixel does not accept expanded keyword arguments")
        arguments = {keyword.arg: _literal(keyword.value)
                     for keyword in expression.keywords}
        if set(arguments) != {"view", "u", "v"}:
            raise ValueError("walk_to_pixel needs view and one pixel")
        view, u, v = arguments["view"], arguments["u"], arguments["v"]
    if view not in PIXEL_VIEWS_QUAD:
        raise ValueError("walk selected view is not front, left, right or rear")
    u_f = _finite_number(u, "walk pixel u")
    v_f = _finite_number(v, "walk pixel v")
    if not (0.0 <= u_f <= 1.0 and 0.0 <= v_f <= 1.0):
        raise ValueError("walk pixel is outside [0, 1]")
    return str(view), u_f, v_f


def _message_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("messages must contain JSON objects")
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    text = item.get("text")
                    if not isinstance(text, str):
                        raise ValueError("message text must be a string")
                    texts.append(text)
    return texts


def _same_number(first: Any, second: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(
            _finite_number(first, "first comparison value"),
            _finite_number(second, "second comparison value"),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except ValueError:
        return False


def _same_pose(first: Any, second: Any, *, tolerance: float = 1e-6) -> bool:
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False
    if set(first) != POSE_KEYS or set(second) != POSE_KEYS:
        return False
    if not all(_same_number(first[key], second[key], tolerance=tolerance)
               for key in ("x_cm", "y_cm", "z_cm")):
        return False
    try:
        first_yaw = _finite_number(first["yaw_deg"], "first pose yaw")
        second_yaw = _finite_number(second["yaw_deg"], "second pose yaw")
    except ValueError:
        return False
    angular_gap = abs((first_yaw - second_yaw + 180.0) % 360.0 - 180.0)
    return angular_gap <= max(tolerance, 1e-4)


def _decode_image_url(value: Any, label: str) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {"type", "image_url"} \
            or value.get("type") != "image_url":
        raise ValueError(f"{label} is not an exact image message")
    image_url = value.get("image_url")
    if not isinstance(image_url, Mapping) or set(image_url) != {"url"}:
        raise ValueError(f"{label} image URL is malformed")
    url = image_url["url"]
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError(f"{label} image URL is incomplete")
    header, separator, encoded = url.partition(",")
    if separator != "," or not header.endswith(";base64"):
        raise ValueError(f"{label} image URL is not base64 data")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} image bytes are invalid") from error


def _attempt_reply(attempt: Mapping[str, Any]) -> str:
    result = attempt.get("result")
    if not isinstance(result, Mapping):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices \
            or not isinstance(choices[0], Mapping):
        raise ValueError("model attempt result has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("model attempt result has no message")
    reply, _reasoning = split_reasoning(dict(message))
    return reply


def _attempt_finish_reason(attempt: Mapping[str, Any]) -> str:
    result = attempt.get("result")
    choices = result.get("choices") if isinstance(result, Mapping) else None
    if not isinstance(choices, list) or not choices \
            or not isinstance(choices[0], Mapping):
        raise ValueError("model attempt result has no finish reason")
    finish = choices[0].get("finish_reason")
    if not isinstance(finish, str) or not finish:
        raise ValueError("model attempt finish reason is missing")
    return finish


def _validate_pool_scenario_contract(root: Mapping[str, Any]) -> None:
    """Re-resolve a pooled order and prove the report names that exact result."""

    if root["pickup_tolerance_cm"] != VALIDATED_POOL_PICKUP_TOLERANCE_CM:
        raise ValueError("report pickup tolerance is not exactly 100 cm")
    raw_scenario = root["scenario"]
    if not isinstance(raw_scenario, Mapping):
        raise ValueError("pooled report scenario must be an object")
    raw_pool = raw_scenario.get("pool")
    if not isinstance(raw_pool, Mapping) \
            or raw_pool.get("schema") not in SUPPORTED_POOL_SCHEMAS:
        raise ValueError("pooled report does not identify the validated pool schema")
    pool_path = raw_pool.get("path")
    if not isinstance(pool_path, str) or not pool_path:
        raise ValueError("pooled report has no order-pool path")
    pool = load_validated_delivery_pool(pool_path)
    if raw_pool.get("schema") != pool.schema:
        raise ValueError("pooled report order-pool schema does not match the file")
    if raw_pool.get("sha256") != pool.sha256:
        raise ValueError("pooled report order-pool SHA-256 does not match the file")

    raw_config = root["config"]
    if not isinstance(raw_config, Mapping):
        raise ValueError("pooled report runtime config must be an object")
    raw_adjustment = raw_config.get("max_navmesh_adjustment_cm")
    if (isinstance(raw_adjustment, bool)
            or not isinstance(raw_adjustment, (int, float))
            or not math.isfinite(raw_adjustment)
            or raw_adjustment < 0.0
            or raw_adjustment > pool.validation.max_navmesh_adjustment_cm):
        raise ValueError(
            "pooled report runtime NavMesh adjustment exceeds its certified limit")

    raw_constraints = raw_scenario.get("constraints")
    if not isinstance(raw_constraints, Mapping):
        raise ValueError("pooled report constraints must be an object")
    try:
        min_delivery_cm = raw_constraints["min_delivery_cm"]
        max_delivery_cm = raw_constraints["max_delivery_cm"]
        min_turns = raw_constraints["min_turns"]
        require_different_streets = raw_constraints["require_different_streets"]
        require_marked_crossing = raw_constraints["require_marked_crossing"]
        if (isinstance(min_delivery_cm, bool)
                or not isinstance(min_delivery_cm, (int, float))
                or isinstance(max_delivery_cm, bool)
                or not isinstance(max_delivery_cm, (int, float))
                or isinstance(min_turns, bool)
                or not isinstance(min_turns, int)
                or not isinstance(require_different_streets, bool)
                or not isinstance(require_marked_crossing, bool)):
            raise ValueError("pooled report constraint types are invalid")
        constraints = OrderConstraints(
            min_delivery_cm=min_delivery_cm,
            max_delivery_cm=max_delivery_cm,
            min_turns=min_turns,
            require_different_streets=require_different_streets,
            require_marked_crossing=require_marked_crossing,
        )
        mode = raw_scenario["mode"]
        seed = raw_scenario["seed"]
        if not isinstance(mode, str) or isinstance(seed, bool) \
                or not isinstance(seed, int):
            raise ValueError("pooled report mode or seed type is invalid")
        raw_request = raw_scenario["request"]
        if not isinstance(raw_request, Mapping) \
                or set(raw_request) != {"spawn_id", "pickup_id", "dropoff_id"}:
            raise ValueError("pooled report order request is incomplete")
        spawn_id = raw_request["spawn_id"]
        pickup_id = raw_request["pickup_id"]
        dropoff_id = raw_request["dropoff_id"]
        if any(value is not None and not isinstance(value, str)
               for value in (spawn_id, pickup_id, dropoff_id)):
            raise ValueError("pooled report order request ids must be text or null")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pooled report scenario metadata is incomplete") from error
    scenario = resolve_delivery_scenario(
        pool,
        mode=mode,
        seed=seed,
        constraints=constraints,
        spawn_id=spawn_id,
        pickup_id=pickup_id,
        dropoff_id=dropoff_id,
    )
    expected_scenario = scenario_report(pool, scenario)
    if raw_scenario != expected_scenario:
        raise ValueError("pooled report scenario does not re-resolve exactly")

    network = build_validated_pool_network(
        build_road_network(MAPS, map_name="citycore-paris"), pool)
    expected_route = selected_route_report(network, pool, scenario)
    if root["route"] != expected_route:
        raise ValueError("pooled report route does not match the selected scenario")

    spawn_node = pool.nodes_by_id[scenario.spawn.node_id]
    expected_region = {
        "name": pool.region.name,
        "agent_spawn_cm": [
            spawn_node.x_cm, spawn_node.y_cm, scenario.spawn.z_cm],
        "agent_yaw_deg": scenario.spawn.yaw_deg,
        "nav_bounds_center_cm": list(pool.region.nav_bounds_center_cm),
        "nav_bounds_extent_cm": list(pool.region.nav_bounds_extent_cm),
    }
    if root["region"] != expected_region:
        raise ValueError("pooled report region does not match the selected spawn")
    setup_request = root["setup_request"]
    if not isinstance(setup_request, Mapping) \
            or setup_request.get("scene") != pool.scene \
            or setup_request.get("agent_spawn_cm") != expected_region["agent_spawn_cm"] \
            or setup_request.get("agent_yaw_deg") != expected_region["agent_yaw_deg"] \
            or setup_request.get("nav_bounds_center_cm") \
                != expected_region["nav_bounds_center_cm"] \
            or setup_request.get("nav_bounds_extent_cm") \
                != expected_region["nav_bounds_extent_cm"]:
        raise ValueError("pooled setup request does not match its certified region")


def validate_front_rear_delivery_report(
    report: dict[str, Any],
    *,
    expected_model_id: str = "qwen3-vl-8b",
) -> dict[str, Any]:
    """Fail closed unless ``report`` is the complete dual-view audit contract."""

    if not expected_model_id:
        raise ValueError("expected model id must be nonempty")

    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("report must be JSON serializable with finite numbers") from error
    report_keys = set(report) if isinstance(report, Mapping) else set()
    if report_keys == REPORT_KEYS:
        root = _exact_keys(report, REPORT_KEYS, "report")
    elif report_keys == POOL_REPORT_KEYS:
        root = _exact_keys(report, POOL_REPORT_KEYS, "report")
        _validate_pool_scenario_contract(root)
    elif report_keys == QUAD_REPORT_KEYS:
        root = _exact_keys(report, QUAD_REPORT_KEYS, "report")
        _validate_pool_scenario_contract(root)
        if tuple(root["pixel_views"]) not in (PIXEL_VIEWS, PIXEL_VIEWS_QUAD):
            raise ValueError("report pixel_views is not the pair or the quad")
        if not isinstance(root["pedestrian_routing"], bool):
            raise ValueError("report pedestrian_routing must be a boolean")
    elif report_keys == TURNING_REPORT_KEYS:
        root = _exact_keys(report, TURNING_REPORT_KEYS, "report")
        route = _exact_keys(root["route"], ROUTE_KEYS, "turning route")
        if route["profile"] != TURNING_ROUTE_PROFILE:
            raise ValueError("turning route profile is not the audited L route")
        if route["pickup_waypoint_id"] != TURNING_PICKUP_NODE_ID \
                or route["dropoff_waypoint_id"] != TURNING_DROPOFF_NODE_ID:
            raise ValueError("turning route endpoints are not exact")
        waypoints = route["waypoints"]
        if not isinstance(waypoints, list) \
                or len(waypoints) != len(TURNING_ROUTE_WAYPOINTS):
            raise ValueError("turning route waypoints are incomplete")
        points: list[tuple[float, float]] = []
        for index, (raw_waypoint, expected) in enumerate(
                zip(waypoints, TURNING_ROUTE_WAYPOINTS), 1):
            waypoint = _exact_keys(
                raw_waypoint, ROUTE_WAYPOINT_KEYS,
                f"turning route waypoint {index}",
            )
            expected_id, expected_x, expected_y, _street_index, expected_role = expected
            if (waypoint["id"] != expected_id
                    or waypoint["role"] != expected_role
                    or not isinstance(waypoint["street"], str)
                    or not waypoint["street"]
                    or not _same_number(waypoint["x_cm"], expected_x)
                    or not _same_number(waypoint["y_cm"], expected_y)):
                raise ValueError("turning route waypoint is not exact")
            points.append((float(waypoint["x_cm"]), float(waypoint["y_cm"])))
        segment_lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        pickup_index = [row[0] for row in TURNING_ROUTE_WAYPOINTS].index(
            TURNING_PICKUP_NODE_ID)
        if (not _same_number(route["planned_total_cm"], sum(segment_lengths))
                or not _same_number(
                    route["planned_delivery_cm"],
                    sum(segment_lengths[pickup_index:]))):
            raise ValueError("turning route planned distance is inconsistent")
    else:
        _exact_keys(report, REPORT_KEYS, "report")
        raise AssertionError("unreachable")
    expected_views = tuple(root.get("pixel_views") or PIXEL_VIEWS)
    quad = expected_views == PIXEL_VIEWS_QUAD
    view_count = len(expected_views)
    if root["action_space"] != ACTION_SPACE_PIXEL_GOAL_FRONT_REAR:
        raise ValueError("report action_space is not pixel_goal_front_rear")
    if root["camera_view"] != CAMERA_VIEW_FRONT_REAR:
        raise ValueError("report camera_view is not front_rear")
    if root["dropoff_tolerance_cm"] != CORRIDOR_DROPOFF_TOLERANCE_CM:
        raise ValueError("report dropoff tolerance is not the harness tolerance")
    model = _exact_keys(root["model"], MODEL_KEYS, "model")
    if model["id"] != expected_model_id:
        raise ValueError(
            f"report model id is not {expected_model_id}")
    if (not isinstance(model["system_prompt"], str)
            or not model["system_prompt"]):
        raise ValueError("model system prompt must be complete")
    if (isinstance(model["max_images_per_prompt"], bool)
            or not isinstance(model["max_images_per_prompt"], int)
            or model["max_images_per_prompt"] < 3):
        raise ValueError("model capacity must allow at least three images")
    history_turns = model["history_turns"]
    if isinstance(history_turns, bool) or not isinstance(history_turns, int) \
            or history_turns < 0:
        raise ValueError("model history_turns must be a non-negative integer")
    setup_request = root["setup_request"]
    if (not isinstance(setup_request, Mapping)
            or setup_request.get("enable_rear_camera") is not True):
        raise ValueError("setup request must enable the rear camera")
    setup_response = root["setup"]
    if (not isinstance(setup_response, Mapping)
            or setup_response.get("rear_camera_enabled") is not True):
        raise ValueError("setup response did not confirm the rear camera")

    transcript = root["transcript"]
    if not isinstance(transcript, list) or not transcript:
        raise ValueError("transcript must contain at least one model turn")
    if root["turns"] != len(transcript):
        raise ValueError("turn count does not match transcript")
    groups: set[str] = set()
    group_order: list[str] = []
    previous_turns: list[tuple[str, str]] = []
    movement_records: list[Mapping[str, Any]] = []
    successful_attempt_count = 0
    truncation_attempt_count = 0
    requery_attempt_count = 0
    budget_clamp_attempt_count = 0
    history_drop_attempt_count = 0
    transport_retry_attempt_count = 0
    timeout_retry_attempt_count = 0
    unparseable_turn_count = 0
    walk_count = 0
    for expected_turn, raw_turn in enumerate(transcript, 1):
        turn = _exact_keys(raw_turn, TURN_KEYS, f"turn {expected_turn}")
        if turn["turn"] != expected_turn:
            raise ValueError("transcript turn numbers are not consecutive")
        input_record = _exact_keys(
            turn["input"], INPUT_KEYS, f"turn {expected_turn} input")
        output = _exact_keys(
            turn["output"], OUTPUT_KEYS, f"turn {expected_turn} output")
        timing = _exact_keys(
            turn["timing"], TIMING_KEYS, f"turn {expected_turn} timing")
        if input_record["system_prompt"] != (
                model["system_prompt"] if expected_turn == 1 else None):
            raise ValueError("per-turn system prompt contract is not exact")
        if not isinstance(input_record["text"], str) or not input_record["text"]:
            raise ValueError("turn input text must be complete")
        messages = input_record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("turn messages must be complete")
        if messages[0] != {"role": "system", "content": model["system_prompt"]}:
            raise ValueError("messages must retain the exact system prompt")
        frames = input_record["frames"]
        if (not isinstance(frames, list)
                or len(frames) not in (view_count, view_count + 1)):
            raise ValueError(
                "turn frame cardinality must be "
                f"{', '.join(expected_views)}, and optional map")
        current_user = messages[-1]
        if (not isinstance(current_user, Mapping)
                or current_user.get("role") != "user"
                or not isinstance(current_user.get("content"), list)):
            raise ValueError("messages do not end with the current user input")
        current_content = current_user["content"]
        if (len(current_content) != len(frames) + 1
                or current_content[0] != {
                    "type": "text", "text": input_record["text"]}):
            raise ValueError("message text or message images are not exact")
        required = {}
        for index, view_name in enumerate(expected_views):
            required[view_name] = _exact_keys(
                frames[index], FRAME_KEYS, f"{view_name} frame")
        if [frame["view_id"] for frame in required.values()] != list(expected_views):
            raise ValueError(
                f"required frames are not ordered {', '.join(expected_views)}")
        front, rear = required["front"], required["rear"]
        group = front["capture_group_id"]
        if (not isinstance(group, str) or not group
                or rear["capture_group_id"] != group):
            raise ValueError("required frames do not share one nonempty capture group")
        turn_groups = [group]
        if quad:
            side_group = required["left"]["capture_group_id"]
            if (not isinstance(side_group, str) or not side_group
                    or required["right"]["capture_group_id"] != side_group
                    or side_group == group):
                raise ValueError(
                    "left and right need their own shared capture group")
            turn_groups.append(side_group)
        if any(item in groups for item in turn_groups):
            raise ValueError("a fresh turn reused an earlier capture group")
        groups.update(turn_groups)
        group_order.append(group)
        snapshots = tuple(frame["camera_snapshot_id"] for frame in required.values())
        if any(not isinstance(value, str) or not value for value in snapshots):
            raise ValueError("required frames need nonempty snapshot identities")
        if len(set(snapshots)) != len(snapshots):
            raise ValueError(f"{' and '.join(expected_views)} need distinct snapshots")
        for view_name, frame in required.items():
            if frame["required_group"] is not True or frame["kind"] != "photograph":
                raise ValueError(f"{view_name} is not a required photograph")
            if not isinstance(frame["path"], str) or not frame["path"]:
                raise ValueError(f"{view_name} frame path is missing")
            if any(token in Path(frame["path"]).name.lower()
                   for token in ("front", "rear", "left", "right")):
                raise ValueError("policy-visible image paths must stay opaque")
            if (not isinstance(frame["width"], int) or frame["width"] <= 0
                    or not isinstance(frame["height"], int)
                    or frame["height"] <= 0):
                raise ValueError("frame dimensions must be positive integers")
            _finite_number(frame["camera_yaw_deg"], "camera yaw")
            _validate_pose(frame["capture_pose"], "frame capture pose")
            if not isinstance(frame["capture_timing"], Mapping):
                raise ValueError("frame capture timing is missing")
            if not isinstance(frame["pair_timing"], Mapping):
                raise ValueError("pair capture timing is missing")
            sha256 = frame["sha256"]
            if (not isinstance(sha256, str) or len(sha256) != 64
                    or any(character not in "0123456789abcdef" for character in sha256)):
                raise ValueError(f"{view_name} frame sha256 is malformed")
        for index, (view_name, frame) in enumerate(required.items(), start=1):
            image = _decode_image_url(current_content[index], view_name)
            if hashlib.sha256(image).hexdigest() != frame["sha256"]:
                raise ValueError(f"{view_name} image hash does not match its frame")
        if len(frames) == view_count + 1:
            phone_map = _exact_keys(frames[view_count], FRAME_KEYS, "phone map frame")
            if (phone_map["kind"] != "map" or phone_map["path"] != ""
                    or not isinstance(phone_map["svg"], str) or not phone_map["svg"]
                    or phone_map["required_group"] is not False
                    or any(phone_map[key] is not None for key in (
                        "view_id", "capture_group_id", "camera_snapshot_id",
                        "camera_intrinsics_id", "camera_yaw_deg", "width",
                        "height", "sha256", "capture_pose", "capture_timing",
                        "pair_timing"))):
                raise ValueError(
                    "frame cardinality permits only one positionally exact optional map")
            _decode_image_url(current_content[view_count + 1], "phone map")

        retained = previous_turns[-history_turns:] if history_turns else []
        expected_messages: list[dict[str, Any]] = [
            {"role": "system", "content": model["system_prompt"]},
        ]
        for prior_text, prior_reply in retained:
            expected_messages.extend((
                {"role": "user", "content": prior_text},
                {"role": "assistant", "content": prior_reply},
            ))
        expected_messages.append(copy.deepcopy(dict(current_user)))
        if messages != expected_messages:
            raise ValueError("messages do not retain exact alternating retained history")
        private_ids = [*turn_groups, *snapshots,
                       *{frame["camera_intrinsics_id"] for frame in required.values()}]
        policy_text = [model["system_prompt"], input_record["text"],
                       *_message_texts(messages)]
        if any(str(private) in text for private in private_ids for text in policy_text):
            raise ValueError("policy text leaks private frame identity")

        raw_reply = output["raw_reply"]
        if not isinstance(raw_reply, str) or not raw_reply:
            raise ValueError("turn raw reply must be complete and nonempty")
        rejected_replies = output["rejected_replies"]
        if (not isinstance(rejected_replies, list)
                or any(not isinstance(reply, str) or not reply
                       for reply in rejected_replies)):
            raise ValueError("rejected model replies must be complete strings")
        attempts = output["model_attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("turn model attempts must be complete and nonempty")
        for attempt_index, raw_attempt in enumerate(attempts):
            attempt = _exact_keys(
                raw_attempt, MODEL_ATTEMPT_KEYS,
                f"turn {expected_turn} model attempt {attempt_index + 1}")
            payload = _exact_keys(
                attempt["payload"], MODEL_PAYLOAD_KEYS,
                f"turn {expected_turn} model attempt payload")
            if payload["model"] != model["id"]:
                raise ValueError("model attempt payload uses the wrong model")
            if (isinstance(payload["max_tokens"], bool)
                    or not isinstance(payload["max_tokens"], int)
                    or payload["max_tokens"] <= 0):
                raise ValueError("model attempt max_tokens is invalid")
            _finite_number(payload["temperature"], "model attempt temperature")
            if attempt_index == 0 and payload["messages"] != messages:
                raise ValueError("first model attempt payload is not the exact turn input")
            result, attempt_error = attempt["result"], attempt["error"]
            if result is None:
                if not isinstance(attempt_error, str) or not attempt_error:
                    raise ValueError("failed model attempt is missing its exact error")
                if attempt["parse_error"] is not None:
                    raise ValueError("transport failure cannot have a parse error")
            else:
                if not isinstance(result, Mapping) or attempt_error is not None:
                    raise ValueError("model attempt raw result/error is inconsistent")
                successful_attempt_count += 1
                finish_reason = _attempt_finish_reason(attempt)
                if finish_reason == "length":
                    truncation_attempt_count += 1
                    if attempt["parse_error"] is not None:
                        raise ValueError("truncated model attempt cannot be a requery")
                elif attempt["parse_error"] is not None:
                    if not isinstance(attempt["parse_error"], str) \
                            or not attempt["parse_error"]:
                        raise ValueError("model attempt parse error is malformed")
                    requery_attempt_count += 1
                    if _attempt_reply(attempt) not in rejected_replies:
                        raise ValueError("requery attempt is absent from rejected replies")
            if attempt_index + 1 < len(attempts):
                following = _exact_keys(
                    attempts[attempt_index + 1], MODEL_ATTEMPT_KEYS,
                    "following model attempt")
                following_payload = _exact_keys(
                    following["payload"], MODEL_PAYLOAD_KEYS,
                    "following model attempt payload")
                prior_messages = payload["messages"]
                if result is None:
                    allowed_messages = [prior_messages]
                    if isinstance(prior_messages, list) and len(prior_messages) > 2:
                        allowed_messages.append(prior_messages[2:])
                    if following_payload["messages"] not in allowed_messages:
                        raise ValueError("model attempt history drop is not exact")
                    if following_payload["messages"] == prior_messages[2:]:
                        history_drop_attempt_count += 1
                    elif following_payload["max_tokens"] < payload["max_tokens"]:
                        budget_clamp_attempt_count += 1
                    elif "timed out" in str(attempt["error"]).lower() \
                            or str(attempt["error"]).startswith("TimeoutError:"):
                        timeout_retry_attempt_count += 1
                    else:
                        transport_retry_attempt_count += 1
                elif _attempt_finish_reason(attempt) == "length":
                    if (following_payload["messages"] != prior_messages
                            or following_payload["max_tokens"]
                            <= payload["max_tokens"]):
                        raise ValueError("model truncation retry is not exact")
                elif attempt["parse_error"] is not None:
                    correction = {
                        "role": "user",
                        "content": (
                            "That reply could not be read as an action: "
                            f"{attempt['parse_error']}\n"
                            "Reply again, in the required shape, with exactly one call."
                        ),
                    }
                    expected_requery = [
                        *prior_messages,
                        {"role": "assistant", "content": _attempt_reply(attempt)},
                        correction,
                    ]
                    if following_payload["messages"] != expected_requery:
                        raise ValueError("model requery messages are not exact")
                else:
                    raise ValueError("model attempts contain an unexplained retry")
        if _attempt_reply(attempts[-1]) != raw_reply:
            raise ValueError("complete raw reply does not match the final model attempt")

        parsed_action = output["parsed_action"]
        status, error = output["status"], output["error"]
        if not isinstance(status, str) or not isinstance(error, str):
            raise ValueError("turn status and error must be typed strings")
        if status in {"accepted", "rejected"}:
            if not isinstance(parsed_action, str) or not parsed_action:
                raise ValueError("accepted or rejected turn needs a parsed action")
            if (status == "accepted" and error != "") \
                    or (status == "rejected" and not error):
                raise ValueError("turn status and error are inconsistent")
        elif status in NON_ACTION_STATUSES:
            if parsed_action is not None or not error:
                raise ValueError("nonaccepted turn status/error/action are inconsistent")
            if status != "truncated":
                unparseable_turn_count += 1
        else:
            raise ValueError("turn status is unknown")
        expected_rejected = [
            _attempt_reply(attempt) for attempt in attempts
            if attempt["parse_error"] is not None
        ]
        terminal_truncation = _attempt_finish_reason(attempts[-1]) == "length"
        if (parsed_action is None and status != "truncated"
                and not terminal_truncation):
            expected_rejected.append(raw_reply)
        if rejected_replies != expected_rejected:
            raise ValueError("rejected replies do not match the exact model attempts")
        if not isinstance(output["feedback"], str) or not output["feedback"]:
            raise ValueError("turn feedback must be retained")
        for key in TIMING_KEYS:
            _finite_number(timing[key], f"turn timing {key}", nonnegative=True)

        previous_turns.append((input_record["text"], raw_reply))
        binding = _walk_binding(parsed_action) if parsed_action is not None else None
        movement = turn["movement"]
        if parsed_action is None:
            if movement is not None:
                raise ValueError("unparseable turn unexpectedly contains movement")
            continue
        if binding is None:
            if movement is not None:
                raise ValueError("non-pixel turn unexpectedly contains movement")
            continue
        walk_count += 1
        movement = _exact_keys(
            movement, MOVEMENT_KEYS, f"turn {expected_turn} movement contract")
        view, u, v = binding
        if movement["selected_view"] != view:
            raise ValueError("movement selected view does not match parsed action")
        pixel = movement["pixel_uv"]
        if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                or [_finite_number(item, "movement pixel") for item in pixel]
                != [u, v]):
            raise ValueError("movement pixel does not match parsed action")
        selected_frame = required[view]
        if movement["capture_group_id"] != selected_frame["capture_group_id"]:
            raise ValueError("movement capture group does not match input pair")
        if movement["camera_snapshot_id"] != selected_frame["camera_snapshot_id"]:
            raise ValueError("movement snapshot does not match selected view")
        if movement["feedback"] != output["feedback"]:
            raise ValueError("movement feedback does not match turn feedback")
        _validate_pose(movement["start_pose"], "movement start pose")
        _validate_pose(movement["end_pose"], "movement end pose")
        if any(not _same_pose(movement["start_pose"], frame["capture_pose"])
               for frame in required.values()):
            raise ValueError("movement start pose does not match pair capture pose")
        walked_cm = _finite_number(
            movement["walked_cm"], "movement walked_cm", nonnegative=True)
        geometric_walked = math.dist(
            (float(movement["start_pose"]["x_cm"]),
             float(movement["start_pose"]["y_cm"])),
            (float(movement["end_pose"]["x_cm"]),
             float(movement["end_pose"]["y_cm"])),
        )
        if not math.isclose(walked_cm, geometric_walked, rel_tol=0.0, abs_tol=0.011):
            raise ValueError("movement walked distance does not match its poses")
        if not isinstance(movement["raw"], Mapping):
            raise ValueError("movement raw record is missing")
        raw_movement = movement["raw"]
        raw_pixel = raw_movement.get("pixel_uv")
        if (raw_movement.get("kind") != movement["kind"]
                or raw_movement.get("selected_view") != view
                or raw_movement.get("capture_group_id") != selected_frame["capture_group_id"]
                or raw_movement.get("camera_snapshot_id")
                    != selected_frame["camera_snapshot_id"]
                or not isinstance(raw_pixel, (list, tuple))
                or len(raw_pixel) != 2
                or [float(raw_pixel[0]), float(raw_pixel[1])] != [u, v]):
            raise ValueError("raw movement record does not match the selected binding")
        if movement["kind"] == "pixel_goal":
            if movement["resolution_outcome"] != "resolved":
                raise ValueError("resolved movement has the wrong resolution outcome")
            controller = movement["controller_outcome"]
            if controller not in CONTROLLER_OUTCOMES:
                raise ValueError("movement controller outcome is invalid")
            if (raw_movement.get("outcome") != controller
                    or raw_movement.get("controller_status") != controller):
                raise ValueError("raw movement outcome does not match normalized outcome")
            expected_turn_outcome = {
                "arrived": ("accepted", ""),
                "stuck": ("rejected", "stuck"),
                "timeout": ("rejected", "walk_timeout"),
            }[controller]
            if (output["status"], output["error"]) != expected_turn_outcome:
                raise ValueError("movement outcome does not match turn status/error")
            if not _same_pose(raw_movement.get("end_pose"), movement["end_pose"]):
                raise ValueError("raw movement end pose does not match normalized end pose")
            if not _same_number(raw_movement.get("walked_cm"), walked_cm):
                raise ValueError("raw movement distance does not match normalized distance")
        elif movement["kind"] == "pixel_refused":
            if (movement["resolution_outcome"] != "rejected"
                    or movement["controller_outcome"] != "not_started"):
                raise ValueError("refused movement outcomes are inconsistent")
            if not isinstance(raw_movement.get("code"), str) \
                    or not raw_movement["code"]:
                raise ValueError("refused raw movement code is missing")
            if (output["status"], output["error"]) != (
                    "rejected", raw_movement["code"]):
                raise ValueError("movement outcome does not match turn status/error")
            if walked_cm != 0.0 or not _same_pose(
                    movement["start_pose"], movement["end_pose"]):
                raise ValueError("refused movement changed the pose")
        else:
            raise ValueError("movement kind is not pixel_goal or pixel_refused")
        movement_records.append(movement)

    backend_events = root["backend_events"]
    if not isinstance(backend_events, list):
        raise ValueError("backend events must be a complete list")
    capture_events: list[Mapping[str, Any]] = []
    action_events: list[Mapping[str, Any]] = []
    valid_backend_outcomes = {
        outcome.value for outcome in ControllerOutcomeCode
    } | {"resolution_rejected", "error"}
    if quad:
        valid_backend_outcomes.add("resolved_only")
    for raw_event in backend_events:
        if not isinstance(raw_event, Mapping):
            raise ValueError("backend event must be a JSON object")
        if raw_event.get("kind") == "view_pair_capture":
            event = _exact_keys(
                raw_event, QUAD_CAPTURE_EVENT_KEYS if quad else CAPTURE_EVENT_KEYS,
                "capture backend event")
            if not isinstance(event["capture_group_id"], str) \
                    or not event["capture_group_id"]:
                raise ValueError("capture backend event group is missing")
            _finite_number(
                event["wall_latency_s"], "capture backend event latency",
                nonnegative=True)
            if (not isinstance(event["ue_timing"], Mapping)
                    or not isinstance(event["views"], Mapping)
                    or set(event["views"]) != set(expected_views)
                    or any(not isinstance(value, Mapping)
                           for value in event["views"].values())):
                raise ValueError("capture backend event timing is incomplete")
            capture_events.append(event)
        elif raw_event.get("kind") == "pixel_action_execute":
            event = _exact_keys(raw_event, ACTION_EVENT_KEYS, "action backend event")
            if event["view_id"] not in {"front", "rear"}:
                # the engine's own view id; a left/right pixel resolves
                # against the second pair's rear/front camera
                raise ValueError("action backend event view is invalid")
            if (not isinstance(event["capture_group_id"], str)
                    or not event["capture_group_id"]
                    or not isinstance(event["camera_snapshot_id"], str)
                    or not event["camera_snapshot_id"]):
                raise ValueError("action backend event binding is incomplete")
            event_pixel = event["pixel_uv"]
            if (not isinstance(event_pixel, list) or len(event_pixel) != 2
                    or any(not 0.0 <= _finite_number(
                        item, "action backend event pixel") <= 1.0
                           for item in event_pixel)):
                raise ValueError("action backend event pixel is invalid")
            _finite_number(
                event["wall_latency_s"], "action backend event latency",
                nonnegative=True)
            if event["outcome"] not in valid_backend_outcomes:
                raise ValueError("action backend event outcome is invalid")
            action_events.append(event)
        elif quad and raw_event.get("kind") == "world_leg":
            event = raw_event
            if not set(LEG_EVENT_KEYS) <= set(event):
                raise ValueError("leg backend event is incomplete")
            _finite_number(event["wall_latency_s"], "leg backend event latency",
                           nonnegative=True)
        elif quad and raw_event.get("kind") == "world_nudge":
            event = _exact_keys(raw_event, NUDGE_EVENT_KEYS, "nudge backend event")
            _finite_number(event["distance_cm"], "nudge distance", nonnegative=True)
            _finite_number(event["residual_cm"], "nudge residual", nonnegative=True)
        else:
            raise ValueError("backend event kind is invalid")
    if [event["capture_group_id"] for event in capture_events] != group_order:
        raise ValueError("backend capture events do not match transcript groups")
    # A left or right pixel resolves against the second pair's rear or
    # front camera, which is the view id the engine event carries.
    engine_view = {"front": "front", "rear": "rear", "left": "rear", "right": "front"}
    for movement in movement_records:
        matches = [
            event for event in action_events
            if (event["view_id"] == engine_view.get(movement["selected_view"])
                and event["capture_group_id"] == movement["capture_group_id"]
                and event["camera_snapshot_id"] == movement["camera_snapshot_id"]
                and event["pixel_uv"] == movement["pixel_uv"])
        ]
        if len(matches) != 1:
            raise ValueError("movement does not have one matching backend action event")
        backend_outcome = matches[0]["outcome"]
        if movement["kind"] == "pixel_refused":
            # under pedestrian routing the engine may have resolved the pixel
            # and the harness refused the destination it named
            allowed = {"resolution_rejected"} | ({"resolved_only"} if quad else set())
            if backend_outcome not in allowed:
                raise ValueError("backend/raw movement outcome is inconsistent")
        elif quad and backend_outcome in ("resolved_only", "resolution_rejected"):
            # the walk itself was the certified legs, recorded as leg events;
            # the engine's verdict was on the pixel's straight line, which the
            # harness does not consult once the hit names a certified node
            continue
        else:
            expected_backend = {
                "arrived": ControllerOutcomeCode.ACCEPTED.value,
                "timeout": ControllerOutcomeCode.EXECUTION_TIMEOUT.value,
                "stuck": ControllerOutcomeCode.CONTROLLER_FAILED.value,
            }[movement["controller_outcome"]]
            if backend_outcome != expected_backend:
                raise ValueError("backend/raw movement outcome is inconsistent")

    latency = root["summary"].get("latency") \
        if isinstance(root["summary"], Mapping) else None
    latency = _exact_keys(latency, LATENCY_KEYS, "summary latency")
    validated_latency: dict[str, Mapping[str, Any]] = {}
    for name in LATENCY_KEYS:
        stats = _exact_keys(latency[name], LATENCY_STAT_KEYS, f"{name} latency")
        validated_latency[name] = stats
        if isinstance(stats["count"], bool) or not isinstance(stats["count"], int) \
                or stats["count"] < 0:
            raise ValueError("summary latency count must be non-negative")
        for key in ("median_s", "p95_s"):
            if stats[key] is None:
                if stats["count"]:
                    raise ValueError("summary latency statistic is missing")
            else:
                _finite_number(stats[key], f"summary latency {key}", nonnegative=True)
        if (stats["median_s"] is not None and stats["p95_s"] is not None
                and float(stats["p95_s"]) < float(stats["median_s"])):
            raise ValueError("summary latency p95 is below its median")
    expected_counts = {
        "view_pair_capture": len(transcript),
        "pixel_action_execute": walk_count,
        "model_inference": len(transcript),
        "turn_wall": len(transcript),
    }
    if any(validated_latency[name]["count"] != expected
           for name, expected in expected_counts.items()):
        raise ValueError("summary latency count does not match the transcript")
    expected_latency = {
        "view_pair_capture": latency_statistics(
            event["wall_latency_s"] for event in capture_events),
        "pixel_action_execute": latency_statistics(
            event["wall_latency_s"] for event in action_events),
        "model_inference": latency_statistics(
            turn["timing"]["model_latency_s"] for turn in transcript),
        "turn_wall": latency_statistics(
            turn["timing"]["turn_wall_latency_s"] for turn in transcript),
    }
    for name, expected in expected_latency.items():
        actual = validated_latency[name]
        matches = actual["count"] == expected["count"]
        for key in ("median_s", "p95_s"):
            if actual[key] is None or expected[key] is None:
                matches = matches and actual[key] is expected[key]
            else:
                matches = matches and math.isclose(
                    float(actual[key]), float(expected[key]),
                    rel_tol=0.0, abs_tol=1e-12)
        if not matches:
            raise ValueError(f"{name} latency does not match retained samples")

    transport = _exact_keys(
        root["model_transport_stats"], TRANSPORT_STATS_KEYS,
        "model transport stats")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in transport.values()):
        raise ValueError("model transport stats must be non-negative integers")
    if (transport["model_calls"] != successful_attempt_count
            or transport["truncations"] != truncation_attempt_count
            or transport["requeries"] != requery_attempt_count
            or transport["budget_clamps"] != budget_clamp_attempt_count
            or transport["history_drops"] != history_drop_attempt_count
            or transport["transport_retries"] != transport_retry_attempt_count
            or transport["timeouts"] != timeout_retry_attempt_count
            or transport["unparseable_turns"] != unparseable_turn_count):
        raise ValueError("model attempt statistics do not match the transcript")
    return {"passed": True, "turn_count": len(transcript),
            "walk_turn_count": walk_count}


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise ValueError("model input frame is not a PNG or JPEG")


def frame_data_url(frame: Frame, *, image_root: Path | None = None) -> str:
    """Encode one policy frame, rasterizing a phone map to an opaque PNG."""

    if frame.path:
        data = Path(frame.path).read_bytes()
    elif frame.svg:
        try:
            import cairosvg
        except ImportError as error:
            raise RuntimeError(
                "the phone map needs cairosvg so all three images reach the model"
            ) from error
        data = cairosvg.svg2png(bytestring=frame.svg.encode())
        if image_root is not None:
            image_root.mkdir(parents=True, exist_ok=True)
            opaque = image_root / f"{hashlib.sha256(data).hexdigest()[:16]}.png"
            if not opaque.exists():
                opaque.write_bytes(data)
    else:
        raise ValueError("model frame has neither a raster path nor SVG data")
    return f"data:{_image_mime(data)};base64," + base64.b64encode(data).decode()


def _pose_dict(env: Any) -> dict[str, float]:
    pose = getattr(env, "ue_pose", None)
    if pose is not None and hasattr(pose, "to_dict"):
        raw = pose.to_dict()
        return {key: float(raw[key]) for key in POSE_KEYS}
    x_cm, y_cm = env.position()
    return {"x_cm": float(x_cm), "y_cm": float(y_cm), "z_cm": 0.0,
            "yaw_deg": float(env.facing() or 0.0)}


def _normalise_movement(
    raw: Mapping[str, Any], *, start_pose: dict[str, float],
    end_pose: dict[str, float], feedback: str,
) -> dict[str, Any]:
    kind = str(raw.get("kind") or "")
    resolution = "resolved" if kind == "pixel_goal" else "rejected"
    controller = (
        raw.get("controller_status") or raw.get("outcome")
        if kind == "pixel_goal" else "not_started"
    )
    walked = raw.get("walked_cm")
    if walked is None:
        walked = math.dist(
            (start_pose["x_cm"], start_pose["y_cm"]),
            (end_pose["x_cm"], end_pose["y_cm"]),
        )
    return {
        "kind": kind,
        "selected_view": raw.get("selected_view"),
        "pixel_uv": list(raw.get("pixel_uv") or ()),
        "capture_group_id": raw.get("capture_group_id"),
        "camera_snapshot_id": raw.get("camera_snapshot_id"),
        "feedback": feedback,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "resolution_outcome": resolution,
        "controller_outcome": str(controller or "not_started"),
        "walked_cm": float(walked),
        "raw": copy.deepcopy(dict(raw)),
    }


def run_model_turns(
    courier_session: Any,
    model: Any,
    *,
    system_prompt: str,
    max_turns: int,
    history_turns: int,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
    image_encoder: Callable[[Frame, int], str] | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Drive the real pending-observation session seam with testable I/O."""

    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if history_turns < 0:
        raise ValueError("history_turns must be non-negative")
    logger = logger or logging.getLogger(__name__)
    encoder = image_encoder or (lambda frame, _index: frame_data_url(frame))
    transcript: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for turn_number in range(1, max_turns + 1):
        if courier_session.finished:
            break
        turn_started = perf_counter_fn()
        observation = courier_session.observe()
        image_urls = [encoder(frame, index)
                      for index, frame in enumerate(observation.frames)]
        content: list[dict[str, Any]] = [
            {"type": "text", "text": observation.text},
            *({"type": "image_url", "image_url": {"url": url}}
              for url in image_urls),
        ]
        history.append({"role": "user", "content": content})
        keep = history_turns * 2 + 1
        recent = history[-keep:] if keep > 0 else [history[-1]]
        stripped: list[dict[str, Any]] = []
        for index, message in enumerate(recent):
            if (message["role"] == "user"
                    and isinstance(message["content"], list)
                    and index < len(recent) - 1):
                text = next(
                    (item["text"] for item in message["content"]
                     if item.get("type") == "text"),
                    "",
                )
                stripped.append({"role": "user", "content": text})
            else:
                stripped.append(message)
        messages = [{"role": "system", "content": system_prompt}, *stripped]
        exact_messages = copy.deepcopy(messages)
        start_pose = _pose_dict(courier_session.env)
        log_start = len(courier_session.env.embodied_log)

        begin_audit = getattr(model, "begin_audit_turn", None)
        if callable(begin_audit):
            begin_audit()
        model_started = perf_counter_fn()
        parse = lambda text: parse_reply(
            text,
            set(courier_session.allowed),
            tools_by_name=getattr(courier_session, "_tools_by_name", None),
        )
        record_requery = getattr(model, "record_requery", None)
        if callable(record_requery):
            reply, parsed, rejected = model.act(
                messages, parse, on_requery=record_requery)
        else:
            reply, parsed, rejected = model.act(messages, parse)
        finish_audit = getattr(model, "finish_audit_turn", None)
        model_attempts = finish_audit() if callable(finish_audit) else []
        if not reply and model_attempts:
            reply = _attempt_reply(model_attempts[-1])
        model_elapsed = perf_counter_fn() - model_started
        if parsed is None:
            logger.warning("turn %d did not contain a parseable action", turn_number)
        history.append({"role": "assistant", "content": reply})
        log = courier_session.step(reply)
        end_pose = _pose_dict(courier_session.env)
        new_events = courier_session.env.embodied_log[log_start:]
        movement_rows = [
            event for event in new_events
            if event.get("kind") in {"pixel_goal", "pixel_refused"}
        ]
        action_value = log.action if isinstance(log.action, str) and log.action else None
        binding = _walk_binding(action_value) if action_value is not None else None
        is_pixel = binding is not None
        matching_rows: list[Mapping[str, Any]] = []
        if binding is not None:
            selected_view, u, v = binding
            selected_frame = next(
                frame for frame in observation.frames
                if frame.view_id == selected_view)
            matching_rows = [
                event for event in movement_rows
                if (event.get("selected_view") == selected_view
                    and list(event.get("pixel_uv") or ()) == [u, v]
                    and event.get("capture_group_id")
                    == selected_frame.capture_group_id
                    and event.get("camera_snapshot_id")
                    == selected_frame.camera_snapshot_id)
            ]
            if len(matching_rows) != 1:
                raise RuntimeError(
                    "walk_to_pixel turn needs one unique matching movement record")
        movement = (
            _normalise_movement(
                matching_rows[0], start_pose=start_pose, end_pose=end_pose,
                feedback=courier_session.feedback,
            )
            if is_pixel else None
        )
        transcript.append({
            "turn": turn_number,
            "input": {
                "system_prompt": system_prompt if turn_number == 1 else None,
                "text": observation.text,
                "frames": [frame.to_dict() for frame in observation.frames],
                "messages": exact_messages,
            },
            "output": {
                "raw_reply": reply,
                "parsed_action": action_value,
                "status": log.status,
                "error": log.error,
                "feedback": courier_session.feedback,
                "rejected_replies": list(rejected),
                "model_attempts": model_attempts,
            },
            "timing": {
                "model_latency_s": model_elapsed,
                "turn_wall_latency_s": perf_counter_fn() - turn_started,
            },
            "movement": movement,
        })
        logger.info(
            "turn %3d %-28s %-13s %.3fs", turn_number, str(action_value),
            log.status, transcript[-1]["timing"]["turn_wall_latency_s"],
        )
    return transcript


def run_delivery(
    args: argparse.Namespace,
    *,
    expected_model_id: str = "qwen3-vl-8b",
    experiment_model_label: str = "Qwen",
) -> dict[str, Any]:
    if args.model != expected_model_id:
        raise ValueError(
            f"the front/rear delivery artifact requires {expected_model_id}")
    needed_images = 5 if getattr(args, "views", "quad") == "quad" else 3
    if args.model_image_capacity < needed_images:
        raise ValueError(
            f"{'four views' if needed_images == 5 else 'front/rear'} plus an "
            f"active phone map requires capacity for {needed_images} images")
    logger = logging.getLogger(__name__)
    simworld_root = Path(args.simworld_root).resolve()
    citycore_content = Path(args.citycore_content).resolve()
    validate_mount_inputs(citycore_content, simworld_root / "SimWorld.uproject")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log = output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)

    pooled_profile = args.route_profile in ("validated_pool", "long_turn")
    delivery_pool = None
    scenario = None
    paris = build_road_network(MAPS, map_name="citycore-paris")
    if pooled_profile:
        delivery_pool = load_validated_delivery_pool(args.order_pool)
        if args.max_navmesh_adjustment_cm \
                > delivery_pool.validation.max_navmesh_adjustment_cm:
            raise ValueError(
                "runtime NavMesh adjustment limit is weaker than the certified "
                "order-pool limit")
        constraints = OrderConstraints(
            min_delivery_cm=args.min_delivery_m * 100.0,
            max_delivery_cm=args.max_delivery_m * 100.0,
            min_turns=args.min_route_turns,
            require_different_streets=args.require_different_streets,
            require_marked_crossing=args.require_marked_crossing,
        )
        scenario = resolve_delivery_scenario(
            delivery_pool,
            mode=args.order_mode,
            seed=args.seed,
            constraints=constraints,
            spawn_id=args.spawn_id,
            pickup_id=args.pickup_id,
            dropoff_id=args.dropoff_id,
        )
        paris = build_validated_pool_network(paris, delivery_pool)
        spawn_node = delivery_pool.nodes_by_id[scenario.spawn.node_id]
        region = replace(
            DELIVERY_REGION,
            name=delivery_pool.region.name,
            agent_spawn_cm=(
                spawn_node.x_cm, spawn_node.y_cm, scenario.spawn.z_cm),
            agent_yaw_deg=scenario.spawn.yaw_deg,
            nav_bounds_center_cm=delivery_pool.region.nav_bounds_center_cm,
            nav_bounds_extent_cm=delivery_pool.region.nav_bounds_extent_cm,
        )
        episode_id = f"pixel-goal-front-rear-{scenario.scenario_id}"
    else:
        region = DELIVERY_REGION
        episode_id = "pixel-goal-front-rear-delivery"

    config = PixelGoalConfig(
        max_navmesh_adjustment_cm=args.max_navmesh_adjustment_cm,
        acceptance_radius_cm=args.acceptance_radius_cm,
        movement_timeout_sim_s=getattr(
            args, "movement_timeout_sim_s", 45.0),
        execution_timeout_s=getattr(
            args, "wall_watchdog_timeout_s",
            getattr(args, "execution_timeout_s", 300.0)),
        poll_interval_s=args.poll_interval_s,
    )
    camera = CameraSpec(
        width=args.capture_width, height=args.capture_height,
        fov_deg=config.fov_degrees,
    )
    setup_request = build_front_rear_setup_request(region)
    if delivery_pool is not None and setup_request.get("scene") != delivery_pool.scene:
        raise ValueError("order pool scene does not match the UE setup scene")
    session = None
    play_started = False
    transcript: list[dict[str, Any]] = []
    capture_gate: CaptureReadinessGate | None = None
    try:
        session = AttachedParisGameSession(args.spear_config)
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        preparation = _prepare_paris_loop(
            endpoint, setup_request,
            readiness_timeout_s=args.navmesh_timeout_s,
            capture_warmup_s=args.capture_warmup_s,
            wait_fn=_wait_for_paris_poc,
            warm_up_fn=_warm_up_paris_capture,
        )
        setup_contract = preparation["setup"].get(
            "rgb_capture_contract_version")
        if setup_contract != "paris-agent-native-rgb-v2":
            raise RuntimeError(
                "UE Pixel Goal camera contract is missing or stale: "
                f"{setup_contract!r}")
        _reset_paris_poc_trial(endpoint, args.navmesh_timeout_s)
        runtime = LivePixelGoalRuntime(endpoint, config)
        idle_waiter = getattr(session, "wait_for_engine_idle", None)
        capture_gate = CaptureReadinessGate(
            oom_monitor=VulkanOomMonitor(os.environ.get("PIXEL_GOAL_UE_LOG")),
            engine_idle_waiter=idle_waiter,
        )
        spawn_pose = Pose(
            x_cm=region.agent_spawn_cm[0],
            y_cm=region.agent_spawn_cm[1],
            z_cm=region.agent_spawn_cm[2],
            yaw_deg=region.agent_yaw_deg,
        )
        quad = args.views == "quad"
        client = SpearTrackBClient(
            runtime,
            agent_tag=AGENT_TAG,
            spawn_pose=spawn_pose,
            capture_gate=capture_gate,
            turner=make_pawn_turner(session, AGENT_TAG) if quad else None,
            mover=make_pawn_mover(session, AGENT_TAG) if quad else None,
            views=PIXEL_VIEWS_QUAD if quad else PIXEL_VIEWS,
        )
        if pooled_profile:
            assert delivery_pool is not None and scenario is not None
            env_class = ValidatedPoolDeliveryEnv
        else:
            env_class = CorridorDeliveryEnv
        env_kwargs: dict[str, Any] = {}
        if pooled_profile:
            env_kwargs.update(
                delivery_pool=delivery_pool,
                scenario=scenario,
            )
        else:
            env_kwargs.update(
                pickup_offset_cm=args.pickup_offset_cm,
                dropoff_offset_cm=args.dropoff_offset_cm,
            )
        env = env_class(
            paris, client,
            street_camera=camera,
            episode_id=episode_id,
            cache_root=output_dir / "album",
            action_space=ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
            camera_view=CAMERA_VIEW_FRONT_REAR,
            pixel_views=PIXEL_VIEWS_QUAD if quad else PIXEL_VIEWS,
            embodiment="human_on_foot",
            difficulty="solo",
            seed=args.seed,
            spawn_z_cm=region.agent_spawn_cm[2],
            max_step_m=args.max_step_m,
            **env_kwargs,
        )
        env.reset()
        courier_session = CourierSession(
            env,
            city="Paris",
            frame_aliases=FrameAliases(output_dir / "frames"),
        )
        system_prompt = courier_session.system_prompt()
        model = AuditedModelClient(
            args.model_endpoint,
            args.model,
            max_tokens=args.max_tokens,
            max_requeries=args.max_requeries,
            max_tokens_ceiling=max(args.max_tokens * 4, 8192),
        )
        transcript = run_model_turns(
            courier_session,
            model,
            system_prompt=system_prompt,
            max_turns=args.max_turns,
            history_turns=args.history_turns,
            image_encoder=lambda frame, _index: frame_data_url(
                frame, image_root=output_dir / "frames"),
            logger=logger,
        )
        summary = env.summary()
        summary_latency = dict(summary.get("latency") or {})
        summary_latency.update({
            "model_inference": latency_statistics(
                turn["timing"]["model_latency_s"] for turn in transcript),
            "turn_wall": latency_statistics(
                turn["timing"]["turn_wall_latency_s"] for turn in transcript),
        })
        summary["latency"] = summary_latency
        report = {
            "experiment": (
                f"{experiment_model_label} front/rear validated-pool pixel-goal "
                "delivery (real SPEAR engine)"
                if pooled_profile
                else f"{experiment_model_label} front/rear pixel-goal delivery "
                "(real SPEAR engine)"
            ),
            "action_space": ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
            "camera_view": CAMERA_VIEW_FRONT_REAR,
            "pixel_views": list(env.pixel_views),
            "pedestrian_routing": bool(getattr(env, "pedestrian_routing", False)),
            "region": region.to_report(),
            "dropoff_tolerance_cm": CORRIDOR_DROPOFF_TOLERANCE_CM,
            "model": {
                "id": args.model,
                "endpoint": args.model_endpoint,
                "max_tokens": args.max_tokens,
                "max_images_per_prompt": args.model_image_capacity,
                "history_turns": args.history_turns,
                "system_prompt": system_prompt,
            },
            "setup_request": setup_request,
            "setup": preparation["setup"],
            "config": {
                "max_navmesh_adjustment_cm": config.max_navmesh_adjustment_cm,
                "acceptance_radius_cm": config.acceptance_radius_cm,
                "movement_timeout_sim_s": config.movement_timeout_sim_s,
                "wall_watchdog_timeout_s": config.execution_timeout_s,
                "timeout_policy": "ue_sim_time_with_host_watchdog_v1",
                "capture_readiness": client.capture_readiness_report,
                "capture_readiness_version": CAPTURE_READINESS_VERSION,
            },
            "turns": len(transcript),
            "transcript": transcript,
            "summary": summary,
            "termination": (
                courier_session.run.termination_reason
                or ("out_of_turns" if not courier_session.finished else "")
            ),
            "model_transport_stats": model.stats.as_dict(),
            "backend_events": list(client.events),
        }
        if pooled_profile:
            assert delivery_pool is not None and scenario is not None
            report.update({
                "pickup_tolerance_cm": VALIDATED_POOL_PICKUP_TOLERANCE_CM,
                "scenario": scenario_report(delivery_pool, scenario),
                "route": selected_route_report(paris, delivery_pool, scenario),
            })
        else:
            report.update({
                "pickup_offset_cm": args.pickup_offset_cm,
                "dropoff_offset_cm": args.dropoff_offset_cm,
            })
        validate_front_rear_delivery_report(
            report, expected_model_id=expected_model_id)
        _write_json_atomic(output_dir / "delivery_report.json", report)
        return report
    except Exception as error:
        failure = {
            "error": f"{type(error).__name__}: {error}",
            "transcript": transcript,
            "region": region.to_report(),
            "route_profile": args.route_profile,
        }
        if delivery_pool is not None and scenario is not None:
            failure["scenario"] = scenario_report(delivery_pool, scenario)
        if capture_gate is not None:
            failure["capture_readiness"] = capture_gate.report()
        _write_json_atomic(output_dir / "delivery_failure.json", failure)
        raise
    finally:
        if session is not None:
            _cleanup_session(
                session,
                launch_mode=args.launch_mode,
                shutdown_attached_editor=args.shutdown_attached_editor,
                play_started=play_started,
            )
        _copy_latest_ue_log(simworld_root, output_dir / "ue.log")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simworld-root", default=".simworld-ue")
    parser.add_argument(
        "--citycore-content",
        default=os.environ.get("CITYCORE_PARIS_CONTENT", ""),
    )
    parser.add_argument("--spear-config")
    parser.add_argument("--launch-mode", choices=("attach",), default="attach")
    parser.add_argument("--shutdown-attached-editor", action="store_true")
    parser.add_argument("--output", default="artifacts/pixel_goal_front_rear_delivery")
    parser.add_argument(
        "--model-endpoint",
        default="http://127.0.0.1:30001/v1/chat/completions",
    )
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--model-image-capacity", type=int, default=5)
    parser.add_argument(
        "--views", choices=("quad", "pair"), default="quad",
        help="quad: four photographs a turn (front, left, right, rear) and a "
             "walk along certified pedestrian ways to the point; pair: the "
             "original front/rear pair and the engine's straight-line walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-requeries", type=int, default=3)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--max-step-m", type=float, default=10.0)
    parser.add_argument(
        "--route-profile",
        choices=("validated_pool", "straight", "long_turn"),
        default="validated_pool",
        help=(
            "Use the certified entrance pool (default), the legacy straight "
            "corridor, or the deprecated long_turn alias for validated_pool."
        ),
    )
    parser.add_argument(
        "--order-pool", default=str(DEFAULT_VALIDATED_ORDER_POOL),
        help="Versioned UE-certified entrance/sidewalk pool JSON.",
    )
    parser.add_argument(
        "--list-order-stops", action="store_true",
        help="Print certified spawn/stop ids from --order-pool and exit.",
    )
    parser.add_argument(
        "--order-mode", choices=("random", "fixed"), default="random",
        help=(
            "Randomly sample a certified order with --seed, or resolve the "
            "explicit --pickup-id/--dropoff-id pair through the same checks."
        ),
    )
    parser.add_argument(
        "--spawn-id",
        help="Optional certified spawn id; omitted means any certified spawn.",
    )
    parser.add_argument(
        "--pickup-id",
        help="Certified pickup stop id (required with --order-mode fixed).",
    )
    parser.add_argument(
        "--dropoff-id",
        help="Certified drop-off stop id (required with --order-mode fixed).",
    )
    parser.add_argument(
        "--min-delivery-m", type=float, default=20.0,
        help="Minimum certified pickup-to-drop-off route length.",
    )
    parser.add_argument(
        "--max-delivery-m", type=float, default=150.0,
        help="Maximum certified pickup-to-drop-off route length.",
    )
    parser.add_argument(
        "--min-route-turns", type=int, default=0,
        help="Minimum number of >=30 degree turns on the delivery route.",
    )
    parser.add_argument(
        "--require-different-streets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require pickup and drop-off to have different street identities.",
    )
    parser.add_argument(
        "--require-marked-crossing", action="store_true",
        help="Require the delivery route to traverse a certified zebra crossing.",
    )
    parser.add_argument("--pickup-offset-cm", type=float, default=DEFAULT_PICKUP_OFFSET_CM)
    parser.add_argument("--dropoff-offset-cm", type=float, default=DEFAULT_DROPOFF_OFFSET_CM)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=360)
    parser.add_argument("--max-navmesh-adjustment-cm", type=float, default=20.0)
    parser.add_argument("--acceptance-radius-cm", type=float, default=15.0)
    parser.add_argument(
        "--movement-timeout-sim-s", type=float, default=45.0,
        help=(
            "Cancel a still-moving controller after this much UE world time; "
            "host load does not consume this budget."
        ),
    )
    parser.add_argument(
        "--wall-watchdog-timeout-s", "--execution-timeout-s",
        dest="wall_watchdog_timeout_s", type=float, default=300.0,
        help=(
            "Abort the rollout as an infrastructure failure if one movement "
            "call remains non-terminal for this much host time."
        ),
    )
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--navmesh-timeout-s", type=float, default=120.0)
    parser.add_argument("--capture-warmup-s", type=float, default=6.0)
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    if args.list_order_stops:
        pool = load_validated_delivery_pool(args.order_pool)
        print(json.dumps({
            "pool": pool.profile,
            "sha256": pool.sha256,
            "spawns": [
                {
                    "id": spawn.id,
                    "route_node_id": spawn.node_id,
                    "verified": spawn.verified,
                }
                for spawn in pool.spawns
            ],
            "stops": [
                {
                    "id": stop.id,
                    "address": stop.text,
                    "building_id": stop.building_id,
                    "roles": list(stop.roles),
                    "surface": stop.surface,
                }
                for stop in pool.stops
            ],
        }, indent=2, sort_keys=True))
        return 0
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model_label = AUDITED_MODEL_LABELS.get(args.model)
    if model_label is None:
        raise ValueError(
            f"front/rear delivery model is not audited: {args.model!r}")
    report = run_delivery(
        args,
        expected_model_id=args.model,
        experiment_model_label=model_label,
    )
    print(json.dumps({
        "turns": report["turns"],
        "termination": report["termination"],
        "delivered": report["summary"].get("delivered"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
