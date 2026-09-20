#!/usr/bin/env python3
"""Scripted live-UE mechanism probe for front/rear Pixel Goal navigation.

This runner has no model client or model endpoint.  It proves one rear-view
reversal and two independently reset, limited front-view steering moves, while
retaining the RPC events, ordered images, trajectory, report, and UE log.
"""

from __future__ import annotations

import argparse
import os
import base64
import copy
import csv
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalConfig,
    PixelGoalViewPair,
)
from embodiedbench.schemas.runtime import ControllerOutcomeCode, NavPixelGoalAction
from tools.run_pixel_goal_1b_closed_loop import _prepare_paris_loop, _write_json_atomic
from tools.run_pixel_goal_1b_poc import (
    AttachedParisGameSession,
    _reset_paris_poc_trial,
    _wait_for_paris_poc,
    _warm_up_paris_capture,
    build_paris_setup_request,
    validate_mount_inputs,
)
from tools.run_pixel_goal_full_delivery import AGENT_TAG, DELIVERY_REGION
from tools.pixel_goal_courier_backend import latency_statistics
from tools.run_pixel_goal_m1a import (
    SpearPixelGoalEndpoint,
    _cleanup_session,
    _copy_latest_ue_log,
    _save_frame,
)


BENCHMARK_KEYS = {
    "cycles", "width_px", "height_px", "fov_degrees", "order",
    "before_status", "after_status", "single_samples", "dual_samples",
    "statistics", "dual_over_single_ratio",
}
SINGLE_SAMPLE_KEYS = {
    "cycle", "sequence", "capture_timing", "client_wall_s",
    "camera_location_cm", "camera_rotation_degrees",
}
DUAL_SAMPLE_KEYS = {
    "cycle", "sequence", "capture_timing", "client_wall_s", "pose", "views",
}
VIEW_POSE_KEYS = {
    "view_id", "camera_location_cm", "camera_rotation_degrees",
}
SINGLE_TIMING_KEYS = {
    "total_ms", "lookup_ms", "init_ms", "capture_read_ms", "encode_ms",
}
DUAL_TIMING_KEYS = {"capture_read_ms", "encode_ms", "wall_ms"}
BENCHMARK_METRICS = (
    "capture_read_ms", "encode_ms", "ue_wall_ms", "client_wall_s",
)
STAT_KEYS = {"count", "median", "p95"}
RATIO_KEYS = {"median", "p95"}
IMAGE_EVIDENCE_KEYS = {
    "pair_index", "view_id", "artifact", "bytes_base64", "sha256",
    "byte_size", "width_px", "height_px", "luma",
}
LUMA_KEYS = {"min", "mean", "max"}


def build_front_rear_setup_request(
    region: Any,
    *,
    base_builder: Callable[[Any], dict[str, object]] = build_paris_setup_request,
) -> dict[str, object]:
    request = dict(base_builder(region))
    request["enable_rear_camera"] = True
    return request


def bearing_degrees(
    start_xy: tuple[float, float], end_xy: tuple[float, float],
) -> float:
    dx = float(end_xy[0]) - float(start_xy[0])
    dy = float(end_xy[1]) - float(start_xy[1])
    if not all(math.isfinite(value) for value in (dx, dy)) or math.hypot(dx, dy) == 0.0:
        raise ValueError("movement bearing needs a finite non-zero displacement")
    return math.degrees(math.atan2(dy, dx)) % 360.0


def signed_bearing_delta_degrees(start_yaw_deg: float, bearing_deg: float) -> float:
    if not all(math.isfinite(float(value))
               for value in (start_yaw_deg, bearing_deg)):
        raise ValueError("bearings must be finite")
    return ((float(bearing_deg) - float(start_yaw_deg) + 180.0) % 360.0) - 180.0


def angular_error_degrees(first: float, second: float) -> float:
    return abs(signed_bearing_delta_degrees(first, second))


def planar_target_dot(
    start_pose: tuple[float, float, float, float],
    target_xy: tuple[float, float],
) -> float:
    if len(start_pose) != 4 or len(target_xy) != 2:
        raise ValueError("dot product needs a four-value pose and planar target")
    values = tuple(float(value) for value in (*start_pose, *target_xy))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dot product inputs must be finite")
    radians = math.radians(values[3])
    displacement = (values[4] - values[0], values[5] - values[1])
    return displacement[0] * math.cos(radians) + displacement[1] * math.sin(radians)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _pose(value: Any, label: str) -> dict[str, float]:
    expected = {"x_cm", "y_cm", "z_cm", "yaw_deg"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} pose is incomplete")
    return {key: _number(value[key], f"{label}.{key}") for key in expected}


def _close(first: float, second: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} does not match the exact probe contract")
    return value


def _numbers(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} is incomplete")
    return [_number(item, f"{label}[{index}]")
            for index, item in enumerate(value)]


def _nonnegative(value: Any, label: str) -> float:
    number = _number(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _timing_values(sample: Mapping[str, Any], condition: str) -> dict[str, float]:
    timing_keys = SINGLE_TIMING_KEYS if condition == "single" else DUAL_TIMING_KEYS
    timing = _exact_keys(
        sample.get("capture_timing"), timing_keys,
        f"{condition} capture timing",
    )
    parsed = {
        key: _nonnegative(value, f"{condition} {key}")
        for key, value in timing.items()
    }
    return {
        "capture_read_ms": parsed["capture_read_ms"],
        "encode_ms": parsed["encode_ms"],
        "ue_wall_ms": parsed[
            "total_ms" if condition == "single" else "wall_ms"],
        "client_wall_s": _nonnegative(
            sample.get("client_wall_s"), f"{condition} client wall"),
    }


def _benchmark_statistics(
    samples: list[Mapping[str, Any]], condition: str,
) -> dict[str, dict[str, int | float]]:
    values = [_timing_values(sample, condition) for sample in samples]
    result: dict[str, dict[str, int | float]] = {}
    for metric in BENCHMARK_METRICS:
        stats = latency_statistics(row[metric] for row in values)
        result[metric] = {
            "count": stats["count"],
            "median": _nonnegative(
                stats["median_s"], f"{condition} {metric} median"),
            "p95": _nonnegative(
                stats["p95_s"], f"{condition} {metric} p95"),
        }
    return result


def _ratio(numerator: Any, denominator: Any, label: str) -> float:
    top = _nonnegative(numerator, f"{label} numerator")
    bottom = _nonnegative(denominator, f"{label} denominator")
    if bottom == 0.0:
        raise ValueError(f"{label} has a zero denominator")
    ratio = top / bottom
    if not math.isfinite(ratio):
        raise ValueError(f"{label} ratio must be finite")
    return ratio


def _benchmark_ratios(
    statistics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            key: _ratio(
                statistics["dual"][metric][key],
                statistics["single"][metric][key],
                f"{metric} {key}",
            )
            for key in ("median", "p95")
        }
        for metric in BENCHMARK_METRICS
    }


def _single_capture_evidence(
    endpoint: Any, frame: Any,
) -> tuple[dict[str, float], list[float], list[float]]:
    calls = getattr(endpoint, "calls", None)
    if not isinstance(calls, list) or not calls \
            or calls[-1][0] != "PixelGoal_CaptureFrameJson":
        raise RuntimeError("single capture omitted its exact endpoint response")
    response = calls[-1][2]
    if not isinstance(response, Mapping):
        raise RuntimeError("single capture response is not an object")
    if (response.get("camera_snapshot_id") != frame.camera_snapshot_id
            or response.get("camera_intrinsics_id")
            != frame.camera_intrinsics_id):
        raise RuntimeError("single capture response does not match its frame")
    timing = _exact_keys(
        response.get("capture_timing"), SINGLE_TIMING_KEYS,
        "single capture timing",
    )
    parsed_timing = {
        key: _nonnegative(value, f"single {key}")
        for key, value in timing.items()
    }
    return (
        parsed_timing,
        _numbers(
            response.get("camera_location_cm"), length=3,
            label="single camera location"),
        _numbers(
            response.get("camera_rotation_degrees"), length=3,
            label="single camera rotation"),
    )


def _dual_capture_evidence(
    endpoint: Any, pair: Any,
) -> tuple[list[float], list[dict[str, Any]]]:
    calls = getattr(endpoint, "calls", None)
    if not isinstance(calls, list) or not calls \
            or calls[-1][0] != "PixelGoal_CaptureViewPairJson":
        raise RuntimeError("dual capture omitted its exact endpoint response")
    response = calls[-1][2]
    if not isinstance(response, Mapping):
        raise RuntimeError("dual capture response is not an object")
    if response.get("capture_group_id") != pair.capture_group_id:
        raise RuntimeError("dual capture response does not match its pair")
    raw_pose = response.get("pose")
    if not isinstance(raw_pose, Mapping):
        raise RuntimeError("dual capture response omitted its actor pose")
    pose = [
        _number(raw_pose.get(key), f"dual actor pose.{key}")
        for key in ("x_cm", "y_cm", "z_cm", "yaw_deg")
    ]
    if tuple(pose) != tuple(pair.pose):
        raise RuntimeError("dual capture response actor pose does not match its pair")
    raw_views = response.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != 2:
        raise RuntimeError("dual capture response omitted front/rear pose evidence")
    view_poses: list[dict[str, Any]] = []
    for expected_view, frame, raw_view in zip(
            ("front", "rear"), (pair.front, pair.rear), raw_views):
        if not isinstance(raw_view, Mapping) \
                or raw_view.get("view_id") != expected_view \
                or raw_view.get("camera_snapshot_id") != frame.camera_snapshot_id:
            raise RuntimeError("dual capture response view identity is not exact")
        view_poses.append({
            "view_id": expected_view,
            "camera_location_cm": _numbers(
                raw_view.get("camera_location_cm"), length=3,
                label=f"dual {expected_view} camera location"),
            "camera_rotation_degrees": _numbers(
                raw_view.get("camera_rotation_degrees"), length=3,
                label=f"dual {expected_view} camera rotation"),
        })
    return pose, view_poses


def run_capture_latency_benchmark(
    runtime: Any,
    endpoint: Any,
    *,
    status_fn: Callable[[], Mapping[str, Any]],
    cycles: int = 20,
    agent_tag: str = AGENT_TAG,
    width_px: int = 640,
    height_px: int = 360,
    fov_degrees: float = 90.0,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Alternately measure one front capture and one same-pose view pair."""

    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles != 20:
        raise ValueError("capture latency benchmark requires exactly 20 cycles")
    if (width_px, height_px) != (640, 360):
        raise ValueError("capture latency benchmark requires 640x360")
    configured_fov = _number(
        getattr(runtime.config, "fov_degrees", None), "configured FOV")
    requested_fov = _number(fov_degrees, "configured FOV")
    if not _close(requested_fov, configured_fov):
        raise ValueError("capture latency benchmark must use the configured FOV")
    if not isinstance(agent_tag, str) or not agent_tag:
        raise ValueError("capture latency benchmark needs an agent tag")

    before_status = copy.deepcopy(dict(status_fn()))
    single_samples: list[dict[str, Any]] = []
    dual_samples: list[dict[str, Any]] = []
    order: list[str] = []
    for cycle in range(1, cycles + 1):
        sequence = (cycle - 1) * 2
        started = perf_counter_fn()
        frame = runtime.capture_frame(
            agent_tag=agent_tag, width_px=width_px, height_px=height_px)
        single_wall_s = perf_counter_fn() - started
        single_timing, single_location, single_rotation = \
            _single_capture_evidence(endpoint, frame)
        single_samples.append({
            "cycle": cycle,
            "sequence": sequence,
            "capture_timing": single_timing,
            "client_wall_s": _nonnegative(
                single_wall_s, "single client wall"),
            "camera_location_cm": single_location,
            "camera_rotation_degrees": single_rotation,
        })
        order.append("single")

        started = perf_counter_fn()
        pair = runtime.capture_view_pair(
            agent_tag=agent_tag,
            width_px=width_px,
            height_px=height_px,
            fov_degrees=requested_fov,
        )
        dual_wall_s = perf_counter_fn() - started
        dual_timing = _exact_keys(
            pair.capture_timing, DUAL_TIMING_KEYS, "dual capture timing")
        dual_pose, dual_views = _dual_capture_evidence(endpoint, pair)
        dual_samples.append({
            "cycle": cycle,
            "sequence": sequence + 1,
            "pose": dual_pose,
            "views": dual_views,
            "capture_timing": {
                key: _nonnegative(value, f"dual {key}")
                for key, value in dual_timing.items()
            },
            "client_wall_s": _nonnegative(dual_wall_s, "dual client wall"),
        })
        order.append("dual")
    after_status = copy.deepcopy(dict(status_fn()))
    statistics = {
        "single": _benchmark_statistics(single_samples, "single"),
        "dual": _benchmark_statistics(dual_samples, "dual"),
    }
    benchmark = {
        "cycles": cycles,
        "width_px": width_px,
        "height_px": height_px,
        "fov_degrees": requested_fov,
        "order": order,
        "before_status": before_status,
        "after_status": after_status,
        "single_samples": single_samples,
        "dual_samples": dual_samples,
        "statistics": statistics,
        "dual_over_single_ratio": _benchmark_ratios(statistics),
    }
    validate_capture_latency_benchmark(benchmark)
    return benchmark


def _status_position(status: Any, label: str) -> list[float]:
    if not isinstance(status, Mapping) or status.get("poc_ready") is not True:
        raise ValueError(f"{label} benchmark status is not ready")
    position = status.get("agent_position_cm")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"{label} benchmark status has no agent position")
    return [_number(value, f"{label} benchmark position") for value in position]


def _same_values(first: list[float], second: list[float]) -> bool:
    return first == second


def validate_capture_latency_benchmark(benchmark: Any) -> None:
    root = _exact_keys(benchmark, BENCHMARK_KEYS, "capture latency benchmark")
    cycles = root["cycles"]
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles != 20:
        raise ValueError("capture latency benchmark requires exactly 20 cycles")
    if (root["width_px"], root["height_px"]) != (640, 360):
        raise ValueError("capture latency benchmark must be 640x360")
    if not _close(
            _number(root["fov_degrees"], "configured FOV"),
            PixelGoalConfig().fov_degrees):
        raise ValueError("capture latency benchmark changed the configured FOV")
    expected_order = [condition for _cycle in range(cycles)
                      for condition in ("single", "dual")]
    if root["order"] != expected_order:
        raise ValueError("capture latency benchmark is not exactly alternating")
    single_samples = root["single_samples"]
    dual_samples = root["dual_samples"]
    if not isinstance(single_samples, list) or len(single_samples) != cycles:
        raise ValueError("capture latency benchmark needs 20 single samples")
    if not isinstance(dual_samples, list) or len(dual_samples) != cycles:
        raise ValueError("capture latency benchmark needs 20 dual samples")

    validated_single: list[Mapping[str, Any]] = []
    validated_dual: list[Mapping[str, Any]] = []
    single_camera_poses: list[tuple[list[float], list[float]]] = []
    dual_actor_poses: list[list[float]] = []
    dual_camera_poses: list[list[dict[str, Any]]] = []
    for index, raw_sample in enumerate(single_samples):
        sample = _exact_keys(raw_sample, SINGLE_SAMPLE_KEYS, "single sample")
        if (sample["cycle"], sample["sequence"]) != (index + 1, index * 2):
            raise ValueError("single sample sequence is not exact")
        _timing_values(sample, "single")
        single_camera_poses.append((
            _numbers(
                sample["camera_location_cm"], length=3,
                label="single sample camera location"),
            _numbers(
                sample["camera_rotation_degrees"], length=3,
                label="single sample camera rotation"),
        ))
        validated_single.append(sample)
    for index, raw_sample in enumerate(dual_samples):
        sample = _exact_keys(raw_sample, DUAL_SAMPLE_KEYS, "dual sample")
        if (sample["cycle"], sample["sequence"]) != (index + 1, index * 2 + 1):
            raise ValueError("dual sample sequence is not exact")
        pose = _numbers(sample["pose"], length=4, label="dual sample pose")
        raw_views = sample["views"]
        if not isinstance(raw_views, list) or len(raw_views) != 2:
            raise ValueError("dual sample needs exact front/rear camera poses")
        parsed_views: list[dict[str, Any]] = []
        for expected_view, raw_view in zip(("front", "rear"), raw_views):
            view = _exact_keys(
                raw_view, VIEW_POSE_KEYS, f"dual {expected_view} camera pose")
            if view["view_id"] != expected_view:
                raise ValueError("dual sample camera poses are not front, rear")
            parsed_views.append({
                "view_id": expected_view,
                "camera_location_cm": _numbers(
                    view["camera_location_cm"], length=3,
                    label=f"dual {expected_view} camera location"),
                "camera_rotation_degrees": _numbers(
                    view["camera_rotation_degrees"], length=3,
                    label=f"dual {expected_view} camera rotation"),
            })
        _timing_values(sample, "dual")
        dual_actor_poses.append(pose)
        dual_camera_poses.append(parsed_views)
        validated_dual.append(sample)

    spawn_pose = dual_actor_poses[0]
    before = _status_position(root["before_status"], "before")
    after = _status_position(root["after_status"], "after")
    if (not _same_values(before, spawn_pose[:3])
            or not _same_values(after, spawn_pose[:3])
            or any(not _same_values(pose, spawn_pose)
                   for pose in dual_actor_poses)):
        raise ValueError("latency benchmark captures did not stay at the same spawn")

    reference_location, reference_rotation = single_camera_poses[0]
    if any(
            not _same_values(location, reference_location)
            or not _same_values(rotation, reference_rotation)
            for location, rotation in single_camera_poses):
        raise ValueError(
            "all single samples must retain the same fixed camera pose")
    for actor_pose, views in zip(dual_actor_poses, dual_camera_poses):
        front, rear = views
        front_location = front["camera_location_cm"]
        rear_location = rear["camera_location_cm"]
        if (not _same_values(front_location, rear_location)
                or not _same_values(front_location, reference_location)):
            raise ValueError(
                "every single/front/rear sample must use the same optical location")
        front_rotation = front["camera_rotation_degrees"]
        rear_rotation = rear["camera_rotation_degrees"]
        if not _same_values(front_rotation, reference_rotation):
            raise ValueError(
                "all single samples must retain the same fixed camera pose")
        if (front_rotation[0] != rear_rotation[0]
                or front_rotation[2] != rear_rotation[2]
                or angular_error_degrees(
                    front_rotation[1], rear_rotation[1]) != 180.0):
            raise ValueError(
                "every dual sample must retain the expected front/rear yaw relationship")
        if angular_error_degrees(actor_pose[3], front_rotation[1]) != 0.0:
            raise ValueError(
                "every dual front camera yaw must agree with the fixed actor pose")

    actual_statistics = _exact_keys(
        root["statistics"], {"single", "dual"}, "benchmark statistics")
    expected_statistics = {
        "single": _benchmark_statistics(validated_single, "single"),
        "dual": _benchmark_statistics(validated_dual, "dual"),
    }
    expected_ratios = _benchmark_ratios(expected_statistics)
    for condition in ("single", "dual"):
        condition_stats = _exact_keys(
            actual_statistics[condition], set(BENCHMARK_METRICS),
            f"{condition} statistics")
        for metric in BENCHMARK_METRICS:
            actual = _exact_keys(
                condition_stats[metric], STAT_KEYS,
                f"{condition} {metric} statistics")
            expected = expected_statistics[condition][metric]
            if (isinstance(actual["count"], bool)
                    or not isinstance(actual["count"], int)
                    or actual["count"] != expected["count"]):
                raise ValueError("benchmark statistics count is not exact")
            for key in ("median", "p95"):
                value = _nonnegative(
                    actual[key], f"benchmark statistics {condition} {metric} {key}")
                expected_value = float(expected[key])
                if not _close(value, expected_value, tolerance=1e-12):
                    raise ValueError("benchmark statistics do not match raw samples")

    actual_ratios = _exact_keys(
        root["dual_over_single_ratio"], set(BENCHMARK_METRICS),
        "benchmark ratio")
    for metric in BENCHMARK_METRICS:
        ratio = _exact_keys(actual_ratios[metric], RATIO_KEYS, "benchmark ratio")
        for key in RATIO_KEYS:
            actual = _nonnegative(
                ratio[key], f"benchmark ratio {metric} {key}")
            expected = expected_ratios[metric][key]
            if not _close(actual, expected, tolerance=1e-12):
                raise ValueError("benchmark ratio does not match raw statistics")


def _image_record(
    data: bytes, *, pair_index: int, view_id: str, artifact: str,
) -> dict[str, Any]:
    try:
        with Image.open(BytesIO(data)) as opened:
            image = opened.convert("RGB")
            image.load()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError("probe image evidence could not decode image bytes") from error
    luma_values = list(image.convert("L").tobytes())
    return {
        "pair_index": pair_index,
        "view_id": view_id,
        "artifact": artifact,
        "bytes_base64": base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_size": len(data),
        "width_px": image.width,
        "height_px": image.height,
        "luma": {
            "min": min(luma_values),
            "mean": sum(luma_values) / len(luma_values),
            "max": max(luma_values),
        },
    }


def analyze_probe_images(
    output_dir: Path,
    image_names: list[str],
    *,
    width_px: int = 640,
    height_px: int = 360,
) -> list[dict[str, Any]]:
    expected = _ordered_probe_image_names()
    if image_names != expected:
        raise ValueError("probe image evidence needs 12 positionally ordered images")
    return _analyze_ordered_probe_images(
        output_dir,
        image_names,
        width_px=width_px,
        height_px=height_px,
    )


def _ordered_probe_image_names() -> list[str]:
    return [
        f"pair-{pair_index:02d}-{view_index:02d}.png"
        for pair_index in range(6) for view_index in range(2)
    ]


def _validate_partial_probe_image_names(image_names: Any) -> list[str]:
    if not isinstance(image_names, list):
        raise ValueError("partial probe images must be an ordered prefix")
    if len(image_names) > 12:
        raise ValueError("partial probe image prefix contains at most 12 images")
    if len(image_names) % 2:
        raise ValueError("partial probe images must be an even-length prefix")
    expected = _ordered_probe_image_names()[:len(image_names)]
    if image_names != expected:
        raise ValueError("partial probe images must be the exact ordered prefix")
    return expected


def _analyze_ordered_probe_images(
    output_dir: Path,
    image_names: list[str],
    *,
    width_px: int,
    height_px: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, artifact in enumerate(image_names):
        pair_index, view_index = divmod(index, 2)
        path = output_dir / artifact
        if not path.is_file():
            raise ValueError(f"probe image artifact is missing: {artifact}")
        record = _image_record(
            path.read_bytes(),
            pair_index=pair_index,
            view_id=("front", "rear")[view_index],
            artifact=artifact,
        )
        if (record["width_px"], record["height_px"]) != (width_px, height_px):
            raise ValueError("probe image artifact dimensions are wrong")
        records.append(record)
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("probe image evidence contains duplicate image bytes")
    return records


def analyze_partial_probe_images(
    output_dir: Path,
    image_names: list[str],
    *,
    width_px: int = 640,
    height_px: int = 360,
) -> list[dict[str, Any]]:
    """Analyze only a complete ordered prefix of the six probe image pairs."""

    _validate_partial_probe_image_names(image_names)
    return _analyze_ordered_probe_images(
        output_dir,
        image_names,
        width_px=width_px,
        height_px=height_px,
    )


def validate_probe_image_evidence(
    evidence: Any,
    *,
    image_names: Any,
    width_px: int,
    height_px: int,
) -> None:
    if not isinstance(evidence, list) or len(evidence) != 12:
        raise ValueError("probe report needs 12 image evidence records")
    if not isinstance(image_names, list) or len(image_names) != 12:
        raise ValueError("probe report needs 12 image artifact names")
    if image_names != _ordered_probe_image_names():
        raise ValueError("probe image evidence position is not exact")
    _validate_probe_image_evidence_records(
        evidence,
        image_names=image_names,
        width_px=width_px,
        height_px=height_px,
    )


def validate_partial_probe_image_evidence(
    evidence: Any,
    *,
    image_names: Any,
    width_px: int,
    height_px: int,
) -> None:
    expected_names = _validate_partial_probe_image_names(image_names)
    if not isinstance(evidence, list) or len(evidence) != len(expected_names):
        raise ValueError(
            "partial probe image evidence must match its ordered image prefix")
    _validate_probe_image_evidence_records(
        evidence,
        image_names=expected_names,
        width_px=width_px,
        height_px=height_px,
    )


def _validate_probe_image_evidence_records(
    evidence: list[Any],
    *,
    image_names: list[str],
    width_px: int,
    height_px: int,
) -> None:
    hashes: list[str] = []
    for index, raw_record in enumerate(evidence):
        record = _exact_keys(raw_record, IMAGE_EVIDENCE_KEYS, "image evidence")
        pair_index, view_index = divmod(index, 2)
        expected_artifact = f"pair-{pair_index:02d}-{view_index:02d}.png"
        if (record["pair_index"] != pair_index
                or record["view_id"] != ("front", "rear")[view_index]
                or record["artifact"] != expected_artifact
                or image_names[index] != expected_artifact):
            raise ValueError("probe image evidence position is not exact")
        encoded = record["bytes_base64"]
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("probe image evidence has no retained bytes")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError("probe image evidence base64 is invalid") from error
        if record["sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError("probe image evidence SHA-256 does not match bytes")
        if record["byte_size"] != len(data) or not data:
            raise ValueError("probe image evidence byte size does not match bytes")
        recomputed = _image_record(
            data,
            pair_index=pair_index,
            view_id=("front", "rear")[view_index],
            artifact=expected_artifact,
        )
        if (record["width_px"], record["height_px"]) != (width_px, height_px) \
                or (recomputed["width_px"], recomputed["height_px"]) \
                != (width_px, height_px):
            raise ValueError("probe image evidence dimensions are wrong")
        luma = _exact_keys(record["luma"], LUMA_KEYS, "image luma")
        expected_luma = recomputed["luma"]
        for key in LUMA_KEYS:
            if not _close(
                    _number(luma[key], "image luma"),
                    float(expected_luma[key]), tolerance=1e-12):
                raise ValueError("probe image luma does not match retained bytes")
        if _number(luma["max"], "image luma") <= 0.0:
            raise ValueError("probe image evidence is black")
        hashes.append(str(record["sha256"]))
    if len(set(hashes)) != len(hashes):
        raise ValueError("probe image evidence contains duplicate bytes")


def validate_front_rear_probe_report(report: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("probe report must be JSON serializable") from error
    if report.get("action_space") != "pixel_goal_front_rear":
        raise ValueError("probe action_space is not pixel_goal_front_rear")
    if report.get("camera_view") != "front_rear":
        raise ValueError("probe camera_view is not front_rear")
    setup = report.get("setup_request")
    if not isinstance(setup, Mapping) or setup.get("enable_rear_camera") is not True:
        raise ValueError("probe setup must enable the rear camera")
    setup_response = report.get("setup")
    if (not isinstance(setup_response, Mapping)
            or setup_response.get("success") is not True
            or setup_response.get("rear_camera_enabled") is not True):
        raise ValueError("probe setup response did not enable the rear camera")
    benchmark = report.get("latency_benchmark")
    validate_capture_latency_benchmark(benchmark)
    reversal = report.get("rear_reversal")
    if not isinstance(reversal, Mapping) or reversal.get("selected_view") != "rear":
        raise ValueError("probe did not select the rear view")
    pixel = reversal.get("pixel_uv")
    if (not isinstance(pixel, list) or len(pixel) != 2
            or not all(0.0 <= _number(value, "rear pixel") <= 1.0
                       for value in pixel)):
        raise ValueError("rear probe pixel is invalid")
    for key in ("capture_group_id", "camera_snapshot_id"):
        if not isinstance(reversal.get(key), str) or not reversal[key]:
            raise ValueError(f"rear probe is missing {key}")
    start = _pose(reversal.get("start_pose"), "rear start")
    end = _pose(reversal.get("end_pose"), "rear end")
    accepted_target = reversal.get("accepted_target_cm")
    if (not isinstance(accepted_target, list) or len(accepted_target) != 3
            or any(not math.isfinite(_number(value, "rear accepted target"))
                   for value in accepted_target)):
        raise ValueError("rear accepted target is incomplete")
    derived_dot = planar_target_dot(
        (start["x_cm"], start["y_cm"], start["z_cm"], start["yaw_deg"]),
        (float(accepted_target[0]), float(accepted_target[1])),
    )
    stored_dot = _number(
        reversal.get("target_dot_body_forward_cm"), "rear target dot")
    if not _close(stored_dot, derived_dot):
        raise ValueError("derived target dot does not match the stored rear target dot")
    if derived_dot >= 0.0:
        raise ValueError("rear target is not behind the initial body-forward vector")
    if reversal.get("controller_outcome") != ControllerOutcomeCode.ACCEPTED.value:
        raise ValueError("rear movement did not reach terminal success")
    derived_displacement = math.dist(
        (start["x_cm"], start["y_cm"]), (end["x_cm"], end["y_cm"]))
    stored_displacement = _number(
        reversal.get("displacement_cm"), "rear displacement")
    if not _close(stored_displacement, derived_displacement):
        raise ValueError("derived rear displacement does not match")
    if derived_displacement <= 0.0:
        raise ValueError("rear movement did not produce non-zero displacement")
    derived_bearing = bearing_degrees(
        (start["x_cm"], start["y_cm"]), (end["x_cm"], end["y_cm"]))
    stored_bearing = _number(
        reversal.get("movement_bearing_deg"), "movement bearing")
    if not _close(stored_bearing, derived_bearing):
        raise ValueError("derived movement bearing does not match the rear record")
    post_yaw = _number(reversal.get("post_front_yaw_deg"), "post front yaw")
    derived_alignment = angular_error_degrees(derived_bearing, post_yaw)
    stored_alignment = _number(
        reversal.get("alignment_error_deg"), "front alignment")
    if not _close(stored_alignment, derived_alignment):
        raise ValueError("derived alignment does not match the rear record")
    if derived_alignment > 25.0:
        raise ValueError("post-move front yaw is not within 25 degrees")
    reversal_reset = reversal.get("trial_reset")
    if not isinstance(reversal_reset, Mapping) \
            or reversal_reset.get("success") is not True:
        raise ValueError("rear probe is missing its independent reset")

    steering = report.get("steering")
    if not isinstance(steering, Mapping) or set(steering) != {
            "front_left", "front_right"}:
        raise ValueError("steering report needs separate front-left and front-right probes")
    deltas: list[float] = []
    reset_starts: list[dict[str, float]] = []
    for name, expected_side in (("front_left", "left"), ("front_right", "right")):
        probe = steering[name]
        if not isinstance(probe, Mapping) or probe.get("selected_view") != "front":
            raise ValueError(f"{name} did not select front")
        uv = probe.get("pixel_uv")
        if not isinstance(uv, list) or len(uv) != 2:
            raise ValueError(f"{name} pixel is invalid")
        u = _number(uv[0], f"{name} u")
        _number(uv[1], f"{name} v")
        if (expected_side == "left" and not u < 0.5) \
                or (expected_side == "right" and not u > 0.5):
            raise ValueError(f"{name} pixel is on the wrong side")
        reset = probe.get("trial_reset")
        if not isinstance(reset, Mapping) or reset.get("success") is not True:
            raise ValueError(f"{name} is missing its independent reset")
        if probe.get("controller_outcome") != ControllerOutcomeCode.ACCEPTED.value:
            raise ValueError(f"{name} steering terminal success is missing")
        if _number(probe.get("displacement_cm"), f"{name} displacement") <= 0.0:
            raise ValueError(f"{name} steering displacement is not positive")
        probe_start = _pose(probe.get("start_pose"), f"{name} start")
        probe_end = _pose(probe.get("end_pose"), f"{name} end")
        reset_starts.append(probe_start)
        derived_probe_displacement = math.dist(
            (probe_start["x_cm"], probe_start["y_cm"]),
            (probe_end["x_cm"], probe_end["y_cm"]),
        )
        if not _close(
                _number(probe.get("displacement_cm"), f"{name} displacement"),
                derived_probe_displacement):
            raise ValueError(f"derived steering displacement does not match {name}")
        derived_probe_bearing = bearing_degrees(
            (probe_start["x_cm"], probe_start["y_cm"]),
            (probe_end["x_cm"], probe_end["y_cm"]),
        )
        if not _close(
                _number(probe.get("movement_bearing_deg"), f"{name} bearing"),
                derived_probe_bearing):
            raise ValueError(f"derived steering bearing does not match {name}")
        derived_delta = signed_bearing_delta_degrees(
            probe_start["yaw_deg"], derived_probe_bearing)
        delta = _number(probe.get("signed_bearing_delta_deg"), f"{name} delta")
        if not _close(delta, derived_delta):
            raise ValueError(f"derived steering delta does not match {name}")
        if not 0.0 < abs(delta) < 90.0:
            raise ValueError(f"{name} is not limited steering")
        deltas.append(delta)
    if not deltas[0] < 0.0 < deltas[1]:
        raise ValueError("steering must have left-negative/right-positive deltas")
    for reset_start in reset_starts:
        if any(not _close(reset_start[key], start[key]) for key in start):
            raise ValueError("independent steering resets did not return to the same spawn")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("probe artifacts are missing")
    images = artifacts.get("images")
    expected_images = [
        f"pair-{pair_index:02d}-{view_index:02d}.png"
        for pair_index in range(6) for view_index in range(2)
    ]
    if images != expected_images or len(set(images)) != len(expected_images):
        raise ValueError("probe needs six ordered pairs of unique image artifacts")
    validate_probe_image_evidence(
        report.get("image_evidence"),
        image_names=images,
        width_px=int(benchmark["width_px"]),
        height_px=int(benchmark["height_px"]),
    )
    for key in ("events", "trajectory", "ue_log"):
        if not isinstance(artifacts.get(key), str) or not artifacts[key]:
            raise ValueError(f"probe artifact {key} is missing")
    return {"passed": True, "rear_reversal": True, "limited_steering": True}


def parse_selection(text: str) -> tuple[str, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("selection must be view,u,v")
    view = parts[0]
    if view not in ("front", "rear"):
        raise ValueError("selection view must be front or rear")
    try:
        u, v = float(parts[1]), float(parts[2])
    except ValueError as error:
        raise ValueError("selection u and v must be numbers") from error
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (u, v)):
        raise ValueError("selection u and v must be finite values in [0, 1]")
    return view, u, v


def _save_pair(
    pair: PixelGoalViewPair, output_dir: Path, pair_index: int,
) -> list[str]:
    paths: list[str] = []
    for view_index, frame in enumerate((pair.front, pair.rear)):
        path = output_dir / f"pair-{pair_index:02d}-{view_index:02d}.png"
        _save_frame(frame.rgb_data_url, path)
        paths.append(path.name)
    return paths


def _vec3(value: Any) -> list[float]:
    return [float(value.x_cm), float(value.y_cm), float(value.z_cm)]


def _execute(
    runtime: LivePixelGoalRuntime,
    pair: PixelGoalViewPair,
    selection: tuple[str, float, float],
) -> tuple[Any, Any, dict[str, Any]]:
    view, u, v = selection
    frame = pair.frame(view)
    started, result = runtime.execute(
        NavPixelGoalAction(target={"u_norm": u, "v_norm": v}), frame)
    target = started.request.projected_target
    if target is None or result.final_pose is None:
        raise RuntimeError("probe action omitted its validated target or final pose")
    start_xy = (pair.pose[0], pair.pose[1])
    end_xy = (result.final_pose.position.x_cm, result.final_pose.position.y_cm)
    displacement_cm = math.dist(start_xy, end_xy)
    movement_bearing = bearing_degrees(start_xy, end_xy)
    record = {
        "selected_view": view,
        "pixel_uv": [u, v],
        "capture_group_id": pair.capture_group_id,
        "camera_snapshot_id": frame.camera_snapshot_id,
        "start_pose": {
            "x_cm": pair.pose[0], "y_cm": pair.pose[1],
            "z_cm": pair.pose[2], "yaw_deg": pair.pose[3],
        },
        "accepted_target_cm": _vec3(target),
        "controller_outcome": result.outcome.value,
        "controller_result": result.controller_result,
        "end_pose": {
            "x_cm": result.final_pose.position.x_cm,
            "y_cm": result.final_pose.position.y_cm,
            "z_cm": result.final_pose.position.z_cm,
            "yaw_deg": result.final_pose.yaw_deg,
        },
        "displacement_cm": displacement_cm,
        "distance_travelled_cm": result.distance_travelled_cm,
        "movement_bearing_deg": movement_bearing,
        "elapsed_sim_s": result.elapsed_sim_s,
    }
    return started, result, record


def _write_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name", "view", "u", "v", "start_x_cm", "start_y_cm", "end_x_cm",
        "end_y_cm", "target_x_cm", "target_y_cm", "displacement_cm",
        "movement_bearing_deg", "controller_outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_row(name: str, action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "view": action["selected_view"],
        "u": action["pixel_uv"][0],
        "v": action["pixel_uv"][1],
        "start_x_cm": action["start_pose"]["x_cm"],
        "start_y_cm": action["start_pose"]["y_cm"],
        "end_x_cm": action["end_pose"]["x_cm"],
        "end_y_cm": action["end_pose"]["y_cm"],
        "target_x_cm": action["accepted_target_cm"][0],
        "target_y_cm": action["accepted_target_cm"][1],
        "displacement_cm": action["displacement_cm"],
        "movement_bearing_deg": action["movement_bearing_deg"],
        "controller_outcome": action["controller_outcome"],
    }


def _attempted_action_record(
    trial_name: str,
    selection: tuple[str, float, float],
    pair: PixelGoalViewPair | None = None,
) -> dict[str, Any]:
    view, u, v = selection
    frame = None if pair is None else (
        pair.front if view == "front" else pair.rear)
    return {
        "trial_name": trial_name,
        "selected_view": view,
        "pixel_uv": [u, v],
        "capture_group_id": None if pair is None else pair.capture_group_id,
        "camera_snapshot_id": (
            None if frame is None else frame.camera_snapshot_id),
    }


def _prepare_output_directory(output_dir: Path) -> None:
    """Create a fresh output or accept only a launcher-created empty one."""

    if output_dir.exists():
        if not output_dir.is_dir() or next(output_dir.iterdir(), None) is not None:
            raise FileExistsError(
                f"probe output directory is non-empty: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=False)


def _exception_traceback(error: BaseException) -> Any:
    return BaseException.__getattribute__(error, "__traceback__")


def _restore_exception_traceback(error: BaseException, traceback: Any) -> None:
    BaseException.__setattr__(error, "__traceback__", traceback)


def _exception_type_name(error: BaseException) -> str:
    try:
        name = type.__getattribute__(type(error), "__name__")
    except BaseException:
        return "BaseException"
    if not isinstance(name, str):
        return "BaseException"
    try:
        exact_name = str.__str__(name)
    except BaseException:
        return "BaseException"
    return exact_name if exact_name else "BaseException"


def _typed_error(error: BaseException) -> str:
    try:
        error_type = _exception_type_name(error)
        try:
            error_text = str.__str__(str(error))
        except BaseException as formatting_error:
            formatting_type = _exception_type_name(formatting_error)
            return (
                error_type
                + ": <exception text unavailable: "
                + formatting_type
                + ">"
            )
        return error_type + ": " + error_text
    except BaseException:
        return "BaseException: <exception text unavailable: BaseException>"


def _log_best_effort(message: str, *args: Any) -> None:
    try:
        logging.error(message, *args)
    except BaseException:
        pass


def _probe_finalizer_actions(
    session: Any,
    *,
    simworld_root: Path,
    output_dir: Path,
    launch_mode: str,
    shutdown_attached_editor: bool,
    play_started: bool,
) -> list[tuple[str, Callable[[], None]]]:
    actions: list[tuple[str, Callable[[], None]]] = []
    if session is not None:
        actions.append((
            "session cleanup",
            lambda: _cleanup_session(
                session,
                launch_mode=launch_mode,
                shutdown_attached_editor=shutdown_attached_editor,
                play_started=play_started,
            ),
        ))
    actions.append((
        "UE log copy",
        lambda: _copy_latest_ue_log(simworld_root, output_dir / "ue.log"),
    ))
    return actions


def _record_finalizer_failure(
    failures: list[tuple[BaseException, Any]],
    label: str,
    error: BaseException,
) -> None:
    failures.append((error, _exception_traceback(error)))
    _log_best_effort(
        "probe finalization failed during %s: %s", label, _typed_error(error))


def _attempt_probe_finalizers(
    session: Any,
    *,
    simworld_root: Path,
    output_dir: Path,
    launch_mode: str,
    shutdown_attached_editor: bool,
    play_started: bool,
) -> list[tuple[BaseException, Any]]:
    """Attempt cleanup then log copy, retaining every error in call order."""

    actions = _probe_finalizer_actions(
        session,
        simworld_root=simworld_root,
        output_dir=output_dir,
        launch_mode=launch_mode,
        shutdown_attached_editor=shutdown_attached_editor,
        play_started=play_started,
    )
    failures: list[tuple[BaseException, Any]] = []
    for label, action in actions:
        try:
            action()
        except BaseException as error:
            _record_finalizer_failure(failures, label, error)
    return failures


def _finalization_error_texts(
    failures: list[tuple[BaseException, Any]],
) -> list[str]:
    return [_typed_error(error) for error, _traceback in failures]


def _attach_finalization_errors(
    error: BaseException, errors: tuple[str, ...],
) -> None:
    try:
        error.finalization_errors = errors
        return
    except BaseException as attachment_error:
        _log_best_effort(
            "failed to attach probe finalization errors: %s",
            _typed_error(attachment_error),
        )
    try:
        error.add_note("probe finalization errors: " + "; ".join(errors))
    except BaseException as note_error:
        _log_best_effort(
            "failed to note probe finalization errors: %s",
            _typed_error(note_error),
        )


def _attempt_probe_finalizers_or_raise(
    session: Any,
    *,
    simworld_root: Path,
    output_dir: Path,
    launch_mode: str,
    shutdown_attached_editor: bool,
    play_started: bool,
    on_failure: Callable[[list[tuple[BaseException, Any]], list[str]], None],
) -> None:
    """Attempt every finalizer and bare-raise the first original failure."""

    actions = _probe_finalizer_actions(
        session,
        simworld_root=simworld_root,
        output_dir=output_dir,
        launch_mode=launch_mode,
        shutdown_attached_editor=shutdown_attached_editor,
        play_started=play_started,
    )
    for index, (label, action) in enumerate(actions):
        try:
            action()
        except BaseException as first_error:
            first_traceback = _exception_traceback(first_error)
            failures: list[tuple[BaseException, Any]] = [
                (first_error, first_traceback)]
            _log_best_effort(
                "probe finalization failed during %s: %s",
                label,
                _typed_error(first_error),
            )
            for later_label, later_action in actions[index + 1:]:
                try:
                    later_action()
                except BaseException as later_error:
                    _record_finalizer_failure(
                        failures, later_label, later_error)
            try:
                error_texts = _finalization_error_texts(failures)
            except BaseException as formatting_error:
                error_texts = []
                _log_best_effort(
                    "failed to format probe finalization errors: %s",
                    _typed_error(formatting_error),
                )
            try:
                on_failure(failures, error_texts)
            except BaseException as evidence_error:
                _log_best_effort(
                    "failed to persist probe finalization evidence: %s",
                    _typed_error(evidence_error),
                )
            _attach_finalization_errors(first_error, tuple(error_texts))
            _restore_exception_traceback(first_error, first_traceback)
            raise


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    rear_selection = parse_selection(args.rear_pixel)
    left_selection = parse_selection(args.front_left_pixel)
    right_selection = parse_selection(args.front_right_pixel)
    if rear_selection[0] != "rear":
        raise ValueError("rear probe selection must use rear")
    if left_selection[0] != "front" or not left_selection[1] < 0.5:
        raise ValueError("front-left probe must use a left-side front pixel")
    if right_selection[0] != "front" or not right_selection[1] > 0.5:
        raise ValueError("front-right probe must use a right-side front pixel")
    simworld_root = Path(args.simworld_root).resolve()
    citycore_content = Path(args.citycore_content).resolve()
    validate_mount_inputs(citycore_content, simworld_root / "SimWorld.uproject")
    output_dir = Path(args.output).resolve()
    _prepare_output_directory(output_dir)
    event_log = output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)
    trajectory_path = output_dir / "trajectory.csv"
    setup_request = build_front_rear_setup_request(DELIVERY_REGION)
    config = PixelGoalConfig(
        max_navmesh_adjustment_cm=args.max_navmesh_adjustment_cm,
        acceptance_radius_cm=args.acceptance_radius_cm,
        execution_timeout_s=args.execution_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    config_report = {
        "max_navmesh_adjustment_cm": config.max_navmesh_adjustment_cm,
        "acceptance_radius_cm": config.acceptance_radius_cm,
        "execution_timeout_s": config.execution_timeout_s,
    }
    session = None
    play_started = False
    images: list[str] = []
    trajectory: list[dict[str, Any]] = []
    failure_stage = "session_initialization"
    attempted_action: dict[str, Any] | None = None
    preparation: dict[str, Any] | None = None
    latency_benchmark: dict[str, Any] | None = None
    image_evidence: list[dict[str, Any]] = []
    secondary_evidence_error: str | None = None

    def failure_report(
        error_text: str,
        *,
        stage: str | None = None,
        finalization_errors: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "experiment": "front/rear scripted Pixel Goal probe",
            "action_space": "pixel_goal_front_rear",
            "camera_view": "front_rear",
            "region": DELIVERY_REGION.to_report(),
            "error": error_text,
            "failure_stage": failure_stage if stage is None else stage,
            "attempted_action": attempted_action,
            "setup_request": setup_request,
            "setup": (
                None if preparation is None else preparation.get("setup")),
            "preparation": preparation,
            "config": config_report,
            "latency_benchmark": latency_benchmark,
            "images": images,
            "image_evidence": image_evidence,
            "trajectory": trajectory,
            "secondary_evidence_error": secondary_evidence_error,
            "finalization_errors": (
                [] if finalization_errors is None else finalization_errors),
        }

    def persist_failure(
        report: dict[str, Any], *, rewrite: bool = False,
    ) -> None:
        try:
            _write_json_atomic(output_dir / "probe_failure.json", report)
        except BaseException as evidence_error:
            _log_best_effort(
                "failed to %s probe failure evidence: %s",
                "rewrite" if rewrite else "persist",
                _typed_error(evidence_error),
            )

    try:
        session = AttachedParisGameSession(args.spear_config)
        failure_stage = "begin_play"
        session.begin_play()
        play_started = True
        failure_stage = "endpoint_initialization"
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        failure_stage = "paris_preparation"
        preparation = _prepare_paris_loop(
            endpoint, setup_request,
            readiness_timeout_s=args.navmesh_timeout_s,
            capture_warmup_s=args.capture_warmup_s,
            wait_fn=_wait_for_paris_poc,
            warm_up_fn=_warm_up_paris_capture,
        )
        runtime = LivePixelGoalRuntime(endpoint, config)

        failure_stage = "benchmark_reset"
        reset = _reset_paris_poc_trial(endpoint, args.navmesh_timeout_s)
        failure_stage = "latency_benchmark"
        latency_benchmark = run_capture_latency_benchmark(
            runtime,
            endpoint,
            status_fn=lambda: endpoint.call(
                "PixelGoal_GetParisPocStatusJson", {}),
            cycles=args.latency_cycles,
            agent_tag=AGENT_TAG,
            width_px=args.capture_width,
            height_px=args.capture_height,
            fov_degrees=config.fov_degrees,
        )
        attempted_action = _attempted_action_record(
            "rear_reversal", rear_selection)
        failure_stage = "rear_reversal.capture_before"
        initial = runtime.capture_view_pair(
            agent_tag=AGENT_TAG,
            width_px=args.capture_width,
            height_px=args.capture_height,
            fov_degrees=config.fov_degrees,
        )
        attempted_action = _attempted_action_record(
            "rear_reversal", rear_selection, initial)
        failure_stage = "rear_reversal.save_before"
        images.extend(_save_pair(initial, output_dir, 0))
        failure_stage = "rear_reversal.execute"
        started, result, reversal = _execute(runtime, initial, rear_selection)
        target = started.request.projected_target
        assert target is not None
        reversal["target_dot_body_forward_cm"] = planar_target_dot(
            initial.pose, (target.x_cm, target.y_cm))
        failure_stage = "rear_reversal.capture_after"
        post_reversal = runtime.capture_view_pair(
            agent_tag=AGENT_TAG,
            width_px=args.capture_width,
            height_px=args.capture_height,
            fov_degrees=config.fov_degrees,
        )
        failure_stage = "rear_reversal.save_after"
        images.extend(_save_pair(post_reversal, output_dir, 1))
        reversal.update({
            "trial_reset": reset,
            "post_capture_group_id": post_reversal.capture_group_id,
            "post_front_snapshot_id": post_reversal.front.camera_snapshot_id,
            "post_front_yaw_deg": post_reversal.front.camera_yaw_deg,
            "alignment_error_deg": angular_error_degrees(
                reversal["movement_bearing_deg"],
                float(post_reversal.front.camera_yaw_deg),
            ),
        })
        trajectory.append(_trajectory_row("rear_reversal", reversal))

        steering: dict[str, dict[str, Any]] = {}
        for offset, (name, selection) in enumerate((
                ("front_left", left_selection),
                ("front_right", right_selection)), start=1):
            attempted_action = _attempted_action_record(name, selection)
            failure_stage = f"{name}.reset"
            independent_reset = _reset_paris_poc_trial(
                endpoint, args.navmesh_timeout_s)
            failure_stage = f"{name}.capture_before"
            pair = runtime.capture_view_pair(
                agent_tag=AGENT_TAG,
                width_px=args.capture_width,
                height_px=args.capture_height,
                fov_degrees=config.fov_degrees,
            )
            attempted_action = _attempted_action_record(name, selection, pair)
            failure_stage = f"{name}.save_before"
            images.extend(_save_pair(pair, output_dir, offset * 2))
            failure_stage = f"{name}.execute"
            _started, _result, action = _execute(runtime, pair, selection)
            action["signed_bearing_delta_deg"] = signed_bearing_delta_degrees(
                pair.pose[3], action["movement_bearing_deg"])
            action["trial_reset"] = independent_reset
            failure_stage = f"{name}.capture_after"
            after = runtime.capture_view_pair(
                agent_tag=AGENT_TAG,
                width_px=args.capture_width,
                height_px=args.capture_height,
                fov_degrees=config.fov_degrees,
            )
            failure_stage = f"{name}.save_after"
            images.extend(_save_pair(after, output_dir, offset * 2 + 1))
            action["post_capture_group_id"] = after.capture_group_id
            steering[name] = action
            trajectory.append(_trajectory_row(name, action))
        attempted_action = None
        failure_stage = "trajectory_write"
        _write_trajectory(trajectory_path, trajectory)
        failure_stage = "image_evidence"
        image_evidence = analyze_probe_images(
            output_dir,
            images,
            width_px=args.capture_width,
            height_px=args.capture_height,
        )
        report = {
            "experiment": "front/rear scripted Pixel Goal probe",
            "action_space": "pixel_goal_front_rear",
            "camera_view": "front_rear",
            "region": DELIVERY_REGION.to_report(),
            "setup_request": setup_request,
            "setup": preparation["setup"],
            "latency_benchmark": latency_benchmark,
            "image_evidence": image_evidence,
            "config": config_report,
            "rear_reversal": reversal,
            "steering": steering,
            "artifacts": {
                "images": images,
                "events": event_log.name,
                "trajectory": trajectory_path.name,
                "ue_log": "ue.log",
            },
        }
        failure_stage = "report_validation"
        validate_front_rear_probe_report(report)
    except BaseException as error:
        primary_traceback = _exception_traceback(error)
        secondary_evidence_error = None
        try:
            image_evidence = analyze_partial_probe_images(
                output_dir,
                images,
                width_px=args.capture_width,
                height_px=args.capture_height,
            )
            validate_partial_probe_image_evidence(
                image_evidence,
                image_names=images,
                width_px=args.capture_width,
                height_px=args.capture_height,
            )
        except BaseException as evidence_error:
            image_evidence = []
            secondary_evidence_error = _typed_error(evidence_error)
        persisted_failure: dict[str, Any] | None = None
        try:
            persisted_failure = failure_report(_typed_error(error))
            persist_failure(persisted_failure)
        except BaseException as evidence_error:
            _log_best_effort(
                "failed to construct primary probe failure evidence: %s",
                _typed_error(evidence_error),
            )
        try:
            finalizer_failures = _attempt_probe_finalizers(
                session,
                simworld_root=simworld_root,
                output_dir=output_dir,
                launch_mode=args.launch_mode,
                shutdown_attached_editor=args.shutdown_attached_editor,
                play_started=play_started,
            )
            if finalizer_failures:
                finalization_errors = _finalization_error_texts(
                    finalizer_failures)
                if persisted_failure is None:
                    persisted_failure = failure_report(_typed_error(error))
                persisted_failure["finalization_errors"] = finalization_errors
                persist_failure(persisted_failure, rewrite=True)
        except BaseException as evidence_error:
            _log_best_effort(
                "failed to augment primary probe failure evidence: %s",
                _typed_error(evidence_error),
            )
        _restore_exception_traceback(error, primary_traceback)
        raise

    def persist_finalization_failure(
        _failures: list[tuple[BaseException, Any]],
        finalization_errors: list[str],
    ) -> None:
        if not finalization_errors:
            return
        persisted_failure = failure_report(
            finalization_errors[0],
            stage="finalization",
            finalization_errors=finalization_errors,
        )
        persist_failure(persisted_failure)

    _attempt_probe_finalizers_or_raise(
        session,
        simworld_root=simworld_root,
        output_dir=output_dir,
        launch_mode=args.launch_mode,
        shutdown_attached_editor=args.shutdown_attached_editor,
        play_started=play_started,
        on_failure=persist_finalization_failure,
    )

    failure_stage = "report_write"
    try:
        _write_json_atomic(output_dir / "probe_report.json", report)
    except BaseException as error:
        primary_traceback = _exception_traceback(error)
        try:
            persist_failure(failure_report(_typed_error(error)))
        except BaseException as evidence_error:
            _log_best_effort(
                "failed to construct report-write failure evidence: %s",
                _typed_error(evidence_error),
            )
        _restore_exception_traceback(error, primary_traceback)
        raise
    return report


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
    parser.add_argument("--output", default="artifacts/pixel_goal_front_rear_probe")
    parser.add_argument("--rear-pixel", default="rear,0.50,0.80")
    parser.add_argument("--front-left-pixel", default="front,0.28,0.80")
    parser.add_argument("--front-right-pixel", default="front,0.60,0.80")
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=360)
    parser.add_argument("--latency-cycles", type=int, choices=(20,), default=20)
    parser.add_argument("--max-navmesh-adjustment-cm", type=float, default=20.0)
    parser.add_argument("--acceptance-radius-cm", type=float, default=15.0)
    parser.add_argument("--execution-timeout-s", type=float, default=45.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--navmesh-timeout-s", type=float, default=120.0)
    parser.add_argument("--capture-warmup-s", type=float, default=6.0)
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_probe(args)
    print(json.dumps(validate_front_rear_probe_report(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
