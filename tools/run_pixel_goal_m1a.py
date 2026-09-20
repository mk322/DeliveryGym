#!/usr/bin/env python3
"""Run the deterministic live-UE Pixel Goal Milestone 1A calibration.

This is deliberately a live-only engineering harness.  A policy supplies only
normalized image points; the UE subsystem retains the associated capture
snapshot, resolves geometry, validates NavMesh reachability, and owns movement.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalConfig,
    PixelGoalEndpoint,
    PixelGoalInProgress,
    PixelGoalRejected,
)
from embodiedbench.schemas.embodiment import ControllerResult
from embodiedbench.schemas.geometry import Vec3
from embodiedbench.schemas.runtime import NavPixelGoalAction
from embodiedbench.schemas.runtime import ControllerOutcomeCode


AGENT_TAG = "PixelGoalCalibrationAgent"
MANUAL_GOALS = [
    (0.50, 0.82),
    (0.28, 0.78),
    (0.72, 0.72),
    (0.10, 0.90),
    (0.85, 1.00),
]
INVALID_GOALS = [
    ("vertical_wall", (0.50, 0.50)),
    ("above_horizon", (0.90, 0.10)),
    ("outside_navmesh", (0.80, 0.55)),
]
TERMINAL_STATES = {"completed", "failed"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Vec3):
        return [value.x_cm, value.y_cm, value.z_cm]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _decode_object_response(
    function_name: str, wire_value: object
) -> dict[str, object]:
    """Normalize reflected UFunction results across SPEAR client versions."""

    if isinstance(wire_value, str):
        response = json.loads(wire_value)
    elif isinstance(wire_value, Mapping):
        response = dict(wire_value)
    else:
        raise RuntimeError(
            f"{function_name} returned {type(wire_value).__name__}, "
            "expected JSON text or an object mapping"
        )
    if not isinstance(response, dict):
        raise RuntimeError(f"{function_name} returned a non-object JSON response")
    return response


class SpearPixelGoalEndpoint(PixelGoalEndpoint):
    """Synchronous JSON calls to the live Pixel Goal UWorld subsystem."""

    def __init__(self, session: Any, event_log: Path) -> None:
        self._instance = session._instance
        game = session._game
        if game is None:
            raise RuntimeError("SpearSession has no live game world")
        with self._instance.begin_frame():
            self._subsystem = game.unreal_service.get_subsystem(
                subsystem_provider_class_name="UWorld",
                subsystem_uclass="USpPixelGoalSubsystem",
                as_unreal_object=True,
            )
        with self._instance.end_frame():
            pass
        if self._subsystem is None:
            raise RuntimeError("USpPixelGoalSubsystem is unavailable")
        self._event_log = event_log
        self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def call(
        self, function_name: str, request: dict[str, object]
    ) -> dict[str, object]:
        request_json = json.dumps(request, separators=(",", ":"), sort_keys=True)
        with self._instance.begin_frame():
            wire_response = self._subsystem.call(
                function_name,
                args={"RequestJson": request_json},
                as_value=True,
            )
        with self._instance.end_frame():
            pass
        response = _decode_object_response(function_name, wire_response)
        self.calls.append((function_name, dict(request), response))
        event = {
            "wall_time_s": time.time(),
            "function": function_name,
            "request": request,
            "response": response,
        }
        with self._event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        return response


def _decode_data_url(data_url: str) -> Image.Image:
    try:
        metadata, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("invalid RGB data URL") from exc
    if ";base64" not in metadata:
        raise ValueError("RGB data URL is not base64 encoded")
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _save_frame(
    data_url: str,
    frame_path: Path,
    selected_path: Path | None = None,
    uv: tuple[float, float] | None = None,
) -> None:
    image = _decode_data_url(data_url)
    image.save(frame_path)
    if selected_path is None or uv is None:
        return
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x = round(uv[0] * (image.width - 1))
    y = round(uv[1] * (image.height - 1))
    radius = max(8, min(image.size) // 40)
    line_width = max(2, radius // 4)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 30, 30),
        width=line_width,
    )
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 30, 30), width=line_width)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 30, 30), width=line_width)
    annotated.save(selected_path)


def _wait_for_navmesh(endpoint: SpearPixelGoalEndpoint, timeout_s: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_status: dict[str, object] = {}
    while time.monotonic() < deadline:
        last_status = endpoint.call("PixelGoal_GetCalibrationStatusJson", {})
        if last_status.get("fixture_ready"):
            return last_status
        time.sleep(0.1)
    raise TimeoutError(f"calibration NavMesh did not become ready: {last_status}")


def _execute_goal(
    runtime: LivePixelGoalRuntime,
    endpoint: SpearPixelGoalEndpoint,
    output_dir: Path,
    index: int,
    uv: tuple[float, float],
) -> dict[str, object]:
    frame = runtime.capture_frame(agent_tag=AGENT_TAG)
    stem = f"frame_{index:02d}"
    frame_path = output_dir / f"{stem}.png"
    selected_path = output_dir / f"{stem}_selected.png"
    _save_frame(frame.rgb_data_url, frame_path, selected_path, uv)

    action = NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]})
    started = runtime.start(action, frame)
    resolve_response = endpoint.calls[-1][2]
    samples: list[list[float]] = []
    initial_feet = resolve_response.get("initial_feet_position_cm")
    if isinstance(initial_feet, list):
        samples.append(initial_feet)
    status_history: list[dict[str, object]] = []
    deadline = time.monotonic() + runtime.config.execution_timeout_s
    final_result: ControllerResult | None = None
    final_status: dict[str, object] | None = None
    while time.monotonic() < deadline:
        status = runtime.poll(started.request.request_id)
        raw_status = endpoint.calls[-1][2]
        status_history.append(raw_status)
        feet = raw_status.get("final_feet_position_cm")
        if isinstance(feet, list) and (not samples or feet != samples[-1]):
            samples.append(feet)
        if isinstance(status, ControllerResult):
            final_result = status
            final_status = raw_status
            break
        if not isinstance(status, PixelGoalInProgress):
            raise RuntimeError(f"unexpected Pixel Goal status {status!r}")
        time.sleep(runtime.config.poll_interval_s)
    if final_result is None or final_status is None:
        cancelled = runtime.cancel(
            started.request.request_id,
            reason="execution_timeout",
        )
        raise TimeoutError(
            f"Pixel Goal {index} timed out and was cancelled: "
            f"controller={cancelled.controller_result}, "
            f"final_feet={cancelled.final_feet_position}, "
            f"error_m={cancelled.execution_error_planar_m}"
        )
    if final_result.controller_result != "success":
        raise RuntimeError(
            f"Pixel Goal {index} failed: {final_result.controller_result} "
            f"({final_result.failure_reason})"
        )

    after_frame = runtime.capture_frame(agent_tag=AGENT_TAG)
    after_path = output_dir / f"{stem}_after.png"
    _save_frame(after_frame.rgb_data_url, after_path)
    if len(samples) < 3:
        raise RuntimeError(
            f"Pixel Goal {index} produced only {len(samples)} distinct feet samples"
        )

    return {
        "index": index,
        "requested_uv": list(uv),
        "camera_snapshot_id": frame.camera_snapshot_id,
        "camera_intrinsics_id": frame.camera_intrinsics_id,
        "raw_world_hit_cm": _jsonable(started.request.raw_world_hit),
        "validated_navigation_target_cm": _jsonable(
            started.request.validated_navigation_target
        ),
        "navmesh_adjustment_cm": started.request.navmesh_adjustment_cm,
        "controller_request_result": started.controller_request_result,
        "controller_result": final_result.controller_result,
        "initial_feet_position_cm": resolve_response["initial_feet_position_cm"],
        "final_feet_position_cm": _jsonable(final_result.final_feet_position),
        "final_agent_position_cm": _jsonable(final_result.final_pose.position),
        "final_yaw_degrees": final_result.final_pose.yaw_deg,
        "execution_error_planar_m": final_result.execution_error_planar_m,
        "execution_error_3d_m": final_result.execution_error_3d_m,
        "distance_travelled_cm": final_result.distance_travelled_cm,
        "elapsed_sim_s": final_result.elapsed_sim_s,
        "feet_position_samples_cm": samples,
        "status_sample_count": len(status_history),
        "input_frame": frame_path.name,
        "selected_frame": selected_path.name,
        "post_move_frame": after_path.name,
        "post_move_camera_snapshot_id": after_frame.camera_snapshot_id,
    }


def _execute_invalid_cases(
    runtime: LivePixelGoalRuntime,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, uv in INVALID_GOALS:
        frame = runtime.capture_frame(agent_tag=AGENT_TAG)
        action = NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]})
        try:
            runtime.start(action, frame)
        except PixelGoalRejected as rejected:
            results.append(
                {
                    "name": name,
                    "requested_uv": list(uv),
                    "accepted": False,
                    "rejection_reason": rejected.reason,
                    "camera_snapshot_id": frame.camera_snapshot_id,
                    "camera_intrinsics_id": frame.camera_intrinsics_id,
                    "raw_world_hit_cm": (
                        _jsonable(rejected.raw_world_hit)
                        if rejected.raw_world_hit is not None
                        else None
                    ),
                    "audit": rejected.audit,
                }
            )
        else:
            raise RuntimeError(f"invalid case {name!r} was unexpectedly accepted")
    return results


def _exercise_timeout_cancellation(
    runtime: LivePixelGoalRuntime,
) -> dict[str, object]:
    """Prove that the timeout path terminates controller state in live UE."""
    uv = MANUAL_GOALS[0]
    frame = runtime.capture_frame(agent_tag=AGENT_TAG)
    started = runtime.start(
        NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]}),
        frame,
    )
    result = runtime.cancel(started.request.request_id, reason="execution_timeout")
    if result.outcome is not ControllerOutcomeCode.EXECUTION_TIMEOUT:
        raise RuntimeError(f"timeout cancellation produced {result.outcome.value}")

    time.sleep(max(0.1, runtime.config.poll_interval_s * 2.0))
    settled = runtime.poll(started.request.request_id)
    if not isinstance(settled, ControllerResult):
        raise RuntimeError("cancelled Pixel Goal resumed movement")
    if result.final_feet_position is None or settled.final_feet_position is None:
        raise RuntimeError("cancelled Pixel Goal omitted final feet")
    drift_cm = result.final_feet_position.distance_cm(settled.final_feet_position)
    if drift_cm > 1.0:
        raise RuntimeError(
            f"cancelled Pixel Goal moved another {drift_cm:.3f} cm after abort"
        )
    return {
        "requested_uv": list(uv),
        "request_id": started.request.request_id,
        "controller_result": result.controller_result,
        "outcome": result.outcome.value,
        "failure_reason": result.failure_reason,
        "final_feet_position_cm": _jsonable(result.final_feet_position),
        "post_cancel_drift_cm": drift_cm,
    }


def _copy_latest_ue_log(simworld_root: Path, destination: Path) -> None:
    candidates = sorted(
        (simworld_root / "Saved" / "Logs").glob("*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        shutil.copy2(candidates[-1], destination)


def _cleanup_session(
    session: Any,
    *,
    launch_mode: str,
    shutdown_attached_editor: bool,
    play_started: bool,
) -> None:
    if launch_mode != "attach" or shutdown_attached_editor:
        session.shutdown()
    elif play_started:
        session.end_play()


def run(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    simworld_root = Path(args.simworld_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log = output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)

    simworld_utils = simworld_root / "utils"
    if not (simworld_utils / "simworld_task" / "session.py").is_file():
        raise FileNotFoundError(f"missing SimWorld task runtime under {simworld_utils}")
    sys.path.insert(0, str(simworld_utils))
    from simworld_task.session import SpearSession  # type: ignore[import-not-found]

    config = PixelGoalConfig(
        max_navmesh_adjustment_cm=args.max_navmesh_adjustment_cm,
        acceptance_radius_cm=args.acceptance_radius_cm,
        execution_timeout_s=args.execution_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    session = None
    play_started = False
    try:
        session = SpearSession(
            spear_config_path=args.spear_config,
            launch_mode=args.launch_mode,
            editor_startup_map="/Game/EmptyLevel.EmptyLevel",
        )
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        fixture = endpoint.call("PixelGoal_SpawnCalibrationFixtureJson", {})
        if fixture.get("success") is not True:
            raise RuntimeError(f"calibration fixture failed: {fixture}")
        calibration_status = _wait_for_navmesh(endpoint, args.navmesh_timeout_s)
        runtime = LivePixelGoalRuntime(endpoint, config)

        timeout_cancellation = _exercise_timeout_cancellation(runtime)
        invalid_cases = _execute_invalid_cases(runtime)
        manual_goals = [
            _execute_goal(runtime, endpoint, output_dir, index, uv)
            for index, uv in enumerate(MANUAL_GOALS, start=1)
        ]
        report = {
            "milestone": "1A",
            "navigation_mode": "nav_pixel_goal",
            "scene": "/Game/EmptyLevel.EmptyLevel",
            "fixture": fixture,
            "calibration_status": calibration_status,
            "config": asdict(config),
            "manual_goals": manual_goals,
            "invalid_cases": invalid_cases,
            "timeout_cancellation": timeout_cancellation,
            "maximum_execution_error_planar_m": max(
                goal["execution_error_planar_m"] for goal in manual_goals
            ),
            "repo_root": str(repo_root),
            "simworld_root": str(simworld_root),
        }
        report_path = output_dir / "calibration_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + "\n",
            encoding="utf-8",
        )
        return report
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--simworld-root", default=".simworld-ue")
    parser.add_argument(
        "--spear-config",
        help="optional SPEAR launch config (useful for private/headless UE builds)",
    )
    parser.add_argument(
        "--launch-mode",
        choices=("launch", "attach"),
        default="launch",
        help="launch a UE editor or attach to an already running calibration editor",
    )
    parser.add_argument(
        "--shutdown-attached-editor",
        action="store_true",
        help=(
            "terminate an attached UE editor after calibration "
            "(default: keep it running)"
        ),
    )
    parser.add_argument("--output", default="artifacts/pixel_goal_m1a")
    parser.add_argument("--max-navmesh-adjustment-cm", type=float, default=10.0)
    parser.add_argument("--acceptance-radius-cm", type=float, default=15.0)
    parser.add_argument("--execution-timeout-s", type=float, default=30.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--navmesh-timeout-s", type=float, default=60.0)
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
