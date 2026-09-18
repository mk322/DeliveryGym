#!/usr/bin/env python3
"""Run a manual no-reset Pixel Goal loop in the local Paris PoC region."""

from __future__ import annotations

import argparse
import os
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalConfig,
    PixelGoalInProgress,
    PixelGoalRejected,
)
from embodiedbench.runtime.pixel_goal_paris_poc import (
    PARIS_POC_SCENE,
    RUE_DE_RIVOLI_SIDEWALK,
)
from embodiedbench.schemas.embodiment import ControllerResult
from embodiedbench.schemas.runtime import NavPixelGoalAction
from tools.run_pixel_goal_m1a import (
    SpearPixelGoalEndpoint,
    _cleanup_session,
    _copy_latest_ue_log,
    _jsonable,
    _save_frame,
)
from tools.run_pixel_goal_1b_poc import (
    STRICT_EXECUTION,
    AttachedParisGameSession,
    _wait_for_paris_poc,
    _warm_up_paris_capture,
    build_paris_setup_request,
    validate_mount_inputs,
)


AGENT_TAG = "PixelGoalParisPocAgent"
VISUAL_CONTEXTS = frozenset(
    {"clear_straight_long", "near_obstacle_short", "adaptive"}
)


@dataclass(frozen=True)
class ManualPixelSelection:
    """Operator input plus audit-only visual context metadata."""

    uv: tuple[float, float]
    visual_context: str = "adaptive"


def _coerce_manual_selection(value) -> ManualPixelSelection:
    if isinstance(value, ManualPixelSelection):
        selection = value
    else:
        selection = ManualPixelSelection(uv=tuple(value))
    if selection.visual_context not in VISUAL_CONTEXTS:
        raise ValueError(
            f"unsupported visual context: {selection.visual_context}"
        )
    if len(selection.uv) != 2:
        raise ValueError("manual Pixel Goal selection needs exactly u and v")
    return selection


class ManualSequenceTerminated(RuntimeError):
    """Accepted controller execution stopped the no-reset sequence."""

    def __init__(
        self,
        reason: str,
        *,
        successful_steps: list[dict[str, object]],
        rejected_attempts: list[dict[str, object]],
        terminal: dict[str, object],
    ) -> None:
        self.reason = reason
        self.successful_steps = list(successful_steps)
        self.rejected_attempts = list(rejected_attempts)
        self.terminal = dict(terminal)
        super().__init__(reason)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simworld-root", default=".simworld-ue")
    parser.add_argument(
        "--citycore-content",
        default=os.environ.get("CITYCORE_PARIS_CONTENT", ""),
    )
    parser.add_argument(
        "--scene", default=PARIS_POC_SCENE, choices=[PARIS_POC_SCENE]
    )
    parser.add_argument("--spear-config")
    parser.add_argument("--launch-mode", choices=("attach",), default="attach")
    parser.add_argument("--shutdown-attached-editor", action="store_true")
    parser.add_argument(
        "--output", default="artifacts/pixel_goal_1b_closed_loop"
    )
    parser.add_argument("--move-count", type=int, default=6)
    parser.add_argument(
        "--max-navmesh-adjustment-cm", type=float, default=10.0
    )
    parser.add_argument("--acceptance-radius-cm", type=float, default=15.0)
    parser.add_argument("--execution-timeout-s", type=float, default=45.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--navmesh-timeout-s", type=float, default=120.0)
    parser.add_argument("--capture-warmup-s", type=float, default=6.0)
    return parser


def _prepare_paris_loop(
    endpoint,
    setup_request: dict[str, object],
    *,
    readiness_timeout_s: float,
    capture_warmup_s: float,
    wait_fn,
    warm_up_fn,
) -> dict[str, object]:
    setup = endpoint.call("PixelGoal_SetupParisPocJson", setup_request)
    valid_setup = (
        setup.get("success") is True
        and setup.get("synthetic_floor_spawned") is False
        and setup.get("dynamic_runtime_generation") is True
        and setup.get("supports_runtime_generation") is True
        and setup.get("navmesh_has_valid_data") is True
        and isinstance(setup.get("active_navmesh_tiles"), int)
        and int(setup["active_navmesh_tiles"]) >= 1
        and setup.get("rgb_capture_mode") == "agent_native"
        and setup.get("rgb_capture_render_state_persistent") is True
    )
    if not valid_setup:
        raise RuntimeError(
            "Paris Pixel Goal setup is not ready for persistent RGB "
            f"closed-loop navigation: {setup}"
        )
    status = wait_fn(endpoint, readiness_timeout_s)
    capture_warmup_ms = warm_up_fn(capture_warmup_s)
    return {
        "setup": setup,
        "status": status,
        "capture_warmup_ms": capture_warmup_ms,
    }


def parse_manual_uv(text: str) -> tuple[float, float]:
    values = text.replace(",", " ").split()
    if len(values) != 2:
        raise ValueError("enter exactly two normalized coordinates: u v")
    try:
        u_norm, v_norm = (float(value) for value in values)
    except ValueError as exc:
        raise ValueError("u and v must be numbers") from exc
    if not all(math.isfinite(value) for value in (u_norm, v_norm)):
        raise ValueError("u and v must be finite")
    if not (0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0):
        raise ValueError("u and v must be in [0, 1]")
    return u_norm, v_norm


def prompt_manual_uv(
    step_index: int,
    before_path: Path,
    *,
    input_fn=input,
    output_fn=print,
) -> tuple[float, float]:
    output_fn(f"STEP {step_index} BEFORE_FPV {before_path}", flush=True)
    while True:
        try:
            return parse_manual_uv(input_fn("manual Pixel Goal u v> "))
        except ValueError as exc:
            output_fn(f"INVALID_INPUT {exc}", flush=True)


def build_sequence_summary(steps, endpoint_calls) -> dict[str, object]:
    trajectory = []
    continuity_errors = []
    planar_errors = []
    if steps:
        trajectory.append(list(steps[0]["initial_feet_position_cm"]))
    for index, step in enumerate(steps):
        final_feet = list(step["final_feet_position_cm"])
        trajectory.append(final_feet)
        planar_errors.append(float(step["execution_error_planar_m"]))
        if index + 1 < len(steps):
            next_initial = list(steps[index + 1]["initial_feet_position_cm"])
            continuity_errors.append(math.dist(final_feet, next_initial))
    function_names = [call[0] for call in endpoint_calls]
    return {
        "trajectory_feet_cm": trajectory,
        "continuity_errors_cm": continuity_errors,
        "reset_calls": function_names.count("PixelGoal_ResetParisPocTrialJson"),
        "resolve_calls": function_names.count("PixelGoal_ResolveAndMoveJson"),
        "mean_execution_error_planar_m": (
            sum(planar_errors) / len(planar_errors) if planar_errors else 0.0
        ),
        "maximum_execution_error_planar_m": max(planar_errors, default=0.0),
    }


def validate_closed_loop_report(
    report, expected_moves: int = 6
) -> dict[str, object]:
    steps = report.get("successful_steps", [])
    summary = report.get("summary", {})
    if report.get("navigation_mode") != "nav_pixel_goal":
        raise ValueError("closed loop must use nav_pixel_goal")
    if len(steps) != expected_moves:
        raise ValueError(f"expected {expected_moves} successful moves")
    if int(summary.get("reset_calls", -1)) != 0:
        raise ValueError("closed loop must contain zero reset calls")
    continuity = [float(value) for value in summary["continuity_errors_cm"]]
    if max(continuity, default=0.0) > 1.0:
        raise ValueError("trajectory continuity exceeds 1 cm")
    for index, step in enumerate(steps):
        if step.get("navigation_mode") != "nav_pixel_goal":
            raise ValueError("every step must use nav_pixel_goal")
        if not step.get("camera_snapshot_id") or not step.get(
            "camera_intrinsics_id"
        ):
            raise ValueError("every step needs a camera snapshot and intrinsics")
        if not step.get("post_move_camera_snapshot_id") or not step.get(
            "post_move_camera_intrinsics_id"
        ):
            raise ValueError(
                "every step needs a post-move camera snapshot and intrinsics"
            )
        if step.get("controller_result") != "success":
            raise ValueError("every accepted step must succeed")
        if float(step["execution_error_planar_m"]) > 0.25:
            raise ValueError("planar execution error exceeds 0.25 m")
        if index + 1 < len(steps):
            if step["after_frame"] != steps[index + 1]["before_frame"]:
                raise ValueError("post-move frame chain is broken")
            if step["post_move_camera_snapshot_id"] != steps[index + 1].get(
                "camera_snapshot_id"
            ):
                raise ValueError("post-move camera snapshot chain is broken")
            if step["post_move_camera_intrinsics_id"] != steps[index + 1].get(
                "camera_intrinsics_id"
            ):
                raise ValueError("post-move camera intrinsics chain is broken")
    return {
        "passed": True,
        "successful_move_count": expected_moves,
        "maximum_execution_error_planar_m": summary[
            "maximum_execution_error_planar_m"
        ],
        "maximum_continuity_error_cm": max(continuity, default=0.0),
    }


def execute_manual_sequence(
    runtime,
    endpoint,
    output_dir: Path,
    move_count: int,
    select_uv,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_frame = runtime.capture_frame(agent_tag=AGENT_TAG)
    current_path = output_dir / "fpv_000.png"
    _save_frame(current_frame.rgb_data_url, current_path)
    successful_steps: list[dict[str, object]] = []
    rejected_attempts: list[dict[str, object]] = []

    while len(successful_steps) < move_count:
        step_index = len(successful_steps) + 1
        attempt_index = 1 + sum(
            attempt["step_index"] == step_index
            for attempt in rejected_attempts
        )
        selection = _coerce_manual_selection(
            select_uv(step_index, current_path)
        )
        uv = selection.uv
        selected_path = output_dir / (
            f"step_{step_index:03d}_attempt_{attempt_index:03d}_selected.png"
        )
        _save_frame(
            current_frame.rgb_data_url,
            current_path,
            selected_path,
            uv,
        )
        try:
            started = runtime.start(
                NavPixelGoalAction(
                    target={"u_norm": uv[0], "v_norm": uv[1]}
                ),
                current_frame,
            )
        except PixelGoalRejected as rejected:
            rejected_attempts.append(
                {
                    "step_index": step_index,
                    "attempt_index": attempt_index,
                    "requested_uv": list(uv),
                    "visual_context": selection.visual_context,
                    "accepted": False,
                    "rejection_reason": rejected.reason,
                    "camera_snapshot_id": current_frame.camera_snapshot_id,
                    "camera_intrinsics_id": (
                        current_frame.camera_intrinsics_id
                    ),
                    "raw_world_hit_cm": (
                        _jsonable(rejected.raw_world_hit)
                        if rejected.raw_world_hit is not None
                        else None
                    ),
                    "audit": rejected.audit,
                    "before_frame": current_path.name,
                    "selected_frame": selected_path.name,
                }
            )
            current_frame = runtime.capture_frame(agent_tag=AGENT_TAG)
            current_path = output_dir / (
                f"fpv_step_{step_index:03d}_retry_{attempt_index:03d}.png"
            )
            _save_frame(current_frame.rgb_data_url, current_path)
            continue
        resolve = endpoint.calls[-1][2]
        initial_feet = list(resolve["initial_feet_position_cm"])
        samples = [initial_feet]
        status_count = 0
        deadline = time.monotonic() + runtime.config.execution_timeout_s
        final: ControllerResult | None = None
        while time.monotonic() < deadline:
            status = runtime.poll(started.request.request_id)
            status_count += 1
            if isinstance(status, ControllerResult):
                final = status
                break
            if not isinstance(status, PixelGoalInProgress):
                raise RuntimeError(f"unexpected Pixel Goal status: {status!r}")
            if status.feet_position is not None:
                feet = _jsonable(status.feet_position)
                if feet != samples[-1]:
                    samples.append(feet)
            time.sleep(runtime.config.poll_interval_s)
        if final is None:
            cancelled = runtime.cancel(
                started.request.request_id,
                reason="execution_timeout",
            )
            raise ManualSequenceTerminated(
                "execution_timeout",
                successful_steps=successful_steps,
                rejected_attempts=rejected_attempts,
                terminal={
                    "step_index": step_index,
                    "requested_uv": list(uv),
                    "visual_context": selection.visual_context,
                    "camera_snapshot_id": current_frame.camera_snapshot_id,
                    "camera_intrinsics_id": current_frame.camera_intrinsics_id,
                    "controller_result": cancelled.controller_result,
                    "failure_reason": cancelled.failure_reason,
                    "final_feet_position_cm": (
                        _jsonable(cancelled.final_feet_position)
                        if cancelled.final_feet_position is not None
                        else None
                    ),
                    "execution_error_planar_m": (
                        cancelled.execution_error_planar_m
                    ),
                    "execution_error_3d_m": cancelled.execution_error_3d_m,
                },
            )
        if final.controller_result != "success":
            raise ManualSequenceTerminated(
                "controller_failed",
                successful_steps=successful_steps,
                rejected_attempts=rejected_attempts,
                terminal={
                    "step_index": step_index,
                    "requested_uv": list(uv),
                    "visual_context": selection.visual_context,
                    "camera_snapshot_id": current_frame.camera_snapshot_id,
                    "camera_intrinsics_id": current_frame.camera_intrinsics_id,
                    "controller_result": final.controller_result,
                    "failure_reason": final.failure_reason,
                    "final_feet_position_cm": (
                        _jsonable(final.final_feet_position)
                        if final.final_feet_position is not None
                        else None
                    ),
                    "execution_error_planar_m": (
                        final.execution_error_planar_m
                    ),
                    "execution_error_3d_m": final.execution_error_3d_m,
                },
            )
        if final.final_feet_position is None or final.final_pose is None:
            raise RuntimeError("successful Pixel Goal omitted its final pose")
        final_feet = _jsonable(final.final_feet_position)
        if final_feet != samples[-1]:
            samples.append(final_feet)

        after_frame = runtime.capture_frame(agent_tag=AGENT_TAG)
        after_path = output_dir / f"fpv_{step_index:03d}.png"
        _save_frame(after_frame.rgb_data_url, after_path)
        successful_steps.append(
            {
                "index": step_index,
                "navigation_mode": "nav_pixel_goal",
                "requested_uv": list(uv),
                "visual_context": selection.visual_context,
                "camera_snapshot_id": current_frame.camera_snapshot_id,
                "camera_intrinsics_id": current_frame.camera_intrinsics_id,
                "raw_world_hit_cm": _jsonable(started.request.raw_world_hit),
                "validated_navigation_target_cm": _jsonable(
                    started.request.validated_navigation_target
                ),
                "navmesh_adjustment_cm": started.request.navmesh_adjustment_cm,
                "initial_feet_position_cm": initial_feet,
                "initial_feet_to_validated_target_planar_m": math.hypot(
                    started.request.validated_navigation_target.x_cm
                    - float(initial_feet[0]),
                    started.request.validated_navigation_target.y_cm
                    - float(initial_feet[1]),
                )
                / 100.0,
                "final_feet_position_cm": final_feet,
                "final_agent_position_cm": _jsonable(
                    final.final_pose.position
                ),
                "final_yaw_degrees": final.final_pose.yaw_deg,
                "controller_request_result": started.controller_request_result,
                "controller_result": final.controller_result,
                "execution_error_planar_m": final.execution_error_planar_m,
                "execution_error_3d_m": final.execution_error_3d_m,
                "distance_travelled_cm": final.distance_travelled_cm,
                "elapsed_sim_s": final.elapsed_sim_s,
                "feet_position_samples_cm": samples,
                "status_sample_count": status_count,
                "before_frame": current_path.name,
                "selected_frame": selected_path.name,
                "after_frame": after_path.name,
                "post_move_camera_snapshot_id": (
                    after_frame.camera_snapshot_id
                ),
                "post_move_camera_intrinsics_id": (
                    after_frame.camera_intrinsics_id
                ),
            }
        )
        current_frame = after_frame
        current_path = after_path

    return successful_steps, rejected_attempts


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.move_count <= 0:
        raise ValueError("move_count must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    simworld_root = Path(args.simworld_root).resolve()
    citycore_content = Path(args.citycore_content).resolve()
    validate_mount_inputs(citycore_content, simworld_root / "SimWorld.uproject")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log = output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)

    config = PixelGoalConfig(
        max_navmesh_adjustment_cm=args.max_navmesh_adjustment_cm,
        acceptance_radius_cm=args.acceptance_radius_cm,
        execution_timeout_s=args.execution_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    session = None
    play_started = False
    try:
        if args.launch_mode != "attach":
            raise ValueError("Paris Pixel Goal closed loop supports attach mode only")
        session = AttachedParisGameSession(args.spear_config)
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        preparation = _prepare_paris_loop(
            endpoint,
            build_paris_setup_request(RUE_DE_RIVOLI_SIDEWALK),
            readiness_timeout_s=args.navmesh_timeout_s,
            capture_warmup_s=args.capture_warmup_s,
            wait_fn=_wait_for_paris_poc,
            warm_up_fn=_warm_up_paris_capture,
        )
        runtime = LivePixelGoalRuntime(endpoint, config)
        try:
            successful_steps, rejected_attempts = execute_manual_sequence(
                runtime,
                endpoint,
                output_dir,
                move_count=args.move_count,
                select_uv=prompt_manual_uv,
            )
        except ManualSequenceTerminated as terminated:
            failure_summary = build_sequence_summary(
                terminated.successful_steps, endpoint.calls
            )
            failure_report: dict[str, object] = {
                "milestone": "Paris manual Pixel Navigation closed loop",
                "navigation_mode": "nav_pixel_goal",
                "scene": args.scene,
                "real_paris_content": True,
                "requested_move_count": args.move_count,
                "trial_protocol": {
                    "single_live_session": True,
                    "reset_between_moves": False,
                    "post_move_frame_is_next_observation": True,
                },
                "region": RUE_DE_RIVOLI_SIDEWALK.to_report(),
                "setup": preparation["setup"],
                "poc_status": preparation["status"],
                "strict_execution": dict(STRICT_EXECUTION),
                "config": asdict(config),
                "successful_steps": terminated.successful_steps,
                "rejected_attempts": terminated.rejected_attempts,
                "terminal": terminated.terminal,
                "summary": failure_summary,
                "acceptance": {
                    "passed": False,
                    "reason": terminated.reason,
                },
            }
            _write_json_atomic(
                output_dir / "closed_loop_failure.json", failure_report
            )
            raise
        summary = build_sequence_summary(successful_steps, endpoint.calls)
        report: dict[str, object] = {
            "milestone": "Paris manual Pixel Navigation closed loop",
            "navigation_mode": "nav_pixel_goal",
            "scene": args.scene,
            "real_paris_content": True,
            "requested_move_count": args.move_count,
            "trial_protocol": {
                "single_live_session": True,
                "reset_between_moves": False,
                "post_move_frame_is_next_observation": True,
            },
            "region": RUE_DE_RIVOLI_SIDEWALK.to_report(),
            "setup": preparation["setup"],
            "poc_status": preparation["status"],
            "rgb_capture": {
                "mode": preparation["setup"].get("rgb_capture_mode"),
                "render_state_persistent": preparation["setup"].get(
                    "rgb_capture_render_state_persistent"
                ),
                "warmup_config_s": args.capture_warmup_s,
                "warmup_wall_ms": preparation["capture_warmup_ms"],
            },
            "strict_execution": dict(STRICT_EXECUTION),
            "config": asdict(config),
            "successful_steps": successful_steps,
            "rejected_attempts": rejected_attempts,
            "summary": summary,
            "repo_root": str(repo_root),
            "simworld_root": str(simworld_root),
            "citycore_content": str(citycore_content),
        }
        report["acceptance"] = validate_closed_loop_report(
            report, expected_moves=args.move_count
        )
        _write_json_atomic(output_dir / "closed_loop_report.json", report)
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


def main() -> int:
    args = _build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run(args)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
