#!/usr/bin/env python3
"""Run the narrow live CityCore_Paris Pixel Goal Milestone 1B PoC."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalConfig,
    PixelGoalInProgress,
    PixelGoalRejected,
)
from embodiedbench.runtime.pixel_goal_paris_poc import (
    PARIS_POC_SCENE,
    RUE_DE_RIVOLI_SIDEWALK,
    ParisPocRegion,
    validate_paris_poc_report,
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


AGENT_TAG = "PixelGoalParisPocAgent"

# Initial choices for the known broad, flat Rue de Rivoli sidewalk.  They are
# policy actions only: no world coordinate, depth, graph node, or route enters
# any request.  Live evidence may justify tuning these normalized points before
# the final acceptance run, but not weakening the acceptance thresholds.
MANUAL_GOALS = [
    (0.50, 0.82),
    (0.62, 0.94),
    (0.50, 0.88),
    (0.48, 0.96),
    (0.52, 0.98),
]
INVALID_GOALS = [
    ("facade", (0.98, 0.45)),
    ("sky", (0.50, 0.05)),
]

STRICT_EXECUTION = {
    "project_goal_location": False,
    "allow_partial_path": False,
    "straight_line_fallback": False,
    "graph_projection": False,
    "pose_lattice_quantization": False,
}


class AttachedParisGameSession:
    """Bind SPEAR directly to an UnrealEditor ``-game`` world.

    Paris source assets are too heavy to safely start a full headless editor UI
    while they compile.  Milestone 1B does not need editor mutation or PIE: its
    UE subsystem creates the local runtime fixture entirely in the already
    playing game world.
    """

    def __init__(self, spear_config_path: str | None) -> None:
        import spear

        user_config_files = [spear_config_path] if spear_config_path else []
        config = spear.get_config(user_config_files=user_config_files)
        config.defrost()
        config.SPEAR.LAUNCH_MODE = "none"
        rpc_port = os.environ.get("SIMWORLD_RPC_PORT")
        if rpc_port:
            config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = int(rpc_port)
        config.freeze()
        spear.configure_system(config=config)
        self._instance = spear.Instance(config=config)
        self._game = None

    def begin_play(self) -> None:
        """Bind the game-scoped services; the process is already playing."""

        self._game = self._instance.get_game()
        if self._game is None:
            raise RuntimeError("spear.Instance.get_game() returned None")

    def end_play(self) -> None:
        """Release the local binding without attempting an editor PIE call."""

        self._game = None

    def wait_for_engine_idle(
        self,
        timeout_s: float,
        *,
        poll_interval_s: float = 0.25,
        consecutive_observations: int = 2,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    ) -> dict[str, object]:
        """Wait for the attached game world's real async/streaming services.

        Two idle observations separated by a short interval prevent a single
        transient zero-work sample from releasing the post-teleport capture.
        The later RGB stability gate remains authoritative for visible output.
        """

        if self._game is None:
            raise RuntimeError("cannot wait for engine idle without a game world")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("engine-idle timeout must be positive and finite")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError(
                "engine-idle poll interval must be positive and finite")
        if consecutive_observations < 1:
            raise ValueError("consecutive idle observations must be positive")
        service = getattr(self._game, "async_loading_service", None)
        if service is None:
            raise RuntimeError("SPEAR async loading service is unavailable")

        started = monotonic_fn()
        for observation in range(consecutive_observations):
            remaining = timeout_s - (monotonic_fn() - started)
            if remaining <= 0:
                raise TimeoutError("timed out waiting for UE engine idle")
            try:
                service.wait_for_engine_idle(
                    max_time_seconds=remaining,
                    sleep_time_seconds=min(poll_interval_s, remaining),
                )
            except AssertionError as error:
                raise TimeoutError(
                    "timed out waiting for UE engine idle") from error
            if observation + 1 < consecutive_observations:
                remaining = timeout_s - (monotonic_fn() - started)
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for UE engine idle")
                sleep_fn(min(poll_interval_s, remaining))
        return {
            "success": True,
            "consecutive_idle_observations": consecutive_observations,
            "wall_ms": max(0.0, (monotonic_fn() - started) * 1000.0),
        }

    def shutdown(self) -> None:
        self._game = None
        self._instance.close(force=False)


def build_paris_setup_request(region: ParisPocRegion) -> dict[str, object]:
    """Build the internal scene setup request, never a policy action."""

    return {
        "scene": PARIS_POC_SCENE,
        "region_name": region.name,
        "agent_spawn_cm": list(region.agent_spawn_cm),
        "agent_yaw_deg": region.agent_yaw_deg,
        "nav_bounds_center_cm": list(region.nav_bounds_center_cm),
        "nav_bounds_extent_cm": list(region.nav_bounds_extent_cm),
    }


def validate_mount_inputs(citycore_content: Path, simworld_project: Path) -> None:
    if not citycore_content.is_dir():
        raise FileNotFoundError(
            f"CityCore_Paris content directory is missing: {citycore_content}"
        )
    paris_map = citycore_content / "Scenes" / "ParisCity_FinalBlueprints.umap"
    if not paris_map.is_file():
        raise FileNotFoundError(f"Paris map asset is missing: {paris_map}")
    if not simworld_project.is_file():
        raise FileNotFoundError(
            f"SimWorld.uproject is missing: {simworld_project}"
        )


def _wait_for_paris_poc(
    endpoint: SpearPixelGoalEndpoint, timeout_s: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_status: dict[str, object] = {}
    while time.monotonic() < deadline:
        last_status = endpoint.call("PixelGoal_GetParisPocStatusJson", {})
        if last_status.get("poc_ready") is True:
            return last_status
        time.sleep(0.1)
    raise TimeoutError(f"Paris local NavMesh did not become ready: {last_status}")


def _warm_up_paris_capture(
    duration_s: float,
    *,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
) -> float:
    """Let the already initialized persistent capture accumulate view state."""

    started = monotonic_fn()
    sleep_fn(duration_s)
    return (monotonic_fn() - started) * 1000.0


def _reset_paris_poc_trial(
    endpoint: SpearPixelGoalEndpoint, timeout_s: float
) -> dict[str, object]:
    """Reset only the live PoC fixture between independent manual trials."""

    reset = endpoint.call("PixelGoal_ResetParisPocTrialJson", {})
    if reset.get("success") is not True:
        raise RuntimeError(f"Paris Pixel Goal trial reset failed: {reset}")
    ready = _wait_for_paris_poc(endpoint, timeout_s)
    return {**reset, **ready}


def _record_valid_goal(
    runtime: LivePixelGoalRuntime,
    endpoint: SpearPixelGoalEndpoint,
    output_dir: Path,
    index: int,
    uv: tuple[float, float],
) -> dict[str, object]:
    frame = runtime.capture_frame(agent_tag=AGENT_TAG)
    stem = f"paris_goal_{index:02d}"
    input_path = output_dir / f"{stem}.png"
    selected_path = output_dir / f"{stem}_selected.png"
    _save_frame(frame.rgb_data_url, input_path, selected_path, uv)

    started = runtime.start(
        NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]}), frame
    )
    resolve = endpoint.calls[-1][2]
    samples: list[list[float]] = []
    initial_feet = resolve.get("initial_feet_position_cm")
    if isinstance(initial_feet, list):
        samples.append(initial_feet)

    deadline = time.monotonic() + runtime.config.execution_timeout_s
    final: ControllerResult | None = None
    status_count = 0
    while time.monotonic() < deadline:
        status = runtime.poll(started.request.request_id)
        status_count += 1
        raw_status = endpoint.calls[-1][2]
        feet = raw_status.get("final_feet_position_cm")
        if isinstance(feet, list) and (not samples or feet != samples[-1]):
            samples.append(feet)
        if isinstance(status, ControllerResult):
            final = status
            break
        if not isinstance(status, PixelGoalInProgress):
            raise RuntimeError(f"unexpected Pixel Goal status: {status!r}")
        time.sleep(runtime.config.poll_interval_s)
    if final is None:
        runtime.cancel(started.request.request_id, reason="execution_timeout")
        raise TimeoutError(f"Paris Pixel Goal {index} timed out")
    if final.controller_result != "success":
        raise RuntimeError(
            f"Paris Pixel Goal {index} failed: {final.controller_result} "
            f"({final.failure_reason})"
        )

    after = runtime.capture_frame(agent_tag=AGENT_TAG)
    after_path = output_dir / f"{stem}_after.png"
    _save_frame(after.rgb_data_url, after_path)
    if len(samples) < 3:
        raise RuntimeError(
            f"Paris Pixel Goal {index} produced only {len(samples)} feet samples"
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
        "initial_feet_position_cm": resolve["initial_feet_position_cm"],
        "final_feet_position_cm": _jsonable(final.final_feet_position),
        "final_agent_position_cm": _jsonable(final.final_pose.position),
        "controller_request_result": started.controller_request_result,
        "controller_result": final.controller_result,
        "execution_error_planar_m": final.execution_error_planar_m,
        "execution_error_3d_m": final.execution_error_3d_m,
        "distance_travelled_cm": final.distance_travelled_cm,
        "elapsed_sim_s": final.elapsed_sim_s,
        "feet_position_samples_cm": samples,
        "status_sample_count": status_count,
        "input_frame": input_path.name,
        "selected_frame": selected_path.name,
        "post_move_frame": after_path.name,
        "post_move_camera_snapshot_id": after.camera_snapshot_id,
    }


def _record_invalid_goals(
    runtime: LivePixelGoalRuntime, output_dir: Path
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, (name, uv) in enumerate(INVALID_GOALS, start=1):
        frame = runtime.capture_frame(agent_tag=AGENT_TAG)
        stem = f"paris_invalid_{index:02d}_{name}"
        input_path = output_dir / f"{stem}.png"
        selected_path = output_dir / f"{stem}_selected.png"
        _save_frame(frame.rgb_data_url, input_path, selected_path, uv)
        try:
            runtime.start(
                NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]}),
                frame,
            )
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
                    "input_frame": input_path.name,
                    "selected_frame": selected_path.name,
                    "audit": rejected.audit,
                }
            )
        else:
            raise RuntimeError(f"invalid Paris case {name!r} was accepted")
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
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
            raise ValueError("Paris Pixel Goal PoC supports attach mode only")
        session = AttachedParisGameSession(args.spear_config)
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        setup = endpoint.call(
            "PixelGoal_SetupParisPocJson",
            build_paris_setup_request(RUE_DE_RIVOLI_SIDEWALK),
        )
        if setup.get("success") is not True:
            raise RuntimeError(f"Paris Pixel Goal setup failed: {setup}")
        if setup.get("synthetic_floor_spawned") is not False:
            raise RuntimeError("Paris setup did not prove synthetic floor absence")
        if (
            setup.get("dynamic_runtime_generation") is not True
            or setup.get("supports_runtime_generation") is not True
            or setup.get("navmesh_has_valid_data") is not True
            or int(setup.get("active_navmesh_tiles", 0)) < 1
        ):
            raise RuntimeError(
                f"Paris setup did not produce a populated dynamic NavMesh: {setup}"
            )
        status = _wait_for_paris_poc(endpoint, args.navmesh_timeout_s)
        capture_warmup_wall_ms = _warm_up_paris_capture(
            args.capture_warmup_s
        )
        runtime = LivePixelGoalRuntime(endpoint, config)

        invalid_cases = _record_invalid_goals(runtime, output_dir)
        manual_goals = []
        for index, uv in enumerate(MANUAL_GOALS, start=1):
            trial_reset = _reset_paris_poc_trial(
                endpoint, args.navmesh_timeout_s
            )
            goal = _record_valid_goal(
                runtime, endpoint, output_dir, index, uv
            )
            goal["trial_reset"] = trial_reset
            manual_goals.append(goal)
        report: dict[str, object] = {
            "milestone": "1B-PoC",
            "navigation_mode": "nav_pixel_goal",
            "scene": args.scene,
            "real_paris_content": True,
            "trial_protocol": {
                "independent_trials": True,
                "reset_between_valid_goals": True,
                "reset_is_policy_action": False,
            },
            "region": RUE_DE_RIVOLI_SIDEWALK.to_report(),
            "setup": setup,
            "poc_status": status,
            "rgb_capture": {
                "mode": setup.get("rgb_capture_mode"),
                "render_state_persistent": setup.get(
                    "rgb_capture_render_state_persistent"
                ),
                "warmup_config_s": args.capture_warmup_s,
                "warmup_wall_ms": capture_warmup_wall_ms,
            },
            "config": asdict(config),
            "strict_execution": dict(STRICT_EXECUTION),
            "manual_goals": manual_goals,
            "invalid_cases": invalid_cases,
            "repo_root": str(repo_root),
            "simworld_root": str(simworld_root),
            "citycore_content": str(citycore_content),
        }
        report["acceptance"] = validate_paris_poc_report(report)
        report_path = output_dir / "paris_poc_report.json"
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
        "--citycore-content",
        default=os.environ.get("CITYCORE_PARIS_CONTENT", ""),
    )
    parser.add_argument("--scene", default=PARIS_POC_SCENE, choices=[PARIS_POC_SCENE])
    parser.add_argument("--spear-config")
    parser.add_argument("--launch-mode", choices=("attach",), default="attach")
    parser.add_argument("--shutdown-attached-editor", action="store_true")
    parser.add_argument("--output", default="artifacts/pixel_goal_1b_poc")
    parser.add_argument("--max-navmesh-adjustment-cm", type=float, default=10.0)
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
    report = run(args)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
