#!/usr/bin/env python3
"""Run one full pickup -> dropoff delivery under real pixel-goal navigation.

This is the "can we actually do a whole delivery this way" proof of concept:
a real VLM drives the full ``CourierSession`` tool set (``walk_to_pixel``,
``collect``, ``hand_over``, ``check_order``, ``wait``, ...) against a real
running SPEAR Paris engine, with every step of locomotion resolved by picking
a visible ground point in the current photograph -- no waypoint-graph
jumping, no coordinate arithmetic.

It reuses, unmodified:

* the SPEAR attach/setup/warm-up/reset boilerplate every other pixel-goal PoC
  script in this directory already uses (``AttachedParisGameSession``,
  ``SpearPixelGoalEndpoint``, ``_prepare_paris_loop``,
  ``_reset_paris_poc_trial``);
* the courier harness's own real-VLM driving loop
  (the evaluator's ``ModelClient`` + ``parse_reply`` +
  ``CourierSession.step`` pattern) -- the model replies in the harness's own
  ``THOUGHT:`` + fenced-Python-call grammar, not the raw-JSON pixel format
  the other PoC scripts use, because it is driving the FULL tool set
  (``collect``/``hand_over`` included), not movement alone.

What is new is only the seam joining them: ``SpearTrackBClient`` (in
``tools/pixel_goal_courier_backend.py``) stands in for the SimWorld2
``UERenderClient`` that ``EmbodiedCourierEnv`` normally talks to over HTTP,
so the real courier task logic (orders, deadlines, ``collect``/
``hand_over``) runs on top of this repo's own working SPEAR RPC instead of
a backend that does not implement ``/walk_pixel`` yet. ``CorridorDeliveryEnv``
pins the spawn and the order's pickup/dropoff onto the one sidewalk corridor
this SPEAR session has a validated navmesh for -- see that module's
docstring for why the compiled graph's own addresses cannot be used as-is.

    python tools/run_pixel_goal_full_delivery.py \\
        --spear-config /path/to/spear/config.yaml \\
        --model-endpoint http://127.0.0.1:8210/v1/chat/completions \\
        --model qwen3-vl-4b
"""

from __future__ import annotations

import argparse
import os
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from embodiedbench.agent.courier.loop import parse_reply
from embodiedbench.agent.courier.model_io import ModelClient
from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL,
    CAMERA_VIEW_FORWARD,
)
from embodiedbench.runtime.live.protocol import CameraSpec, Pose
from embodiedbench.runtime.pixel_goal import LivePixelGoalRuntime, PixelGoalConfig
from embodiedbench.runtime.pixel_goal_paris_poc import PARIS_POC_SCENE, ParisPocRegion
from tools.pixel_goal_courier_backend import (
    DEFAULT_DROPOFF_OFFSET_CM,
    DEFAULT_PICKUP_OFFSET_CM,
    CorridorDeliveryEnv,
    SpearTrackBClient,
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
from tools.run_pixel_goal_m1a import (
    SpearPixelGoalEndpoint,
    _cleanup_session,
    _copy_latest_ue_log,
)

# Hardcoded engine-side on the Paris PoC spawn path (SpPixelGoalSubsystem.cpp's
# ParisPocAgentTag) -- not a free-choice label. Every other Paris PoC script in
# this directory uses this same literal string for the same reason.
AGENT_TAG = "PixelGoalParisPocAgent"

MAPS = REPO_ROOT / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"

# The same recon-corrected sidewalk X and a navmesh extent proven to build
# successfully (this is exactly what
# tmp/paris-interior-start-audit/run_audit.py's REGION used, before that
# script's later, separate widening of the bounds past this Setup call).
# See CorridorDeliveryEnv's docstring for why the compiled graph's own
# addresses cannot supply the pickup/dropoff kerb instead.
SIDEWALK_X_CM = -5_934.8349
SPAWN_Y_CM = -9_700.0
HEADING_DEGREES = 90.0


# Two real runs walked the model clean off the sidewalk corridor and onto a
# real street intersection (BP_CustomIntersectionGeneration7) just past the
# corridor's old north edge -- nine, then ten, straight walk_to_pixel hops
# covering 40-48 m. A first widening attempt kept the box CENTERED ON SPAWN
# and just grew it, which (a) tripped SpPixelGoalSubsystem's own hard cap of
# 5_000 cm per extent axis ("nav_bounds_not_local" -- a real UE-side limit,
# not a tuning knob) and (b) even at that cap, still didn't reach: the
# intersection's rejected hits landed as far as Y=-4_582 and X=-5_015,
# outside a box merely grown around Y=-9_700/X=-5_934.83. So this box is
# RECENTERED instead of just grown, splitting the difference between the
# spawn and the intersection so both fit inside one 5_000-cm-capped axis
# with margin (the engine caps each nav-bounds axis at 5000 cm).
NAV_BOUNDS_CENTER_CM = (-5_600.0, -7_100.0, 100.0)
NAV_BOUNDS_EXTENT_CM = (1_000.0, 5_000.0, 300.0)

DELIVERY_REGION = ParisPocRegion(
    name="paris_pixel_goal_full_delivery_corridor",
    agent_spawn_cm=(SIDEWALK_X_CM, SPAWN_Y_CM, 90.0),
    agent_yaw_deg=HEADING_DEGREES,
    nav_bounds_center_cm=NAV_BOUNDS_CENTER_CM,
    nav_bounds_extent_cm=NAV_BOUNDS_EXTENT_CM,
)


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def run_delivery(args: argparse.Namespace) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    repo_root = REPO_ROOT
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
    camera = CameraSpec(width=args.capture_width, height=args.capture_height,
                        fov_deg=config.fov_degrees)

    session = None
    play_started = False
    transcript: list[dict[str, Any]] = []
    report: dict[str, Any] | None = None
    try:
        session = AttachedParisGameSession(args.spear_config)
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        preparation = _prepare_paris_loop(
            endpoint, build_paris_setup_request(DELIVERY_REGION),
            readiness_timeout_s=args.navmesh_timeout_s,
            capture_warmup_s=args.capture_warmup_s,
            wait_fn=_wait_for_paris_poc, warm_up_fn=_warm_up_paris_capture,
        )
        _reset_paris_poc_trial(endpoint, args.navmesh_timeout_s)

        runtime = LivePixelGoalRuntime(endpoint, config)
        spawn_pose = Pose(
            x_cm=DELIVERY_REGION.agent_spawn_cm[0],
            y_cm=DELIVERY_REGION.agent_spawn_cm[1],
            z_cm=DELIVERY_REGION.agent_spawn_cm[2],
            yaw_deg=DELIVERY_REGION.agent_yaw_deg,
        )
        client = SpearTrackBClient(runtime, agent_tag=AGENT_TAG, spawn_pose=spawn_pose)

        paris = build_road_network(MAPS, map_name="citycore-paris")
        env = CorridorDeliveryEnv(
            paris, client,
            street_camera=camera,
            episode_id="pixel-goal-full-delivery",
            cache_root=output_dir / "album",
            action_space=ACTION_SPACE_PIXEL_GOAL,
            camera_view=CAMERA_VIEW_FORWARD,
            embodiment="human_on_foot",
            difficulty="solo",
            seed=args.seed,
            max_step_m=args.max_step_m,
            pickup_offset_cm=args.pickup_offset_cm,
            dropoff_offset_cm=args.dropoff_offset_cm,
        )
        env.reset()
        courier_session = CourierSession(env, city="Paris")
        system_prompt = courier_session.system_prompt()

        model = ModelClient(
            args.model_endpoint, args.model,
            max_tokens=args.max_tokens, max_requeries=args.max_requeries,
            max_tokens_ceiling=max(args.max_tokens * 4, 8192),
        )

        history: list[dict[str, Any]] = []
        for turn in range(1, args.max_turns + 1):
            if courier_session.finished:
                break
            observation = courier_session.observe()
            images = [data_url(Path(p)) for p in observation.image_paths]
            content: list[dict[str, Any]] = [{"type": "text", "text": observation.text}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
            history.append({"role": "user", "content": content})
            keep = args.history_turns * 2
            recent = history[-keep:] if keep > 0 else [history[-1]]
            stripped = []
            for i, message in enumerate(recent):
                if (message["role"] == "user" and isinstance(message["content"], list)
                        and i < len(recent) - 1):
                    text = next((c["text"] for c in message["content"]
                               if c["type"] == "text"), "")
                    stripped.append({"role": "user", "content": text})
                else:
                    stripped.append(message)
            messages = [{"role": "system", "content": system_prompt}, *stripped]

            started = time.time()
            try:
                reply, parsed, rejected = model.act(
                    messages,
                    lambda text: parse_reply(text, set(courier_session.allowed)))
                if parsed is None:
                    reply = reply or "(no parseable action)"
                infra_error = None
            except Exception as error:  # noqa: BLE001 — a failed request is not a model output
                infra_error = f"{type(error).__name__}: {error}"
                reply = None

            if infra_error is not None:
                transcript.append({
                    "turn": turn, "status": "infra_error", "action": None,
                    "error": infra_error, "latency_s": round(time.time() - started, 1),
                    "reply": None,
                })
                logger.error("turn %d: request failed: %s", turn, infra_error)
                break

            history.append({"role": "assistant", "content": reply})
            log = courier_session.step(reply)
            transcript.append({
                "turn": turn, "status": log.status, "action": log.action,
                "error": log.error, "latency_s": round(time.time() - started, 1),
                "reply": reply[:400], "requeries": len(rejected),
                "image_paths": observation.image_paths,
            })
            logger.info("turn %3d %-28s %-13s %5.1fs",
                       turn, str(log.action), log.status, time.time() - started)

        summary = env.summary()
        report = {
            "experiment": "Qwen pixel-goal full delivery (real SPEAR engine)",
            "region": DELIVERY_REGION.to_report(),
            "pickup_offset_cm": args.pickup_offset_cm,
            "dropoff_offset_cm": args.dropoff_offset_cm,
            "model": {"id": args.model, "endpoint": args.model_endpoint,
                     "max_tokens": args.max_tokens, "system_prompt": system_prompt},
            "setup": preparation["setup"],
            "config": {"max_navmesh_adjustment_cm": config.max_navmesh_adjustment_cm,
                      "acceptance_radius_cm": config.acceptance_radius_cm,
                      "execution_timeout_s": config.execution_timeout_s},
            "turns": len(transcript),
            "transcript": transcript,
            "summary": summary,
            "termination": (courier_session.run.termination_reason
                           or ("out_of_turns" if not courier_session.finished else "")),
            "model_transport_stats": model.stats.as_dict(),
        }
        _write_json_atomic(output_dir / "delivery_report.json", report)
        return report
    except Exception:
        failure = {"transcript": transcript, "region": DELIVERY_REGION.to_report()}
        _write_json_atomic(output_dir / "delivery_failure.json", failure)
        raise
    finally:
        if session is not None:
            _cleanup_session(
                session, launch_mode=args.launch_mode,
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
    parser.add_argument("--output", default="artifacts/pixel_goal_full_delivery")
    parser.add_argument("--model-endpoint",
                        default="http://127.0.0.1:8210/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-vl-4b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-requeries", type=int, default=3)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--max-step-m", type=float, default=10.0)
    parser.add_argument("--pickup-offset-cm", type=float, default=DEFAULT_PICKUP_OFFSET_CM)
    parser.add_argument("--dropoff-offset-cm", type=float, default=DEFAULT_DROPOFF_OFFSET_CM)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=360)
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
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_delivery(args)
    print(json.dumps({
        "turns": report["turns"], "termination": report["termination"],
        "delivered": report["summary"].get("delivered"),
        "orders_issued": report["summary"].get("orders_issued"),
        "sim_seconds": report["summary"].get("sim_seconds"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
