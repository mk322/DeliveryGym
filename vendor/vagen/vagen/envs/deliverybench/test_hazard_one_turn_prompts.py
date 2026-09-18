"""One-turn FPV hazard prompt tests for DeliveryBench.

This is a deterministic, no-model harness for checking the *inputs* and one-step
action mechanics used by VLM rollouts:

* obstacle front view -> PASSBY() succeeds without collision
* red traffic-light front view -> WAIT(minutes=1) succeeds
* green traffic-light front view -> MOVE(direction="forward") succeeds without
  traffic violation

Each case places the agent as if a NAVIGATE route wants the forward edge, then
builds the real system prompt + observation, including the FPV cross and PIL
2D-map image. Results and prompt/image fixtures are written to
``outputs/hazard_one_turn_<timestamp>/``.

Run:
    PYTHONPATH=. python -m vagen.envs.deliverybench.test_hazard_one_turn_prompts
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .deliverybench_env import DeliveryBench, NAV_PRESET
from .vlm_delivery.actions.move import available_moves
from .vlm_delivery.base.defs import TransportMode
from .vlm_delivery.gameplay.action_space import effective_enabled_actions
from .vlm_delivery.utils.hazards import TrafficController, normalize_axis
from .vlm_delivery.utils.transport import transport_set_mode


_HERE = Path(__file__).resolve().parent
_FPV_ROOT = _HERE / "deliverybench_fpv" / "small-city-11-new"
_HIRES = _FPV_ROOT / "main_base_floor_road_full_1280x960"
_MANIFEST = _HIRES / "manifest.jsonl"
_OBSTACLES = _HIRES / "obstacles.json"
_IMAGES = _FPV_ROOT / "images"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _image_path_for_row(fpv_dir: Path, row: Mapping[str, Any]) -> Path:
    wp_id = str(row["waypoint_id"])
    kind, num = wp_id.rsplit("_", 1)
    dir_name = f"{kind}_{int(num):03d}"
    return fpv_dir / "images" / dir_name / Path(str(row.get("image_path") or "")).name


def _prepare_fpv_dir() -> Path:
    """Build a coherent temp dataset: hi-res manifest + current image tree."""
    if not _MANIFEST.exists():
        raise FileNotFoundError(f"missing manifest: {_MANIFEST}")
    if not _OBSTACLES.exists():
        raise FileNotFoundError(f"missing obstacles sidecar: {_OBSTACLES}")
    if not _IMAGES.exists():
        raise FileNotFoundError(f"missing image tree: {_IMAGES}")

    root = Path(tempfile.mkdtemp(prefix="vagen_hazard_one_turn_"))
    shutil.copy2(_MANIFEST, root / "manifest.jsonl")
    shutil.copy2(_OBSTACLES, root / "obstacles.json")
    (root / "images").symlink_to(_IMAGES, target_is_directory=True)
    return root


def _base_cfg(fpv_dir: Path):
    return replace(
        NAV_PRESET,
        map_name="small-city-11",
        max_steps=8,
        render_mode="vision",
        enable_fpv=True,
        fpv_dir=str(fpv_dir),
        enable_map_images=True,
        map_renderer="pil",
        use_gmaps_renderer=True,
        gmaps_out_scale=0.22,
        enable_obstacles=True,
        enable_traffic_lights=True,
        enable_pedestrian_traffic_lights=False,
        traffic_light_require_visible_signal_view=False,
        enabled_actions=["MOVE", "WAIT", "NAVIGATE"],
        initial_transport_mode="walk",
        enable_feasible_orders=False,
        enable_infeasible_orders=True,
    )


def _setup_env(fpv_dir: Path) -> DeliveryBench:
    """Use synchronous reset to avoid executor hangs in some minimal shells."""
    cfg = _base_cfg(fpv_dir)
    env = DeliveryBench(cfg)
    env._env = env._create_env()
    env._env.reset(seed=90)
    env.total_reward = 0.0
    env.last_action = None
    env.last_action_result = {}
    env.deliveries_completed = 0
    env._recent_parsed_actions = []
    env._load_fpv_lookup()

    dm = env._env.dms[0]
    dm.cfg["enabled_actions"] = cfg.enabled_actions
    dm.cfg["enable_obstacles"] = True
    dm.cfg["enable_traffic_lights"] = True
    dm.cfg["enable_pedestrian_traffic_lights"] = False
    dm.cfg["passby_cost_scale"] = cfg.passby_cost_scale
    dm.cfg["traffic_lights"] = {
        "control_radius_cm": cfg.traffic_light_control_radius_cm,
        "red_light_penalty_s": cfg.traffic_light_red_penalty_s,
        "red_light_energy_multiplier": cfg.traffic_light_red_energy_multiplier,
        "require_visible_signal_view": False,
        "fpv_yaw_offset_deg": cfg.fpv_yaw_offset_deg,
    }
    dm.collision_count = 0
    dm.traffic_violation_count = 0
    env._load_hazards(dm)
    transport_set_mode(dm, TransportMode("walk"))
    return env


def _set_clock(dm: Any, sim_s: float) -> None:
    dm.clock._base_real = time.monotonic()
    dm.clock._base_sim = float(sim_s)


def _snap_yaw(yaw: float) -> float:
    return min(
        [0.0, 90.0, 180.0, 270.0],
        key=lambda y: min(abs(y - float(yaw)), 360.0 - abs(y - float(yaw))),
    )


def _stored_yaw_for_compass(compass_deg: float, offset: float = 90.0) -> float:
    return _snap_yaw((float(offset) - float(compass_deg)) % 360.0)


def _state_time(axis: str, state: str) -> float:
    axis = normalize_axis(axis)
    state = str(state).lower()
    # even minute: NS green / EW red; odd minute: NS red / EW green
    if axis == "NS":
        return 60.0 if state == "red" else 0.0
    return 0.0 if state == "red" else 60.0


def _place_agent(env: DeliveryBench, x_cm: float, y_cm: float, facing_deg: float) -> Optional[Dict[str, Any]]:
    dm = env._env.dms[0]
    city_map = dm.city_map
    node = city_map.nearest_waypoint(float(x_cm), float(y_cm))
    if node is None:
        return None
    dm.x = float(node.position.x)
    dm.y = float(node.position.y)
    dm.facing_deg = float(facing_deg) % 360.0
    env._prev_dm_x = float(dm.x)
    env._prev_dm_y = float(dm.y)
    dm.collision_count = 0
    dm.traffic_violation_count = 0
    dm.energy_pct = float(dm.cfg.get("energy_pct_max", 100.0))
    dm.is_rescued = False
    dm._hospital_ctx = None
    dm._bus_ctx = None
    dm.vlm_clear_errors()

    forward = available_moves(dm).get("forward")
    if forward is None:
        dm._nav_target_node = None
        dm._nav_route_path = []
        dm._nav_route_color = None
        return {"node": node, "forward": None}

    target = forward["node"]
    dm._nav_target_node = target
    dm._nav_route_color = (26, 115, 232, 255)
    try:
        path, _ = city_map.waypoint_graph.shortest_path_nodes(node, target)
    except Exception:
        path = [node, target]
    dm._nav_route_path = path or [node, target]
    dm.vlm_ephemeral["navigation"] = (
        "[navigation]\n"
        "mode: walk\n"
        f"from: {getattr(node, 'waypoint_name', getattr(node, 'waypoint_id', '?'))}\n"
        f"to: {getattr(target, 'waypoint_name', getattr(target, 'waypoint_id', '?'))}\n"
        f"distance_m: {float(forward.get('dist_m', 0.0)):.1f}\n"
        "estimated_time: one-step test route\n"
        "next_move: move forward"
    )
    return {"node": node, "forward": forward}


def _build_obs(env: DeliveryBench) -> Dict[str, Any]:
    return asyncio.run(env._build_observation({}, {}, init_obs=False))


def _step(env: DeliveryBench, action: str) -> Tuple[Optional[str], Dict[str, Any]]:
    dm = env._env.dms[0]
    _obs, _reward, _terminated, _truncated, raw_info = env._env.step(action)
    err = raw_info.get("error") or getattr(dm, "vlm_errors", None)
    if getattr(dm, "vlm_errors", None):
        dm.vlm_clear_errors()
    metrics = {
        "collisions": int(getattr(dm, "collision_count", 0)),
        "traffic_violations": int(getattr(dm, "traffic_violation_count", 0)),
        "sim_s": round(float(dm.clock.now_sim()), 3),
        "x_cm": round(float(dm.x), 1),
        "y_cm": round(float(dm.y), 1),
    }
    return err, metrics


def _obs_images(obs: Mapping[str, Any]) -> List[Any]:
    mmi = obs.get("multi_modal_input") or {}
    return [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]


def _write_case_fixture(
    *,
    out_dir: Path,
    case_id: str,
    system_prompt: str,
    obs: Mapping[str, Any],
    save_images: bool,
) -> Dict[str, Any]:
    case_dir = out_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(
        system_prompt + "\n\n### observation\n" + str(obs.get("obs_str", "")),
        encoding="utf-8",
    )
    images_meta = []
    for idx, im in enumerate(_obs_images(obs)):
        rel = Path("cases") / case_id / f"image_{idx}.png"
        if save_images:
            im.save(out_dir / rel)
        images_meta.append({"path": str(rel), "size": list(im.size)})
    return {"prompt": str(Path("cases") / case_id / "prompt.txt"), "images": images_meta}


def _case_ok_prompt(system_prompt: str, obs: Mapping[str, Any], required: Iterable[str]) -> bool:
    text = system_prompt + "\n" + str(obs.get("obs_str", ""))
    return all(token in text for token in required)


def run(save_images: bool = True) -> int:
    fpv_dir = _prepare_fpv_dir()
    out_dir = _HERE / "outputs" / f"hazard_one_turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    rows = _load_jsonl(fpv_dir / "manifest.jsonl")
    obstacle_rows = [r for r in rows if str(r.get("render_kind", "")).lower() == "obstacle"]
    light_rows = [r for r in rows if str(r.get("render_kind", "")).lower() == "traffic_light"]
    missing_images = [
        {
            "waypoint_id": r.get("waypoint_id"),
            "render_kind": r.get("render_kind"),
            "signal_state": r.get("signal_state"),
            "yaw": r.get("yaw"),
            "expected_path": str(_image_path_for_row(fpv_dir, r)),
        }
        for r in obstacle_rows + light_rows
        if not _image_path_for_row(fpv_dir, r).exists()
    ]

    env = _setup_env(fpv_dir)
    system_prompt = asyncio.run(env.system_prompt())["obs_str"]
    action_set = effective_enabled_actions(env.config.__dict__) or []
    summary = {
        "manifest": str(_MANIFEST),
        "fpv_dir": str(fpv_dir),
        "output_dir": str(out_dir),
        "action_set_has_passby": "PASSBY" in set(action_set),
        "action_set_has_wait": "WAIT" in set(action_set),
        "obstacle_rows": len(obstacle_rows),
        "traffic_light_rows": len(light_rows),
        "missing_hazard_images": len(missing_images),
        "cases": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "missing_images_sample": missing_images[:10],
    }

    with results_path.open("w", encoding="utf-8") as fh:
        def record(rec: Dict[str, Any]) -> None:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            summary["cases"] += 1
            summary[rec["status"]] += 1

        # Obstacles: one row per blocked front view.
        for idx, row in enumerate(obstacle_rows):
            case_id = f"obstacle_{idx:03d}_{row['waypoint_id']}_yaw_{int(float(row['yaw'])):03d}"
            image_path = _image_path_for_row(fpv_dir, row)
            if not image_path.exists():
                record({"case_id": case_id, "kind": "obstacle", "status": "skipped", "reason": "missing_image", "image": str(image_path)})
                continue

            bearing = float(row.get("bearing_deg", (90.0 - float(row["yaw"])) % 360.0))
            placed = _place_agent(env, float(row["x_cm"]), float(row["y_cm"]), bearing)
            if not placed or placed.get("forward") is None:
                record({"case_id": case_id, "kind": "obstacle", "status": "skipped", "reason": "no_forward_edge"})
                continue

            _set_clock(env._env.dms[0], 0.0)
            obs = _build_obs(env)
            fixture = _write_case_fixture(
                out_dir=out_dir,
                case_id=case_id,
                system_prompt=system_prompt,
                obs=obs,
                save_images=save_images,
            )
            err, metrics = _step(env, "PASSBY()")
            prompt_ok = _case_ok_prompt(system_prompt, obs, ["PASSBY", "next_move: move forward"])
            ok = err is None and metrics["collisions"] == 0 and prompt_ok
            record({
                "case_id": case_id,
                "kind": "obstacle",
                "expected_action": "PASSBY()",
                "status": "passed" if ok else "failed",
                "error": err,
                "metrics": metrics,
                "prompt_ok": prompt_ok,
                "fixture": fixture,
            })

        # Traffic lights: every red/green row whose stored yaw corresponds to a
        # real forward edge. Red expects WAIT; green expects MOVE with no violation.
        for idx, row in enumerate(light_rows):
            state = str(row.get("signal_state", "")).lower()
            yaw = float(row["yaw"])
            case_id = (
                f"traffic_{state}_{idx:03d}_{row['waypoint_id']}"
                f"_yaw_{int(yaw):03d}"
            )
            image_path = _image_path_for_row(fpv_dir, row)
            if not image_path.exists():
                record({"case_id": case_id, "kind": "traffic_light", "status": "skipped", "reason": "missing_image", "image": str(image_path)})
                continue

            facing = (90.0 - yaw) % 360.0
            placed = _place_agent(env, float(row["x_cm"]), float(row["y_cm"]), facing)
            if not placed or placed.get("forward") is None:
                record({"case_id": case_id, "kind": "traffic_light", "status": "skipped", "reason": "no_forward_edge"})
                continue

            axis = normalize_axis(row.get("signal_axis"))
            _set_clock(env._env.dms[0], _state_time(axis, state))
            # Sanity-check the runtime state for the intended forward bearing.
            forward_bearing = float(placed["forward"].get("bearing_deg", facing))
            runtime_state = env._env.dms[0]._traffic.light_for_bearing(
                forward_bearing,
                float(env._env.dms[0].clock.now_sim()),
            )
            obs = _build_obs(env)
            fixture = _write_case_fixture(
                out_dir=out_dir,
                case_id=case_id,
                system_prompt=system_prompt,
                obs=obs,
                save_images=save_images,
            )

            if state == "red":
                err, metrics = _step(env, 'WAIT(minutes=1)')
                expected_action = 'WAIT(minutes=1)'
                prompt_ok = _case_ok_prompt(system_prompt, obs, ["WAIT", "next_move: move forward"])
                ok = err is None and metrics["traffic_violations"] == 0 and prompt_ok and runtime_state == "red"
            elif state == "green":
                err, metrics = _step(env, 'MOVE(direction="forward")')
                expected_action = 'MOVE(direction="forward")'
                prompt_ok = _case_ok_prompt(system_prompt, obs, ["MOVE", "next_move: move forward"])
                ok = err is None and metrics["traffic_violations"] == 0 and prompt_ok and runtime_state == "green"
            else:
                record({"case_id": case_id, "kind": "traffic_light", "status": "skipped", "reason": f"unknown_state:{state}"})
                continue

            record({
                "case_id": case_id,
                "kind": "traffic_light",
                "signal_state": state,
                "signal_axis": axis,
                "runtime_state": runtime_state,
                "expected_action": expected_action,
                "status": "passed" if ok else "failed",
                "error": err,
                "metrics": metrics,
                "prompt_ok": prompt_ok,
                "fixture": fixture,
            })

    env._env.close()
    summary["results_jsonl"] = str(results_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(run(save_images=True))
