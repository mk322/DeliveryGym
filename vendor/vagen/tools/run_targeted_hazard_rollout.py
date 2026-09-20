"""Targeted local-Qwen rollout for DeliveryBench hazard checkpoints.

This runner uses the real DeliveryBench environment and real FPV hazard images,
but stages a short, deterministic "Path A" through hand-picked checkpoints:

1. A blocked forward edge, where the safe action is PASSBY().
2. A signalized forward crossing on red, where the safe action is WAIT(minutes=...).
3. The same signalized crossing after the wait flips it green, where the safe
   action is MOVE(direction="forward").

It writes a JSONL trace and the images sent to the model under
vagen/envs/deliverybench/outputs/targeted_hazard_rollout_<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import AsyncOpenAI

from vagen.envs.deliverybench.deliverybench_env import DeliveryBench, STAGE_1_CONFIG
from vagen.envs.deliverybench.vlm_delivery.actions.move import available_moves
from vagen.envs.deliverybench.vlm_delivery.base.defs import TransportMode
from vagen.envs.deliverybench.vlm_delivery.utils.hazards import normalize_axis
from vagen.envs.deliverybench.vlm_delivery.utils.transport import transport_set_mode


HERE = Path(__file__).resolve().parents[1] / "vagen" / "envs" / "deliverybench"
FPV_ROOT = HERE / "deliverybench_fpv" / "small-city-11"
HAZARD_ROOT = FPV_ROOT / "main_base_floor_road_full_1280x960"
OUT_ROOT = HERE / "outputs"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_fpv_dir() -> Path:
    manifest = HAZARD_ROOT / "manifest.jsonl"
    obstacles = HAZARD_ROOT / "obstacles.json"
    images = FPV_ROOT / "images"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not obstacles.exists():
        raise FileNotFoundError(obstacles)
    if not images.exists():
        raise FileNotFoundError(images)

    root = Path(tempfile.mkdtemp(prefix="vagen_targeted_hazard_"))
    shutil.copy2(manifest, root / "manifest.jsonl")
    shutil.copy2(obstacles, root / "obstacles.json")
    (root / "images").symlink_to(images, target_is_directory=True)
    return root


def base_config(fpv_dir: Path):
    return replace(
        STAGE_1_CONFIG,
        map_name="small-city-11",
        max_steps=12,
        render_mode="vision",
        enable_fpv=True,
        fpv_dir=str(fpv_dir),
        enable_map_images=False,
        map_renderer="pil",
        use_gmaps_renderer=False,
        enable_obstacles=True,
        enable_traffic_lights=True,
        enable_pedestrian_traffic_lights=False,
        traffic_light_require_visible_signal_view=False,
        enabled_actions=["MOVE", "WAIT", "NAVIGATE", "PASSBY"],
        initial_transport_mode="walk",
        enable_feasible_orders=False,
        enable_infeasible_orders=True,
    )


def set_clock(dm: Any, sim_s: float) -> None:
    dm.clock._base_real = time.monotonic()
    dm.clock._base_sim = float(sim_s)


def state_time(axis: str, state: str) -> float:
    axis = normalize_axis(axis)
    state = str(state).lower()
    # even minute: NS green / EW red; odd minute: NS red / EW green
    if axis == "NS":
        return 60.0 if state == "red" else 0.0
    return 0.0 if state == "red" else 60.0


def place_agent(env: DeliveryBench, x_cm: float, y_cm: float, facing_deg: float) -> Optional[Dict[str, Any]]:
    dm = env._env.dms[0]
    node = dm.city_map.nearest_waypoint(float(x_cm), float(y_cm))
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
        path, _ = dm.city_map.waypoint_graph.shortest_path_nodes(node, target)
    except Exception:
        path = [node, target]
    dm._nav_route_path = path or [node, target]
    dm.vlm_ephemeral["navigation"] = (
        "[navigation]\n"
        "mode: walk\n"
        f"from: {getattr(node, 'waypoint_name', getattr(node, 'waypoint_id', '?'))}\n"
        f"to: {getattr(target, 'waypoint_name', getattr(target, 'waypoint_id', '?'))}\n"
        f"distance_m: {float(forward.get('dist_m', 0.0)):.1f}\n"
        "estimated_time: one-step Path A checkpoint\n"
        "next_move: move forward"
    )
    return {"node": node, "forward": forward}


def facing_for_row(row: Mapping[str, Any]) -> float:
    if "bearing_deg" in row:
        return float(row["bearing_deg"]) % 360.0
    return (90.0 - float(row["yaw"])) % 360.0


def choose_case(env: DeliveryBench, rows: Iterable[Dict[str, Any]], *, kind: str, state: Optional[str] = None) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("render_kind", "")).lower() != kind:
            continue
        if state is not None and str(row.get("signal_state", "")).lower() != state:
            continue
        placed = place_agent(env, float(row["x_cm"]), float(row["y_cm"]), facing_for_row(row))
        if placed and placed.get("forward") is not None:
            return row
    raise RuntimeError(f"no usable {kind} row found")


def images_from_obs(obs: Mapping[str, Any]) -> List[Any]:
    mmi = obs.get("multi_modal_input") or {}
    return [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]


def image_data_url(img: Any, max_side: int) -> str:
    im = img.convert("RGB")
    scale = min(1.0, float(max_side) / max(im.size))
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_user_text(obs: Mapping[str, Any], stage_prompt: str = "") -> str:
    text = str(obs.get("obs_str", "")).replace("<image>", "").strip()
    return text


def obs_to_content(obs: Mapping[str, Any], stage_prompt: str, max_image_side: int) -> List[Dict[str, Any]]:
    user_text = build_user_text(obs, stage_prompt)
    parts: List[Dict[str, Any]] = []
    for img in images_from_obs(obs):
        parts.append({"type": "image_url", "image_url": {"url": image_data_url(img, max_image_side)}})
    parts.append({"type": "text", "text": user_text})
    return parts


def save_images(obs: Mapping[str, Any], case_dir: Path) -> List[str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, img in enumerate(images_from_obs(obs)):
        path = case_dir / f"image_{idx}.png"
        img.save(path)
        paths.append(str(path))
    return paths


def hazard_state(env: DeliveryBench, placed: Mapping[str, Any]) -> Dict[str, Any]:
    dm = env._env.dms[0]
    forward = placed.get("forward") or {}
    node = placed.get("node")
    target = forward.get("node")
    tx = float(getattr(getattr(target, "position", None), "x", dm.x)) if target else float(dm.x)
    ty = float(getattr(getattr(target, "position", None), "y", dm.y)) if target else float(dm.y)
    out = {
        "from_waypoint": getattr(node, "waypoint_id", None),
        "to_waypoint": getattr(target, "waypoint_id", None) if target else None,
        "forward_bearing_deg": float(forward.get("bearing_deg", dm.facing_deg)) if forward else None,
    }
    obstacles = getattr(dm, "_obstacle_field", None)
    if obstacles is not None and target is not None:
        out["obstacle_forward"] = obstacles.obstacle_on(float(dm.x), float(dm.y), tx, ty)
    traffic = getattr(dm, "_traffic", None)
    if traffic is not None:
        controlled = traffic.is_signalised(float(dm.x), float(dm.y))
        out["traffic_controlled_here"] = controlled
        if controlled and forward:
            out["traffic_forward_state"] = traffic.light_for_bearing(
                float(forward.get("bearing_deg", dm.facing_deg)),
                float(dm.clock.now_sim()),
            )
    return out


async def run_case(
    *,
    env: DeliveryBench,
    client: AsyncOpenAI,
    model: str,
    system_text: str,
    stage_id: str,
    stage_prompt: str,
    placed: Mapping[str, Any],
    out_dir: Path,
    max_image_side: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    obs = await env._build_observation({}, {}, init_obs=False)
    image_paths = save_images(obs, out_dir / "images" / stage_id)
    user_text = build_user_text(obs, stage_prompt)
    input_dir = out_dir / "inputs" / stage_id
    input_dir.mkdir(parents=True, exist_ok=True)
    system_prompt_path = input_dir / "system_prompt.txt"
    user_prompt_path = input_dir / "user_prompt.txt"
    request_path = input_dir / "request.json"
    system_prompt_path.write_text(system_text, encoding="utf-8")
    user_prompt_path.write_text(user_text, encoding="utf-8")
    request_path.write_text(
        json.dumps(
            {
                "stage_id": stage_id,
                "system_prompt_path": str(system_prompt_path),
                "user_prompt_path": str(user_prompt_path),
                "image_paths": image_paths,
                "system_prompt": system_text,
                "user_text": user_text,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": obs_to_content(obs, stage_prompt, max_image_side)},
    ]
    before = hazard_state(env, placed)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    model_text = response.choices[0].message.content or ""
    obs2, reward, done, info = await env.step(model_text)
    dm = env._env.dms[0]
    metrics = (info.get("metrics") or {}).get("traj_metrics", {})
    return {
        "stage_id": stage_id,
        "stage_prompt": stage_prompt,
        "input": {
            "system_prompt_path": str(system_prompt_path),
            "user_prompt_path": str(user_prompt_path),
            "request_path": str(request_path),
            "image_paths": image_paths,
            "user_text": user_text,
        },
        "hazard_before": before,
        "model_response": model_text,
        "parsed_action": (info.get("parsed") or {}).get("action"),
        "action_error": info.get("action_error"),
        "reward": reward,
        "done": done,
        "metrics": metrics,
        "position_after_cm": [round(float(dm.x), 1), round(float(dm.y), 1)],
        "facing_after_deg": round(float(dm.facing_deg), 1),
        "sim_s_after": round(float(dm.clock.now_sim()), 3),
        "images": image_paths,
        "obs_after_excerpt": str(obs2.get("obs_str", ""))[:1000],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    fpv_dir = prepare_fpv_dir()
    run_dir = OUT_ROOT / f"targeted_hazard_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    env = DeliveryBench(base_config(fpv_dir))
    system_text = (await env.system_prompt())["obs_str"]
    await env.reset(seed=90)
    dm = env._env.dms[0]
    transport_set_mode(dm, TransportMode("walk"))

    rows = load_jsonl(fpv_dir / "manifest.jsonl")
    obstacle_row = choose_case(env, rows, kind="obstacle")
    red_row = choose_case(env, rows, kind="traffic_light", state="red")

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180)
    records: List[Dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as fh:
        def emit(record: Dict[str, Any]) -> None:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(record)

        emit({
            "event": "run_meta",
            "run_dir": str(run_dir),
            "model": args.model,
            "base_url": args.base_url,
            "fpv_dir": str(fpv_dir),
            "path": "Path A: blocked forward edge, red signalized crossing, green crossing after WAIT(minutes=...)",
            "obstacle_row": obstacle_row,
            "traffic_red_row": red_row,
        })

        placed = place_agent(env, float(obstacle_row["x_cm"]), float(obstacle_row["y_cm"]), facing_for_row(obstacle_row))
        set_clock(dm, 0.0)
        emit({"event": "stage_start", "stage_id": "A1_obstacle", "row": obstacle_row})
        rec = await run_case(
            env=env,
            client=client,
            model=args.model,
            system_text=system_text,
            stage_id="A1_obstacle",
            stage_prompt="",
            placed=placed or {},
            out_dir=run_dir,
            max_image_side=args.max_image_side,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        emit({"event": "stage_result", **rec})
        if rec.get("action_error"):
            placed = place_agent(env, float(obstacle_row["x_cm"]), float(obstacle_row["y_cm"]), facing_for_row(obstacle_row))
            set_clock(dm, 0.0)
            emit({"event": "stage_start", "stage_id": "A1_obstacle_followup", "row": obstacle_row})
            rec = await run_case(
                env=env,
                client=client,
                model=args.model,
                system_text=system_text,
                stage_id="A1_obstacle_followup",
                stage_prompt="",
                placed=placed or {},
                out_dir=run_dir,
                max_image_side=args.max_image_side,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            emit({"event": "stage_result", **rec})

        placed = place_agent(env, float(red_row["x_cm"]), float(red_row["y_cm"]), facing_for_row(red_row))
        axis = normalize_axis(red_row.get("signal_axis"))
        set_clock(dm, state_time(axis, "red"))
        emit({"event": "stage_start", "stage_id": "A2_red_light", "row": red_row})
        rec = await run_case(
            env=env,
            client=client,
            model=args.model,
            system_text=system_text,
            stage_id="A2_red_light",
            stage_prompt="",
            placed=placed or {},
            out_dir=run_dir,
            max_image_side=args.max_image_side,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        emit({"event": "stage_result", **rec})
        red_light_waited = (
            rec.get("action_error") is None
            and str(rec.get("parsed_action") or "").upper().startswith("WAIT")
        )

        placed_after_wait = place_agent(env, float(red_row["x_cm"]), float(red_row["y_cm"]), facing_for_row(red_row))
        if not red_light_waited:
            set_clock(dm, state_time(axis, "green"))
        emit({"event": "stage_start", "stage_id": "A3_after_wait_cross", "row": red_row})
        rec = await run_case(
            env=env,
            client=client,
            model=args.model,
            system_text=system_text,
            stage_id="A3_after_wait_cross",
            stage_prompt="",
            placed=placed_after_wait or {},
            out_dir=run_dir,
            max_image_side=args.max_image_side,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        emit({"event": "stage_result", **rec})

        summary = {
            "event": "summary",
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "stages": [
                {
                    "stage_id": r["stage_id"],
                    "parsed_action": r.get("parsed_action"),
                    "action_error": r.get("action_error"),
                    "hazard_before": r.get("hazard_before"),
                    "metrics": r.get("metrics"),
                }
                for r in records
                if r.get("event") == "stage_result"
            ],
        }
        emit(summary)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
