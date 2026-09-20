"""Evaluate traffic-light FPV cases without copying image files.

This runs two passes over sampled red/green traffic-light rows from
deliverybench_fpv, excluding small-city-11-old:

1. Rollout-style prompt: the real DeliveryBench system prompt + observation,
   with the traffic-light timing sentence appended by the environment.
2. Color-only prompt: the same front traffic-light source image, asking only
   whether the light is red or green.

The output JSONL records source image paths and prompts. No images are copied
or written to the output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError
from PIL import Image

from vagen.envs.deliverybench.deliverybench_env import DeliveryBench, STAGE_1_CONFIG
from vagen.envs.deliverybench.vlm_delivery.actions.move import available_moves
from vagen.envs.deliverybench.vlm_delivery.base.defs import TransportMode
from vagen.envs.deliverybench.vlm_delivery.utils.hazards import normalize_axis
from vagen.envs.deliverybench.vlm_delivery.utils.transport import transport_set_mode


HERE = Path(__file__).resolve().parents[1] / "vagen" / "envs" / "deliverybench"
FPV_BASE = HERE / "deliverybench_fpv"
OUT_ROOT = HERE / "outputs"
EXCLUDED_MAPS = {"small-city-11-old"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def map_name_from_manifest(path: Path) -> str:
    return path.relative_to(FPV_BASE).parts[0]


def candidate_manifests() -> List[Path]:
    out: List[Path] = []
    for path in sorted(FPV_BASE.glob("*/main*/manifest.jsonl")):
        map_name = map_name_from_manifest(path)
        if map_name in EXCLUDED_MAPS:
            continue
        if not (HERE / "maps" / map_name).exists():
            continue
        if not (path.parent / "images").exists():
            continue
        rows = load_jsonl(path)
        if any(str(r.get("render_kind", "")).lower() == "traffic_light" for r in rows):
            out.append(path)
    return out


def source_image_path(manifest: Path, row: Mapping[str, Any]) -> Optional[Path]:
    recorded = Path(str(row.get("image_path") or ""))
    if recorded.exists():
        return recorded
    wp_id = str(row["waypoint_id"])
    kind, num = wp_id.rsplit("_", 1)
    dirname = f"{kind}_{int(num):03d}"
    basename = recorded.name if recorded.name else f"yaw_{int(float(row['yaw'])):03d}_{row.get('signal_state', 'green')}.png"
    local = manifest.parent / "images" / dirname / basename
    return local if local.exists() else None


def base_config(map_name: str, fpv_dir: Path):
    return replace(
        STAGE_1_CONFIG,
        map_name=map_name,
        max_steps=8,
        render_mode="vision",
        enable_fpv=True,
        fpv_dir=str(fpv_dir),
        enable_map_images=False,
        map_renderer="pil",
        use_gmaps_renderer=False,
        enable_obstacles=False,
        enable_traffic_lights=True,
        enable_pedestrian_traffic_lights=False,
        traffic_light_require_visible_signal_view=False,
        enabled_actions=["MOVE", "WAIT", "NAVIGATE"],
        initial_transport_mode="walk",
        enable_feasible_orders=False,
        enable_infeasible_orders=True,
    )


def state_time(axis: str, state: str) -> float:
    axis = normalize_axis(axis)
    state = str(state).lower()
    if axis == "NS":
        return 60.0 if state == "red" else 0.0
    return 0.0 if state == "red" else 60.0


def set_clock(dm: Any, sim_s: float) -> None:
    dm.clock._base_real = time.monotonic()
    dm.clock._base_sim = float(sim_s)


def facing_for_row(row: Mapping[str, Any]) -> float:
    if "bearing_deg" in row:
        return float(row["bearing_deg"]) % 360.0
    return (90.0 - float(row["yaw"])) % 360.0


def nonarrival_nav_target(dm: Any, start: Any, first_step: Any) -> Any:
    graph = getattr(getattr(dm, "city_map", None), "waypoint_graph", None)
    if graph is None:
        return first_step
    arrive_tol = max(
        float((getattr(dm, "cfg", {}) or {}).get("arrive_tolerance_cm", 500.0)),
        float((getattr(dm, "cfg", {}) or {}).get("door_tolerance_cm", 1000.0)),
    )
    queue = [first_step]
    seen = {start}
    best = first_step
    visits = 0
    while queue and visits < 80:
        cur = queue.pop(0)
        if cur in seen:
            continue
        visits += 1
        seen.add(cur)
        best = cur
        try:
            dist = float(start.position.distance(cur.position))
            path, _ = graph.shortest_path_nodes(start, cur)
        except Exception:
            path = []
            dist = 0.0
        if dist > arrive_tol and len(path) >= 2 and path[1] is first_step:
            return cur
        for nb in graph.adjacency_list.get(cur, []):
            if nb not in seen:
                queue.append(nb)
    return best


def place_agent(env: DeliveryBench, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    dm = env._env.dms[0]
    node = dm.city_map.nearest_waypoint(float(row["x_cm"]), float(row["y_cm"]))
    if node is None:
        return None
    dm.x = float(node.position.x)
    dm.y = float(node.position.y)
    dm.facing_deg = facing_for_row(row)
    env._prev_dm_x = float(dm.x)
    env._prev_dm_y = float(dm.y)
    dm.collision_count = 0
    dm.traffic_violation_count = 0
    dm.energy_pct = float(dm.cfg.get("energy_pct_max", 100.0))
    dm.is_rescued = False
    dm.vlm_clear_errors()

    forward = available_moves(dm).get("forward")
    if forward is None:
        return None
    first_step = forward["node"]
    nav_target = nonarrival_nav_target(dm, node, first_step)
    dm._nav_target_node = nav_target
    dm._nav_route_color = (26, 115, 232, 255)
    try:
        path, _ = dm.city_map.waypoint_graph.shortest_path_nodes(node, nav_target)
    except Exception:
        path = [node, first_step, nav_target] if nav_target is not first_step else [node, first_step]
    dm._nav_route_path = path or [node, first_step]
    dm.vlm_ephemeral["navigation"] = (
        "[navigation]\n"
        "mode: walk\n"
        f"from: {getattr(node, 'waypoint_name', getattr(node, 'waypoint_id', '?'))}\n"
        f"to: {getattr(nav_target, 'waypoint_name', getattr(nav_target, 'waypoint_id', '?'))}\n"
        f"distance_m: {float(forward.get('dist_m', 0.0)):.1f}\n"
        "estimated_time: one-step traffic-light test\n"
        "next_move: move forward"
    )
    return {"node": node, "forward": forward}


def images_from_obs(obs: Mapping[str, Any]) -> List[Any]:
    mmi = obs.get("multi_modal_input") or {}
    return [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]


def image_data_url_from_image(img: Any, max_side: int) -> str:
    im = img.convert("RGB")
    scale = min(1.0, float(max_side) / max(im.size))
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_data_url_from_path(path: Path, max_side: int) -> str:
    return image_data_url_from_image(Image.open(path).convert("RGB"), max_side)


def parse_jsonish(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def parse_action_text(text: str) -> Optional[str]:
    data = parse_jsonish(text)
    action = data.get("action") if data else None
    return str(action) if action is not None else None


def parse_reason_text(text: str) -> Optional[str]:
    data = parse_jsonish(text)
    for key in ("reasoning_and_reflection", "reasoning", "reason"):
        if key in data:
            return str(data[key])
    return None


def action_kind(action: Optional[str]) -> str:
    s = str(action or "").strip().upper()
    if s.startswith("WAIT"):
        return "WAIT"
    if s.startswith("MOVE"):
        return "MOVE"
    if s.startswith("PASSBY") or s.startswith("BYPASS"):
        return "PASSBY"
    return s.split("(", 1)[0] if s else "NONE"


def parse_color(text: str) -> Optional[str]:
    data = parse_jsonish(text)
    color = str(data.get("color", "")).lower() if data else ""
    if color in {"red", "green"}:
        return color
    low = text.lower()
    if "green" in low and "red" not in low:
        return "green"
    if "red" in low and "green" not in low:
        return "red"
    return None


def parse_rollout_light(text: str) -> Optional[str]:
    data = parse_jsonish(text)
    light = str(data.get("light", "")).lower() if data else ""
    if light in {"red", "green"}:
        return light
    return None


def traffic_prompt_text(user_text: str, mode: str) -> str:
    if mode == "default":
        return user_text
    pattern = (
        r"You are at a traffic light crossing\. The light will change in about "
        r"(?P<minutes>[^.]+) minute\(s\)\."
    )
    match = re.search(pattern, user_text)
    if not match:
        return user_text
    minutes_text = match.group("minutes")
    sentence = (
        f"You are at a traffic light crossing. The light will change in about "
        f"{minutes_text} minute(s). Check out the light before making action"
    )
    if mode == "check_light_schema":
        sentence += (
            ". Your JSON output must include a top-level \"light\" field with "
            "value exactly \"green\" or \"red\""
        )
    sentence += "."
    return re.sub(pattern, sentence, user_text)


def system_prompt_for_mode(system_text: str, mode: str) -> str:
    if mode != "check_light_schema":
        return system_text
    old = '''Return ONLY a valid JSON object with the following three keys:
{
"reasoning_and_reflection": "<=60 tokens: state the relevant observation and why the chosen action follows the rules>",
"action": "Your next action as a single-line function call, strictly following the Action space specification",
"future_plan": "<=40 tokens: a concise next-step plan in natural language"
}
Do not include prose, code fences, or text outside the JSON.'''
    new = '''Return ONLY a valid JSON object with the following four keys:
{
"light": "The traffic light color you see in the FRONT view; exactly green or red",
"reasoning_and_reflection": "<=60 tokens: state the relevant observation, light color, and why the chosen action follows the rules>",
"action": "Your next action as a single-line function call, strictly following the Action space specification",
"future_plan": "<=40 tokens: a concise next-step plan in natural language"
}
Do not include prose, code fences, or text outside the JSON.'''
    if old in system_text:
        return system_text.replace(old, new)
    return system_text + "\n\n" + new


def selected_cases(n: int) -> List[Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {"red": [], "green": []}
    for manifest in candidate_manifests():
        map_name = map_name_from_manifest(manifest)
        for idx, row in enumerate(load_jsonl(manifest)):
            if str(row.get("render_kind", "")).lower() != "traffic_light":
                continue
            label = str(row.get("signal_state", "")).lower()
            if label not in by_label:
                continue
            image_path = source_image_path(manifest, row)
            if image_path is None:
                continue
            by_label[label].append({
                "map_name": map_name,
                "manifest": str(manifest),
                "manifest_dir": str(manifest.parent),
                "row_index": idx,
                "row": row,
                "front_source_image_path": str(image_path),
                "expected_color": label,
            })

    def take_round_robin(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            buckets[item["map_name"]].append(item)
        out: List[Dict[str, Any]] = []
        while len(out) < n and any(buckets.values()):
            for map_name in sorted(list(buckets)):
                if buckets[map_name] and len(out) < n:
                    out.append(buckets[map_name].pop(0))
        return out

    red = take_round_robin(by_label["red"])
    green = take_round_robin(by_label["green"])
    if len(red) < n or len(green) < n:
        raise RuntimeError(f"not enough cases: red={len(red)} green={len(green)} requested={n}")
    return red + green


def panel_source_paths(env: DeliveryBench, row: Mapping[str, Any], dm: Any) -> Dict[str, Optional[str]]:
    pos_key = (round(float(row["x_cm"]), 1), round(float(row["y_cm"]), 1))
    off = float(getattr(env.config, "fpv_yaw_offset_deg", 90.0))
    facing = float(getattr(dm, "facing_deg", facing_for_row(row)))
    panels_dir = {
        "front": (facing + 0.0) % 360.0,
        "right": (facing + 270.0) % 360.0,
        "back": (facing + 180.0) % 360.0,
        "left": (facing + 90.0) % 360.0,
    }
    out: Dict[str, Optional[str]] = {}
    traffic = getattr(dm, "_traffic", None)
    for label, compass_dir in panels_dir.items():
        yaw = env._snap_to_fpv_yaw((off - compass_dir) % 360.0)
        path = None
        if traffic is not None and env._fpv_light_lookup and pos_key in env._fpv_light_lookup and traffic.is_signalised(pos_key[0], pos_key[1]):
            state = traffic.light(traffic.axis_of(compass_dir), float(dm.clock.now_sim()))
            path = env._fpv_light_lookup[pos_key].get((yaw, state))
        if path is None:
            path = (env._fpv_lookup or {}).get(pos_key, {}).get(yaw)
        out[label] = str(path) if path is not None else None
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--rollout-max-tokens", type=int, default=512)
    parser.add_argument("--color-max-tokens", type=int, default=128)
    parser.add_argument("--resume-log", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--traffic-prompt-mode",
        choices=["default", "check_light", "check_light_schema"],
        default="default",
    )
    parser.add_argument("--skip-color", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument(
        "--color-image-source",
        choices=["front", "rollout"],
        default="front",
        help="Use the clean front source image or the rollout composite image for the color-only prompt.",
    )
    args = parser.parse_args()

    cases = selected_cases(args.n)
    completed_indices = set()
    if args.resume_log is not None:
        log_path = args.resume_log
        out_dir = log_path.parent
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("event") == "case_result":
                    completed_indices.add(int(rec.get("case_index", -1)))
    else:
        out_dir = OUT_ROOT / f"traffic_light_rollout_color_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = out_dir / "log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180)
    results: List[Dict[str, Any]] = []
    color_system = "You are a careful visual classifier for traffic-light color."
    color_user = (
        "Look at the traffic light in the image. Identify only the currently "
        "illuminated traffic-light color. Return JSON only with keys reasoning "
        "and color. The color value must be exactly red or green."
    )

    mode = "a" if args.resume_log is not None else "w"
    with log_path.open(mode, encoding="utf-8") as fh:
        def emit(record: Dict[str, Any]) -> None:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(record)

        emit({
            "event": "run_meta",
            "run_dir": str(out_dir),
            "log_path": str(log_path),
            "base_url": args.base_url,
            "model": args.model,
            "n_per_color": args.n,
            "excluded_maps": sorted(EXCLUDED_MAPS),
            "selected_counts": dict(Counter(c["expected_color"] for c in cases)),
            "selected_maps": dict(Counter(c["map_name"] for c in cases)),
            "resume_log": str(args.resume_log) if args.resume_log else None,
            "completed_before_resume": len(completed_indices),
            "traffic_prompt_mode": args.traffic_prompt_mode,
            "skip_color": args.skip_color,
            "skip_rollout": args.skip_rollout,
            "color_image_source": args.color_image_source,
            "sampling": {
                "greedy": False,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "presence_penalty": args.presence_penalty,
                "rollout_max_tokens": args.rollout_max_tokens,
                "color_max_tokens": args.color_max_tokens,
            },
        })

        env_cache: Dict[str, Tuple[DeliveryBench, str]] = {}

        async def env_for(case: Mapping[str, Any]) -> Tuple[DeliveryBench, str]:
            key = str(case["manifest_dir"])
            if key not in env_cache:
                env = DeliveryBench(base_config(str(case["map_name"]), Path(key)))
                system_text = (await env.system_prompt())["obs_str"]
                await env.reset(seed=90)
                transport_set_mode(env._env.dms[0], TransportMode("walk"))
                env_cache[key] = (env, system_text)
            return env_cache[key]

        async def chat_with_retries(**kwargs):
            last_exc = None
            for attempt in range(max(1, int(args.retries))):
                try:
                    return await client.chat.completions.create(**kwargs)
                except (APIConnectionError, APITimeoutError) as exc:
                    last_exc = exc
                    await asyncio.sleep(min(10.0, 1.5 * (attempt + 1)))
            raise last_exc

        for idx, case in enumerate(cases):
            if idx in completed_indices:
                continue
            label = str(case["expected_color"])
            row = case["row"]
            env, system_text = await env_for(case)
            system_text = system_prompt_for_mode(system_text, args.traffic_prompt_mode)
            placed = place_agent(env, row)
            if not placed:
                emit({"event": "case_skipped", "case_index": idx, "reason": "could_not_place_agent", **case})
                continue
            dm = env._env.dms[0]
            set_clock(dm, state_time(str(row.get("signal_axis")), label))
            obs = await env._build_observation({}, {}, init_obs=False)
            user_text = str(obs.get("obs_str", "")).replace("<image>", "").strip()
            user_text = traffic_prompt_text(user_text, args.traffic_prompt_mode)
            rollout_images = images_from_obs(obs)
            panels = panel_source_paths(env, row, dm)
            case_id = f"{label}_{idx:03d}_{case['map_name']}_{row['waypoint_id']}_yaw_{int(float(row['yaw'])):03d}"

            extra_body: Dict[str, Any] = {}
            if args.top_k is not None:
                extra_body["top_k"] = args.top_k
            if args.repetition_penalty is not None:
                extra_body["repetition_penalty"] = args.repetition_penalty
            sampling_kwargs: Dict[str, Any] = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "extra_body": extra_body or None,
            }
            if args.presence_penalty is not None:
                sampling_kwargs["presence_penalty"] = args.presence_penalty

            rollout_text = ""
            if not args.skip_rollout:
                rollout_sampling_kwargs = dict(sampling_kwargs)
                rollout_sampling_kwargs["max_tokens"] = args.rollout_max_tokens
                rollout_resp = await chat_with_retries(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_text},
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {"type": "image_url", "image_url": {"url": image_data_url_from_image(img, args.max_image_side)}}
                                    for img in rollout_images
                                ],
                                {"type": "text", "text": user_text},
                            ],
                        },
                    ],
                    **rollout_sampling_kwargs,
                )
                rollout_text = rollout_resp.choices[0].message.content or ""
            rollout_action = parse_action_text(rollout_text)

            color_image_path = Path(str(case["front_source_image_path"]))
            color_text = ""
            color_input_image_path = str(color_image_path)
            if not args.skip_color:
                if args.color_image_source == "rollout":
                    color_image_content = [
                        {"type": "image_url", "image_url": {"url": image_data_url_from_image(img, args.max_image_side)}}
                        for img in rollout_images
                    ]
                    color_input_image_path = "<rollout_composite_from_observation>"
                else:
                    color_image_content = [
                        {"type": "image_url", "image_url": {"url": image_data_url_from_path(color_image_path, args.max_image_side)}}
                    ]
                color_sampling_kwargs = dict(sampling_kwargs)
                color_sampling_kwargs["max_tokens"] = args.color_max_tokens
                color_resp = await chat_with_retries(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": color_system},
                        {
                            "role": "user",
                            "content": [
                                *color_image_content,
                                {"type": "text", "text": color_user},
                            ],
                        },
                    ],
                    **color_sampling_kwargs,
                )
                color_text = color_resp.choices[0].message.content or ""

            emit({
                "event": "case_result",
                "case_id": case_id,
                "case_index": idx,
                "expected_color": label,
                "map_name": case["map_name"],
                "manifest": case["manifest"],
                "manifest_row_index": case["row_index"],
                "manifest_row": row,
                "rollout": {
                    "system_prompt": system_text,
                    "user_prompt": user_text,
                    "model_response": rollout_text,
                    "reasoning": parse_reason_text(rollout_text),
                    "light": parse_rollout_light(rollout_text),
                    "action": rollout_action,
                    "action_kind": action_kind(rollout_action),
                    "panel_source_image_paths": panels,
                    "front_source_image_path": panels.get("front") or case["front_source_image_path"],
                    "hazard_before": {
                        "traffic_forward_state": label,
                        "traffic_controlled_here": True,
                    },
                },
                "color_identification": {
                    "system_prompt": color_system,
                    "user_prompt": color_user,
                    "input_image_path": color_input_image_path,
                    "image_source": args.color_image_source,
                    "panel_source_image_paths": panels if args.color_image_source == "rollout" else None,
                    "model_response": color_text,
                    "reasoning": parse_jsonish(color_text).get("reasoning"),
                    "predicted_color": parse_color(color_text),
                    "skipped": bool(args.skip_color),
                },
            })

        for env, _system_text in env_cache.values():
            await env.close()

        all_rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        case_results = [r for r in all_rows if r.get("event") == "case_result"]
        summary = {
            "event": "summary",
            "run_dir": str(out_dir),
            "log_path": str(log_path),
            "n": len(case_results),
            "rollout": {
                "actions_by_expected": {
                    color: dict(Counter(r["rollout"].get("action_kind") for r in case_results if r.get("expected_color") == color))
                    for color in ("red", "green")
                },
                "lights_by_expected": {
                    color: dict(Counter(r["rollout"].get("light") for r in case_results if r.get("expected_color") == color))
                    for color in ("red", "green")
                },
            },
            "color_identification": {
                "overall_correct": sum(
                    1 for r in case_results
                    if r["color_identification"].get("predicted_color") == r.get("expected_color")
                ),
                "predictions_by_expected": {
                    color: dict(Counter(
                        r["color_identification"].get("predicted_color")
                        for r in case_results
                        if r.get("expected_color") == color
                    ))
                    for color in ("red", "green")
                },
            },
        }
        emit(summary)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
