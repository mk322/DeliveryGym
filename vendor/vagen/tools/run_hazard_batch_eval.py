"""Batch local-Qwen hazard eval across DeliveryBench FPV maps.

Runs:
  * obstacle cases: first action, plus feedback follow-up when first action fails
  * red traffic-light cases
  * green traffic-light cases

Every case records system prompt, user prompt, image(s), raw model response, parsed
action, and environment result under
vagen/envs/deliverybench/outputs/hazard_batch_eval_<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
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
FPV_BASE = HERE / "deliverybench_fpv"
OUT_ROOT = HERE / "outputs"
EXCLUDED_MAPS = {"small-city-11-old"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def map_name_from_manifest(path: Path) -> str:
    return path.relative_to(FPV_BASE).parts[0]


def candidate_manifests() -> List[Path]:
    out = []
    for path in sorted(FPV_BASE.glob("*/main*/manifest.jsonl")):
        map_name = map_name_from_manifest(path)
        if map_name in EXCLUDED_MAPS:
            continue
        if not (HERE / "maps" / map_name).exists():
            continue
        rows = load_jsonl(path)
        kinds = {str(r.get("render_kind", "")).lower() for r in rows}
        if {"obstacle", "traffic_light"} <= kinds:
            out.append(path)
    return out


def source_image_path(manifest: Path, row: Mapping[str, Any]) -> Optional[Path]:
    recorded = Path(str(row.get("image_path") or ""))
    if recorded.exists():
        return recorded

    root = manifest.parents[1]
    wp_id = str(row["waypoint_id"])
    kind, num = wp_id.rsplit("_", 1)
    dir_name = f"{kind}_{int(num):03d}"
    basename = recorded.name if recorded.name else f"yaw_{int(float(row['yaw'])):03d}.png"
    local = root / "images" / dir_name / basename
    if local.exists():
        return local
    return None


def materialize_fpv_root(manifest: Path) -> Path:
    rows = load_jsonl(manifest)
    root = Path(tempfile.mkdtemp(prefix=f"vagen_batch_{map_name_from_manifest(manifest)}_"))
    (root / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, root / "manifest.jsonl")
    obstacles = manifest.parent / "obstacles.json"
    if obstacles.exists():
        shutil.copy2(obstacles, root / "obstacles.json")

    for row in rows:
        src = source_image_path(manifest, row)
        if src is None:
            continue
        wp_id = str(row["waypoint_id"])
        kind, num = wp_id.rsplit("_", 1)
        dst_dir = root / "images" / f"{kind}_{int(num):03d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
    return root


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
    if axis == "NS":
        return 60.0 if state == "red" else 0.0
    return 0.0 if state == "red" else 60.0


def facing_for_row(row: Mapping[str, Any]) -> float:
    if "bearing_deg" in row:
        return float(row["bearing_deg"]) % 360.0
    return (90.0 - float(row["yaw"])) % 360.0


def nonarrival_nav_target(dm: Any, start: Any, first_step: Any) -> Any:
    """Pick a route target beyond the first forward edge.

    Some dock-to-neighbor obstacle rows are inside DeliveryBench's arrival
    tolerance, so the live navigation hint becomes "you have arrived". Keep the
    tested first edge unchanged, but set a farther target whose shortest path
    still starts with that edge.
    """
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


def place_agent(
    env: DeliveryBench,
    row: Mapping[str, Any],
    *,
    force_nonarrival_route: bool = True,
) -> Optional[Dict[str, Any]]:
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
    target = forward["node"]
    nav_target = nonarrival_nav_target(dm, node, target) if force_nonarrival_route else target
    dm._nav_target_node = nav_target
    dm._nav_route_color = (26, 115, 232, 255)
    try:
        path, _ = dm.city_map.waypoint_graph.shortest_path_nodes(node, nav_target)
    except Exception:
        path = [node, target, nav_target] if nav_target is not target else [node, target]
    dm._nav_route_path = path or [node, target]
    dm.vlm_ephemeral["navigation"] = (
        "[navigation]\n"
        "mode: walk\n"
        f"from: {getattr(node, 'waypoint_name', getattr(node, 'waypoint_id', '?'))}\n"
        f"to: {getattr(nav_target, 'waypoint_name', getattr(nav_target, 'waypoint_id', '?'))}\n"
        f"distance_m: {float(forward.get('dist_m', 0.0)):.1f}\n"
        "estimated_time: one-step hazard test\n"
        "next_move: move forward"
    )
    return {"node": node, "forward": forward}


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


def build_user_text(obs: Mapping[str, Any], stage_prompt: str) -> str:
    text = str(obs.get("obs_str", "")).replace("<image>", "").strip()
    return (
        f"{stage_prompt}\n\n"
        "Return JSON only with keys reasoning_and_reflection, action, future_plan. "
        "Use exactly one valid DeliveryBench action.\n\n"
        f"{text}"
    )


def obs_to_content(obs: Mapping[str, Any], stage_prompt: str, max_image_side: int) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for img in images_from_obs(obs):
        parts.append({"type": "image_url", "image_url": {"url": image_data_url(img, max_image_side)}})
    parts.append({"type": "text", "text": build_user_text(obs, stage_prompt)})
    return parts


def save_images(obs: Mapping[str, Any], case_dir: Path) -> List[str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, img in enumerate(images_from_obs(obs)):
        path = case_dir / f"image_{idx}.png"
        img.save(path)
        out.append(str(path))
    return out


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


def action_kind(action: Optional[str]) -> str:
    s = str(action or "").strip().upper()
    if s.startswith("PASSBY") or s.startswith("BYPASS"):
        return "PASSBY"
    if s.startswith("MOVE"):
        return "MOVE"
    if s.startswith("WAIT"):
        return "WAIT"
    return s.split("(", 1)[0] or "NONE"


async def run_model_case(
    *,
    env: DeliveryBench,
    client: AsyncOpenAI,
    model: str,
    out_dir: Path,
    case_id: str,
    system_text: str,
    stage_prompt: str,
    placed: Mapping[str, Any],
    max_image_side: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    obs = await env._build_observation({}, {}, init_obs=False)
    image_paths = save_images(obs, out_dir / "images" / case_id)
    user_text = build_user_text(obs, stage_prompt)
    input_dir = out_dir / "inputs" / case_id
    input_dir.mkdir(parents=True, exist_ok=True)
    system_prompt_path = input_dir / "system_prompt.txt"
    user_prompt_path = input_dir / "user_prompt.txt"
    request_path = input_dir / "request.json"
    system_prompt_path.write_text(system_text, encoding="utf-8")
    user_prompt_path.write_text(user_text, encoding="utf-8")
    request_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
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
    before = hazard_state(env, placed)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": obs_to_content(obs, stage_prompt, max_image_side)},
        ],
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    model_text = resp.choices[0].message.content or ""
    obs2, reward, done, info = await env.step(model_text)
    dm = env._env.dms[0]
    parsed_action = (info.get("parsed") or {}).get("action")
    return {
        "event": "case_result",
        "case_id": case_id,
        "hazard_before": before,
        "model_response": model_text,
        "parsed_action": parsed_action,
        "action_kind": action_kind(parsed_action),
        "action_error": info.get("action_error"),
        "reward": reward,
        "done": done,
        "metrics": (info.get("metrics") or {}).get("traj_metrics", {}),
        "position_after_cm": [round(float(dm.x), 1), round(float(dm.y), 1)],
        "facing_after_deg": round(float(dm.facing_deg), 1),
        "sim_s_after": round(float(dm.clock.now_sim()), 3),
        "input": {
            "system_prompt_path": str(system_prompt_path),
            "user_prompt_path": str(user_prompt_path),
            "request_path": str(request_path),
            "image_paths": image_paths,
        },
        "obs_after_excerpt": str(obs2.get("obs_str", ""))[:500],
    }


async def build_usable_cases(limit_each: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Path]]:
    manifests = candidate_manifests()
    fpv_roots: Dict[str, Path] = {}
    cases = {"obstacle": [], "red": [], "green": []}
    for manifest in manifests:
        map_name = map_name_from_manifest(manifest)
        fpv_root = materialize_fpv_root(manifest)
        fpv_roots[map_name] = fpv_root
        env = DeliveryBench(base_config(map_name, fpv_root))
        await env.reset(seed=90)
        dm = env._env.dms[0]
        transport_set_mode(dm, TransportMode("walk"))
        for idx, row in enumerate(load_jsonl(manifest)):
            kind = str(row.get("render_kind", "")).lower()
            if kind not in ("obstacle", "traffic_light"):
                continue
            placed = place_agent(env, row, force_nonarrival_route=False)
            if not placed:
                continue
            rec = {
                "map_name": map_name,
                "manifest": str(manifest),
                "row_index": idx,
                "row": row,
            }
            if kind == "obstacle":
                cases["obstacle"].append(rec)
            elif str(row.get("signal_state", "")).lower() == "red":
                cases["red"].append(rec)
            elif str(row.get("signal_state", "")).lower() == "green":
                cases["green"].append(rec)
        await env.close()

    def take_round_robin(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_map[item["map_name"]].append(item)
        selected: List[Dict[str, Any]] = []
        while len(selected) < limit_each and any(by_map.values()):
            for map_name in sorted(list(by_map)):
                if by_map[map_name] and len(selected) < limit_each:
                    selected.append(by_map[map_name].pop(0))
        return selected

    return (
        take_round_robin(cases["obstacle"]),
        take_round_robin(cases["red"]),
        take_round_robin(cases["green"]),
        fpv_roots,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-image-side", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--only", choices=["all", "obstacle", "traffic_red", "traffic_green"], default="all")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"hazard_batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180)
    obstacle_cases, red_cases, green_cases, fpv_roots = await build_usable_cases(args.n)
    if args.only == "obstacle":
        obstacle_cases = obstacle_cases[args.skip:]
        red_cases = []
        green_cases = []
    elif args.only == "traffic_red":
        obstacle_cases = []
        red_cases = red_cases[args.skip:]
        green_cases = []
    elif args.only == "traffic_green":
        obstacle_cases = []
        red_cases = []
        green_cases = green_cases[args.skip:]
    elif args.skip:
        raise ValueError("--skip is only supported with --only obstacle|traffic_red|traffic_green")
    results: List[Dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as fh:
        def emit(record: Dict[str, Any]) -> None:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(record)

        emit({
            "event": "run_meta",
            "run_dir": str(out_dir),
            "base_url": args.base_url,
            "model": args.model,
            "n": args.n,
            "only": args.only,
            "skip": args.skip,
            "excluded_maps": sorted(EXCLUDED_MAPS),
            "selected_counts": {
                "obstacle": len(obstacle_cases),
                "traffic_red": len(red_cases),
                "traffic_green": len(green_cases),
            },
            "selected_maps": sorted({c["map_name"] for c in obstacle_cases + red_cases + green_cases}),
        })

        async def new_env(map_name: str) -> Tuple[DeliveryBench, str]:
            env = DeliveryBench(base_config(map_name, fpv_roots[map_name]))
            system_text = (await env.system_prompt())["obs_str"]
            await env.reset(seed=90)
            transport_set_mode(env._env.dms[0], TransportMode("walk"))
            return env, system_text

        for i, case in enumerate(obstacle_cases):
            map_name = case["map_name"]
            env, system_text = await new_env(map_name)
            row = case["row"]
            placed = place_agent(env, row)
            set_clock(env._env.dms[0], 0.0)
            case_id = f"obstacle_{i:03d}_{map_name}_{row['waypoint_id']}_yaw_{int(float(row['yaw'])):03d}"
            first = await run_model_case(
                env=env,
                client=client,
                model=args.model,
                out_dir=out_dir,
                case_id=case_id,
                system_text=system_text,
                stage_prompt=(
                    "Obstacle test: the planned route goes forward. Inspect the FRONT FPV image "
                    "for a visible obstacle or road block before choosing the action."
                ),
                placed=placed or {},
                max_image_side=args.max_image_side,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            first.update({"test_type": "obstacle_first", "map_name": map_name, "row": row})
            emit(first)
            if first.get("action_error"):
                env2, system_text2 = await new_env(map_name)
                placed2 = place_agent(env2, row)
                set_clock(env2._env.dms[0], 0.0)
                follow = await run_model_case(
                    env=env2,
                    client=client,
                    model=args.model,
                    out_dir=out_dir,
                    case_id=case_id + "_followup",
                    system_text=system_text2,
                    stage_prompt=(
                        f"Previous action failed: {first['action_error']}\n"
                        "You are still at the same blocked forward edge. Choose the best follow-up action."
                    ),
                    placed=placed2 or {},
                    max_image_side=args.max_image_side,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                follow.update({"test_type": "obstacle_followup", "map_name": map_name, "row": row, "first_case_id": case_id})
                emit(follow)
                await env2.close()
            await env.close()

        for label, selected in (("traffic_red", red_cases), ("traffic_green", green_cases)):
            state = "red" if label.endswith("red") else "green"
            for i, case in enumerate(selected):
                map_name = case["map_name"]
                env, system_text = await new_env(map_name)
                row = case["row"]
                placed = place_agent(env, row)
                set_clock(env._env.dms[0], state_time(normalize_axis(row.get("signal_axis")), state))
                case_id = f"{label}_{i:03d}_{map_name}_{row['waypoint_id']}_yaw_{int(float(row['yaw'])):03d}"
                rec = await run_model_case(
                    env=env,
                    client=client,
                    model=args.model,
                    out_dir=out_dir,
                    case_id=case_id,
                    system_text=system_text,
                    stage_prompt=(
                        "Traffic-light test: the planned route goes forward through a traffic light. "
                        "Inspect the FRONT FPV signal and use the traffic-light timing if you decide to wait."
                    ),
                    placed=placed or {},
                    max_image_side=args.max_image_side,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                rec.update({"test_type": label, "map_name": map_name, "row": row})
                emit(rec)
                await env.close()

        case_results = [r for r in results if r.get("event") == "case_result"]
        def counts_for(test_type: str) -> Dict[str, Any]:
            rows = [r for r in case_results if r.get("test_type") == test_type]
            return {
                "n": len(rows),
                "actions": dict(Counter(r.get("action_kind") for r in rows)),
                "errors": sum(1 for r in rows if r.get("action_error")),
                "maps": dict(Counter(r.get("map_name") for r in rows)),
            }

        first_rows = [r for r in case_results if r.get("test_type") == "obstacle_first"]
        follow_rows = [r for r in case_results if r.get("test_type") == "obstacle_followup"]
        summary = {
            "event": "summary",
            "run_dir": str(out_dir),
            "log_path": str(log_path),
            "obstacle_first_passby": sum(1 for r in first_rows if r.get("action_kind") == "PASSBY"),
            "obstacle_first_total": len(first_rows),
            "obstacle_feedback_followup_passby": sum(1 for r in follow_rows if r.get("action_kind") == "PASSBY"),
            "obstacle_feedback_followup_total": len(follow_rows),
            "by_test": {
                "obstacle_first": counts_for("obstacle_first"),
                "obstacle_followup": counts_for("obstacle_followup"),
                "traffic_red": counts_for("traffic_red"),
                "traffic_green": counts_for("traffic_green"),
            },
        }
        emit(summary)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
