"""Capture full per-step traces for a few representative DeliveryBench rollouts.

Same scaffolded harness as qwen35_lora_rollout_eval.py (oracle drives the
WORKFLOW transitions; the MODEL drives every MOVE during the two navigation
legs), but here we RECORD each step so we can render a human-viewable trajectory:

    step -> [observation image] + model's chosen direction + oracle's correct
            direction + match?  (the model runs with close-thinking, so its
            generation is just the action JSON — there is no free reasoning text)

Writes per-episode trace JSON + the step images into <trace-dir>/, consumed by
build_rollout_trace_html.py to produce a self-contained HTML.

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m vagen.envs.deliverybench.tools.qwen35_lora_rollout_trace \
    --checkpoint <run>/checkpoint-280 --trace-dir <run>/rollout_eval/trace \
    --episodes small-city-15:9000 medium-city-22:10001 large-city-30:11000 large-city-30:11002
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from typing import Any, Dict, List

import torch

from ..deliverybench_env import DeliveryBench
from .build_balanced_visual_sft_data import make_env_config
from .generate_visual_sft_data import (
    _first_available_order, _active_order, _node_target, _delivery_complete, _quote_arg,
)
from .qwen35_lora_rollout_eval import _load_model, _model_move, DEFAULT_MODEL

_DIR_FROM_MOVE = {"move forward": "forward", "move backward": "backward",
                  "turn left": "left", "turn right": "right"}


def _save_obs_image(obs: Dict[str, Any], trace_dir: Path, name: str) -> str | None:
    pil = [im for v in (obs.get("multi_modal_input") or {}).values()
           for im in (v if isinstance(v, list) else [v])]
    if not pil:
        return None
    p = trace_dir / f"{name}.png"
    pil[0].save(p)
    return p.name


async def _model_walk_traced(env, model, processor, system_text, obs, device, tmpdir,
                             trace_dir, max_moves, step0, leg, ep_tag, trace: List[Dict[str, Any]]):
    """Model picks MOVEs until the active route reports arrival; record every step."""
    step = step0
    parsed = 0
    moves = 0
    for _ in range(max_moves):
        nav = env._visual_route_following_info()
        if nav.get("route_arrived"):
            return obs, step, moves, parsed, True
        oracle_dir = _DIR_FROM_MOVE.get(nav.get("oracle_next_move") or "")
        img_name = _save_obs_image(obs, trace_dir, f"{ep_tag}_step{step:02d}")
        d, raw = _model_move(model, processor, system_text, obs, device, tmpdir, step)
        moves += 1
        parsed_ok = d is not None
        if parsed_ok:
            parsed += 1
        trace.append({
            "step": step, "leg": leg, "image": img_name,
            "model_direction": d, "oracle_direction": oracle_dir,
            "match": (d == oracle_dir) if (d and oracle_dir) else None,
            "parsed": parsed_ok, "raw": raw.strip()[:300],
        })
        eff = d if parsed_ok else "forward"  # unparseable -> safe no-op move (counted as miss)
        step += 1
        obs, _r, done, _info = await env.step(json.dumps({"action": f'MOVE(direction="{eff}")'}))
        if done:
            return obs, step, moves, parsed, env._visual_route_following_info().get("route_arrived", False)
    return obs, step, moves, parsed, env._visual_route_following_info().get("route_arrived", False)


async def run_episode(env_cfg, model, processor, device, seed, max_moves, tmpdir, trace_dir, ep_tag):
    env = DeliveryBench(env_cfg)
    rec: Dict[str, Any] = {"seed": seed, "map": env_cfg.map_name, "tag": ep_tag,
                           "setup_ok": False, "trace": []}
    trace = rec["trace"]
    try:
        system_text = (await env.system_prompt())["obs_str"]
        obs, _ = await env.reset(seed=seed)
        obs, _r, _d, _i = await env.step(json.dumps({"action": "VIEW_ORDERS()"}))
        sel = _first_available_order(env); oid = int(getattr(sel, "id"))
        obs, _r, _d, _i = await env.step(json.dumps({"action": f"ACCEPT_ORDER({oid})"}))
        rec["setup_ok"] = True
        order = _active_order(env, oid); pickup = _node_target(getattr(order, "pickup_node", None))
        obs, _r, _d, _i = await env.step(json.dumps({"action": f'NAVIGATE(target={_quote_arg(pickup)}, mode="walk")'}))
        obs, step, mv1, pa1, arr1 = await _model_walk_traced(
            env, model, processor, system_text, obs, device, tmpdir, trace_dir,
            max_moves, 0, "to_pickup", ep_tag, trace)
        rec["arrived_pickup"] = bool(arr1); rec["moves_leg1"] = mv1; rec["parsed_leg1"] = pa1
        if arr1:
            obs, _r, _d, _i = await env.step(json.dumps({"action": f"PICKUP(orders=[{oid}])"}))
            order = _active_order(env, oid); dropoff = _node_target(getattr(order, "dropoff_node", None))
            obs, _r, _d, _i = await env.step(json.dumps({"action": f'NAVIGATE(target={_quote_arg(dropoff)}, mode="walk")'}))
            obs, step, mv2, pa2, arr2 = await _model_walk_traced(
                env, model, processor, system_text, obs, device, tmpdir, trace_dir,
                max_moves, step, "to_dropoff", ep_tag, trace)
            rec["arrived_dropoff"] = bool(arr2); rec["moves_leg2"] = mv2; rec["parsed_leg2"] = pa2
            if arr2:
                await env.step(json.dumps({"action": f"DROP_OFF(oid={oid})"}))
        dm = env._env.dms[0]
        rec["deliveries"] = len(getattr(dm, "completed_orders", []) or [])
        rec["success"] = bool(_delivery_complete(env, oid))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["success"] = False
        rec.setdefault("deliveries", 0)
    finally:
        await env.close()
    return rec


async def amain(args):
    import tempfile
    device = torch.device(args.device)
    model, processor = _load_model(Path(args.checkpoint), Path(args.model_path), device)
    trace_dir = Path(args.trace_dir); trace_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for spec in args.episodes:
        city, seed = spec.rsplit(":", 1)
        pairs.append((city, int(seed)))
    episodes: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for i, (city, seed) in enumerate(pairs):
            cfg = make_env_config(map_name=city, max_steps=args.max_steps,
                                  feasible_order_step_budget=args.feasible_budget, enable_fpv=args.enable_fpv)
            ep_tag = f"ep{i}_{city}_{seed}"
            rec = await run_episode(cfg, model, processor, device, seed, args.max_moves, tmpdir, trace_dir, ep_tag)
            nsteps = len(rec["trace"])
            nmatch = sum(1 for t in rec["trace"] if t.get("match"))
            print(f"[{ep_tag}] success={rec.get('success')} pickup={rec.get('arrived_pickup')} "
                  f"dropoff={rec.get('arrived_dropoff')} steps={nsteps} match={nmatch}/{nsteps} "
                  f"err={rec.get('error','')}", flush=True)
            episodes.append(rec)
    out = trace_dir / "trace.json"
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint),
                               "checkpoint_step": Path(args.checkpoint).name.split("-")[-1],
                               "episodes": episodes}, indent=2))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--episodes", nargs="+", required=True, help="city:seed pairs, e.g. small-city-15:9000")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--feasible-budget", type=int, default=20)
    ap.add_argument("--max-moves", type=int, default=40)
    ap.add_argument("--enable-fpv", action="store_true", default=False)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
