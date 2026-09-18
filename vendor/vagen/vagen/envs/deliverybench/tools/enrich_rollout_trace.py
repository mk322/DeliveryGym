"""Enrich a rollout trace with per-step accepted/blocked flags (env-only, no GPU).

The trace records the model's chosen MOVE per step but not whether the env
accepted it. Because the env is deterministic given (seed, action sequence), we
replay the exact recorded directions and capture info["action_error"] per step,
so the HTML can show which moves actually advanced vs were rejected (e.g. the
model choosing forward when only left/right are legal). Writes trace_enriched.json.
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path

from ..deliverybench_env import DeliveryBench
from .build_balanced_visual_sft_data import make_env_config
from .generate_visual_sft_data import _first_available_order, _active_order, _node_target, _quote_arg


def _short_err(e):
    if not e:
        return None
    s = str(e)
    m = re.search(r"Directions you can move:\s*([a-z, ]+)", s)
    if "no reachable waypoint" in s and m:
        return f"blocked; legal: {m.group(1).strip()}"
    return s[:80]


async def _feed_leg(env, steps, oid_state):
    """Replay recorded directions for one leg; annotate each with move_ok/err. Returns arrived."""
    for t in steps:
        d = t.get("model_direction") or "forward"  # matches harness fallback
        nav = env._visual_route_following_info()
        if nav.get("route_arrived"):
            t["move_ok"] = None; t["err"] = None
            return True
        _o, _r, done, info = await env.step(json.dumps({"action": f'MOVE(direction="{d}")'}))
        err = info.get("action_error")
        t["move_ok"] = not err
        t["err"] = _short_err(err)
        if done:
            return env._visual_route_following_info().get("route_arrived", False)
    return env._visual_route_following_info().get("route_arrived", False)


async def enrich_episode(ep):
    cfg = make_env_config(map_name=ep["map"], max_steps=25, feasible_order_step_budget=20, enable_fpv=False)
    env = DeliveryBench(cfg)
    try:
        await env.system_prompt()
        await env.reset(seed=ep["seed"])
        await env.step(json.dumps({"action": "VIEW_ORDERS()"}))
        sel = _first_available_order(env); oid = int(getattr(sel, "id"))
        await env.step(json.dumps({"action": f"ACCEPT_ORDER({oid})"}))
        order = _active_order(env, oid); pickup = _node_target(getattr(order, "pickup_node", None))
        await env.step(json.dumps({"action": f'NAVIGATE(target={_quote_arg(pickup)}, mode="walk")'}))
        leg1 = [t for t in ep["trace"] if t["leg"] == "to_pickup"]
        leg2 = [t for t in ep["trace"] if t["leg"] == "to_dropoff"]
        arr1 = await _feed_leg(env, leg1, oid)
        if arr1 and leg2:
            await env.step(json.dumps({"action": f"PICKUP(orders=[{oid}])"}))
            order = _active_order(env, oid); dropoff = _node_target(getattr(order, "dropoff_node", None))
            await env.step(json.dumps({"action": f'NAVIGATE(target={_quote_arg(dropoff)}, mode="walk")'}))
            await _feed_leg(env, leg2, oid)
    finally:
        await env.close()


async def amain(args):
    src = Path(args.trace)
    data = json.loads(src.read_text())
    for ep in data["episodes"]:
        await enrich_episode(ep)
        ok = sum(1 for t in ep["trace"] if t.get("move_ok"))
        n = len(ep["trace"])
        print(f"[{ep['tag']}] moves accepted {ok}/{n}", flush=True)
    out = src.with_name("trace_enriched.json")
    out.write_text(json.dumps(data, indent=2))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="path to trace.json")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
