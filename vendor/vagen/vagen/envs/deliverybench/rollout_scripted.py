"""
rollout_scripted.py
───────────────────
Deterministic, no-API "autopilot" rollout for the direction-based MOVE world.
It drives the env exactly like a model would (only via env.step(<action JSON>))
but chooses each MOVE(direction) by translating the shortest waypoint path into
forward/left/right/backward relative to the agent's current facing. Useful to
(a) sanity-check the whole walking + pickup/drop-off loop end-to-end and
(b) produce a human-readable log + per-step images to inspect.

Run:
    PYTHONPATH=. python3 -m vagen.envs.deliverybench.rollout_scripted
Outputs: outputs/scripted_<timestamp>/  (log.jsonl + images/)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from .deliverybench_env import DeliveryBench, NAV_PRESET
from .vlm_delivery.actions.move import available_moves

# Optional hard cap on total steps (0 = run the whole delivery). e.g. MAX_STEPS=15
_MAX_STEPS = int(os.environ.get("MAX_STEPS", "0") or "0")


class _StepCap(Exception):
    """Raised to stop the rollout once _MAX_STEPS is reached."""

_RUN = Path(__file__).parent / "outputs" / f"scripted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
_IMG = _RUN / "images"
_RUN.mkdir(parents=True, exist_ok=True)
_IMG.mkdir(parents=True, exist_ok=True)
_fh = (_RUN / "log.jsonl").open("w", encoding="utf-8")


def _emit(rec):
    _fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _fh.flush()


def _save_imgs(obs, step):
    mmi = obs.get("multi_modal_input") or {}
    out = []
    imgs = [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]
    for i, im in enumerate(imgs):
        p = _IMG / f"step_{step:03d}_img{i}.png"
        im.save(p)
        out.append(str(p.relative_to(_RUN)))
    return out


def _next_direction(dm, target_node):
    """Direction to step next along the shortest path to target, or None if arrived."""
    cm = dm.city_map
    cur = cm.nearest_waypoint(float(dm.x), float(dm.y))
    if cur is target_node:
        return None
    path, _ = cm.waypoint_graph.shortest_path_nodes(cur, target_node)
    if not path or len(path) < 2:
        return None
    nxt = path[1]
    for d, a in available_moves(dm).items():
        if a and a.get("node") is nxt:
            return d
    return None


async def _walk_to(env, target_node, step0, max_steps=60):
    """Autopilot: MOVE step-by-step until standing on target_node."""
    dm = env._env.dms[0]
    step = step0
    while step < step0 + max_steps:
        d = _next_direction(dm, target_node)
        if d is None:
            return step  # arrived
        step += 1
        obs, r, done, info = await env.step(json.dumps({"action": f'MOVE(direction="{d}")'}))
        imgs = _save_imgs(obs, step)
        print(f"  step {step:03d}  MOVE({d:8})  facing={dm.facing_deg:5.0f}  "
              f"pos=({dm.x/100:.0f},{dm.y/100:.0f})m  err={info.get('action_error')}")
        _emit({"step": step, "action": f"MOVE({d})", "facing_deg": dm.facing_deg,
               "pos_m": [round(dm.x/100, 1), round(dm.y/100, 1)],
               "action_error": info.get("action_error"), "images": imgs,
               "sim_hours": round(info["metrics"]["traj_metrics"]["sim_hours"], 3)})
        if _MAX_STEPS and step >= _MAX_STEPS:
            raise _StepCap(step)
        if done:
            break
    return step


async def main():
    cfg = NAV_PRESET
    cfg.render_mode = "vision"; cfg.enable_fpv = True; cfg.enable_map_images = True
    cfg.use_gmaps_renderer = True; cfg.gmaps_out_scale = 0.5; cfg.map_name = "small-city-11"
    env = DeliveryBench(cfg)
    await env.system_prompt()
    obs, _ = await env.reset(seed=42)
    _save_imgs(obs, 0)
    dm = env._env.dms[0]
    step = 0

    async def do(action):
        nonlocal step
        step += 1
        o, r, dn, i = await env.step(json.dumps({"action": action}))
        imgs = _save_imgs(o, step)
        print(f"  step {step:03d}  {action:28}  err={i.get('action_error')}")
        _emit({"step": step, "action": action, "facing_deg": dm.facing_deg,
               "action_error": i.get("action_error"), "images": imgs})
        if _MAX_STEPS and step >= _MAX_STEPS:
            raise _StepCap(step)
        return o, i

    capped = False
    try:
        print("=== accept order 0 ===")
        await do("VIEW_ORDERS()")
        await do("ACCEPT_ORDER(0)")
        order = dm.active_orders[0]
        print(f"order #{order.id}: pickup_node={getattr(order.pickup_node,'waypoint_id','?')} "
              f"dropoff_node={getattr(order.dropoff_node,'waypoint_id','?')}")

        print("=== NAVIGATE (route + estimate, no chain) ===")
        o, _ = await do('NAVIGATE(target="restaurant 1")')

        print("=== walk to pickup dock ===")
        step = await _walk_to(env, order.pickup_node, step)
        o, i = await do("PICKUP(orders=[0])")
        print(f"  picked_up={bool(getattr(order, 'has_picked_up', False))}")

        print("=== walk to dropoff dock ===")
        step = await _walk_to(env, order.dropoff_node, step)
        o, i = await do("DROP_OFF(oid=0)")
    except _StepCap as e:
        capped = True
        step = int(e.args[0]) if e.args else step
        print(f"\n[stopped at MAX_STEPS={_MAX_STEPS}]")

    deliveries = len(getattr(dm, "completed_orders", []) or [])
    earn = float(getattr(dm, "earnings_total", 0.0))
    print(f"\nRESULT: deliveries={deliveries}  earnings=${earn:.2f}  steps={step}  "
          f"sim={env._get_sim_hours():.2f}h  capped={capped}")
    _emit({"event": "summary", "deliveries": deliveries, "earnings": round(earn, 2),
           "steps": step, "sim_hours": round(env._get_sim_hours(), 3), "capped": capped})
    _fh.close()
    print(f"log dir: {_RUN}")
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
