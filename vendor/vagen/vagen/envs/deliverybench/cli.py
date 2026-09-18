"""
DeliveryBench debugging CLI.

Drive the env by hand (REPL) or from a scripted action list, inspect raw
state, and dump per-step images — without any model in the loop.

Usage:
    # Interactive REPL, nav preset on small-city-11, text mode
    python -m vagen.envs.deliverybench.cli

    # Vision mode with FPV + GMaps map, dump images per step
    python -m vagen.envs.deliverybench.cli --render vision --fpv

    # Scripted probe (actions separated by ';')
    python -m vagen.envs.deliverybench.cli --actions 'VIEW_ORDERS(); ACCEPT_ORDER(1)'

    # Scripted probe from file (one action per line, '#' comments allowed)
    python -m vagen.envs.deliverybench.cli --script probe.txt

REPL commands (anything else is sent to the env as an action):
    :obs     reprint last observation
    :sys     print system prompt
    :state   dump raw dm state (position, waypoint, orders, sim time, ...)
    :orders  dump raw order-pool state
    :help    list commands
    :quit    exit
"""

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from .deliverybench_env import (
    DeliveryBench,
    DeliveryBenchEnvConfig,
    PRESETS,
)


def build_config(args: argparse.Namespace) -> DeliveryBenchEnvConfig:
    cfg = dataclasses.replace(PRESETS[args.preset])
    cfg.map_name = args.map
    cfg.max_steps = args.max_steps
    cfg.render_mode = args.render
    if args.render == "vision":
        cfg.enable_fpv = args.fpv
        cfg.enable_map_images = True
        cfg.use_gmaps_renderer = True
        cfg.map_global_only = True
        # Vision mode also gets the visual navigation tool.
        if cfg.enabled_actions and "VISUAL_NAVIGATE_WALK" not in cfg.enabled_actions:
            cfg.enabled_actions = cfg.enabled_actions + ["VISUAL_NAVIGATE_WALK"]
    return cfg


def dump_state(env: DeliveryBench) -> str:
    """Raw internals the observation may not show — for debugging only."""
    if env._env is None or not env._env.dms:
        return "(env not reset)"
    dm = env._env.dms[0]
    lines = [
        f"pos: ({dm.x / 100:.2f}m, {dm.y / 100:.2f}m)   waypoint: {env._current_waypoint_id()}",
        f"sim_time: {env._get_sim_hours():.3f}h   earnings: ${getattr(dm, 'earnings_total', 0):.2f}",
        f"mode: {getattr(dm, 'mode', '?')}   energy: {getattr(dm, 'energy_pct', '?')}",
        f"carrying: {getattr(dm, 'carrying', [])}",
        f"active_orders: {[getattr(o, 'id', '?') for o in (getattr(dm, 'active_orders', []) or [])]}",
        f"completed: {[getattr(o, 'id', '?') for o in (getattr(dm, 'completed_orders', []) or [])]}",
    ]
    return "\n".join(lines)


def dump_orders(env: DeliveryBench) -> str:
    if env._env is None or not env._env.dms:
        return "(env not reset)"
    om = getattr(env._env.dms[0], "_order_manager", None)
    if om is None:
        return "(no order manager)"
    lines = []
    for o in getattr(om, "_orders", []):
        lines.append(
            f"#{getattr(o, 'id', '?')}: {getattr(o, 'pickup_road_name', '?')} -> "
            f"{getattr(o, 'dropoff_road_name', '?')}  "
            f"items={len(getattr(o, 'items', []) or [])}  "
            f"earn=${getattr(o, 'earning', 0):.2f}  state={getattr(o, 'state', '?')}"
        )
    return "\n".join(lines) or "(pool empty)"


def save_images(obs: dict, out_dir: Path, step: int) -> list:
    mmi = obs.get("multi_modal_input")
    if not mmi:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    imgs = [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]
    for i, img in enumerate(imgs):
        p = out_dir / f"step_{step:03d}_img{i}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def print_step(step: int, obs: dict, reward: float, done: bool, info: dict, img_paths: list):
    print(f"\n──── step {step} ────  reward={reward:.3f}  done={done}", flush=True)
    if info.get("action_error"):
        print(f"  ⚠ action_error: {info['action_error']}")
    tm = info.get("metrics", {}).get("turn_metrics", {})
    if tm:
        print(f"  valid={tm.get('action_is_valid')}  effective={tm.get('action_is_effective')}"
              f"  is_tool={info.get('is_tool')}")
    if img_paths:
        print(f"  images: {', '.join(img_paths)}")
    print(obs.get("obs_str", "(no obs)"))


async def run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    env = DeliveryBench(cfg)
    out_dir = Path(args.out)

    sys_prompt = (await env.system_prompt())["obs_str"]
    if args.show_sys:
        print("════ SYSTEM PROMPT ════")
        print(sys_prompt)
        print("═══════════════════════")

    obs, _ = await env.reset(seed=args.seed)
    img_paths = save_images(obs, out_dir, 0)
    print_step(0, obs, 0.0, False, {}, img_paths)
    print("\n" + dump_state(env))

    # Build the action source: scripted list or stdin REPL.
    scripted = []
    if args.script:
        for line in Path(args.script).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                scripted.append(line)
    elif args.actions:
        scripted = [a.strip() for a in args.actions.split(";") if a.strip()]

    step = 0
    while True:
        if scripted:
            if step >= len(scripted):
                break
            action = scripted[step]
            print(f"\n>>> {action}")
        else:
            try:
                action = input("\naction> ").strip()
            except EOFError:
                break
            if not action:
                continue
            if action in (":quit", ":q", "quit", "exit"):
                break
            if action == ":obs":
                print(obs.get("obs_str", ""))
                continue
            if action == ":sys":
                print(sys_prompt)
                continue
            if action == ":state":
                print(dump_state(env))
                continue
            if action == ":orders":
                print(dump_orders(env))
                continue
            if action == ":help":
                print(":obs :sys :state :orders :quit — anything else is an action")
                continue

        step += 1
        obs, reward, done, info = await env.step(action)
        img_paths = save_images(obs, out_dir, step)
        print_step(step, obs, reward, done, info, img_paths)
        if args.verbose_state:
            print("\n" + dump_state(env))
        if done:
            print("\n══ EPISODE DONE ══")
            print(dump_state(env))
            break

    await env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="DeliveryBench debugging CLI")
    p.add_argument("--preset", default="nav", choices=sorted(PRESETS))
    p.add_argument("--map", default="small-city-11")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--render", default="text", choices=["text", "vision"])
    p.add_argument("--fpv", action="store_true", help="enable first-person view (vision mode)")
    p.add_argument("--actions", help="';'-separated scripted actions")
    p.add_argument("--script", help="file with one action per line")
    p.add_argument("--out", default="/tmp/deliverybench_cli", help="image dump dir")
    p.add_argument("--show-sys", action="store_true", help="print system prompt at start")
    p.add_argument("--verbose-state", action="store_true", help="dump raw state every step")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
