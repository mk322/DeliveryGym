"""Generate visual route-following SFT data from oracle DeliveryBench rollouts.

The exporter drives the real DeliveryBench visual-route workflow with oracle
actions and writes one policy-visible decision sample per row. Oracle fields are
used only to choose labels and metadata; they are never written into the prompt.

Example:
    PYTHONPATH=. python -m vagen.envs.deliverybench.tools.generate_visual_sft_data \
        --output-dir /tmp/deliverybench_visual_sft \
        --seeds 200-263
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..deliverybench_env import DeliveryBench, DeliveryBenchEnvConfig, VISUAL_ROUTE_FOLLOWING_CONFIG


_LEAK_TOKENS = ("next_move", "oracle_next_move", "oracle_next_action")


def parse_seed_list(text: str) -> List[int]:
    """Parse comma-separated seeds/ranges, e.g. ``100,105-107``."""
    out: List[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid seed range: {part}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("at least one seed is required")
    return out


def load_env_config(path: Optional[Path], *, max_steps: Optional[int] = None) -> DeliveryBenchEnvConfig:
    """Load an env config from a rollout YAML, or use the visual SFT default."""
    if path is None:
        cfg = dataclasses.replace(
            VISUAL_ROUTE_FOLLOWING_CONFIG,
            max_orders_in_pool=1,
            enable_feasible_orders=True,
            enable_infeasible_orders=False,
        )
    else:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        env_raw = dict(raw.get("env", raw) or {})
        cfg = DeliveryBenchEnvConfig(**env_raw)

    if max_steps is not None:
        cfg = dataclasses.replace(cfg, max_steps=int(max_steps))
    return cfg


def _json_response(action: str, *, stage: str) -> str:
    return json.dumps(
        {
            "reasoning_and_reflection": _reason_for_stage(stage, action),
            "action": action,
            "future_plan": _plan_for_stage(stage),
        },
        ensure_ascii=False,
    )


def _reason_for_stage(stage: str, action: str) -> str:
    if stage == "move" and str(action).startswith("MOVE_TO"):
        return ("The reachable waypoints are numbered on the first-person "
                "views and in the waypoint_marks list; pick the marker that "
                "follows the active blue route.")
    if stage == "move":
        return "The active blue route is visible; choose the primitive MOVE that follows it."
    if stage == "navigate_pickup":
        return "An order is accepted and the pickup route is not active yet."
    if stage == "navigate_dropoff":
        return "The order is picked up and the dropoff route is not active yet."
    if stage == "pickup":
        return "The pickup location has been reached; collect the ready order."
    if stage == "dropoff":
        return "The dropoff location has been reached while carrying the order."
    if stage == "accept":
        return "Choose the available feasible order for this single-order trajectory."
    return f"Follow the delivery workflow with {action.split('(', 1)[0]}."


def _plan_for_stage(stage: str) -> str:
    return {
        "view_orders": "select an order",
        "accept": "navigate to pickup",
        "navigate_pickup": "follow the route to pickup",
        "pickup": "navigate to dropoff",
        "navigate_dropoff": "follow the route to dropoff",
        "move": "continue route following",
        "dropoff": "finish the delivery",
    }.get(stage, "continue the delivery")


def _assert_no_policy_leak(text: str) -> None:
    lowered = str(text or "").lower()
    leaked = [tok for tok in _LEAK_TOKENS if tok in lowered]
    if leaked:
        raise AssertionError(f"policy prompt leaked {leaked}: {text[:500]}")


def _obs_images(obs: Dict[str, Any]) -> List[Any]:
    mmi = obs.get("multi_modal_input") or {}
    imgs: List[Any] = []
    for value in mmi.values():
        if isinstance(value, list):
            imgs.extend(value)
        elif value is not None:
            imgs.append(value)
    return imgs


def _save_pre_action_images(
    obs: Dict[str, Any],
    *,
    image_dir: Path,
    seed: int,
    turn_index: int,
) -> List[Dict[str, str]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, str]] = []
    for idx, img in enumerate(_obs_images(obs)):
        path = image_dir / f"seed_{seed}_turn_{turn_index:03d}_img{idx}.png"
        img.save(path)
        out.append({"image": str(path.resolve())})
    return out


def _count_image_placeholders(text: str, placeholder: str = "<image>") -> int:
    return str(text or "").count(placeholder)


def _validate_prompt_images(user_text: str, images: Sequence[Dict[str, str]]) -> None:
    expected = _count_image_placeholders(user_text)
    if expected != len(images):
        raise AssertionError(f"placeholder/image mismatch: {expected} placeholders vs {len(images)} images")
    for image in images:
        path = Path(image["image"])
        if not path.exists():
            raise FileNotFoundError(path)


def _quote_arg(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _node_target(node: Any) -> str:
    for attr in ("address", "waypoint_name", "waypoint_id"):
        value = str(getattr(node, attr, "") or "").strip()
        if value:
            return value
    raise RuntimeError("navigation target node has no address/name/id")


def _first_available_order(env: DeliveryBench) -> Any:
    dm = env._env.dms[0]
    orders = list(getattr(dm._order_manager, "_orders", []) or [])
    if not orders:
        raise RuntimeError("VIEW_ORDERS produced no available orders")
    return orders[0]


def _active_order(env: DeliveryBench, oid: int) -> Any:
    dm = env._env.dms[0]
    for order in list(getattr(dm, "active_orders", []) or []):
        if int(getattr(order, "id", -1)) == int(oid):
            return order
    raise RuntimeError(f"accepted order #{oid} not found in active_orders")


def _delivery_complete(env: DeliveryBench, oid: int) -> bool:
    dm = env._env.dms[0]
    for order in list(getattr(dm, "completed_orders", []) or []):
        if int(getattr(order, "id", -1)) == int(oid):
            return True
    return int(getattr(env, "deliveries_completed", 0) or 0) > 0


async def export_dataset(
    *,
    output_dir: Path,
    seeds: Iterable[int],
    env_config: DeliveryBenchEnvConfig,
    parquet_name: str = "visual_sft.parquet",
    move_repeat: int = 1,
    include_non_move: bool = True,
) -> Path:
    """Generate oracle SFT rows and write a parquet dataset."""
    output_dir = output_dir.resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(seed) for seed in seeds]

    rows: List[Dict[str, Any]] = []
    for seed in seed_list:
        seed_rows = await _run_oracle_episode(
            seed=int(seed),
            env_config=env_config,
            image_dir=image_dir,
            move_repeat=max(1, int(move_repeat)),
            include_non_move=bool(include_non_move),
        )
        rows.extend(seed_rows)

    if not rows:
        raise RuntimeError("no SFT rows generated")

    out_path = output_dir / parquet_name
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    manifest = {
        "num_rows": len(rows),
        "seeds": seed_list,
        "env_config": dataclasses.asdict(env_config),
        "parquet": str(out_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


async def _run_oracle_episode(
    *,
    seed: int,
    env_config: DeliveryBenchEnvConfig,
    image_dir: Path,
    move_repeat: int,
    include_non_move: bool,
    waypoint_marks: bool = False,
) -> List[Dict[str, Any]]:
    env = DeliveryBench(env_config)
    rows: List[Dict[str, Any]] = []
    turn_index = 0
    try:
        # Match rollout_qwen.py's prompt order: system prompt is captured before reset.
        system_text = (await env.system_prompt())["obs_str"]
        _assert_no_policy_leak(system_text)
        obs, _ = await env.reset(seed=seed)

        async def emit_and_step(
            action: str, stage: str, extra: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            nonlocal obs, turn_index
            turn_index += 1
            images = _save_pre_action_images(obs, image_dir=image_dir, seed=seed, turn_index=turn_index)
            user_text = obs.get("obs_str", "")
            _assert_no_policy_leak(user_text)
            _validate_prompt_images(user_text, images)
            assistant_text = _json_response(action, stage=stage)
            dm0 = env._env.dms[0] if env._env and env._env.dms else None

            repeats = move_repeat if stage == "move" else 1
            if include_non_move or stage == "move":
                for repeat_idx in range(repeats):
                    row = {
                        "messages": [
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": assistant_text},
                        ],
                        "images": images,
                        "seed": seed,
                        "turn_index": turn_index,
                        "stage": stage,
                        "action": action,
                        "repeat_index": repeat_idx,
                        # pre-action agent state: lets audits replay the node
                        # sequence without re-running the env (e.g. the F1
                        # MOVE≡MOVE_TO oracle-identity gate).
                        "dm_x": float(getattr(dm0, "x", 0.0)) if dm0 else 0.0,
                        "dm_y": float(getattr(dm0, "y", 0.0)) if dm0 else 0.0,
                        "gt_rel_direction": str((extra or {}).get("gt_rel_direction", "")),
                    }
                    rows.append(row)

            obs, _reward, done, info = await env.step(assistant_text)
            if info.get("action_error"):
                raise RuntimeError(f"oracle action failed at seed={seed}, turn={turn_index}: {info['action_error']}")
            return obs, info

        await emit_and_step("VIEW_ORDERS()", "view_orders")
        selected = _first_available_order(env)
        oid = int(getattr(selected, "id"))
        await emit_and_step(f"ACCEPT_ORDER({oid})", "accept")

        order = _active_order(env, oid)
        pickup = _node_target(getattr(order, "pickup_node", None))
        await emit_and_step(f"NAVIGATE(target={_quote_arg(pickup)}, mode=\"walk\")", "navigate_pickup")
        await _follow_active_route(env, emit_and_step, waypoint_marks=waypoint_marks)

        await emit_and_step(f"PICKUP(orders=[{oid}])", "pickup")

        order = _active_order(env, oid)
        dropoff = _node_target(getattr(order, "dropoff_node", None))
        await emit_and_step(f"NAVIGATE(target={_quote_arg(dropoff)}, mode=\"walk\")", "navigate_dropoff")
        await _follow_active_route(env, emit_and_step, waypoint_marks=waypoint_marks)

        await emit_and_step(f"DROP_OFF(oid={oid})", "dropoff")
        if not _delivery_complete(env, oid):
            raise RuntimeError(f"oracle episode seed={seed} did not complete delivery #{oid}")
        return rows
    finally:
        await env.close()


def _oracle_move_to_action(env: DeliveryBench, phrase: Optional[str]) -> Tuple[str, str]:
    """Translate the oracle's next_move phrase into the equivalent MOVE_TO(k).

    Deliberately routed through available_moves: the marks oracle steps the
    SAME edge the MOVE oracle would, expressed as the enumerate_candidates
    mark index — so the two oracles' node sequences are identical by
    construction. Returns (action, relative_direction); the latter is the
    ground-truth balancing bucket (kills the forward-heavy skew without
    upsampling).
    """
    from ..vlm_delivery.actions.move import available_moves, enumerate_candidates

    direction = {
        "move forward": "forward",
        "move backward": "backward",
        "turn left": "left",
        "turn right": "right",
    }.get(str(phrase or ""))
    if direction is None:
        raise RuntimeError(f"marks oracle: unmapped next_move phrase {phrase!r}")
    dm = env._env.dms[0]
    target = available_moves(dm).get(direction)
    if target is None:
        raise RuntimeError(f"marks oracle: no reachable waypoint to the {direction}")
    node = target.get("node")
    for cand in enumerate_candidates(dm):
        if cand.get("node") is node:
            return f"MOVE_TO({cand['index']})", direction
    raise RuntimeError("marks oracle: MOVE target missing from candidates")


async def _follow_active_route(env: DeliveryBench, emit_and_step, *,
                               waypoint_marks: bool = False) -> None:
    """Execute oracle MOVE/MOVE_TO actions until the active NAVIGATE route
    reports arrival."""
    max_moves = int(getattr(env.config, "max_steps", 100))
    for _ in range(max_moves):
        nav = env._visual_route_following_info()
        if nav.get("route_arrived"):
            return
        if waypoint_marks:
            action, rel_dir = _oracle_move_to_action(env, nav.get("oracle_next_move"))
            await emit_and_step(action, "move", extra={"gt_rel_direction": rel_dir})
            continue
        action = nav.get("oracle_next_action")
        if not action:
            raise RuntimeError(f"active route has no oracle MOVE action: {nav}")
        await emit_and_step(str(action), "move")
    raise RuntimeError(f"route did not arrive within {max_moves} oracle MOVE actions")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="directory for parquet, manifest, and images")
    parser.add_argument("--seeds", default="200-263", help="comma-separated seeds/ranges, e.g. 200-263,300")
    parser.add_argument("--config-yaml", default=None, help="optional rollout YAML; its env section is used")
    parser.add_argument("--max-steps", type=int, default=None, help="override env max_steps")
    parser.add_argument("--parquet-name", default="visual_sft.parquet")
    parser.add_argument("--move-repeat", type=int, default=1, help="duplicate MOVE rows this many times")
    parser.add_argument("--move-only", action="store_true", help="write only MOVE rows while still driving full workflow")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_env_config(
        Path(args.config_yaml).resolve() if args.config_yaml else None,
        max_steps=args.max_steps,
    )
    parquet = asyncio.run(
        export_dataset(
            output_dir=Path(args.output_dir),
            seeds=parse_seed_list(args.seeds),
            env_config=cfg,
            parquet_name=args.parquet_name,
            move_repeat=args.move_repeat,
            include_non_move=not args.move_only,
        )
    )
    print(f"wrote {parquet}")


if __name__ == "__main__":
    main()
