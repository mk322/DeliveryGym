"""
Regression tests for NAVIGATE target resolution against active order endpoints.

Run:
    PYTHONPATH=. python3 -m vagen.envs.deliverybench.test_navigation_target_resolution
"""

import asyncio
import json
import sys
from pathlib import Path


def _act(s: str) -> str:
    return json.dumps({"action": s})


def _cfg(**over):
    base = dict(
        map_name="small-city-11",
        render_mode="text",
        enable_map_images=False,
        enable_fpv=False,
        use_gmaps_renderer=False,
        enable_battery=True,
        enable_walking_energy=False,
        enable_food_temperature=False,
        enable_food_smell=False,
        enable_food_fragility=False,
        enable_bag_compartments=False,
        enable_advanced_transport=False,
        enable_delivery_methods=False,
        enable_special_notes=False,
        enable_multi_agent=False,
        initial_transport_mode="e-scooter",
        deadline_multiplier=1.5,
        max_orders_in_pool=3,
        num_restaurants=1,
        num_customers=1,
        require_single_item=True,
        enable_earning_jitter=False,
        fixed_spawn_position=[-17.0, 256.58],
        enable_prep_time=False,
        enabled_actions=[
            "VIEW_ORDERS",
            "ACCEPT_ORDER",
            "PICKUP",
            "DROP_OFF",
            "WAIT",
            "MOVE",
            "NAVIGATE",
        ],
        max_steps=20,
    )
    base.update(over)
    return base


async def test_seed72_active_order_address_targets_pickup_node():
    from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig
    from vagen.envs.deliverybench.deliverybench_env import DeliveryBench as DeliveryBenchClass
    from vagen.envs.deliverybench.vlm_delivery.actions._nav_helpers import (
        order_endpoint_by_target,
        resolve_navigation_target,
    )
    from vagen.envs.deliverybench.vlm_delivery.utils.vlm_prompt import _live_next_move_line

    if not Path(DeliveryBenchEnvConfig().base_dir).exists():
        print("SKIP test_seed72_active_order_address_targets_pickup_node (no map data)")
        return

    env = DeliveryBench(_cfg())
    try:
        await env.reset(seed=72)
        dm = env._env.dms[0]
        om = getattr(dm, "_order_manager", None)
        pool = list(getattr(om, "_orders", []) or [])
        assert pool, "seed72 should expose an order pool"
        assert getattr(pool[0], "id", None) == 0, "seed72 regression expects order #0 first"

        _, _, _, info = await env.step(_act("ACCEPT_ORDER(0)"))
        assert not info.get("action_error"), info.get("action_error")
        order = dm.active_orders[0]
        pickup = order.pickup_node
        pickup_addr = getattr(pickup, "address", "")
        assert pickup_addr == "200 Cherry St", pickup_addr

        assert order_endpoint_by_target(dm, pickup_addr) is pickup
        assert resolve_navigation_target(dm, pickup_addr) is pickup
        assert resolve_navigation_target(dm, "pickup of order #0") is pickup

        _, _, _, info = await env.step(
            _act(f'NAVIGATE(target="{pickup_addr}", mode="e-scooter")')
        )
        assert not info.get("action_error"), info.get("action_error")
        assert getattr(dm, "_nav_target_node", None) is pickup

        move_by_hint = {
            "next_move: move forward": 'MOVE(direction="forward")',
            "next_move: move backward": 'MOVE(direction="backward")',
            "next_move: turn left": 'MOVE(direction="left")',
            "next_move: turn right": 'MOVE(direction="right")',
        }
        for _ in range(40):
            live = _live_next_move_line(dm)
            if live == "next_move: you have arrived":
                break
            action = move_by_hint.get(live)
            assert action, live
            _, _, _, info = await env.step(_act(action))
            assert not info.get("action_error"), info.get("action_error")

        assert _live_next_move_line(dm) == "next_move: you have arrived"
        assert getattr(dm, "_nav_target_node", None) is pickup
        assert DeliveryBenchClass._live_nav_route(dm) is None
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            res = close()
            if asyncio.iscoroutine(res):
                await res


_ASYNC = [test_seed72_active_order_address_targets_pickup_node]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[4]))
    failed = []
    for t in _ASYNC:
        try:
            asyncio.run(t())
            print(f"PASS {t.__name__}")
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} failed: {failed}")
        sys.exit(1)
    print(f"\nAll {len(_ASYNC)} tests passed.")
