# actions/view_orders.py
# -*- coding: utf-8 -*-

from typing import Any, List
from ..base.defs import DMAction
from ..utils.order_feasibility import (
    DEFAULT_NON_MOVE_ACTIONS,
    DEFAULT_ORDER_STEP_BUDGET,
    is_order_feasible,
)


def _order_allowed(dm: Any, order: Any, city_map: Any, step_budget: int, non_move_actions: int) -> bool:
    cfg = getattr(dm, "cfg", {}) or {}
    enable_feasible = bool(cfg.get("enable_feasible_orders", True))
    enable_infeasible = bool(cfg.get("enable_infeasible_orders", True))
    feasible, est = is_order_feasible(
        city_map=city_map,
        start_x_cm=float(getattr(dm, "x", 0.0)),
        start_y_cm=float(getattr(dm, "y", 0.0)),
        order=order,
        step_budget=step_budget,
        non_move_actions=non_move_actions,
    )
    try:
        order.feasibility_estimate = est
        order.is_feasible_order = feasible
    except Exception:
        pass
    return (feasible and enable_feasible) or ((not feasible) and enable_infeasible)


def _apply_feasibility_filter(dm: Any, om: Any) -> bool:
    cfg = getattr(dm, "cfg", {}) or {}
    enable_feasible = bool(cfg.get("enable_feasible_orders", True))
    enable_infeasible = bool(cfg.get("enable_infeasible_orders", True))
    if enable_feasible and enable_infeasible:
        return True
    if not enable_feasible and not enable_infeasible:
        dm.vlm_add_error(
            "view_orders failed: enable_feasible_orders and enable_infeasible_orders "
            "cannot both be false"
        )
        return False

    city_map = getattr(om, "_city_map", None) or getattr(dm, "city_map", None)
    world_nodes = getattr(om, "_world_nodes", None) or getattr(dm, "world_nodes", None)
    if city_map is None or world_nodes is None:
        return True

    step_budget = int(cfg.get("feasible_order_step_budget", DEFAULT_ORDER_STEP_BUDGET))
    non_move_actions = int(cfg.get("feasible_order_non_move_actions", DEFAULT_NON_MOVE_ACTIONS))
    capacity = int(getattr(om, "capacity", 0) or 0)
    max_attempts = max(200, capacity * 300)

    with om._lock:
        kept: List[Any] = [
            order for order in list(getattr(om, "_orders", []) or [])
            if _order_allowed(dm, order, city_map, step_budget, non_move_actions)
        ]
        om._orders = kept[:capacity] if capacity > 0 else kept

        attempts = 0
        while capacity > 0 and len(om._orders) < capacity and attempts < max_attempts:
            attempts += 1
            order = om._spawn_one_order(city_map, world_nodes)
            if _order_allowed(dm, order, city_map, step_budget, non_move_actions):
                om._orders.append(order)

        if capacity > 0 and len(om._orders) < capacity:
            dm.vlm_add_error(
                "view_orders warning: could not fully refill the feasibility-filtered "
                f"order pool after {attempts} attempts"
            )
    return True


def handle_view_orders(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Display current order-pool information if available.
    """
    om = act.data.get("order_manager") or dm._order_manager

    # Retrieve order pool text if the manager provides it
    if om and hasattr(om, "orders_text"):
        if not _apply_feasibility_filter(dm, om):
            dm._finish_action(success=False)
            return
        pool_text = om.orders_text()
        if pool_text:
            dm.vlm_add_ephemeral("order_pool", pool_text)
            dm._log("view orders")

    dm._finish_action(success=True)
