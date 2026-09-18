# -*- coding: utf-8 -*-
"""Oracle-style order feasibility estimates for DeliveryBench."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


DEFAULT_ORDER_STEP_BUDGET = 20
DEFAULT_NON_MOVE_ACTIONS = 6


def estimate_order_steps(
    *,
    city_map: Any,
    start_x_cm: float,
    start_y_cm: float,
    order: Any,
    non_move_actions: int = DEFAULT_NON_MOVE_ACTIONS,
) -> Dict[str, Any]:
    """Estimate one-order completion steps from a current agent position.

    The estimate is structural and assumes perfect navigation:
    VIEW_ORDERS + ACCEPT_ORDER + NAVIGATE pickup + PICKUP + NAVIGATE dropoff
    + DROP_OFF, plus shortest-path waypoint edge counts for movement.
    """
    try:
        start = city_map.nearest_waypoint(float(start_x_cm), float(start_y_cm))
        pickup_node = getattr(order, "pickup_node", None)
        dropoff_node = getattr(order, "dropoff_node", None)
        if start is None or pickup_node is None or dropoff_node is None:
            raise RuntimeError("missing start, pickup, or dropoff node")

        approach_path, approach_dist_cm = city_map.shortest_path_nodes(start, pickup_node)
        delivery_path, delivery_dist_cm = city_map.shortest_path_nodes(pickup_node, dropoff_node)
        approach_moves = max(0, len(approach_path) - 1)
        delivery_moves = max(0, len(delivery_path) - 1)
        total_steps = int(non_move_actions) + approach_moves + delivery_moves
        return {
            "feasible_estimate_valid": True,
            "approach_moves": approach_moves,
            "delivery_moves": delivery_moves,
            "total_steps": total_steps,
            "approach_m": float(approach_dist_cm) / 100.0,
            "delivery_m": float(delivery_dist_cm) / 100.0,
        }
    except Exception as exc:
        return {
            "feasible_estimate_valid": False,
            "approach_moves": math.inf,
            "delivery_moves": math.inf,
            "total_steps": math.inf,
            "approach_m": math.inf,
            "delivery_m": math.inf,
            "error": str(exc),
        }


def is_order_feasible(
    *,
    city_map: Any,
    start_x_cm: float,
    start_y_cm: float,
    order: Any,
    step_budget: int = DEFAULT_ORDER_STEP_BUDGET,
    non_move_actions: int = DEFAULT_NON_MOVE_ACTIONS,
) -> Tuple[bool, Dict[str, Any]]:
    estimate = estimate_order_steps(
        city_map=city_map,
        start_x_cm=start_x_cm,
        start_y_cm=start_y_cm,
        order=order,
        non_move_actions=non_move_actions,
    )
    feasible = bool(estimate["total_steps"] <= int(step_budget))
    return feasible, estimate
