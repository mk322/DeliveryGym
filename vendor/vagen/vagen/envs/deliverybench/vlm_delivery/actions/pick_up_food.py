# actions/pick_up_food.py
# -*- coding: utf-8 -*-

from typing import Any, List
import math

from ..base.defs import DMAction
from ..utils.util import get_tol, xy_of_node, is_at_xy


def handle_pickup_food(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles picking up food orders that are ready and located at the agent’s position.
    """
    dm.vlm_clear_ephemeral()
    orders = list(act.data.get("orders") or [])
    tol = float(act.data.get("tol_cm", get_tol(dm.cfg, "nearby")))

    # Find orders that are ready for pickup and within range
    here_orders = []
    for o in orders:
        if getattr(o, "has_picked_up", False):
            continue

        pu_xy = xy_of_node(getattr(o, "pickup_node", None))
        if pu_xy and is_at_xy(dm, pu_xy[0], pu_xy[1], tol_cm=tol):
            # Determine readiness using active-time logic
            is_ready = (
                True
                if not hasattr(o, "is_ready_for_pickup")
                else o.is_ready_for_pickup(o.active_now())
            )
            if is_ready:
                here_orders.append(o)

    if not here_orders:
        # Build a diagnostic message: distinguish "not ready" from "too far".
        hints = []
        for o in orders:
            if getattr(o, "has_picked_up", False):
                continue
            pu_xy = xy_of_node(getattr(o, "pickup_node", None))
            if pu_xy is None:
                continue
            dist_cm = math.hypot(dm.x - pu_xy[0], dm.y - pu_xy[1])
            oid = getattr(o, "id", "?")

            if dist_cm <= tol:
                # At the door — the only reason we're here is readiness.
                remain_s = 0.0
                if hasattr(o, "remaining_prep_s"):
                    try:
                        remain_s = float(o.remaining_prep_s(o.active_now()))
                    except Exception:
                        remain_s = 0.0
                if remain_s > 0:
                    mins = max(1, int(round(remain_s / 60.0)))
                    hints.append(
                        f"Order #{oid}: you are at the pickup door, but the food is "
                        f"still being prepared (~{mins} min). WAIT and try PICKUP again."
                    )
                else:
                    hints.append(
                        f"Order #{oid}: you are at the pickup door, but it is not "
                        f"ready for pickup yet. WAIT and try PICKUP again."
                    )
                continue
            
            door_x_m = pu_xy[0] / 100.0
            door_y_m = pu_xy[1] / 100.0
            hints.append(
                f"Order #{oid}: pickup door is {dist_cm/100:.1f}m away at "
                f"({door_x_m:.2f}m, {door_y_m:.2f}m) — tolerance is {tol/100:.1f}m. "
                "Continue following the highlighted route with MOVE(direction=...) until [pickup_hint] appears."
            )
        msg = "Nothing to pick up here. " + " ".join(hints) if hints else "Nothing to pick up here."
        dm.vlm_add_error(msg)
        dm._finish_action(success=False)
        return

    amb_pickup = dm.ambient_temp_c     # ambient temp at pickup
    k = dm.k_food_per_s                # decay constant

    picked_ids: List[int] = []
    for o in here_orders:
        # Optional locking for safety
        lock = getattr(o, "_lock", None)
        ctx = lock if lock is not None else None

        if ctx is not None:
            # If lock exists, use context manager
            with ctx:
                _pickup_one_order(dm, o, amb_pickup, k, picked_ids)
        else:
            _pickup_one_order(dm, o, amb_pickup, k, picked_ids)

    if picked_ids:
        dm._log(
            f"picked order #{picked_ids[0]}"
            if len(picked_ids) == 1
            else f"picked orders {picked_ids}"
        )

    # Only require bag placement when bag compartments are enabled.
    # When disabled, items are already in dm.carrying — no placement step needed.
    if dm.cfg.get("enable_bag_compartments", True):
        dm._force_place_food_now = True
        dm.vlm_ephemeral["bag_hint"] = dm._build_bag_place_hint()

    dm._finish_action(success=True)


def _pickup_one_order(dm: Any, o: Any, amb_pickup: float, k: float, picked_ids: List[int]) -> None:
    """
    Handles pickup logic for a single order, including temperature decay and state updates.
    """
    oid = int(getattr(o, "id", -1))
    items = list(getattr(o, "items", []) or [])

    now_active = o.active_now()
    ready_at = o.ready_at()

    # Compute active-time offset since ready, clipped to non-negative
    time_from_ready = max(
        0.0,
        float(getattr(o, "sim_elapsed_active_s", 0.0))
        - float(getattr(o, "prep_longest_s", 0.0)),
    )

    for it in items:
        # Initialize temperature if missing or NaN
        tc = float(getattr(it, "temp_c", float("nan")))
        if math.isnan(tc):
            it.temp_c = float(getattr(it, "serving_temp_c", 25.0))

        # Record active-time-based timestamps if fields exist
        if hasattr(it, "prepared_at_sim"):
            it.prepared_at_sim = float(ready_at)
        if hasattr(it, "picked_at_sim"):
            it.picked_at_sim = float(now_active)

        # Restaurant holds food at serving temp until pickup; decay starts on carry

    # Mark order as picked up
    o.has_picked_up = True
    if oid not in dm.carrying:
        dm.carrying.append(oid)

    # Queue items for bag placement only when compartments are enabled.
    if dm.cfg.get("enable_bag_compartments", True):
        cur = dm._pending_food_by_order.get(oid, [])
        cur += items
        dm._pending_food_by_order[oid] = cur

    picked_ids.append(oid)
