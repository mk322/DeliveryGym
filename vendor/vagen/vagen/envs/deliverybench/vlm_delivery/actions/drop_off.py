# actions/drop_off.py
# -*- coding: utf-8 -*-

from typing import Any
import math

from ..base.defs import DMAction, VALID_DELIVERY_METHODS
from ..gameplay.settlement import compute_settlement
from ..gameplay.comms import get_comms, HelpType

from ..utils.util import get_tol, xy_of_node, is_at_xy

# How long the customer takes to reach the door after the courier knocks
# (sim seconds). The first hand_to_customer call at the dock acts as the knock;
# the agent WAITs this out, then hands off.
_CUSTOMER_COME_OUT_S = 30.0


def handle_drop_off(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles dropping off an order with a specified delivery method.
    """
    # 1) Parse inputs
    try:
        oid = int(act.data.get("oid"))
    except Exception:
        dm.vlm_add_error("drop_off failed: need integer 'oid'")
        dm._finish_action(success=False)
        return

    method = str(act.data.get("method", "")).strip().lower()
    if method not in VALID_DELIVERY_METHODS:
        dm.vlm_add_error(
            "drop_off failed: invalid 'method' "
            "(use one of leave_at_door|knock|call|hand_to_customer)"
        )
        dm._finish_action(success=False)
        return

    tol = float(act.data.get("tol_cm", get_tol(dm.cfg, "nearby")))

    # 2) Determine whether this is the agent's own order or a helper order
    order_obj = next(
        (o for o in dm.active_orders if int(getattr(o, "id", -1)) == oid),
        None,
    )
    is_helper = False
    if order_obj is None:
        order_obj = dm.help_orders.get(int(oid))
        is_helper = order_obj is not None

    if order_obj is None:
        dm.vlm_add_error(f"drop_off failed: order #{oid} not found on this agent")
        dm._finish_action(success=False)
        return

    # 3) Validate order state and agent location
    if not getattr(order_obj, "has_picked_up", False):
        dm.vlm_add_error("drop_off failed: order not picked up yet")
        dm._finish_action(success=False)
        return

    if getattr(order_obj, "has_delivered", False):
        dm.vlm_add_error("drop_off failed: order already delivered")
        dm._finish_action(success=False)
        return

    # Determine permitted delivery locations
    allowed_methods = getattr(order_obj, "allowed_delivery_methods", [])
    is_handoff_allowed = "hand_to_customer" in allowed_methods
    handoff_address = getattr(order_obj, "handoff_address", None)

    # Check dropoff_node location
    dxy = xy_of_node(getattr(order_obj, "dropoff_node", None))
    at_dropoff = bool(dxy and is_at_xy(dm, dxy[0], dxy[1], tol_cm=tol))

    # Check handoff address location
    at_handoff = False
    if handoff_address:
        hx, hy = float(handoff_address.x), float(handoff_address.y)
        at_handoff = is_at_xy(dm, hx, hy, tol_cm=tol)

    # Validate location based on delivery method
    if method == "hand_to_customer":
        # Hand-off happens at the dropoff DOCK (same place the agent can stand for
        # every other method). The customer must come to the door first: the first
        # hand_to_customer call at the dock "knocks" and starts a short come-out
        # timer; the agent WAITs it out, then hands the food over directly.
        if not at_dropoff and not at_handoff:
            dm.vlm_add_error(
                "drop_off failed: go to the drop-off location (its dock) before handing to the customer."
            )
            dm._finish_action(success=False)
            return
        now = float(dm.clock.now_sim())
        out_at = getattr(order_obj, "_customer_out_at_s", None)
        if out_at is None:
            order_obj._customer_out_at_s = now + _CUSTOMER_COME_OUT_S
            dm.vlm_add_error(
                f"You knocked. The customer needs ~{int(_CUSTOMER_COME_OUT_S)}s to reach the door. "
                "WAIT, then DROP_OFF(hand_to_customer) again to hand it over."
            )
            dm._finish_action(success=False)
            return
        if now < float(out_at):
            remain = int(round(float(out_at) - now))
            dm.vlm_add_error(
                f"The customer is still coming to the door (~{remain}s left). WAIT, then try again."
            )
            dm._finish_action(success=False)
            return
        # Customer is out → fall through to unload + settle.
    else:
        # Other methods allow either dropoff_node or handoff_address
        if not at_dropoff and not at_handoff:
            dm.vlm_add_error("drop_off failed: not at the drop-off location")
            dm._finish_action(success=False)
            return

    # 4) Record delivery method
    try:
        setattr(order_obj, "delivery_method", method)
    except Exception:
        pass

    # 5) Perform physical unload
    dropoff_physical_unload(dm, order_obj)

    # 6) Settlement: own order → settle locally; helper → push to comms
    if not is_helper:
        # Own order: settle and record
        dropoff_settle_record(dm, order_obj)
        dm._finish_action(success=True)

        # Remove customer entity if applicable
        if is_handoff_allowed and handoff_address and getattr(dm, "_ue", None):
            dm._ue.destroy_customer(order_obj.id)
        return

    # Helper logic: notify publisher and update helper state
    comms = get_comms()
    if not comms:
        dm.vlm_add_error("drop_off failed: comms unavailable for helper delivery")
        dm._finish_action(success=False)
        return

    req_id = int(dm._help_delivery_req_by_oid.get(int(oid), 0))
    if req_id <= 0:
        dm.vlm_add_error("drop_off failed: no req_id bound for this helper delivery")
        dm._finish_action(success=False)
        return

    ok, msg = comms.push_helper_delivered(
        req_id=req_id,
        by_agent=str(dm.agent_id),
        order_id=int(oid),
        at_xy=(dm.x, dm.y),
    )
    if not ok:
        dm.vlm_add_error(f"drop_off failed: {msg}")
        dm._finish_action(success=False)
        return

    # Update helper-side records
    dm.help_orders.pop(int(oid), None)
    dm.help_orders_completed.add(int(oid))
    dm._helping_wait_ack_oids.add(int(oid))
    dm._log(
        f"helper delivered order #{oid} with method '{method}', "
        f"pushed to Comm (req #{req_id})"
    )

    dm._finish_action(success=True)

    # Remove customer entity if applicable
    if is_handoff_allowed and handoff_address and getattr(dm, "_ue", None):
        dm._ue.destroy_customer(order_obj.id)


# ===== Physical Unload =====
def dropoff_physical_unload(dm: Any, order: Any) -> None:
    """
    Removes the order’s items from the insulated bag, carrying list,
    and pending-food queue.
    """
    oid = int(getattr(order, "id", -1))
    items = list(getattr(order, "items", []) or [])

    if dm.insulated_bag and hasattr(dm.insulated_bag, "remove_items") and items:
        dm.insulated_bag.remove_items(items)

    if oid in dm.carrying:
        dm.carrying.remove(oid)

    if dm._pending_food_by_order and oid in dm._pending_food_by_order:
        dm._pending_food_by_order.pop(oid, None)


# ===== Settlement and Record =====
def dropoff_settle_record(dm: Any, order: Any) -> None:
    """
    Performs settlement calculation, records delivery statistics,
    updates earnings, and logs results.
    """
    oid = getattr(order, "id", None)

    # Mark delivered and set timestamps
    order.has_delivered = True
    now_sim = float(dm.clock.now_sim())
    order.sim_delivered_s = now_sim
    for it in getattr(order, "items", []) or []:
        it.delivered_at_sim = now_sim

    # Compute settlement
    duration_s = float(getattr(order, "sim_elapsed_active_s", 0.0) or 0.0)
    time_limit_s = float(getattr(order, "time_limit_s", 0.0) or 0.0)
    base_earn = float(getattr(order, "earnings", 0.0) or 0.0)
    items = list(getattr(order, "items", []) or [])

    settle_res = compute_settlement(
        order_base_earnings=base_earn,
        duration_s=duration_s,
        time_limit_s=time_limit_s,
        items=items,
        order_allowed_delivery_methods=getattr(order, "allowed_delivery_methods", []),
        actual_delivery_method=getattr(order, "delivery_method", None),
        config=dm.cfg.get("settlement"),
    )
    dm.add_earnings(settle_res.total_pay)

    # Extract breakdown fields
    _bd = settle_res.breakdown or {}
    _time_star = int((_bd.get("time") or {}).get("time_star", 0))
    _food_star = int((_bd.get("food") or {}).get("food_star", 0))
    _method_star = int((_bd.get("method") or {}).get("method_star", 0))
    _flags = dict((_bd.get("flags") or {}))
    _on_time = bool(_flags.get("on_time", True))
    _temp_ok = bool(_flags.get("temp_ok_all", True))
    _odor_ok = bool(_flags.get("odor_ok_all", True))
    _dmg_ok = bool(_flags.get("damage_ok_all", True))

    _flags_detail = (
        f" [on_time={'Y' if _on_time else 'N'}, "
        f"temp={'OK' if _temp_ok else 'BAD'}, "
        f"odor={'OK' if _odor_ok else 'BAD'}, "
        f"damage={'OK' if _dmg_ok else 'BAD'}]"
    )

    # Append completion record
    dm.completed_orders.append(
        dict(
            id=oid,
            duration_s=duration_s,
            time_limit_s=time_limit_s,
            pick_score=float(getattr(order, "pick_score", 0.0) or 0.0),
            rating=float(settle_res.stars),
            earnings=float(settle_res.base_pay),
            bonus_extra=float(settle_res.extra_pay),
            paid_total=float(settle_res.total_pay),
            breakdown=settle_res.breakdown,
            pickup=getattr(order, "pickup_road_name", ""),
            dropoff=getattr(order, "dropoff_road_name", ""),
            allowed_delivery_methods=list(
                getattr(order, "allowed_delivery_methods", []) or []
            ),
            delivery_method=getattr(order, "delivery_method", None),
            stars=dict(
                overall=int(settle_res.stars),
                time=_time_star,
                food=_food_star,
                method=_method_star,
            ),
            flags=dict(
                on_time=_on_time,
                temp_ok_all=_temp_ok,
                odor_ok_all=_odor_ok,
                damage_ok_all=_dmg_ok,
            ),
        )
    )

    extra_str = f" (extra {settle_res.extra_pay:+.2f}, stars={settle_res.stars})"
    star_detail = f" [time={_time_star}, food={_food_star}, method={_method_star}]"

    dm._log(
        (
            f"dropped off order #{oid}{extra_str}{star_detail}{_flags_detail}"
            if oid is not None
            else f"dropped off order{extra_str}{star_detail}{_flags_detail}"
        )
    )

    # Remove from active_orders
    dm.active_orders = [
        o for o in dm.active_orders if getattr(o, "id", None) != oid
    ]

    # Clear bag hints if nothing pending
    if not dm._pending_food_by_order:
        dm._force_place_food_now = False
        dm.vlm_ephemeral.pop("bag_hint", None)

    dm.vlm_clear_errors()


# ===== Auto-dropoff for helper deliveries =====
def auto_try_dropoff(dm: Any) -> None:
    """
    Automatically processes completed helper deliveries based on messages
    from the communication system.
    """
    tol = get_tol(dm.cfg, "nearby")
    comms = get_comms()

    # Consume comm messages indicating helper-completed deliveries
    if comms:
        msgs = comms.pop_msgs_for_publisher(str(dm.agent_id))
        for m in msgs:
            if m.get("type") != "HELP_DELIVERY_DONE":
                continue
            oid = int(m.get("order_id", -1))
            if oid <= 0 or oid in dm.help_completed_order_ids:
                continue

            # Lookup active order still pending settlement
            order_obj = next(
                (
                    o
                    for o in dm.active_orders
                    if int(getattr(o, "id", -1)) == oid
                    and not getattr(o, "has_delivered", False)
                ),
                None,
            )
            if order_obj is None:
                continue

            # Settle only; physical unload was done by helper
            dropoff_settle_record(dm, order_obj)
            dm.help_completed_order_ids.add(oid)

    # Clean up helper-side orders that have already been settled
    for oid, o in list(dm.help_orders.items()):
        if getattr(o, "has_delivered", False):
            dm._helping_wait_ack_oids.discard(int(oid))
            dm.help_orders.pop(int(oid), None)