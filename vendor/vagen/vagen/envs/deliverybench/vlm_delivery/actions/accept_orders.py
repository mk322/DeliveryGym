# actions/accept_orders.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_accept_order(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles accepting one or more orders and attaching them to the agent.
    """
    dm.vlm_clear_ephemeral()

    om = dm._order_manager
    if om is None:
        dm.vlm_add_error("accept_order failed: no order manager")
        dm._finish_action(success=False)
        return

    # ------------------------------------------------------
    # 1) Normalize input into a deduplicated list of order IDs
    # ------------------------------------------------------
    if "oids" in act.data:
        oids = [int(x) for x in (act.data.get("oids") or [])]
    elif "oid" in act.data:
        oids = [int(act.data.get("oid"))]
    else:
        dm.vlm_add_error("accept_order failed: need 'oid' or 'oids'")
        dm._finish_action(success=False)
        return

    oids = list(dict.fromkeys([i for i in oids if isinstance(i, int)]))
    if not oids:
        dm.vlm_add_error("accept_order failed: empty ids")
        dm._finish_action(success=False)
        return

    # ------------------------------------------------------
    # 2) Call the order manager to accept orders in batch
    # ------------------------------------------------------
    accepted_ids, failed_ids = om.accept_order(oids)
    accepted_ids = list(accepted_ids or [])
    failed_ids   = list(failed_ids or [])

    # ------------------------------------------------------
    # 3) Attach accepted orders to active_orders
    #    and remove them from the global order pool
    # ------------------------------------------------------
    active_seeds = [
        (float(o2.pickup_address.x), float(o2.pickup_address.y))
        for o2 in dm.active_orders
    ]

    # Compute relative scores for accepted orders
    rel_scores = om.relative_scores_for(
        agent_xy=(dm.x, dm.y),
        active_seeds_xy=active_seeds,
        order_ids=accepted_ids,
    )

    # Add to active_orders with assigned scores
    for oid, s in zip(accepted_ids, rel_scores):
        o = om.get(int(oid))
        if o is not None and all(o is not x for x in dm.active_orders):
            dm._log(f"order #{oid} relative score = {s:.2f}")
            o.pick_score = float(s)
            dm.active_orders.append(o)

    # Remove accepted orders from global pool
    if accepted_ids:
        om.remove_order(accepted_ids, dm.city_map, dm.world_nodes)

    # ------------------------------------------------------
    # 4) Logging summary
    # ------------------------------------------------------
    acc_txt = " ".join(f"#{i}" for i in accepted_ids) if accepted_ids else "none"

    if failed_ids:
        fail_txt = " ".join(f"#{i}" for i in failed_ids)
        msg = (
            f"accept orders: accepted {acc_txt}; failed {fail_txt} "
            f"(not found or already accepted by others)"
        )
    else:
        msg = f"accept orders: accepted {acc_txt}"

    dm._log(msg)

    # ------------------------------------------------------
    # 5) Final status
    # ------------------------------------------------------
    if accepted_ids:
        dm._finish_action(success=True)
    else:
        dm.vlm_add_error(f"accept_order failed: {msg}")
        dm._finish_action(success=False)