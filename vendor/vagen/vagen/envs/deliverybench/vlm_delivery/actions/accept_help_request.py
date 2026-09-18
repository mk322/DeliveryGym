# actions/accept_help_request.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction
from ..gameplay.comms import get_comms, HelpType


def _attach_helper_order(dm: Any, order_obj: Any) -> None:
    """
    Local replacement for DeliveryMan._attach_helper_order(self, order_obj).
    Here we bind the order to a *local helper* object (dm instead of self).

    This function:
    - Stores another agent's order reference in the helper-specific container.
    - Does NOT modify `is_accepted` or the global order pool.
    - Ensures helper-side bookkeeping without affecting global state.
    """
    oid = int(getattr(order_obj, "id", -1))
    if oid <= 0:
        return

    # Store into the helper's order dictionary
    dm.help_orders[oid] = order_obj
    dm.helping_order_ids.add(oid)  # Keep compatibility with previous marking mechanism

    # Local-only start time for display purposes (not used for billing, not written back to pool)
    if getattr(order_obj, "sim_started_s", None) is None:
        order_obj.sim_started_s = float(dm.clock.now_sim())
        order_obj.sim_elapsed_active_s = 0.0

    dm._log(f"attached helper order #{oid} (kept outside active_orders)")


def handle_accept_help_request(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Equivalent to DeliveryMan._handle_accept_help_request(self, act, _allow_interrupt),
    but rewritten so that `self` becomes `dm`. No logic changes.

    Handles:
    - Accepting a help request via comms.
    - Recording request metadata.
    - Attaching helper-side order references when applicable.
    - Updating helper bookkeeping and finishing the action.
    """
    comms = get_comms()
    if not comms:
        dm.vlm_add_error("accept_help_request failed: no comms")
        dm._finish_action(success=False)
        return

    req_id = int(act.data.get("req_id"))
    ok, msg = comms.accept_request(req_id=req_id, helper_id=dm.agent_id)
    if not ok:
        dm.vlm_add_error(f"accept_help_request failed: {msg}")
        dm._finish_action(success=False)
        return

    dm._log(f"accepted help request #{req_id}")

    # Map HELP_DELIVERY / HELP_PICKUP <-> order_id, and attach helper-side order object
    det = get_comms().get_request_detail(req_id=req_id) or {}
    kind = det.get("kind")
    kind_str = str(kind)

    if det.get("order_id") is not None and kind_str in (
        HelpType.HELP_DELIVERY.value,
        HelpType.HELP_DELIVERY.name,
        HelpType.HELP_PICKUP.value,
        HelpType.HELP_PICKUP.name,
    ):
        oid = int(det["order_id"])
        dm.helping_order_ids.add(oid)

        # Store mapping for later use (e.g., during helper DROP_OFF)
        dm._help_delivery_req_by_oid[oid] = int(det.get("id", req_id))

        order_obj = det.get("order_ref")
        if order_obj is not None:
            _attach_helper_order(dm, order_obj)

    # Record into accepted_help for UI/display
    req_obj = get_comms().get_request(req_id)
    if req_obj is not None:
        dm.accepted_help[int(req_id)] = req_obj

    # Update recorder if enabled
    if dm._recorder:
        dm._recorder.inc("help_accepted", 1)

    dm._finish_action(success=True)