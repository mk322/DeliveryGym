# actions/take_from_temp_box.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List
from ..base.defs import DMAction, TransportMode
from ..gameplay.comms import get_comms
from ..utils.util import is_at_xy, get_tol, fmt_xy_m


def _fmt_inv_compact(inv: Dict[str, int]) -> str:
    """
    Compact inventory formatter used internally for logging.
    """
    if not inv:
        return "empty"
    parts = [f"{k} x{int(v)}" for k, v in inv.items() if int(v) > 0]
    return ", ".join(parts) if parts else "empty"


def handle_take_from_temp_box(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles retrieving items from a TempBox for a given help request.
    """
    comms = get_comms()
    if not comms:
        dm.vlm_add_error("take_from_temp_box failed: no comms")
        dm._finish_action(success=False)
        return

    req_id = int(act.data.get("req_id"))
    tol = float(act.data.get("tol_cm", get_tol(dm.cfg, "nearby")))

    # 1) Determine the agent's role in this request to decide which TempBox to use
    det = comms.get_request_detail(req_id) or {}
    me = str(dm.agent_id)
    role = None

    if str(det.get("accepted_by", "")) == me and str(det.get("publisher_id", "")) != me:
        role = "helper"
        expect_key = "publisher_box"  # Helper takes from the publisher's TempBox
    elif str(det.get("publisher_id", "")) == me:
        role = "publisher"
        expect_key = "helper_box"  # Publisher takes from the helper's TempBox
    else:
        dm.vlm_add_error("take_from_temp_box failed: not a participant of this request")
        dm._finish_action(success=False)
        return

    info = comms.get_temp_box_info(req_id) or {}
    box = info.get(expect_key) or {}

    # 2) Validate TempBox position and content
    if not box.get("xy"):
        dm.vlm_add_error("take_from_temp_box failed: temp box not available yet")
        dm._finish_action(success=False)
        return

    bx, by = box["xy"]
    if not is_at_xy(dm, float(bx), float(by), tol_cm=tol):
        dm.vlm_add_error(
            f"take_from_temp_box failed: not near the TempBox (at {fmt_xy_m(bx, by)}). "
            "MOVE there first."
        )
        # Provide a hint to the VLM about where to go
        dm.vlm_ephemeral["tempbox_hint"] = (
            f"Go to the TempBox at {fmt_xy_m(bx, by)} for request #{req_id}."
        )
        dm._finish_action(success=False)
        return

    if not box.get("has_content", False):
        dm.vlm_add_error("take_from_temp_box failed: the TempBox is empty")
        dm.vlm_ephemeral["tempbox_hint"] = (
            f"[Help #{req_id}] This TempBox is currently empty."
        )
        dm._finish_action(success=False)
        return

    # 3) Perform the take operation
    ok, msg, payload = comms.take_from_temp_box(req_id=req_id, by_agent=me)
    if not ok:
        dm.vlm_add_error(f"take_from_temp_box failed: {msg}")
        dm._finish_action(success=False)
        return

    # 4) Empty payload is treated as failure to avoid a no-op success
    if not (
        payload.get("inventory")
        or payload.get("food_by_order")
        or (payload.get("escooter") is not None)
    ):
        dm.vlm_add_error("take_from_temp_box failed: TempBox is empty")
        dm._finish_action(success=False)
        return

    _inv_before = dict(dm.inventory)
    _had_scooter_before = dm._has_any_scooter()

    # Merge inventory from TempBox into local inventory
    inv = dict(payload.get("inventory") or {})
    for k, q in inv.items():
        dm.inventory[str(k)] = int(dm.inventory.get(str(k), 0)) + int(q)

    # Handle e-scooter handover
    if payload.get("escooter") is not None:
        sc = payload["escooter"]

        # Own scooter
        if getattr(sc, "owner_id", None) == str(dm.agent_id):
            com = get_comms()
            if com:
                canon = com.get_scooter_by_owner(str(dm.agent_id)) or sc
                dm.e_scooter = canon
            else:
                dm.e_scooter = sc

            # Restore parked state and mark as with_owner
            try:
                dm.e_scooter.unpark()
            except Exception:
                dm.e_scooter.park_xy = None
            setattr(dm.e_scooter, "with_owner", True)

        # Assisting someone else's scooter
        else:
            if dm.assist_scooter is None:
                dm.assist_scooter = sc
                setattr(dm.assist_scooter, "proxy_helper_id", str(dm.agent_id))
                try:
                    dm.assist_scooter.unpark()
                except AttributeError:
                    dm.assist_scooter.park_xy = None

                dm.set_mode(TransportMode.DRAG_SCOOTER)

                # If own scooter is currently unparked and still with the owner, park it at current location
                if (
                    dm.e_scooter
                    and dm.e_scooter.park_xy is None
                    and getattr(dm.e_scooter, "with_owner", True)
                ):
                    dm.e_scooter.park_here(dm.x, dm.y)
            else:
                dm._log(
                    "take_from_temp_box: already assisting another scooter; "
                    "ignoring extra"
                )

    # Merge food items by order and update carrying / pending state
    fbo = payload.get("food_by_order") or {}
    if fbo:
        now_sim = dm.clock.now_sim()
        for oid, items in fbo.items():
            oid = int(oid)
            items_list = list(items or [])
            if not items_list:
                continue

            order_obj = dm.help_orders.get(oid)

            for it in items_list:
                if hasattr(it, "picked_at_sim"):
                    it.picked_at_sim = float(now_sim)

            if order_obj is not None:
                order_obj.has_picked_up = True
                if oid not in dm.carrying:
                    dm.carrying.append(oid)

            if oid not in dm.carrying:
                dm.carrying.append(oid)

            cur = dm._pending_food_by_order.get(oid, [])
            cur += items_list
            dm._pending_food_by_order[oid] = cur

        dm._force_place_food_now = True
        dm.vlm_ephemeral["bag_hint"] = dm._build_bag_place_hint()

    # Logging for inventory changes
    if inv:
        dm._log(
            f"TempBox[#{req_id}] took inventory: "
            f"{_fmt_inv_compact(_inv_before)} -> {_fmt_inv_compact(dm.inventory)}"
        )

    # Logging for food retrieval by order
    if fbo:
        for _oid in sorted(fbo.keys()):
            dm._log(f"TempBox[#{req_id}] took food for order #{int(_oid)}")

    # Logging for scooter retrieval
    if "escooter" in payload:
        _had_scooter_after = dm._has_any_scooter()
        dm._log(
            f"TempBox[#{req_id}] took e-scooter: "
            f"{'present' if _had_scooter_before else 'absent'} -> "
            f"{'present' if _had_scooter_after else 'absent'}"
        )

    dm._log(f"took items from TempBox for request #{req_id}")
    dm._finish_action(success=True)