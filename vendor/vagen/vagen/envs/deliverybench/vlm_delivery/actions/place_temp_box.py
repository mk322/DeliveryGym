# actions/place_temp_box.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List
from ..base.defs import DMAction, TransportMode
from ..gameplay.comms import get_comms
from ..utils.util import is_at_xy, get_tol


def _fmt_inv_compact(inv: Dict[str, int]) -> str:
    """Formats an inventory dictionary into a compact string."""
    if not inv:
        return "empty"
    parts = [f"{k} x{int(v)}" for k, v in inv.items() if int(v) > 0]
    return ", ".join(parts) if parts else "empty"


def handle_place_temp_box(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Places items into a temporary box for a help request.

    Accepted content fields:
      - inventory:  {item_id: qty, ...}   (deducted from local inventory on success)
      - food:       key presence indicates placing *all* undelivered food items
      - escooter:   key presence indicates placing an e-scooter (assist or own)

    Only on successful placement will local states be updated.
    """

    comms = get_comms()
    if not comms:
        dm.vlm_add_error("place_temp_box failed: no comms")
        dm._finish_action(success=False)
        return

    req_id = int(act.data.get("req_id"))
    location_xy = tuple(act.data.get("location_xy") or (dm.x, dm.y))
    content_req = dict(act.data.get("content") or {})

    payload: Dict[str, Any] = {}

    # -----------------------------------------------------
    # 1) Validate and assemble inventory section
    # -----------------------------------------------------
    inv_req = {str(k): int(v) for k, v in (content_req.get("inventory") or {}).items()}

    for k, q in inv_req.items():
        if int(dm.inventory.get(k, 0)) < int(q):
            dm.vlm_add_error(f"place_temp_box failed: lacking '{k}' x{int(q)}")
            dm._finish_action(success=False)
            return

    if inv_req:
        payload["inventory"] = dict(inv_req)

    # -----------------------------------------------------
    # 2) Assemble food section — if "food" key exists,
    #    place *all* picked-up and undelivered items
    # -----------------------------------------------------
    want_food = ("food" in content_req)
    food_by_order: Dict[int, List[Any]] = {}

    if want_food:
        for o in list(dm.active_orders or []):
            if getattr(o, "has_picked_up", False) and not getattr(o, "has_delivered", False):
                items = list(getattr(o, "items", []) or [])
                if items:
                    food_by_order[int(getattr(o, "id", -1))] = items

        if not food_by_order:
            dm.vlm_add_error("place_temp_box failed: no food to place")
            dm._finish_action(success=False)
            return

        payload["food_by_order"] = {int(k): list(v) for k, v in food_by_order.items()}

    # -----------------------------------------------------
    # 3) Assemble e-scooter section — presence of "escooter"
    #    means placing the full scooter
    # -----------------------------------------------------
    give_scooter = ("escooter" in content_req)

    if give_scooter:
        scooter_to_place = (
            dm.assist_scooter if dm.assist_scooter is not None else dm.e_scooter
        )

        if scooter_to_place is None:
            dm.vlm_add_error("place_temp_box failed: no e-scooter to place")
            dm._finish_action(success=False)
            return

        is_my_scooter = (
            getattr(scooter_to_place, "owner_id", None) == str(dm.agent_id)
        )

        # Prevent placing the same owned scooter twice
        if is_my_scooter and not getattr(dm.e_scooter, "with_owner", True):
            dm.vlm_add_error("place_temp_box failed: your e-scooter has already been handed off")
            dm._finish_action(success=False)
            return

        # If scooter is parked, ensure the agent is near it
        if scooter_to_place.park_xy:
            px, py = scooter_to_place.park_xy
            if not is_at_xy(dm, px, py, tol_cm=get_tol(dm.cfg, "nearby")):
                dm.vlm_add_error("place_temp_box failed: not near the e-scooter to place")
                dm._finish_action(success=False)
                return
            scooter_to_place.unpark()

        lx, ly = location_xy
        scooter_to_place.park_here(float(lx), float(ly))
        payload["escooter"] = scooter_to_place

        # Mark the owned scooter as handed off
        if is_my_scooter:
            setattr(dm.e_scooter, "with_owner", False)

    # -----------------------------------------------------
    # No content selected
    # -----------------------------------------------------
    if not (
        payload.get("inventory")
        or payload.get("food_by_order")
        or payload.get("escooter")
    ):
        dm.vlm_add_error("place_temp_box failed: empty content")
        dm._finish_action(success=False)
        return

    # -----------------------------------------------------
    # 4) Call Comms to place the TempBox
    # -----------------------------------------------------
    _inv_before = dict(dm.inventory)
    _had_scooter_before = dm._has_scooter()

    ok, msg = comms.place_temp_box(
        req_id=req_id,
        by_agent=str(dm.agent_id),
        location_xy=location_xy,
        content=payload,
    )
    if not ok:
        dm.vlm_add_error(f"place_temp_box failed: {msg}")
        dm._finish_action(success=False)
        return

    # -----------------------------------------------------
    # 5) Update local states on success
    # -----------------------------------------------------

    # Deduct inventory
    for k, q in inv_req.items():
        dm.inventory[k] = int(dm.inventory.get(k, 0)) - int(q)
        if dm.inventory[k] <= 0:
            dm.inventory.pop(k, None)

    # Remove food items from bag and pending lists
    if want_food:
        if dm.insulated_bag:
            all_items: List[Any] = []
            for items in food_by_order.values():
                all_items.extend(items)
            if all_items:
                dm.insulated_bag.remove_items(all_items)

        for oid in list(food_by_order.keys()):
            dm._pending_food_by_order.pop(oid, None)
            if oid in dm.carrying:
                dm.carrying.remove(oid)

    # Reset scooter state
    if give_scooter:
        was_using = (dm.mode in (TransportMode.SCOOTER, TransportMode.DRAG_SCOOTER))

        if dm.assist_scooter is not None:
            dm.assist_scooter = None

        if was_using:
            dm.set_mode(TransportMode.WALK)

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    dm._log(f"placed TempBox for request #{req_id}")

    if inv_req:
        dm._log(
            f"TempBox[#{req_id}] placed inventory: "
            f"{_fmt_inv_compact(_inv_before)} -> {_fmt_inv_compact(dm.inventory)}"
        )

    if want_food:
        for _oid in sorted(food_by_order.keys()):
            dm._log(f"TempBox[#{req_id}] placed food for order #{int(_oid)}")

    if give_scooter:
        _had_scooter_after = dm._has_scooter()
        dm._log(
            f"TempBox[#{req_id}] placed e-scooter: "
            f"{'present' if _had_scooter_before else 'absent'} -> "
            f"{'present' if _had_scooter_after else 'absent'}"
        )

    dm._finish_action(success=True)