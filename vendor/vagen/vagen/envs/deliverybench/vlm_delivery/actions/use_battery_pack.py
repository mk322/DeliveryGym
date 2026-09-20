# actions/use_battery_pack.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, TransportMode, ITEM_ESC_BATTERY_PACK
from ..entities.escooter import ScooterState
from ..utils.util import is_at_xy, get_tol, fmt_xy_m


def handle_use_battery_pack(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Use an e-scooter battery pack to fully recharge the scooter.
    """
    dm.vlm_clear_ephemeral()

    # Basic validation: scooter must exist and belong to this agent
    if not dm.e_scooter:
        dm.vlm_add_error("use_battery_pack failed: no e-scooter")
        dm._finish_action(success=False)
        return

    if not getattr(dm.e_scooter, "with_owner", True):
        dm.vlm_add_error(
            "use_battery_pack failed: your e-scooter is currently handed off "
            "(in a TempBox). Retrieve it first."
        )
        dm._finish_action(success=False)
        return

    item_id = act.data.get("item_id", ITEM_ESC_BATTERY_PACK)
    cnt = int(dm.inventory.get(item_id, 0))
    if cnt <= 0:
        dm.vlm_add_error(f"use_battery_pack failed: inventory=0 ({item_id})")
        dm._finish_action(success=False)
        return

    # Check proximity: must be riding/dragging the scooter, or standing near the parked scooter
    tol = float(act.data.get("tol_cm", get_tol(dm.cfg, "nearby")))
    own_scooter_in_hand = (
        getattr(dm.e_scooter, "with_owner", True)
        and (
            dm.mode == TransportMode.SCOOTER
            or (dm.mode == TransportMode.DRAG_SCOOTER and dm.assist_scooter is None)
        )
    )

    scooter_is_with_me = own_scooter_in_hand
    scooter_is_parked_nearby = False

    if getattr(dm.e_scooter, "with_owner", True) and dm.e_scooter.park_xy:
        px, py = dm.e_scooter.park_xy
        scooter_is_parked_nearby = is_at_xy(dm, float(px), float(py), tol_cm=tol)

    if not (scooter_is_with_me or scooter_is_parked_nearby):
        if dm.e_scooter.park_xy:
            px, py = dm.e_scooter.park_xy
            dm.vlm_add_error(
                "use_battery_pack failed: not near your e-scooter "
                f"(parked at {fmt_xy_m(px, py)}). MOVE there first."
            )
        else:
            dm.vlm_add_error("use_battery_pack failed: scooter location unknown")

        # Soft hint to guide the model to move near the scooter
        dm.vlm_ephemeral["charging_hint"] = (
            "Go to your parked scooter (MOVE to its coordinates) "
            "before using a battery pack."
        )
        dm._finish_action(success=False)
        return

    # Consume one battery pack and recharge
    dm.inventory[item_id] = cnt - 1
    target = float(
        dm.cfg.get("items", {})
        .get(ITEM_ESC_BATTERY_PACK, {})
        .get("target_charge_pct", 100)
    )
    dm.e_scooter.charge_to(target)

    # Update scooter state and agent mode after charging
    # - If dragging: switch to riding since the scooter is now usable
    # - If riding: keep riding
    # - If parked nearby: remain parked (do not mount automatically)
    if dm.mode == TransportMode.DRAG_SCOOTER:
        dm.e_scooter.state = ScooterState.USABLE
        dm.set_mode(TransportMode.SCOOTER)
    elif dm.mode == TransportMode.SCOOTER:
        dm.e_scooter.state = ScooterState.USABLE
    else:
        dm.e_scooter.state = ScooterState.PARKED

    dm._log(
        f"used '{item_id}': scooter battery -> 100% "
        f"(remaining {dm.inventory.get(item_id, 0)})"
    )
    dm._recorder.inc_preventive("use_scooter_battery_pack")
    dm._finish_action(success=True)