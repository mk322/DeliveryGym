# actions/use_ice_pack.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, ITEM_ICE_PACK
from ..entities.insulated_bag import IcePack


def handle_use_ice_pack(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Insert an ice pack into a specified insulated bag compartment.
    """
    dm.vlm_clear_ephemeral()

    # The agent must have an insulated bag
    if not dm.insulated_bag:
        dm.vlm_add_error("use_ice_pack failed: no insulated bag")
        dm._finish_action(success=False)
        return

    # Check item availability
    cnt = int(dm.inventory.get(ITEM_ICE_PACK, 0))
    if cnt <= 0:
        dm.vlm_add_error("use_ice_pack failed: inventory=0 (ice_pack)")
        dm._finish_action(success=False)
        return

    # Compartment label, must be a letter (A/B/C...); default to "A"
    lab = str(act.data.get("comp") or "").strip().upper() or "A"

    try:
        # Delegate validation and insertion to InsulatedBag
        dm.insulated_bag.add_misc_item(lab, IcePack())
    except Exception as e:
        dm.vlm_add_error(f"use_ice_pack failed: {e}")
        dm._finish_action(success=False)
        return

    # Deduct one ice pack from inventory
    dm.inventory[ITEM_ICE_PACK] = cnt - 1

    dm._log(
        f"inserted 'ice_pack' into compartment {lab} "
        f"(remaining {dm.inventory.get(ITEM_ICE_PACK, 0)})"
    )
    dm._finish_action(success=True)