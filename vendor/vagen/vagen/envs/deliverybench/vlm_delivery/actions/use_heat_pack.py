# actions/use_heat_pack.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, ITEM_HEAT_PACK
from ..entities.insulated_bag import HeatPack


def handle_use_heat_pack(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Insert a heat pack into a specified insulated bag compartment.
    """
    dm.vlm_clear_ephemeral()

    # The agent must have an insulated bag
    if not dm.insulated_bag:
        dm.vlm_add_error("use_heat_pack failed: no insulated bag")
        dm._finish_action(success=False)
        return

    # Check item availability
    cnt = int(dm.inventory.get(ITEM_HEAT_PACK, 0))
    if cnt <= 0:
        dm.vlm_add_error("use_heat_pack failed: inventory=0 (heat_pack)")
        dm._finish_action(success=False)
        return

    # Compartment label; default to "A"
    lab = str(act.data.get("comp") or "").strip().upper() or "A"

    # Attempt to insert a HeatPack object into the corresponding compartment
    try:
        dm.insulated_bag.add_misc_item(lab, HeatPack())
    except Exception as e:
        dm.vlm_add_error(f"use_heat_pack failed: {e}")
        dm._finish_action(success=False)
        return

    # Deduct one heat pack from inventory
    dm.inventory[ITEM_HEAT_PACK] = cnt - 1

    dm._log(
        f"inserted 'heat_pack' into compartment {lab} "
        f"(remaining {dm.inventory.get(ITEM_HEAT_PACK, 0)})"
    )
    dm._finish_action(success=True)