# actions/use_energy_drink.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, ITEM_ENERGY_DRINK


def handle_use_energy_drink(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Use an energy drink to restore the agent's energy.
    """
    dm.vlm_clear_ephemeral()

    # Cannot use energy drinks while in hospital rescue
    if dm._hospital_ctx is not None:
        dm.vlm_add_error("use_energy_drink failed: in hospital rescue")
        dm._finish_action(success=False)
        return

    # Energy is already full
    if float(dm.energy_pct) >= 100.0 - 1e-6:
        dm.vlm_add_error("use_energy_drink failed: Your energy is full.")
        dm._finish_action(success=False)
        return

    # Check item availability
    item_id = act.data.get("item_id", ITEM_ENERGY_DRINK)
    cnt = int(dm.inventory.get(item_id, 0))
    if cnt <= 0:
        dm.vlm_add_error(f"use_energy_drink failed: inventory=0 ({item_id})")
        dm._finish_action(success=False)
        return

    # Consume the drink
    dm.inventory[item_id] = cnt - 1
    gain = float(
        dm.cfg.get("items", {})
        .get(ITEM_ENERGY_DRINK, {})
        .get("energy_gain_pct", 50)
    )

    # Apply energy gain with upper bound
    before = float(dm.energy_pct)
    dm.energy_pct = float(
        min(dm.cfg.get("energy_pct_max", 100), before + gain)
    )

    dm._log(
        f"used '{item_id}': energy {before:.0f}% -> {dm.energy_pct:.0f}% "
        f"(remaining {dm.inventory[item_id]})"
    )
    dm._recorder.inc_preventive("use_energy_drink")
    dm._finish_action(success=True)