# actions/view_bag.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_view_bag(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Show the current layout and temperature information of the insulated bag.
    """
    try:
        # If the agent has no insulated bag, return a simple message
        if not dm.insulated_bag:
            dm.vlm_add_ephemeral("bag_layout", "(no insulated bag)")
            dm._log("view bag (no bag)")
            dm._finish_action(success=True)
            return

        # Get textual representation of bag layout
        layout = dm.insulated_bag.list_items()

        # Retrieve per-compartment temperatures, if available
        temps = getattr(dm.insulated_bag, "_comp_temp_c", None)
        if isinstance(temps, (list, tuple)):
            temps_txt = " | ".join(
                [f"Comp {i}: {float(t):.1f}°C" for i, t in enumerate(temps)]
            )
        else:
            temps_txt = "(no per-compartment temps)"

        # Construct final view text
        text = f"{layout}\n\n[compartment temps] {temps_txt}"

        dm.vlm_add_ephemeral("bag_layout", text)
        dm._log("view bag")
        dm._finish_action(success=True)

    except Exception as e:
        dm.vlm_add_error(f"view_bag failed: {e}")
        dm._finish_action(success=False)