# actions/rest.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction
from ..utils.util import nearest_poi_xy, get_tol


def handle_rest(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Initiates a rest action to restore energy when the agent is near a rest area.
    """

    dm.vlm_clear_ephemeral()

    # Cannot rest while in hospital rescue state
    if dm.is_rescued:
        dm.vlm_add_error("rest failed: in hospital rescue")
        dm._finish_action(success=False)
        return

    # Must be located near a designated rest area
    if nearest_poi_xy(dm, "rest_area", tol_cm=get_tol(dm.cfg, "nearby")) is None:
        dm.vlm_add_error(
            "rest failed: not near a rest area; you should first go to a rest_area"
        )
        dm._finish_action(success=False)
        return

    # Target energy percentage
    target = float(
        act.data.get(
            "target_pct",
            dm.cfg.get(
                "rest_target_pct",
                dm.cfg.get("defaults", {}).get("rest_target_pct", 100.0),
            ),
        )
    )

    before = float(dm.energy_pct)

    # Skip if already at or above target
    if target <= before + 1e-6:
        dm._log(f"rest skipped: already at {before:.0f}%")
        dm._finish_action(success=True)
        return

    # Validate rest rate
    rate = float(dm.rest_rate_pct_per_min)
    if rate <= 0.0:
        dm.vlm_add_error("rest failed: invalid rest rate")
        dm._finish_action(success=False)
        return

    # Text-only mode: rest is instantaneous, but advances simulation time.
    duration_sim_s = (target - before) / rate * 60.0
    duration_sim_s = max(0.0, float(duration_sim_s))
    if hasattr(dm.clock, "advance"):
        dm.clock.advance(duration_sim_s)

    dm.energy_pct = float(target)
    dm._rest_ctx = None

    dm._log(
        f"rest: {before:.0f}% -> {target:.0f}% "
        f"(+{duration_sim_s/60.0:.1f} min @virtual)"
    )

    rec = getattr(dm, "_recorder", None)
    if rec:
        rec.tick_inactive("rest", duration_sim_s)
        rec.inc_preventive("early_rest")

    dm._finish_action(success=True)