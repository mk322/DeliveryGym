# actions/wait.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_wait(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handle wait action for either fixed duration or until charging is completed.
    """

    # Text-only mode: waiting is instantaneous, but advances simulation time.
    # Case 0: WAIT("traffic_light") - advance to the start of the next wall-minute
    # so the traffic signal flips (see OBSTACLE_TRAFFIC_DESIGN.md). Deterministic:
    # always advances a positive amount, even when the light is already green.
    if str(act.data.get("until") or "").lower() == "traffic_light":
        traffic = getattr(dm, "_traffic", None)
        period = float(getattr(traffic, "_period", 60.0)) if traffic is not None else 60.0
        now_sim = float(dm.clock.now_sim())
        rem = now_sim % period
        dt = period - rem if rem > 1e-9 else period
        if hasattr(dm.clock, "advance"):
            dm.clock.advance(dt)
        dm._wait_ctx = None
        dm._log(f"wait for traffic light: +{dt:.1f}s (to next minute)")
        rec = getattr(dm, "_recorder", None)
        if rec:
            rec.tick_inactive("wait", dt)
        dm._finish_action(success=True)
        return

    # Case 1: WAIT(\"charge_done\") - advance until the active charge context completes.
    if str(act.data.get("until") or "").lower() == "charge_done":
        if dm._charge_ctx is None:
            dm._log("wait skipped: not currently charging")
            dm._finish_action(success=True)
            return

        end_sim = dm._charge_ctx.get("end_sim")
        if end_sim is None:
            dm._log("wait skipped: charge has no end time")
            dm._finish_action(success=True)
            return

        now_sim = float(dm.clock.now_sim())
        dt = max(0.0, float(end_sim) - now_sim)
        if hasattr(dm.clock, "advance"):
            dm.clock.advance(dt)
        dm._wait_ctx = None
        dm._log(f"wait until charge done: +{dt:.1f}s @virtual")
        dm._finish_action(success=True)
        return

    # Case 2: Wait for a fixed duration.
    duration_s = float(act.data.get("duration_s", 0.0))
    if duration_s <= 0.0:
        dm._log("wait skipped: duration <= 0s")
        dm._finish_action(success=True)
        return

    if hasattr(dm.clock, "advance"):
        dm.clock.advance(float(duration_s))
    dm._wait_ctx = None
    dm._log(f"wait: +{duration_s:.1f}s (~{duration_s/60.0:.1f} min) @virtual")
    rec = getattr(dm, "_recorder", None)
    if rec:
        rec.tick_inactive("wait", duration_s)
    dm._finish_action(success=True)