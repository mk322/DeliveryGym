# utils/ui.py
# -*- coding: utf-8 -*-

"""
UI helper functions for DeliveryMan progress indicators.

Each function takes a DeliveryMan-like object `dm` and inspects its
internal context fields to compute normalized progress and related
metadata for:
- Charging an e-scooter
- Resting to recover personal energy
- Being rescued / hospitalized

All helpers return either a dictionary with progress information
or None if no corresponding activity is in progress.
"""

from typing import Any, Dict, Optional


def ui_charging_progress(dm: Any) -> Optional[Dict[str, Any]]:
    """
    Compute the current charging progress for the DeliveryMan's e-scooter.

    Reads `_charge_ctx` from `dm` and returns:
      - progress: float in [0, 1], normalized completion ratio
      - current_pct: current estimated battery percentage
      - target_pct: target battery percentage for this charging session
      - xy: current or parking coordinates of the scooter
      - which: string tag indicating whose scooter is being charged
               (e.g., "own")
    Returns None if there is no active charging context.
    """
    ctx = getattr(dm, "_charge_ctx", None)
    if ctx and ctx.get("scooter_ref"):
        now = dm.clock.now_sim()
        sc = ctx["scooter_ref"]
        t0, t1 = ctx["start_sim"], ctx["end_sim"]
        p0, pt = ctx["start_pct"], ctx["target_pct"]

        if t1 <= t0:
            cur = pt
            prog = 1.0
        else:
            r = max(0.0, min(1.0, (now - t0) / (t1 - t0)))
            cur = p0 + (pt - p0) * r
            prog = 0.0 if pt <= p0 else (cur - p0) / max(1e-9, pt - p0)

        xy = sc.park_xy if sc.park_xy else (dm.x, dm.y)
        return dict(
            progress=float(max(0.0, min(1.0, prog))),
            current_pct=float(cur),
            target_pct=float(pt),
            xy=xy,
            which=ctx.get("which", "own"),
        )
    return None


def ui_resting_progress(dm: Any) -> Optional[Dict[str, Any]]:
    """
    Compute the DeliveryMan's rest/recovery progress.

    Reads `_rest_ctx` from `dm` and returns:
      - progress: float in [0, 1], normalized completion ratio
      - current_pct: current estimated personal energy percentage
      - target_pct: target recovery percentage
      - xy: current (x, y) position of the DeliveryMan
    Returns None if there is no active rest context.
    """
    ctx = getattr(dm, "_rest_ctx", None)
    if ctx:
        now = dm.clock.now_sim()
        t0, t1 = ctx["start_sim"], ctx["end_sim"]
        e0, et = ctx["start_pct"], ctx["target_pct"]

        if t1 <= t0:
            cur = et
            prog = 1.0
        else:
            r = max(0.0, min(1.0, (now - t0) / (t1 - t0)))
            cur = e0 + (et - e0) * r
            prog = 0.0 if et <= e0 else (cur - e0) / max(1e-9, et - e0)

        return dict(
            progress=float(max(0.0, min(1.0, prog))),
            current_pct=float(cur),
            target_pct=float(et),
            xy=(dm.x, dm.y),
        )
    return None


def ui_rescue_progress(dm: Any) -> Optional[Dict[str, Any]]:
    """
    Compute the DeliveryMan's rescue/hospitalization progress.

    Reads `_hospital_ctx` from `dm` and returns:
      - progress: float in [0, 1], normalized completion ratio
      - xy: current (x, y) position of the DeliveryMan
    Returns None if there is no active hospital context.
    """
    ctx = getattr(dm, "_hospital_ctx", None)
    if ctx:
        now = dm.clock.now_sim()
        t0, t1 = ctx["start_sim"], ctx["end_sim"]
        r = 0.0 if t1 <= t0 else (now - t0) / (t1 - t0)
        r = max(0.0, min(1.0, r))
        return dict(progress=r, xy=(dm.x, dm.y))
    return None