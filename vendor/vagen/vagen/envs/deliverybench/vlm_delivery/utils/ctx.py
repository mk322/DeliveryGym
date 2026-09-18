# utils/ctx.py
# -*- coding: utf-8 -*-

"""
Context time adjustment utilities.

These helpers manage simple pause/resume behavior for time-based
contexts. A context `ctx` is expected to be a dictionary that contains
timestamps such as:
    - "start_sim": simulation start time
    - "end_sim":   simulation end time
    - "paused_at": time when the context was paused (added dynamically)

Both functions mutate `ctx` in place and safely handle None inputs.
"""

from typing import Optional, Dict


def ctx_mark_pause(ctx: Optional[Dict[str, float]], now: float) -> None:
    """
    Mark the context as paused at the given time.

    If `ctx` exists and has not yet recorded a pause time,
    a new field `"paused_at"` is added to store the pause timestamp.
    """
    if ctx is not None and "paused_at" not in ctx:
        ctx["paused_at"] = float(now)


def ctx_mark_resume(ctx: Optional[Dict[str, float]], now: float) -> None:
    """
    Resume a paused context by shifting its timeline forward.

    If the context was previously paused, the elapsed pause duration
    is added to both `"start_sim"` and `"end_sim"`. The `"paused_at"`
    field is then removed.
    """
    if ctx is not None and ctx.get("paused_at") is not None:
        delta = float(now) - float(ctx["paused_at"])
        ctx["start_sim"] += delta
        ctx["end_sim"]   += delta
        ctx.pop("paused_at", None)