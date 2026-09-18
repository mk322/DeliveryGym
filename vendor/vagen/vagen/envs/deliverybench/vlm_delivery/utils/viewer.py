# utils/viewer.py
# -*- coding: utf-8 -*-

"""
Viewer binding utilities for DeliveryMan agents.

This module provides helper functions to:
- Bind a DeliveryMan instance to a map viewer and register callbacks.
- Handle viewer-driven events (e.g., movement completion, blocked paths)
  and propagate them back to the agent state.
"""

from typing import Any, Dict


def viewer_bind_viewer(dm: Any, viewer: Any) -> None:
    """
    Bind a viewer instance to a DeliveryMan-like object and register callbacks.

    This:
      - Stores the viewer reference on the agent.
      - Assigns a viewer-specific agent ID.
      - Registers the agent with the viewer (if supported).
      - Hooks a callback so that animation completion events can update
        the agent's position and trigger follow-up logic.
    """
    dm._viewer = viewer
    dm._viewer_agent_id = str(dm.agent_id)

    if hasattr(viewer, "add_agent"):
        def _proxy_on_done(aid, *args):
            """
            Local proxy to adapt viewer callbacks into a unified event handler.

            The viewer may call this with:
              - (aid, fx, fy, event) or
              - (aid, fx, fy)
            This helper normalizes the arguments and forwards them
            to `viewer_on_view_event`.
            """
            event = "move"
            if len(args) == 3:
                fx, fy, event = args
            elif len(args) == 2:
                fx, fy = args
            else:
                fx = fy = None
            viewer_on_view_event(dm, aid, event, {"x": fx, "y": fy})

        viewer.add_agent(
            dm._viewer_agent_id,
            dm.x,
            dm.y,
            speed_cm_s=dm.get_current_speed_for_viewer(),
            label_text=f"{dm.agent_id}",
            on_anim_done=_proxy_on_done,
        )

    if hasattr(dm._viewer, "register_delivery_man"):
        dm._viewer.register_delivery_man(dm)


def viewer_on_view_event(
    dm: Any,
    agent_id: str,
    event: str,
    payload: Dict[str, Any],
) -> None:
    """
    Handle events sent from the viewer for a given DeliveryMan agent.

    The viewer notifies about completed animations or blocked movements.
    This handler:
      - Updates the agent's (x, y) position from the payload.
      - On "move": refreshes nearby POI hints.
      - On "blocked": logs the issue, optionally attaches a VLM error,
        and marks the movement context as blocked.
    """
    fx = float(payload.get("x", dm.x)) if payload.get("x", dm.x) is not None else dm.x
    fy = float(payload.get("y", dm.y)) if payload.get("y", dm.y) is not None else dm.y
    dm.x, dm.y = fx, fy

    if event == "move":
        # Auto drop-off logic was intentionally left disabled in the original code:
        # dm._auto_try_dropoff()
        dm._refresh_poi_hints_nearby()
        return

    if event == "blocked":
        if dm._interrupt_reason == "escooter_depleted":
            dm._log("movement blocked (ESCOOTER depleted) -> re-decide")
        else:
            dm.vlm_add_error("movement blocked")
            dm._log("movement blocked")
        dm._interrupt_reason = None
        if dm._move_ctx is not None:
            dm._move_ctx["blocked"] = 1.0
        return