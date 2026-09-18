# actions/say.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction
from ..gameplay.comms import get_comms


def handle_say(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Sends a chat message from the agent. Supports both direct messages and broadcasts.
    """

    text = str(act.data.get("text", "") or "").strip()
    to   = act.data.get("to", None)

    # Validate text content
    if not text:
        dm.vlm_add_error("say failed: empty text")
        dm._finish_action(success=False)
        return

    # Ensure communication channel is available
    comms = get_comms()
    if not comms:
        dm.vlm_add_error("say failed: comms not ready")
        dm._finish_action(success=False)
        return

    # Determine whether to broadcast or send to a specific agent
    is_broadcast = (to is None) or (str(to).upper() in ("ALL", "*"))
    target_id = None if is_broadcast else str(to)

    # Send chat message
    ok, msg, _ = comms.send_chat(
        from_agent=str(dm.agent_id),
        text=text,
        to_agent=target_id,
        broadcast=is_broadcast,
    )
    if not ok:
        dm.vlm_add_error(f"say failed: {msg}")
        dm._finish_action(success=False)
        return

    # Logging and ephemeral state
    if is_broadcast:
        dm._log(f"chat broadcast: {text}")
        dm.vlm_ephemeral["chat_sent"] = f"(broadcast) {text}"
    else:
        dm._log(f"chat to agent {target_id}: {text}")
        dm.vlm_ephemeral["chat_sent"] = f"to {target_id}: {text}"

    # Recorder metrics
    if dm._recorder:
        dm._recorder.inc("say", 1)

    dm._finish_action(success=True)