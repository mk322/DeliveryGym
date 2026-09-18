# actions/view_help_board.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction
from ..gameplay.comms import get_comms


def handle_view_help_board(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Display the current help board as text.
    """
    try:
        comms = get_comms()
        if comms is None:
            # No comms available — still treated as success
            text = "(help board unavailable: comms not initialized)"
            dm.vlm_add_ephemeral("help_board", text)
            dm._log("view help board (comms not ready)")
            dm._finish_action(success=True)
            return

        # Display options (fixed defaults)
        include_active = False
        include_completed = False
        max_items = 50

        # Retrieve formatted help-board text from comms
        text = comms.board_to_text(
            include_active=include_active,
            include_completed=include_completed,
            max_items=max_items,
            exclude_publisher=str(dm.agent_id),
        )

        # Store for VLM prompt usage
        dm.vlm_add_ephemeral("help_board", text)
        dm._log("view help board")
        dm._finish_action(success=True)

    except Exception as e:
        dm.vlm_add_error(f"view_help_board failed: {e}")
        dm._finish_action(success=False)