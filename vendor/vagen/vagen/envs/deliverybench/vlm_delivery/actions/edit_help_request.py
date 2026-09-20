# actions/edit_help_request.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction
from ..gameplay.comms import get_comms


def handle_edit_help_request(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles editing the parameters of an existing help request.
    """
    comms = get_comms()
    req_id = int(act.data.get("req_id"))
    new_bounty = act.data.get("new_bounty", None)
    new_ttl_s = act.data.get("new_ttl_s", None)

    ok, msg = comms.modify_request(
        publisher_id=str(dm.agent_id),
        req_id=req_id,
        reward=new_bounty,
        time_limit_s=new_ttl_s,
    )
    if not ok:
        dm.vlm_add_error(f"edit_help_request failed: {msg}")
        dm._finish_action(success=False)
        return

    dm._log(f"edited help request #{req_id}")
    dm._finish_action(success=True)