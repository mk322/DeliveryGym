# actions/report_help_finished.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction
from ..gameplay.comms import get_comms


def handle_report_help_finished(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Reports the completion of a help request to the communication system.
    """

    comms = get_comms()
    if not comms:
        dm.vlm_add_error("report_help_finished failed: no comms")
        dm._finish_action(success=False)
        return

    req_id = int(act.data.get("req_id"))
    ok, msg, _res = comms.report_help_finished(
        req_id=req_id,
        by_agent=str(dm.agent_id),
        at_xy=(dm.x, dm.y),
    )

    if not ok:
        dm.vlm_add_error(f"report_help_finished failed: {msg}")
        dm._finish_action(success=False)
        return

    dm._log(f"reported help finished for request #{req_id}")
    dm._finish_action(success=True)