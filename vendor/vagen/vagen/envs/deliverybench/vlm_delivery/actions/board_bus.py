# actions/board_bus.py
# -*- coding: utf-8 -*-

import math
from typing import Any

from ..base.defs import DMAction, TransportMode


def handle_board_bus(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles boarding a bus and selecting a target stop.
    """
    dm.vlm_clear_ephemeral()

    if not dm._bus_manager:
        dm.vlm_add_error("board_bus failed: no bus manager")
        dm._finish_action(success=False)
        return

    bus_id = act.data.get("bus_id")
    target_stop_id = act.data.get("target_stop_id")

    if not bus_id:
        dm.vlm_add_error("board_bus failed: need bus_id")
        dm._finish_action(success=False)
        return

    if not target_stop_id:
        dm.vlm_add_error("board_bus failed: need target_stop")
        dm._finish_action(success=False)
        return

    bus = dm._bus_manager.get_bus(bus_id)
    if not bus:
        dm.vlm_add_error(f"board_bus failed: bus {bus_id} not found")
        dm._finish_action(success=False)
        return

    # Validate that the bus is currently at a stop and close enough to board
    if not bus.is_at_stop() or math.hypot(bus.x - dm.x, bus.y - dm.y) > 1000.0:
        dm.vlm_add_error(f"board_bus failed: bus {bus_id} not at stop")
        dm._finish_action(success=False)
        return

    # Ensure the target stop exists on the bus route
    target_stop = None
    for stop in bus.route.stops:
        if stop.id == target_stop_id:
            target_stop = stop
            break

    if not target_stop:
        dm.vlm_add_error(
            f"board_bus failed: target stop {target_stop_id} not on bus route"
        )
        dm._finish_action(success=False)
        return

    # Prevent boarding if already at the target stop
    current_stop = bus.get_current_stop()
    if current_stop and current_stop.id == target_stop_id:
        dm.vlm_add_error(
            f"board_bus failed: already at target stop {target_stop_id}"
        )
        dm._finish_action(success=False)
        return

    # Check if the agent has enough balance ($1 required)
    if float(dm.earnings_total) + 1e-9 < 1.0:
        dm.vlm_add_error("board_bus failed: insufficient funds ($1 required)")
        dm._finish_action(success=False)
        return

    # Attempt to board the bus
    if bus.board_passenger(str(dm.agent_id)):
        # Boarding successful — deduct fare
        old_balance = float(dm.earnings_total)
        dm.earnings_total = max(0.0, old_balance - 1.0)

        if dm._recorder:
            dm._recorder.inc("bus_board", 1)

        # Store bus boarding context
        dm._bus_ctx = {
            "bus_id": bus_id,
            "boarding_stop": current_stop.id if current_stop else "",
            "target_stop": target_stop_id,
            "transport_mode": dm.mode,
            "boarded_time": dm.clock.now_sim(),
        }

        dm.set_mode(TransportMode.BUS)
        dm._log(
            f"boarded bus {bus_id} at "
            f"{current_stop.id if current_stop else 'unknown'} "
            f"heading to {target_stop_id}"
        )
        dm._register_success(f"boarded bus {bus_id}")
    else:
        dm.vlm_add_error("board_bus failed: could not board")
        dm._finish_action(success=False)