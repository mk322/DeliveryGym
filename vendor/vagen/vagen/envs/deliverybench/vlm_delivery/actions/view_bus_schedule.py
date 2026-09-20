# actions/view_bus_schedule.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction


def handle_view_bus_schedule(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Display the available bus routes and current bus status.
    """
    try:
        # If no bus manager is registered, return placeholder info
        if not dm._bus_manager:
            dm.vlm_add_ephemeral("bus_schedule", "(no bus schedule)")
            dm._log("view bus schedule (no bus manager)")
            dm._finish_action(success=True)
            return

        # Retrieve all bus routes
        routes_info = dm._bus_manager.get_all_routes_info()

        # Retrieve current status of all buses
        buses_status = dm._bus_manager.get_all_buses_status()

        # Construct schedule text
        schedule_text = ""

        # Add route information
        if routes_info:
            schedule_text += "Routes:\n"
            for route_id, route_info in routes_info.items():
                schedule_text += f"Route {route_info['name']}:\n"
                schedule_text += f"  Stops ({len(route_info['stops'])}):\n"
                for stop in route_info["stops"]:
                    schedule_text += (
                        f"  {stop['name']} - Wait: {stop['wait_time_s']:.1f}s\n"
                    )
        else:
            schedule_text += "No routes available.\n"

        # Add current bus statuses
        if buses_status:
            schedule_text += "\nCurrent bus status:\n"
            for status in buses_status:
                schedule_text += f"  {status}\n"
        else:
            schedule_text += "\nNo buses currently running.\n"

        # Optional: print for debugging
        # print(schedule_text)

        # Store final text into ephemeral memory
        dm.vlm_add_ephemeral("bus_schedule", schedule_text)
        dm._log("view bus schedule")
        dm._finish_action(success=True)

    except Exception as e:
        dm.vlm_add_error(f"view_bus_schedule failed: {e}")
        dm._finish_action(success=False)