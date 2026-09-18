# actions/return_car.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, TransportMode
from ..utils.util import nearest_poi_xy, get_tol


def handle_return_car(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Returns a rented car when the agent is at a valid car rental location.
    """

    # Ensure the agent is near a car rental station
    tol_xy = nearest_poi_xy(dm, "car_rental", tol_cm=get_tol(dm.cfg, "nearby"))
    if tol_xy is None:
        dm.vlm_add_error("return_car failed: not at car_rental")
        dm._finish_action(success=False)
        return

    # Must currently have a rented car to return
    if not dm.car:
        dm.vlm_add_error("return_car failed: no car")
        dm._finish_action(success=False)
        return

    # Switch back to walking mode if currently driving
    if dm.mode == TransportMode.CAR:
        dm.set_mode(TransportMode.WALK)

    dm._log("return car: stop billing")

    # Clear car and billing context
    dm.car = None
    dm._rental_ctx = None

    dm._finish_action(success=True)