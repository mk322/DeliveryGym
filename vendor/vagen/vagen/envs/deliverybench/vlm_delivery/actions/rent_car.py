# actions/rent_car.py
# -*- coding: utf-8 -*-

from typing import Any
from ..base.defs import DMAction, TransportMode
from ..entities.car import Car, CarState
from ..utils.util import nearest_poi_xy, get_tol


def handle_rent_car(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """
    Handles renting a car when the agent is located at a valid car rental spot.
    """

    # Ensure the agent is near a car rental point
    tol_xy = nearest_poi_xy(dm, "car_rental", tol_cm=get_tol(dm.cfg, "nearby"))
    if tol_xy is None:
        dm.vlm_add_error("rent_car failed: not at car_rental")
        dm._finish_action(success=False)
        return

    # Cannot rent a second car
    if dm.car is not None:
        dm.vlm_add_error("rent_car failed: already have a car")
        dm._finish_action(success=False)
        return

    # If currently using an e-scooter, park it before renting the car
    if dm.e_scooter and (
        dm.mode == TransportMode.SCOOTER
        or (dm.mode == TransportMode.DRAG_SCOOTER and dm.assist_scooter is None)
    ):
        dm.e_scooter.park_here(dm.x, dm.y)

    # Load default rental parameters
    defs = dm.cfg.get("rent_car_defaults", {})
    rate = float(act.data.get("rate_per_min", defs.get("rate_per_min", 1.0)))
    speed = float(act.data.get("avg_speed_cm_s", defs.get("avg_speed_cm_s", 1200)))

    # Create the rental car instance
    dm.car = Car(
        owner_id=str(dm.agent_id),
        avg_speed_cm_s=speed,
        rate_per_min=rate,
        state=CarState.USABLE,
        park_xy=None,
    )

    # Switch movement mode to CAR
    dm.set_mode(TransportMode.CAR)

    # Initialize rental billing context
    dm._rental_ctx = {
        "last_tick_sim": dm.clock.now_sim(),
        "rate_per_min": float(dm.car.rate_per_min),
    }

    dm._log(f"rent car @ ${dm.car.rate_per_min:.2f}/min")
    dm._finish_action(success=True)