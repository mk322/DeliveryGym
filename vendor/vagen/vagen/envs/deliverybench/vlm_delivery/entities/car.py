# entities/car.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class CarState(str, Enum):
    """
    High-level availability state of a rental car.

    Attributes:
        USABLE: The car is available for use and not currently parked.
        PARKED: The car is parked at a specific location and must be picked up
                from that position.
    """
    USABLE = "usable"
    PARKED = "parked"


@dataclass
class Car:
    """
    Simple car model used by the delivery agent.

    The car tracks:
      - Ownership (which agent rented or owns it).
      - Average travel speed (used for ETA / cost estimation).
      - Rental rate per minute.
      - Current state (usable or parked) and park position.

    All distances are in centimeters, and time is assumed to be in seconds
    when combined with avg_speed_cm_s.
    """
    owner_id: str
    avg_speed_cm_s: float = 2000.0     # ≈ 72 km/h
    rate_per_min: float = 1.0          # Rental cost in $ per minute
    state: CarState = CarState.USABLE
    park_xy: Optional[Tuple[float, float]] = None  # (x_cm, y_cm) if parked

    def park_here(self, x: float, y: float):
        """
        Park the car at the given world position.

        This sets the state to PARKED and records the parking coordinates.

        Args:
            x: X coordinate in world space (centimeters).
            y: Y coordinate in world space (centimeters).
        """
        self.state = CarState.PARKED
        self.park_xy = (float(x), float(y))

    def unpark(self):
        """
        Mark the car as usable again and clear the parking position.

        Typically called when the owner has reached the parked car and
        starts driving it.
        """
        self.state = CarState.USABLE
        self.park_xy = None