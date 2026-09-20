# entities/escooter.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ScooterState(str, Enum):
    """
    High-level availability state of an e-scooter.

    USABLE   : The scooter has sufficient battery and can be ridden.
    DEPLETED : The scooter is out of battery and cannot be ridden.
    PARKED   : The scooter is parked at a fixed world position (static).
    """
    USABLE   = "usable"
    DEPLETED = "depleted"
    PARKED   = "parked"


@dataclass
class EScooter:
    """
    Lightweight e-scooter model.

    All distances are in centimeters; time is assumed to be in seconds when
    combined with avg_speed_cm_s. Battery is represented as a percentage.

    Attributes:
        avg_speed_cm_s:
            Nominal travel speed in cm/s used for planning.
        min_speed_cm_s:
            Lower bound on allowed speed (for clamping).
        max_speed_cm_s:
            Upper bound on allowed speed (for clamping).

        battery_max_pct:
            Maximum battery capacity in percent (usually 100).
        battery_pct:
            Current battery level in percent [0, 100].
        charge_rate_pct_per_min:
            Charging rate in percent per minute (used by higher-level logic).

        state:
            Current scooter state (USABLE / DEPLETED / PARKED).
        park_xy:
            Parking coordinates (x_cm, y_cm) when the scooter is PARKED.

        owner_id:
            Optional owner/agent identifier (used by access control).
        with_owner:
            Whether the scooter is currently with the owner (True) or left
            somewhere in the world (False).
    """
    # Speed configuration (cm/s)
    avg_speed_cm_s: float = 800.0
    min_speed_cm_s: float = 300.0
    max_speed_cm_s: float = 1500.0

    # Battery configuration (%)
    battery_max_pct: float = 100.0
    battery_pct: float = 100.0
    charge_rate_pct_per_min: float = 25.0

    # State & ownership
    state: ScooterState = ScooterState.USABLE
    park_xy: Optional[Tuple[float, float]] = None
    owner_id: str = ""
    with_owner: bool = True

    # === Speed helpers ===
    def clamp_speed(self, v: float) -> float:
        """
        Clamp a raw speed value into [min_speed_cm_s, max_speed_cm_s].

        Args:
            v: Speed in cm/s.

        Returns:
            The clamped speed.
        """
        v = float(v)
        if v < self.min_speed_cm_s:
            return self.min_speed_cm_s
        if v > self.max_speed_cm_s:
            return self.max_speed_cm_s
        return v

    def set_speed(self, v: float) -> float:
        """
        Set avg_speed_cm_s, automatically clamped to the valid range.

        Returns:
            The resulting clamped speed.
        """
        self.avg_speed_cm_s = self.clamp_speed(v)
        return self.avg_speed_cm_s

    # === Battery helpers ===
    def set_battery_pct(self, pct: float) -> float:
        """
        Set the battery percentage and update the scooter state accordingly.

        Battery value is clamped to [0, battery_max_pct]. If the scooter is
        not PARKED, its state is automatically updated to USABLE or DEPLETED
        depending on the resulting battery level.
        """
        p = float(pct)
        if p < 0.0:
            p = 0.0
        if p > self.battery_max_pct:
            p = self.battery_max_pct
        self.battery_pct = p
        # Synchronize state unless explicitly PARKED
        if self.state != ScooterState.PARKED:
            self.state = (
                ScooterState.DEPLETED
                if self.battery_pct <= 0.0
                else ScooterState.USABLE
            )
        return self.battery_pct

    def consume_pct(self, delta_pct: float) -> float:
        """Consume a given percentage of battery and update state accordingly."""
        return self.set_battery_pct(self.battery_pct - float(delta_pct))

    def charge_to(self, target_pct: float) -> float:
        """
        Charge the scooter to the given battery percentage.

        This is applied immediately; higher-level logic (e.g., WAIT actions)
        is responsible for simulating the passage of time.
        """
        return self.set_battery_pct(target_pct)

    # === Parking / retrieval ===
    def park_here(self, x: float, y: float):
        """
        Park the scooter at the given world coordinates.

        Sets state to PARKED and records the parking position.
        """
        self.state = ScooterState.PARKED
        self.park_xy = (float(x), float(y))

    def unpark(self):
        """
        Mark the scooter as taken from its parking spot.

        After unparking, the scooter state becomes USABLE or DEPLETED
        depending on the current battery level, and park_xy is cleared.
        """
        self.park_xy = None
        self.state = (
            ScooterState.DEPLETED
            if self.battery_pct <= 0.0
            else ScooterState.USABLE
        )

    def can_be_ridden_by(self, agent_id) -> bool:
        """
        Check whether a given agent is allowed to ride this scooter.

        If owner_id is empty, any agent is allowed. Otherwise, only the
        matching owner_id is allowed to ride.
        """
        return str(agent_id) == str(self.owner_id) if self.owner_id else True