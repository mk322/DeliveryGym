# entities/bus.py
# -*- coding: utf-8 -*-

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import List, Tuple, Optional, Dict, Any

from ..base.timer import VirtualClock


class BusState(str, Enum):
    """High-level operational state of the bus."""
    STOPPED = "stopped"      # Stopped at a bus stop
    MOVING = "moving"        # Moving along the route
    WAITING = "waiting"      # Waiting for departure (idle)


@dataclass
class BusStop:
    """
    Representation of a bus stop in the world.

    Attributes:
        id: Unique identifier for the stop.
        x, y: Position of the stop in world coordinates (centimeters).
        name: Optional human-readable stop name.
        wait_time_s: Planned dwell time at this stop in seconds.
    """
    id: str
    x: float
    y: float
    name: str = ""
    wait_time_s: float = 10.0  # Dwell time at stop (seconds)


@dataclass
class BusRoute:
    """
    Bus route definition, including its stops and underlying path.

    Attributes:
        id: Unique identifier for the route.
        name: Human-readable route name.
        stops: Ordered list of BusStop objects along this route.
        path_points: Low-level path points along the route.
                     Typically interleaves stops and intermediate points.
        speed_cm_s: Travel speed in centimeters per second.
        direction: Current traversal direction, 1 for forward and -1 for reverse.
    """
    id: str
    name: str
    stops: List[BusStop]
    path_points: List[Tuple[float, float]]  # Route path points
    speed_cm_s: float = 1200.0  # Travel speed (cm/s)
    direction: int = 1  # 1: forward, -1: reverse


@dataclass
class Bus:
    """
    Simulated bus that follows a BusRoute using a VirtualClock.

    This class keeps track of:
      - The bus position and state (stopped/moving/waiting).
      - Current stop and path indices.
      - Time-based movement between stops (departure/arrival times).
      - On-board passengers (by passenger_id).
    """
    id: str
    route: BusRoute
    clock: VirtualClock = field(default_factory=lambda: VirtualClock())

    # Position and state
    x: float = 0.0
    y: float = 0.0
    state: BusState = BusState.WAITING

    # Route progress
    # current_stop_index: index of the last stop reached.
    # current_stop_index + 1 is the stop the bus is moving toward.
    current_stop_index: int = -1
    current_path_index: int = 0      # Index of the last path point reached
    progress_to_next: float = 0.0    # Progress toward the next path point [0, 1]

    # Time-related fields
    stop_start_time: float = 0.0     # Simulation time when the bus started dwelling at the current stop
    last_update_time: float = 0.0    # Simulation time of the last update call

    # Time-based movement control
    departure_time: float = 0.0      # Scheduled departure time from the current stop
    arrival_time: float = 0.0        # Scheduled arrival time at the target stop
    target_stop_index: int = -1      # Index of the stop currently being approached

    # Passengers
    passengers: List[str] = field(default_factory=list)  # List of passenger IDs currently on board

    def __post_init__(self):
        """Initialize runtime state and lock after dataclass construction."""
        # Dedicated lock to guard passenger list and other shared state.
        self._lock = Lock()

        # Initialize bus position at the starting point of the route.
        # Prefer the first path point if available.
        if self.route.path_points:
            first_point = self.route.path_points[0]
            self.x = first_point[0]
            self.y = first_point[1]
            self.current_path_index = 0
            self.progress_to_next = 0.0
            # Start in MOVING state from the first path point.
            self.state = BusState.MOVING
        elif self.route.stops:
            # Fallback: no explicit path points, snap to the first stop.
            first_stop = self.route.stops[0]
            self.x = first_stop.x
            self.y = first_stop.y
            self.state = BusState.STOPPED
            self.stop_start_time = self.clock.now_sim()

        self.last_update_time = self.clock.now_sim()

    def update(self):
        """
        Main update entry point.

        Advances the bus simulation based on the current simulation time:
          - If STOPPED: handle dwell time at the current stop.
          - If MOVING: handle time-based movement toward the next stop.
        """
        now = self.clock.now_sim()
        self.last_update_time = now

        if self.state == BusState.STOPPED:
            self._update_stopped(now)
        elif self.state == BusState.MOVING:
            self._update_moving_time_based(now)

    def _update_stopped(self, now: float):
        """
        Handle STOPPED state (dwell at a bus stop).

        When the dwell time at the current stop reaches wait_time_s,
        the bus departs and transitions to MOVING.
        """
        if not self.route.stops:
            return

        current_stop = self.route.stops[self.current_stop_index]
        stop_duration = now - self.stop_start_time

        if stop_duration >= current_stop.wait_time_s:
            # Dwell time elapsed -> depart from the current stop.
            self._depart_from_stop()

    def _update_moving_time_based(self, now: float):
        """
        Handle MOVING state using time-based movement.

        The bus moves along the route based purely on scheduled
        departure/arrival times. It does not integrate per-frame movement
        here; instead, it snaps to the target stop on arrival.
        """
        if not self.route.stops:
            return

        # If the scheduled arrival time has passed and we have a valid target stop,
        # we consider the bus as having reached that stop.
        if now >= self.arrival_time and self.target_stop_index >= 0:
            self._arrive_at_stop(self.target_stop_index)
            return

        # If we do not yet have a target stop, schedule the next one.
        if self.target_stop_index < 0:
            next_stop_index = self.current_stop_index + 1
            if next_stop_index < len(self.route.stops):
                self._set_target_stop(next_stop_index, now)
            else:
                # We have reached the end of the route; reverse direction and
                # start from the beginning of the reversed list.
                self._reverse_route()
                if self.route.stops:
                    self._set_target_stop(0, now)

    def _set_target_stop(self, stop_index: int, now: float):
        """
        Set the next target stop and compute its scheduled arrival time.

        Args:
            stop_index: Index of the target stop in route.stops.
            now: Current simulation time.
        """
        if stop_index >= len(self.route.stops):
            return

        self.target_stop_index = stop_index
        target_stop = self.route.stops[stop_index]

        # Compute path distance from current position to the target stop.
        distance_cm = self._calculate_path_distance_to_stop(stop_index)

        # Convert distance to travel time using the configured speed.
        if self.route.speed_cm_s > 0:
            travel_time = distance_cm / self.route.speed_cm_s
        else:
            travel_time = 0.0

        # Set departure and arrival timestamps (simulation time).
        self.departure_time = now
        self.arrival_time = now + travel_time

    def _calculate_path_distance_to_stop(self, stop_index: int) -> float:
        """
        Compute the travel distance along path_points to reach a target stop.

        The path is assumed to follow this pattern:
          path[0], stop0, path[1], stop1, path[2], stop2, ...

        In other words, stops are located at odd indices (1, 3, 5, ...) in path_points.

        Args:
            stop_index: Index of the target stop in route.stops.

        Returns:
            Total distance in centimeters along the path from the current position
            up to the target stop's path point.
        """
        if not self.route.path_points or stop_index >= len(self.route.stops):
            return 0.0

        total_distance = 0.0

        # Start from the current position (x, y).
        current_x, current_y = self.x, self.y

        # Determine the path_points index corresponding to this stop.
        # For the pattern [path0, stop0, path1, stop1, ...],
        # stops are at odd indices: 1, 3, 5, ...
        target_path_index = stop_index * 2 + 1

        # If the computed index goes out of range, clamp to the last path point.
        if target_path_index >= len(self.route.path_points):
            target_path_index = len(self.route.path_points) - 1

        # Sum the distances from the current_path_index to the target_path_index.
        for i in range(self.current_path_index, target_path_index + 1):
            if i < len(self.route.path_points):
                next_point = self.route.path_points[i]
                segment_distance = math.hypot(next_point[0] - current_x, next_point[1] - current_y)
                total_distance += segment_distance
                current_x, current_y = next_point[0], next_point[1]

        return total_distance

    def _reverse_route(self):
        """
        Reverse the route direction.

        This:
          - Reverses the order of stops and path_points.
          - Flips the route direction flag.
          - Resets indices so that the bus restarts from the "new" beginning.
        """
        # Reverse stops and path points in-place.
        self.route.stops.reverse()
        self.route.path_points.reverse()
        self.route.direction *= -1

        # Reset routing state so we start again from the new first path point.
        self.current_path_index = 0
        self.current_stop_index = -1
        self.progress_to_next = 0.0
        self.target_stop_index = -1

    def _arrive_at_stop(self, stop_index: int):
        """
        Handle arrival at a stop.

        This method:
          - Updates current_stop_index.
          - Switches state to STOPPED.
          - Snaps the bus position to the stop position.
          - Resets target/arrival/departure times.
          - Aligns current_path_index to the corresponding stop path index.
        """
        self.current_stop_index = stop_index
        self.state = BusState.STOPPED
        self.stop_start_time = self.clock.now_sim()

        # Snap to the stop position.
        stop = self.route.stops[stop_index]
        self.x = stop.x
        self.y = stop.y

        # Reset target stop and timings.
        self.target_stop_index = -1
        self.departure_time = 0.0
        self.arrival_time = 0.0

        # Align the path index to the path point corresponding to this stop (odd index).
        expected_path_index = stop_index * 2 + 1
        if expected_path_index < len(self.route.path_points):
            self.current_path_index = expected_path_index

        # print(f"Bus {self.id} arrived at stop {stop.name or stop.id}")

    def _depart_from_stop(self):
        """
        Handle departure from the current stop.

        This method:
          - Switches state to MOVING.
          - Schedules the next stop as the target.
          - Reverses the route at the end of the line.
        """
        self.state = BusState.MOVING

        # Schedule the next stop as the new target.
        next_stop_index = self.current_stop_index + 1
        if next_stop_index < len(self.route.stops):
            self._set_target_stop(next_stop_index, self.clock.now_sim())
        else:
            # Reached the terminal stop: reverse the route and start from the first stop.
            self._reverse_route()
            if self.route.stops:
                self._set_target_stop(0, self.clock.now_sim())

        # print(f"Bus {self.id} departed from stop {self.route.stops[self.current_stop_index].name or self.route.stops[self.current_stop_index].id}")

    def get_next_stop(self) -> Optional[BusStop]:
        """
        Get the upcoming stop for this bus, if any.

        Returns:
            The BusStop the bus is currently moving toward, if in MOVING state;
            otherwise the next stop after the current one; or None if there is
            no next stop.
        """
        if self.state == BusState.MOVING and self.target_stop_index >= 0:
            return self.route.stops[self.target_stop_index]
        elif not self.route.stops or self.current_stop_index + 1 >= len(self.route.stops):
            return None
        else:
            return self.route.stops[self.current_stop_index + 1]

    def get_current_stop(self) -> Optional[BusStop]:
        """
        Get the stop where the bus is currently dwelling.

        Returns:
            The current BusStop if current_stop_index is valid and the bus is
            considered at that stop; otherwise None.
        """
        if not self.route.stops or self.current_stop_index >= len(self.route.stops):
            return None
        return self.route.stops[self.current_stop_index]

    def is_at_stop(self) -> bool:
        """
        Check if the bus is currently stopped at a bus stop.

        Returns:
            True if the bus state is STOPPED, False otherwise.
        """
        return self.state == BusState.STOPPED

    def board_passenger(self, passenger_id: str) -> bool:
        """
        Attempt to board a passenger onto the bus.

        A passenger may only board when:
          - The bus is currently at a stop (is_at_stop() is True).
          - The passenger is not already on board.

        Returns:
            True if the passenger successfully boarded, otherwise False.
        """
        with self._lock:
            if self.is_at_stop() and passenger_id not in self.passengers:
                self.passengers.append(passenger_id)
                # print(f"Passenger {passenger_id} boarded bus {self.id}")
                return True
            return False

    def alight_passenger(self, passenger_id: str) -> bool:
        """
        Attempt to have a passenger alight from the bus.

        Returns:
            True if the passenger was on board and has been removed,
            otherwise False.
        """
        with self._lock:
            if passenger_id in self.passengers:
                self.passengers.remove(passenger_id)
                # print(f"Passenger {passenger_id} alighted from bus {self.id}")
                return True
            return False

    def _calculate_time_to_next_stop(self) -> Optional[float]:
        """
        Estimate remaining time (in seconds) to reach the next stop.

        This method prefers the precomputed arrival_time when the bus is in
        MOVING state with a valid target stop. If that is not available,
        it falls back to a direct distance-based estimate using the current
        position and route speed.

        Returns:
            Remaining time in seconds, or None if not applicable.
        """
        if self.state == BusState.MOVING and self.target_stop_index >= 0:
            # Use scheduled arrival time if we already have it.
            now = self.clock.now_sim()
            remaining_time = self.arrival_time - now
            return max(0.0, remaining_time)
        elif not self.route.stops or self.state != BusState.MOVING:
            return None

        next_stop = self.get_next_stop()
        if not next_stop:
            return None

        # Fallback: direct distance-based estimate from current position to
        # the next stop's coordinates.
        distance_cm = math.hypot(next_stop.x - self.x, next_stop.y - self.y)

        if distance_cm > 0 and self.route.speed_cm_s > 0:
            return distance_cm / self.route.speed_cm_s

        return None

    def _get_remaining_stop_time(self) -> Optional[float]:
        """
        Get the remaining dwell time (in seconds) at the current stop.

        Returns:
            Remaining dwell time in seconds if the bus is STOPPED at a valid stop;
            otherwise None. The returned value is clamped to be non-negative.
        """
        if self.state != BusState.STOPPED:
            return None

        current_stop = self.get_current_stop()
        if not current_stop:
            return None

        now = self.clock.now_sim()
        elapsed_time = now - self.stop_start_time
        remaining_time = current_stop.wait_time_s - elapsed_time

        # Clamp to zero if we have already exceeded the dwell time.
        return max(0.0, remaining_time)

    def get_status_text(self) -> str:
        """
        Construct a concise, human-readable status string for this bus.

        The string includes:
          - Bus ID and route name.
          - Whether the bus is stopped or moving (and where).
          - Remaining dwell time or ETA where applicable.
        """
        status_parts = [f"Bus {self.id}"]

        # Route name
        status_parts.append(f"route: {self.route.name}")

        if self.state == BusState.STOPPED:
            stop = self.get_current_stop()
            stop_name = stop.name if stop else "None"
            status_parts.append(f"stopped at {stop_name}")

            # Remaining dwell time at the current stop
            remaining_time = self._get_remaining_stop_time()
            if remaining_time is not None:
                status_parts.append(f"departing in: {remaining_time:.1f}s")

        elif self.state == BusState.MOVING:
            next_stop = self.get_next_stop()
            next_name = next_stop.name if next_stop else "None"
            status_parts.append(f"moving to {next_name}")

            # Estimated time of arrival at the next stop
            time_to_next = self._calculate_time_to_next_stop()
            if time_to_next is not None:
                status_parts.append(f"ETA: {time_to_next:.1f}s")

        else:
            # Fallback: explicitly show state if neither STOPPED nor MOVING.
            status_parts.append(f"state: {self.state.value}")

        # Optional debugging details:
        # status_parts.append(f"passengers: {len(self.passengers)}")
        # status_parts.append(f"pos: ({self.x/100:.1f}m, {self.y/100:.1f}m)")

        return " | ".join(status_parts)