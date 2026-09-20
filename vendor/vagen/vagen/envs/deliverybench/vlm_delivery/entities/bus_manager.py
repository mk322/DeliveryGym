# entities/bus_manager.py
# -*- coding: utf-8 -*-

import json
import math
import copy
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .bus import Bus, BusRoute, BusStop
from ..base.timer import VirtualClock


class BusManager:
    """
    Central manager for the bus system.

    Responsibilities:
      - Load bus routes from world data.
      - Create and maintain Bus instances bound to those routes.
      - Provide query utilities (nearby buses, buses at a stop, nearest stop).
      - Update all buses based on a shared VirtualClock.
    """

    def __init__(
        self,
        clock: Optional[VirtualClock] = None,
        waiting_time_s: float = 10.0,
        speed_cm_s: float = 1200.0,
        num_buses: int = 1,
    ):
        """
        Initialize the bus manager.

        Args:
            clock: Shared VirtualClock for all buses. A new one is created if None.
            waiting_time_s: Default dwell time at each stop (seconds).
            speed_cm_s: Default travel speed for buses (centimeters per second).
            num_buses: Number of buses to spawn on initialization (all on the first route).
        """
        self.clock = clock if clock is not None else VirtualClock()
        self.routes: Dict[str, BusRoute] = {}
        self.buses: Dict[str, Bus] = {}
        self._update_timer = None
        self.waiting_time_s = waiting_time_s
        self.speed_cm_s = speed_cm_s
        self.num_buses = num_buses
        # Sim-time anchor for deterministic schedule forecasting. Set once
        # by init_bus_system() to clock.now_sim() at the moment buses are
        # instantiated (Bus.__post_init__ uses the same clock and starts in
        # state=MOVING heading toward stop[0]). Treated as the canonical
        # birth time t₀ for all buses on this manager. Never mutated
        # afterward; the query-only NAVIGATE(mode="bus") forecaster reads this.
        self._birth_sim_time: Optional[float] = None

    def init_bus_system(self, world_data: Dict[str, Any]):
        """
        Initialize the bus system from world data.

        This:
          - Records the canonical bus birth time t₀ (clock.now_sim() at the
            moment buses are instantiated). Deterministic schedule forecasts
            anchor on this time.
          - Loads all routes defined under world_data["bus_routes"].
          - Spawns `num_buses` buses on the first available route.
        """
        self._birth_sim_time = float(self.clock.now_sim())
        self.load_routes_from_world_data(world_data)
        for i in range(self.num_buses):
            self.create_bus(f"bus_{i+1}", list(self.routes.keys())[0])

    def load_routes_from_world_data(self, world_data: Dict[str, Any]):
        """
        Load bus routes from world data.

        Expected world_data structure (simplified):
            {
                "nodes": [...],
                "bus_routes": [
                    {
                        "id": "...",
                        "name": "...",
                        "station_ids": [...],
                        "path": [...],  # optional
                        ...
                    },
                    ...
                ]
            }

        For each route:
          - A BusRoute is created.
          - BusStop instances are built from "station_ids" and "nodes".
          - path_points are built from the route "path" and interleaved with stops
            as [path[0], stop0, path[1], stop1, ...] when both are available.
        """
        bus_routes_data = world_data.get("bus_routes", [])

        for route_data in bus_routes_data:
            route_id = route_data.get("id", f"route_{len(self.routes)}")
            route_name = route_data.get("name", f"{route_id}")

            # Build BusStop objects for each station_id in the route.
            stops: List[BusStop] = []
            station_ids = route_data.get("station_ids", [])

            # Build a quick lookup from node id to node.
            world_nodes = world_data.get("nodes", [])
            node_map = {node.get("id"): node for node in world_nodes}

            for station_id in station_ids:
                if station_id in node_map:
                    node = node_map[station_id]
                    stop = BusStop(
                        id=self._extract_station_name(station_id),
                        x=float(
                            node.get("properties", {})
                            .get("location", {})
                            .get("x", 0)
                        ),
                        y=float(
                            node.get("properties", {})
                            .get("location", {})
                            .get("y", 0)
                        ),
                        # Use a human-readable name derived from station_id.
                        name=self._extract_station_name(station_id),
                        # Default dwell time at each stop.
                        wait_time_s=self.waiting_time_s,
                    )
                    stops.append(stop)

            # Build route path points.
            path_points: List[Tuple[float, float]] = []
            path_data = route_data.get("path", [])

            # Construct an interleaved path:
            #   path[0], stop0, path[1], stop1, ...
            if path_data and stops:
                # Use the longer length to ensure we include all points/stops.
                max_length = max(len(path_data), len(stops))

                for i in range(max_length):
                    # Add a path point if available.
                    if i < len(path_data):
                        point = path_data[i]
                        if isinstance(point, dict):
                            # path points are given in meters, convert to centimeters.
                            x = float(point.get("x", 0)) * 100
                            y = float(point.get("y", 0)) * 100
                        elif isinstance(point, (list, tuple)) and len(point) >= 2:
                            x = float(point[0]) * 100
                            y = float(point[1]) * 100
                        else:
                            continue
                        path_points.append((x, y))

                    # Follow each path point with the corresponding stop position if available.
                    if i < len(stops):
                        stop = stops[i]
                        path_points.append((stop.x, stop.y))

            elif path_data:
                # Case: only path points, no explicit stops.
                # Use raw path points (converted from meters to centimeters).
                for point in path_data:
                    if isinstance(point, dict):
                        x = float(point.get("x", 0)) * 100
                        y = float(point.get("y", 0)) * 100
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        x = float(point[0]) * 100
                        y = float(point[1]) * 100
                    else:
                        continue
                    path_points.append((x, y))

            elif stops:
                # Case: only stops, no explicit path; use the stop positions as the path.
                path_points = [(stop.x, stop.y) for stop in stops]

            # Create the BusRoute for this route definition.
            route = BusRoute(
                id=route_id,
                name=route_name,
                stops=stops,
                path_points=path_points,
                speed_cm_s=self.speed_cm_s,  # default speed
                direction=1,
            )

            self.routes[route_id] = route
            print(f"Loaded bus route: {route_name} with {len(stops)} stops")

    def create_bus(self, bus_id: str, route_id: str) -> Optional[Bus]:
        """
        Create a bus instance on a given route.

        Each bus has its own deep-copied BusRoute so that per-bus
        state (e.g., reversed direction) does not affect other buses.

        Args:
            bus_id: Unique identifier for the bus.
            route_id: ID of the route to attach this bus to.

        Returns:
            The created Bus instance, or None if the route does not exist.
        """
        if route_id not in self.routes:
            print(f"Route {route_id} not found")
            return None

        # Use a deep copy so multiple buses can share a logical route
        # without sharing the same mutable BusRoute object.
        original_route = self.routes[route_id]
        route_copy = copy.deepcopy(original_route)

        bus = Bus(
            id=bus_id,
            route=route_copy,  # independent route copy for this bus
            clock=self.clock,
        )

        self.buses[bus_id] = bus
        print(f"Created bus {bus_id} on route {route_copy.name}")
        return bus

    def get_bus(self, bus_id: str) -> Optional[Bus]:
        """Retrieve a bus by bus_id."""
        return self.buses.get(bus_id)

    def get_route(self, route_id: str) -> Optional[BusRoute]:
        """Retrieve a bus route by route_id."""
        return self.routes.get(route_id)

    def find_nearby_buses(
        self,
        x: float,
        y: float,
        radius_cm: float = 1000.0,
    ) -> List[Bus]:
        """
        Find buses whose positions are within the given radius of (x, y).

        Args:
            x, y: Reference position in world coordinates (centimeters).
            radius_cm: Search radius in centimeters.

        Returns:
            A list of Bus instances within the given radius.
        """
        nearby_buses: List[Bus] = []
        for bus in self.buses.values():
            distance = math.hypot(bus.x - x, bus.y - y)
            if distance <= radius_cm:
                nearby_buses.append(bus)
        return nearby_buses

    def find_buses_at_stop(self, stop_id: str) -> List[Bus]:
        """
        Find all buses currently stopped at a given stop.

        Args:
            stop_id: ID of the BusStop to search for.

        Returns:
            A list of Bus instances that are currently at that stop.
        """
        buses_at_stop: List[Bus] = []
        for bus in self.buses.values():
            if bus.is_at_stop():
                current_stop = bus.get_current_stop()
                if current_stop and current_stop.id == stop_id:
                    buses_at_stop.append(bus)
        return buses_at_stop

    def find_nearest_bus_stop(
        self,
        x: float,
        y: float,
    ) -> Optional[Tuple[BusStop, float]]:
        """
        Find the nearest bus stop to a given position.

        Args:
            x, y: Reference position in world coordinates (centimeters).

        Returns:
            (nearest_stop, distance_cm) if any stop exists, otherwise None.
        """
        nearest_stop: Optional[BusStop] = None
        nearest_distance = float("inf")

        for route in self.routes.values():
            for stop in route.stops:
                distance = math.hypot(stop.x - x, stop.y - y)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_stop = stop

        if nearest_stop:
            return nearest_stop, nearest_distance
        return None

    def update_all_buses(self):
        """
        Advance the simulation for all buses.

        Delegates to Bus.update() for each registered bus, using
        the shared VirtualClock held by this manager.
        """
        for bus in self.buses.values():
            bus.update()

    def get_all_buses_status(self) -> List[str]:
        """
        Get a human-readable status line for every bus.

        Returns:
            A list of strings, one per bus, as produced by Bus.get_status_text().
        """
        return [bus.get_status_text() for bus in self.buses.values()]

    def get_route_info(self, route_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a structured description of a specific route.

        Returns:
            A dictionary describing the route (id, name, stops, path_points,
            speed, direction), or None if the route does not exist.
        """
        route = self.routes.get(route_id)
        if not route:
            return None

        return {
            "id": route.id,
            "name": route.name,
            "stops": [
                {
                    "id": stop.id,
                    "name": stop.name,
                    "x": stop.x,
                    "y": stop.y,
                    "wait_time_s": stop.wait_time_s,
                }
                for stop in route.stops
            ],
            "path_points": route.path_points,
            "speed_cm_s": route.speed_cm_s,
            "direction": route.direction,
        }

    def get_all_routes_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get structured descriptions of all known routes.

        Returns:
            A mapping from route_id to the same structure returned by get_route_info().
        """
        return {
            route_id: self.get_route_info(route_id)
            for route_id in self.routes.keys()
        }

    def _extract_station_name(self, station_id: str) -> str:
        """
        Derive a human-friendly station name from a raw station_id.

        Special handling:
            station_id like "GEN_POI_BusStation_0_598"
              -> "bus_station 1"

        For unknown formats, the original station_id is returned unchanged.
        """
        # Example: "GEN_POI_BusStation_0_598" -> "bus_station 1"
        if station_id.startswith("GEN_POI_BusStation_"):
            parts = station_id.split("_")
            if len(parts) >= 4:
                # Use the numeric part as a 0-based index and expose a 1-based label.
                station_number = str(int(parts[3]) + 1)
                return f"bus_station {station_number}"

        # Fallback: return the raw ID if it does not match the expected pattern.
        return station_id