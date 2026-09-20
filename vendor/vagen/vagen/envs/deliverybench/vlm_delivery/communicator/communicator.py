# communicator/communicator.py
# -*- coding: utf-8 -*-

"""
Communicator.py

This module provides a single UnrealCV-based communicator instance shared by
multiple agents. All Unreal Engine calls are protected by a unified RLock.
Each delivery man uses an independent walking thread to enable parallel
route-following. Supports robust parsing of location/orientation responses
and provides high-level movement utilities.
"""

import math
import random
import re
import threading
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from .unrealcv_delivery import UnrealCvDelivery
from ..base.types import Vector

# Default acceleration and braking values (cm/s^2) for near-instant start/stop.
DEFAULT_MAX_ACCEL_CM_S2 = 80000.0
DEFAULT_BRAKE_DECEL_CM_S2 = 80000.0

# Legacy constant kept for backward compatibility.
SPEED_KEEPALIVE_SEC = 0.8


class Communicator(UnrealCvDelivery):
    def __init__(self, port: int, ip: str, resolution: Tuple[int, int], cfg: Dict[str, Any],):
        """Initialize communicator with a shared RLock for all UE commands."""
        super().__init__(port, ip, resolution)

        self.delivery_manager_name: Optional[str] = None
        self.delivery_man_id_to_name: Dict[str, str] = {}

        self._walkers: Dict[str, threading.Thread] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

        self._send_lock = self.lock  # unified send lock

        self.cfg = cfg

    # ------------------------------------------------------
    # ID → Actor name mapping
    # ------------------------------------------------------
    def get_delivery_man_name(self, agent_id: Any) -> str:
        key = str(agent_id)
        name = self.delivery_man_id_to_name.get(key)
        if not name:
            name = f"GEN_DELIVERY_MAN_{key}"
            self.delivery_man_id_to_name[key] = name
        return name

    # ------------------------------------------------------
    # Actor spawning
    # ------------------------------------------------------
    def spawn_delivery_man(self, agent_id: Any, x: float, y: float) -> None:
        key = str(agent_id)
        name = f"GEN_DELIVERY_MAN_{key}"
        self.delivery_man_id_to_name[key] = name

        model_name = self.cfg["ue_models"]["delivery_man"]
        with self._send_lock:
            self.spawn_bp_asset(model_name, name)
            self.set_location((float(x), float(y), 110.0), name)
            self.set_orientation((0.0, 0.0, 0.0), name)
            self.set_scale((1.0, 1.0, 1.0), name)
            self.set_collision(name, True)
            self.set_movable(name, True)

    def spawn_delivery_manager(self) -> None:
        self.delivery_manager_name = "GEN_DeliveryManager"
        with self._send_lock:
            self.spawn_bp_asset(
                self.cfg["ue_models"]["delivery_manager"], self.delivery_manager_name
            )

    def spawn_customer(self, order_id: int, x: float, y: float) -> None:
        name = f"GEN_CUSTOMER_{order_id}"
        with self._send_lock:
            self.spawn_bp_asset(self.cfg["ue_models"]["customer"], name)
            self.set_location((float(x), float(y), 110.0), name)
            self.set_orientation((0.0, random.uniform(0, 360), 0.0), name)
            self.set_scale((1.0, 1.0, 1.0), name)
            self.set_collision(name, True)
            self.set_movable(name, True)

    def destroy_customer(self, order_id: int) -> None:
        name = f"GEN_CUSTOMER_{order_id}"
        with self._send_lock:
            self.destroy(name)

    # ------------------------------------------------------
    # Vec3 parsing + readiness checks
    # ------------------------------------------------------
    def _parse_vec3(self, raw: Any) -> Optional[Tuple[float, float, float]]:
        """Parse various UnrealCV-returned vector formats into (x, y, z)."""
        if raw is None:
            return None

        # Sequence: list/tuple/ndarray etc.
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            try:
                vals = list(raw)
                if len(vals) >= 2:
                    x = float(vals[0])
                    y = float(vals[1])
                    z = float(vals[2]) if len(vals) > 2 else 0.0
                    return (x, y, z)
            except Exception:
                return None

        # Generic iterable object.
        if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes, dict)):
            try:
                it = iter(raw)
                x = float(next(it))
                y = float(next(it))
                try:
                    z = float(next(it))
                except StopIteration:
                    z = 0.0
                return (x, y, z)
            except Exception:
                pass

        # String formats.
        if isinstance(raw, str):
            s = raw.strip()
            if not s or s == "{}":
                return None

            # Match X=..., Y=..., Z=...
            mX = re.search(r"[Xx]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            mY = re.search(r"[Yy]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            mZ = re.search(r"[Zz]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            if mX and mY:
                x = float(mX.group(1))
                y = float(mY.group(1))
                z = float(mZ.group(1)) if mZ else 0.0
                return (x, y, z)

            # Plain numbers
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if len(nums) >= 2:
                x = float(nums[0])
                y = float(nums[1])
                z = float(nums[2]) if len(nums) > 2 else 0.0
                return (x, y, z)

        return None

    def wait_actor_ready(self, delivery_man_id: Any, timeout: float = 5.0, interval: float = 0.1) -> bool:
        """Poll until both valid location and orientation are returned."""
        name = self.get_delivery_man_name(delivery_man_id)
        end = time.time() + float(timeout)

        while time.time() < end:
            with self._send_lock:
                loc_raw = self.get_location(name)
                rot_raw = self.get_orientation(name)

            loc = self._parse_vec3(loc_raw)
            rot = self._parse_vec3(rot_raw)
            if loc and rot:
                return True
            time.sleep(interval)

        return False

    # ------------------------------------------------------
    # Position and heading query
    # ------------------------------------------------------
    def get_position_and_direction(self, delivery_man_id: Any) -> Dict[str, Tuple[Vector, float]]:
        """Return {"id": (Vector(x,y), yaw)} or {} if invalid."""
        try:
            name = self.get_delivery_man_name(delivery_man_id)

            with self._send_lock:
                loc_raw = self.get_location(name)
                rot_raw = self.get_orientation(name)

            loc = self._parse_vec3(loc_raw)
            rot = self._parse_vec3(rot_raw)
            if not loc or not rot:
                print(f"Warning: Could not retrieve location/orientation for {name}")
                return {}

            px, py = float(loc[0]), float(loc[1])
            yaw = float(rot[1]) if len(rot) >= 2 else 0.0

            return {str(delivery_man_id): (Vector(px, py), yaw)}

        except Exception as e:
            print(f"Error in get_position_and_direction: {e}")
            return {}

    # ------------------------------------------------------
    # Basic controlled actions (locked)
    # ------------------------------------------------------
    def set_orientation_safe(self, name: str, euler_xyz: List[float]) -> None:
        with self._send_lock:
            self.set_orientation(euler_xyz, name)

    def delivery_man_move_forward(self, delivery_man_id: Any) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_move_forward(name)

    def delivery_man_stop(self, delivery_man_id: Any) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_stop(name)

    def delivery_man_step_forward(self, delivery_man_id: Any, speed: float, time: float) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_step_forward(name, speed, time)

    def delivery_man_turn_around(self, delivery_man_id: Any, angle: float, direction: str) -> None:
        name = self.get_delivery_man_name(delivery_man_id)
        with self._send_lock:
            self.d_turn_around(name, angle, direction)

    # ------------------------------------------------------
    # Speed / acceleration configuration
    # ------------------------------------------------------
    def configure_speed_profile(
        self,
        delivery_man_id: Any,
        speed_cm_s: Optional[float] = None,
        accel_cm_s2: Optional[float] = None,
        decel_cm_s2: Optional[float] = None,
    ) -> None:
        """Set speed, acceleration, and braking (defaults approximate constant speed)."""
        try:
            with self._send_lock:
                if speed_cm_s is not None:
                    self.set_max_speed(delivery_man_id, float(speed_cm_s))

                self.set_max_accel(
                    delivery_man_id,
                    float(accel_cm_s2 if accel_cm_s2 is not None else DEFAULT_MAX_ACCEL_CM_S2),
                )
                self.set_braking_decel(
                    delivery_man_id,
                    float(decel_cm_s2 if decel_cm_s2 is not None else DEFAULT_BRAKE_DECEL_CM_S2),
                )
        except Exception as e:
            print(f"[UE] configure_speed_profile failed on {delivery_man_id}: {e}")

    # ------------------------------------------------------
    # Asynchronous route following
    # ------------------------------------------------------
    def stop_go_to(self, delivery_man_id: Any) -> None:
        """Signal an active walking thread to stop."""
        key = str(delivery_man_id)
        with self._lock:
            ev = self._stops.get(key)
            if ev:
                ev.set()

    def go_to_xy_async(
        self,
        delivery_man_id: Any,
        route: List[Tuple[float, float]],
        speed_cm_s: Optional[float] = None,
        accel_cm_s2: Optional[float] = None,
        decel_cm_s2: Optional[float] = None,
        arrive_tolerance_cm: float = 300.0,
    ) -> None:
        """Launch a background thread to follow the full route."""
        if not route:
            return

        key = str(delivery_man_id)
        self.stop_go_to(key)

        stop_event = threading.Event()
        t = threading.Thread(
            target=self._walk_route,
            args=(key, list(route), stop_event, speed_cm_s, accel_cm_s2, decel_cm_s2, float(arrive_tolerance_cm)),
            name=f"UEWalker-{key}",
            daemon=True,
        )

        with self._lock:
            self._stops[key] = stop_event
            self._walkers[key] = t

        t.start()

    # Backward-compatible synchronous wrapper
    def go_to_xy(
        self,
        delivery_man_id: Any,
        route: List[Tuple[float, float]],
        speed_cm_s: Optional[float] = None,
        accel_cm_s2: Optional[float] = None,
        decel_cm_s2: Optional[float] = None,
        arrive_tolerance_cm: float = 300.0,
    ) -> None:
        self.go_to_xy_async(
            delivery_man_id,
            route,
            speed_cm_s=speed_cm_s,
            accel_cm_s2=accel_cm_s2,
            decel_cm_s2=decel_cm_s2,
            arrive_tolerance_cm=arrive_tolerance_cm,
        )

    # ------------------------------------------------------
    # Teleportation
    # ------------------------------------------------------
    def teleport_xy(self, delivery_man_id: Any, x: float, y: float) -> None:
        """Teleport instantly to (x, y), stopping any ongoing movement."""
        name = self.get_delivery_man_name(delivery_man_id)

        self.stop_go_to(delivery_man_id)

        t = self._walkers.pop(str(delivery_man_id), None)
        if t and t.is_alive():
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

        try:
            with self._send_lock:
                self.d_stop(name)
        except Exception:
            pass

        cmd = f"vbp {name} SetLocation {float(x):.2f} {float(y):.2f}"
        print("cmd:", cmd)
        with self._send_lock:
            self.client.request(cmd)

    # ------------------------------------------------------
    # Core route-walking implementation
    # ------------------------------------------------------
    def _walk_route(
        self,
        delivery_man_id: str,
        route: List[Tuple[float, float]],
        stop_event: threading.Event,
        speed_cm_s: Optional[float],
        accel_cm_s2: Optional[float],
        decel_cm_s2: Optional[float],
        arrive_tolerance_cm: float,
    ) -> None:
        """
        Minimal route-following implementation relying on UE's
        "Orient Rotation to Movement":
        - Configure speed & acceleration once.
        - Issue one MoveToNextPoint per waypoint.
        - Poll distance; advance when within tolerance.
        - Detect stalled movement to avoid deadlock.
        """
        name = self.get_delivery_man_name(delivery_man_id)
        tol = float(arrive_tolerance_cm)

        def _get_pos() -> Optional["Vector"]:
            cur = self.get_position_and_direction(delivery_man_id)
            if not cur:
                return None
            pos, _ = cur.get(delivery_man_id, (None, None))
            return pos

        def _dist_to(tx: float, ty: float) -> Optional[float]:
            p = _get_pos()
            if not p:
                return None
            return math.hypot(float(p.x) - float(tx), float(p.y) - float(ty))

        def _issue_move(x: float, y: float, tolerance: float) -> None:
            cmd = f"vbp {name} MoveToNextPoint {x:.2f} {y:.2f} {tolerance:.1f}"
            with self._send_lock:
                self.client.request(cmd)

        # Configure movement profile once.
        self.configure_speed_profile(
            delivery_man_id,
            speed_cm_s=speed_cm_s,
            accel_cm_s2=accel_cm_s2,
            decel_cm_s2=decel_cm_s2,
        )

        try:
            for (tx, ty) in route:
                if stop_event.is_set():
                    break

                _issue_move(tx, ty, tol)

                stagnant = 0
                max_stagnant = 10
                eps = 1e-3
                prev_d = None

                while not stop_event.is_set():
                    d = _dist_to(tx, ty)

                    if d is not None and d <= tol:
                        break

                    if d is None:
                        stagnant = 0
                    else:
                        if prev_d is not None and abs(d - prev_d) <= eps:
                            stagnant += 1
                        else:
                            stagnant = 0
                        prev_d = d

                        if stagnant >= max_stagnant:
                            print(f"[{name}] No progress for {max_stagnant} ticks; breaking.")
                            break

                    time.sleep(0.05)

        except Exception as e:
            try:
                self.delivery_man_stop(delivery_man_id)
            except Exception:
                pass
            print(f"[{name}] walk_route error: {e}")