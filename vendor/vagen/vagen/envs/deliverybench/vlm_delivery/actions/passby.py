# actions/passby.py
# -*- coding: utf-8 -*-

"""
PASSBY(): move forward (same result as MOVE(forward)) at 1.5x time/energy cost.

No arguments. The agent steps to the adjacent forward waypoint and keeps its
current facing, but the energy consumed and clock time advanced are both
multiplied by 1.5 compared to a normal MOVE(forward).
"""

from typing import Any

from ..base.defs import DMAction
from .move import available_moves

_PASSBY_COST_SCALE_DEFAULT = 1.5

_DIR_COMPASS = {0.0: "N", 90.0: "E", 180.0: "S", 270.0: "W"}


def _passby_cost_scale(dm: Any) -> float:
    cfg = getattr(dm, "cfg", None)
    if isinstance(cfg, dict):
        try:
            return float(cfg.get("passby_cost_scale", _PASSBY_COST_SCALE_DEFAULT))
        except (TypeError, ValueError):
            pass
    return _PASSBY_COST_SCALE_DEFAULT


def handle_passby(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """Execute PASSBY(): step forward at 1.5x time/energy cost."""
    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "nearest_waypoint"):
        dm.vlm_add_error("PASSBY: map has no waypoint graph.")
        dm._finish_action(success=False)
        return

    moves = available_moves(dm)
    target_match = moves.get("forward")
    if target_match is None:
        facing = float(getattr(dm, "facing_deg", 0.0)) % 360.0
        legal = ", ".join(d for d, v in moves.items() if v is not None) or "(none)"
        dm.vlm_add_error(
            f"PASSBY: no reachable waypoint forward "
            f"(facing {_DIR_COMPASS.get(facing, f'{facing:.0f}deg')}). "
            f"Directions you can move: {legal}."
        )
        dm._finish_action(success=False)
        return

    target_node = target_match["node"]
    dist_cm = float(target_match.get("dist_m", 0.0)) * 100.0
    if dist_cm <= 0.0:
        dist_cm = city_map.nearest_waypoint(float(dm.x), float(dm.y)).position.distance(
            target_node.position
        )

    tx, ty = float(target_node.position.x), float(target_node.position.y)
    scale = _passby_cost_scale(dm)
    dm._log(
        f"PASSBY: -> {getattr(target_node, 'waypoint_id', '?')} "
        f"({getattr(target_node, 'waypoint_name', '')}) {dist_cm / 100.0:.1f}m "
        f"({scale:g}x cost)"
    )

    # Traffic light: bypassing across a signalised intersection on red is still a
    # violation (it moves, but counts), same as a normal MOVE.
    traffic = getattr(dm, "_traffic", None)
    if traffic is not None and traffic.is_signalised(float(dm.x), float(dm.y)):
        bearing = float(target_match.get("bearing_deg", getattr(dm, "facing_deg", 0.0)))
        if traffic.light_for_bearing(bearing, float(dm.clock.now_sim())) == "red":
            dm.traffic_violation_count = int(getattr(dm, "traffic_violation_count", 0)) + 1

    dm.x, dm.y = tx, ty
    try:
        dm.on_move_consumed(float(dist_cm) * scale)
    except Exception:
        pass
    try:
        speed = max(1e-6, float(dm.speed_cm_s) * float(dm._pace_scale()))
        travel_s = max(0.0, float(dist_cm) / speed) * scale
        if hasattr(dm.clock, "advance"):
            dm.clock.advance(travel_s)
    except Exception:
        pass

    # Facing is unchanged (moving forward).
    tgt = getattr(dm, "_nav_target_node", None)
    if tgt is not None and target_node is tgt:
        dm._nav_route_path = []
        dm._nav_route_color = None
        dm._nav_target_node = None

    dm._move_ctx = None
    dm._finish_action(success=True)
