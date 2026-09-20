# actions/move.py
# -*- coding: utf-8 -*-

"""
Direction-based locomotion: MOVE(direction="forward"|"left"|"right"|"backward").

The agent has a persistent compass facing (`dm.facing_deg`, 0=N/90=E/180=S/270=W).
Each waypoint has at most one reachable neighbour in each of the four directions
relative to that facing. MOVE picks the neighbour in the requested direction,
walks one edge to it, and **rotates the facing** accordingly:

    forward  -> facing unchanged          (go to the neighbour ahead)
    right    -> facing + 90               (turn right, then step)
    backward -> facing + 180              (turn around, then step)
    left     -> facing + 270 (= -90)      (turn left, then step)

On the bundled grid maps every edge is exactly cardinal, so the requested
direction resolves to exactly one neighbour (or none → a clear error). The same
energy/clock accounting as the old STEP_TO is used, so time/reward mechanics are
unchanged; only the *interface* (id-free, FPV-driven) differs.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base.defs import DMAction


# Relative bearing added to the current facing for each direction.
_DIR_OFFSET = {"forward": 0.0, "right": 90.0, "backward": 180.0, "left": 270.0}
# How close (deg) a neighbour's bearing must be to the requested direction.
# Keep this below 45° so diagonal corner connectors are not legal MOVE edges.
_BIN_TOL_DEG = 30.0
_DIR_COMPASS = {  # for readable messages
    0.0: "N", 90.0: "E", 180.0: "S", 270.0: "W",
}


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def available_moves(dm: Any) -> Dict[str, Optional[Dict[str, Any]]]:
    """Map {forward,left,right,backward} -> the adjacent waypoint dict in that
    direction relative to ``dm.facing_deg`` (or None when nothing is reachable).

    Shared by the MOVE handler and the observation builder so the text hint and
    the actual action always agree. Each value, when present, is the dict that
    ``city_map.adjacents`` returns (id/name/kind/dist_m/bearing_deg/compass/
    road_name/node).
    """
    out: Dict[str, Optional[Dict[str, Any]]] = {d: None for d in _DIR_OFFSET}
    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "nearest_waypoint"):
        return out
    current = city_map.nearest_waypoint(float(dm.x), float(dm.y))
    if current is None:
        return out
    facing = float(getattr(dm, "facing_deg", 0.0)) % 360.0
    adj = city_map.adjacents(current) or []
    for direction, offset in _DIR_OFFSET.items():
        want = (facing + offset) % 360.0
        best = None
        best_d = _BIN_TOL_DEG
        for a in adj:
            diff = _ang_diff(float(a.get("bearing_deg", 0.0)), want)
            if diff <= best_d:
                best, best_d = a, diff
        out[direction] = best
    return out


def enumerate_candidates(dm: Any) -> List[Dict[str, Any]]:
    """Stable numbered one-hop candidates for marked-waypoint navigation
    (enable_waypoint_marks).

    Sorted by (absolute bearing_deg, dist, id) so the numbering is
    deterministic and permanently stable per waypoint. The FPV marker
    renderer, the per-step candidate text, and the MOVE_TO validator all call
    this one function, so number k always means the same waypoint everywhere.
    Returns [{index, id, name, road_name, bearing_deg, dist_cm, node}, ...]
    with index starting at 1.
    """
    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "nearest_waypoint"):
        return []
    current = city_map.nearest_waypoint(float(dm.x), float(dm.y))
    if current is None:
        return []
    adj = city_map.adjacents(current) or []
    rows: List[Dict[str, Any]] = []
    for a in adj:
        rows.append({
            "id": str(a.get("id") or a.get("name") or ""),
            "name": str(a.get("name") or ""),
            "road_name": str(a.get("road_name") or ""),
            "bearing_deg": float(a.get("bearing_deg", 0.0)) % 360.0,
            "dist_cm": float(a.get("dist_m", 0.0)) * 100.0,
            "node": a.get("node"),
        })
    rows.sort(key=lambda r: (r["bearing_deg"], r["dist_cm"], r["id"]))
    for i, r in enumerate(rows, start=1):
        r["index"] = i
    return rows


def _execute_edge_step(dm: Any, current_node: Any, target_match: Dict[str, Any],
                       new_facing: float, log_label: str) -> None:
    """Shared MOVE/MOVE_TO edge-step mechanics: traffic-light check, snap to
    the target waypoint, energy/battery/clock accounting (identical to the
    former STEP_TO so reward/time are stable), facing update, NAVIGATE-route
    bookkeeping, fragility bump, and action finish. Callers have already
    resolved `target_match` (an adjacents/candidate dict) and run their own
    obstacle check."""
    target_node = target_match["node"]
    dist_cm = float(target_match.get("dist_cm") or 0.0)
    if dist_cm <= 0.0:
        dist_cm = float(target_match.get("dist_m", 0.0)) * 100.0
    if dist_cm <= 0.0:
        dist_cm = current_node.position.distance(target_node.position)

    tx, ty = float(target_node.position.x), float(target_node.position.y)

    # Traffic light: crossing a signalised intersection while the light for the
    # *travel axis* is red still moves the agent at normal cost but counts a
    # violation. It is vision-only (the red FPV is the only signal) and, for now,
    # purely a counter — no time/energy penalty. The axis is taken from the move
    # bearing, so the front and perpendicular directions are scored against
    # opposite signals (front red ⇒ counted; right green ⇒ free).
    traffic = getattr(dm, "_traffic", None)
    if traffic is not None and traffic.is_signalised(float(dm.x), float(dm.y)):
        bearing = float(target_match.get("bearing_deg", new_facing))
        if traffic.light_for_bearing(bearing, float(dm.clock.now_sim())) == "red":
            dm.traffic_violation_count = int(getattr(dm, "traffic_violation_count", 0)) + 1

    facing = float(getattr(dm, "facing_deg", 0.0)) % 360.0
    dm._log(
        f"{log_label}: -> {getattr(target_node, 'waypoint_id', '?')} "
        f"({getattr(target_node, 'waypoint_name', '')}) {dist_cm/100.0:.1f}m; "
        f"facing {facing:.0f}->{new_facing:.0f}"
    )

    # Snap to the target waypoint, then charge energy/battery and advance the
    # clock — identical accounting to the former STEP_TO so reward/time are stable.
    dm.x, dm.y = tx, ty
    try:
        dm.on_move_consumed(float(dist_cm))
    except Exception:
        pass
    try:
        speed = max(1e-6, float(dm.speed_cm_s) * float(dm._pace_scale()))
        travel_s = max(0.0, float(dist_cm) / speed)
        if hasattr(dm.clock, "advance"):
            dm.clock.advance(travel_s)
    except Exception:
        pass

    # Rotate facing to the direction actually travelled.
    dm.facing_deg = new_facing

    # The planned NAVIGATE route persists on the map across steps; clear it only
    # once the agent has REACHED the navigation destination. (A new NAVIGATE call
    # overwrites it.)
    tgt = getattr(dm, "_nav_target_node", None)
    if tgt is not None and target_node is tgt:
        dm._nav_route_path = []
        dm._nav_route_color = None
        # Keep _nav_target_node and the [navigation] block so the next
        # observation can explicitly say "next_move: you have arrived".
        # A later NAVIGATE call overwrites this finished target.

    if (dm.cfg.get("enable_fragility_compartment_damage")
            and dm.cfg.get("enable_food_fragility", True)
            and dm.insulated_bag is not None):
        try:
            dm.insulated_bag.bump_motion_damage_shared(
                every_n=int(dm.cfg.get("fragility_bump_every_n_moves", 4))
            )
        except Exception:
            pass

    dm._move_ctx = None
    dm._finish_action(success=True)


def handle_move(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """Execute MOVE(direction=...): step to the neighbour in that direction and
    rotate the facing."""
    direction = str(act.data.get("direction", "")).strip().lower()
    if direction not in _DIR_OFFSET:
        dm.vlm_add_error(
            'MOVE needs direction="forward"|"left"|"right"|"backward".'
        )
        dm._finish_action(success=False)
        return

    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "nearest_waypoint"):
        dm.vlm_add_error("MOVE: map has no waypoint graph.")
        dm._finish_action(success=False)
        return
    current_node = city_map.nearest_waypoint(float(dm.x), float(dm.y))
    if current_node is None:
        dm.vlm_add_error("MOVE: could not locate the current waypoint.")
        dm._finish_action(success=False)
        return

    facing = float(getattr(dm, "facing_deg", 0.0)) % 360.0
    new_facing = (facing + _DIR_OFFSET[direction]) % 360.0

    moves = available_moves(dm)
    target_match = moves.get(direction)
    if target_match is None:
        legal = ", ".join(d for d, v in moves.items() if v is not None) or "(none)"
        dm.vlm_add_error(
            f"MOVE: no reachable waypoint to your {direction} "
            f"(facing {_DIR_COMPASS.get(facing, f'{facing:.0f}deg')}). "
            f"Directions you can move: {legal}."
        )
        dm._finish_action(success=False)
        return

    # --- Pluggable hazards (default-off; see OBSTACLE_TRAFFIC_DESIGN.md) ---
    # Both mechanics are driven by the single TrafficController / ObstacleField
    # runtime (utils/hazards.py); there is no second traffic-light code path
    # (the traffic-light check lives in _execute_edge_step).
    #
    # Obstacle: a static block on the forward edge. MOVE("forward") into it is a
    # collision (no move, counter++); the agent must use BYPASS() to pass. Only
    # the forward edge is checked — that is the head-on case the FPV shows.
    # (MOVE_TO checks its *chosen* edge instead; see handle_move_to.)
    obstacles = getattr(dm, "_obstacle_field", None)
    if direction == "forward" and obstacles is not None:
        target_node = target_match["node"]
        tx, ty = float(target_node.position.x), float(target_node.position.y)
        if obstacles.obstacle_on(float(dm.x), float(dm.y), tx, ty) is not None:
            dm.vlm_add_error("blocked, unable to proceed")
            dm.collision_count = int(getattr(dm, "collision_count", 0)) + 1
            dm._finish_action(success=False)
            return

    _execute_edge_step(dm, current_node, target_match, new_facing,
                       f"MOVE {direction}")


def _resolve_move_to_target(cands: List[Dict[str, Any]],
                            target: Any) -> Optional[Dict[str, Any]]:
    """Match a MOVE_TO argument against the numbered candidates: an int (or
    digit string) is a mark index; any other string matches the waypoint id or
    name case-insensitively. Returns the candidate row or None."""
    if isinstance(target, bool):
        return None
    if isinstance(target, (int, float)):
        if float(target) != int(target):
            return None
        k = int(target)
        for c in cands:
            if c["index"] == k:
                return c
        return None
    if isinstance(target, str):
        t = target.strip()
        if t.isdigit():
            k = int(t)
            for c in cands:
                if c["index"] == k:
                    return c
            return None
        tl = t.lower()
        for c in cands:
            if c["id"].lower() == tl or (c["name"] and c["name"].lower() == tl):
                return c
    return None


def handle_move_to(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """Execute MOVE_TO(k) / MOVE_TO("dock_94") (enable_waypoint_marks): step
    along the chosen edge to any one-hop adjacent waypoint and rotate the
    facing onto that edge's bearing.

    Unlike MOVE, the candidate set is the FULL adjacency list — numbered by
    enumerate_candidates, the same function that renders the FPV markers and
    writes the per-step candidate text — so non-cardinal edges are reachable
    and mark k always refers to the same waypoint the agent saw."""
    if not ((getattr(dm, "cfg", None) or {}).get("enable_waypoint_marks", False)):
        dm.vlm_add_error("MOVE_TO is not available in this configuration.")
        dm._finish_action(success=False)
        return

    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "nearest_waypoint"):
        dm.vlm_add_error("MOVE_TO: map has no waypoint graph.")
        dm._finish_action(success=False)
        return
    current_node = city_map.nearest_waypoint(float(dm.x), float(dm.y))
    if current_node is None:
        dm.vlm_add_error("MOVE_TO: could not locate the current waypoint.")
        dm._finish_action(success=False)
        return

    cands = enumerate_candidates(dm)
    if not cands:
        dm.vlm_add_error("MOVE_TO: no adjacent waypoint is reachable from here.")
        dm._finish_action(success=False)
        return

    chosen = _resolve_move_to_target(cands, act.data.get("target"))
    if chosen is None:
        valid = "; ".join(f"{c['index']}={c['id']}" for c in cands)
        dm.vlm_add_error(
            f"MOVE_TO: {act.data.get('target')!r} does not match any numbered "
            f"waypoint mark. Valid marks this step: {valid}. "
            f"Reply with MOVE_TO(<mark number>)."
        )
        dm._finish_action(success=False)
        return

    # Obstacle check on the CHOSEN edge. MOVE only checks the head-on forward
    # edge (the case the FPV front panel shows), but MOVE_TO lets the agent
    # pick any edge and the marked FPV shows a barrier in whichever panel
    # faces it — so a blocked chosen edge fails the same way: in place, with
    # the collision counted. BYPASS stays forward-bound.
    obstacles = getattr(dm, "_obstacle_field", None)
    if obstacles is not None:
        node = chosen["node"]
        if obstacles.obstacle_on(float(dm.x), float(dm.y),
                                 float(node.position.x),
                                 float(node.position.y)) is not None:
            dm.vlm_add_error("blocked, unable to proceed")
            dm.collision_count = int(getattr(dm, "collision_count", 0)) + 1
            dm._finish_action(success=False)
            return

    new_facing = float(chosen.get("bearing_deg", 0.0)) % 360.0
    _execute_edge_step(dm, current_node, chosen, new_facing,
                       f"MOVE_TO {chosen['index']}={chosen['id']}")
