# actions/_nav_helpers.py
# -*- coding: utf-8 -*-

"""
Shared helpers for query-only waypoint-graph navigation actions.

Used by:
  - actions/navigate.py             (waypoint chain output)
  - actions/navigate_directions.py  (turn-by-turn output)

These helpers are pure reads against ``dm.city_map`` and never modify
agent or environment state. Lifted out of navigate.py as a mechanical
no-behavior-change refactor so both action handlers can share them
without diverging.
"""

from typing import Any, List


def _fmt_time(t_s: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if t_s >= 3600.0:
        return f"{t_s / 3600.0:.1f}h"
    if t_s >= 60.0:
        return f"{t_s / 60.0:.0f}min"
    return f"{t_s:.0f}s"


def resolve_token(dm: Any, token: str):
    """Resolve a free-form waypoint token (id / address / name) to a Node."""
    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "resolve_waypoint"):
        return None
    return city_map.resolve_waypoint(str(token))


def _norm_token(text: Any) -> str:
    return str(text or "").strip().lower()


def _node_tokens(node: Any) -> set:
    vals = {
        getattr(node, "address", ""),
        getattr(node, "waypoint_name", ""),
        getattr(node, "waypoint_id", ""),
    }
    return {_norm_token(v) for v in vals if v}


def order_endpoint_by_target(dm: Any, target_token: str):
    """Resolve target text against active order endpoints before the global map.

    Agent-facing observations expose pickup/dropoff endpoints as human addresses,
    while the map can contain ordinary waypoints with similar labels. Once an
    order is active, navigation to its address should target the order endpoint.
    """
    t = _norm_token(target_token)
    if not t:
        return None

    orders = list(getattr(dm, "active_orders", []) or [])
    orders.extend(list((getattr(dm, "help_orders", {}) or {}).values()))

    # Explicit semantic forms, e.g. "pickup of order #0" or "dropoff order 4".
    for order in orders:
        oid = getattr(order, "id", None)
        if oid is None:
            continue
        oid_forms = {f"order #{oid}", f"order {oid}", f"#{oid}", str(oid)}
        if any(form in t for form in oid_forms):
            if "pickup" in t or "pick up" in t:
                return getattr(order, "pickup_node", None)
            if "dropoff" in t or "drop-off" in t or "drop off" in t or "delivery" in t:
                return getattr(order, "dropoff_node", None)

    # Exact endpoint-address/name/id forms shown in the active-order text.
    for order in orders:
        pickup = getattr(order, "pickup_node", None)
        dropoff = getattr(order, "dropoff_node", None)
        picked = bool(getattr(order, "has_picked_up", False))

        # If pickup and dropoff labels ever collide, prefer the endpoint relevant
        # to the order's current phase.
        if not picked and pickup is not None and t in _node_tokens(pickup):
            return pickup
        if picked and dropoff is not None and t in _node_tokens(dropoff):
            return dropoff
        if pickup is not None and t in _node_tokens(pickup):
            return pickup
        if dropoff is not None and t in _node_tokens(dropoff):
            return dropoff

    return None


def resolve_navigation_target(dm: Any, token: str):
    """Resolve a NAVIGATE target using task endpoints before global waypoints."""
    return order_endpoint_by_target(dm, token) or resolve_token(dm, token)


def semantic_target_label(dm: Any, node: Any) -> str:
    """Semantic name of a navigation target node ("pickup for order #0",
    "charging_station 1"), or "" when it has none. Pure read."""
    if node is None:
        return ""

    orders = list(getattr(dm, "active_orders", []) or [])
    orders.extend(list((getattr(dm, "help_orders", {}) or {}).values()))
    for order in orders:
        oid = getattr(order, "id", None)
        if oid is None:
            continue
        if getattr(order, "pickup_node", None) is node:
            return f"pickup for order #{oid}"
        if getattr(order, "dropoff_node", None) is node:
            return f"dropoff for order #{oid}"

    city_map = getattr(dm, "city_map", None)
    for meta in getattr(city_map, "poi_meta", None) or []:
        if node in (meta.get("dock_node"), meta.get("door_node"), meta.get("node")):
            disp = getattr(meta.get("node"), "display_name", "") or ""
            if disp:
                return disp
    return ""


def current_waypoint(dm: Any):
    """Waypoint nearest to the agent's current (x, y) position."""
    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "nearest_waypoint"):
        return None
    return city_map.nearest_waypoint(float(dm.x), float(dm.y))


def waypoint_label(city_map: Any, node: Any) -> str:
    """Render a waypoint node as ``"id (name)"``."""
    wp_id = getattr(node, "waypoint_id", "") or ""
    wp_name = getattr(node, "waypoint_name", "") or ""
    if wp_id and wp_name:
        return f"{wp_id} ({wp_name})"
    return wp_id or wp_name or str(node)


def edge_distance_m(city_map: Any, u: Any, v: Any) -> float:
    """Edge length in meters, preferring stored ``dist_cm`` meta over Euclidean."""
    graph = getattr(city_map, "waypoint_graph", None)
    meta = graph.get_edge_meta(u, v) if graph is not None else None
    if meta and "dist_cm" in meta:
        return float(meta["dist_cm"]) / 100.0
    try:
        return float(u.position.distance(v.position)) / 100.0
    except Exception:
        return 0.0


def path_distance_m(city_map: Any, path: List[Any]) -> float:
    """Sum of per-edge distances along a path."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += edge_distance_m(city_map, u, v)
    return total


_COMPASS_FULL = {
    0: "North", 45: "Northeast", 90: "East", 135: "Southeast",
    180: "South", 225: "Southwest", 270: "West", 315: "Northwest",
}
# Relative turn for a heading change (new - prev) snapped to 90° bins, given the
# MOVE facing convention (right=+90, back=180, left=270/-90).
_REL_TURN = {0: "go straight", 90: "turn right", 180: "turn around", 270: "turn left"}


def _edge_bearing(city_map: Any, u: Any, v: Any) -> float:
    """Compass bearing (0=N, 90=E) from waypoint ``u`` to neighbour ``v``."""
    adj = getattr(city_map, "adjacents", None)
    if adj is not None:
        for a in (adj(u) or []):
            if a.get("node") is v and "bearing_deg" in a:
                return float(a["bearing_deg"]) % 360.0
    import math
    dx = float(v.position.x) - float(u.position.x)   # east
    dy = float(v.position.y) - float(u.position.y)   # north
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _edge_road(city_map: Any, u: Any, v: Any) -> str:
    adj = getattr(city_map, "adjacents", None)
    if adj is not None:
        for a in (adj(u) or []):
            if a.get("node") is v:
                return a.get("road_name", "") or ""
    return ""


_MOVE_PHRASE = {0: "move forward", 90: "turn right", 180: "move backward", 270: "turn left"}


def route_directions_text(city_map: Any, path: List[Any], start_facing_deg: float) -> str:
    """The single next MOVE to take along ``path`` (relative to current facing).

    Returns one line, e.g. ``"next_move: turn left"`` — "move forward",
    "turn left", "turn right", or "move backward". No compass/distance/road.
    """
    if not path or len(path) < 2:
        return ""
    prev = round(float(start_facing_deg) / 90.0) * 90 % 360
    deg = round(_edge_bearing(city_map, path[0], path[1]) / 90.0) * 90 % 360
    return f"next_move: {_MOVE_PHRASE.get((deg - prev) % 360, 'move forward')}"


def mode_estimates_text(dm: Any, distance_m: float) -> str:
    """
    Build a ``mode_estimates:`` text block comparing walk vs. e-scooter.

    Uses the agent's current speed/energy config and pace scale to compute
    per-mode estimates of travel time and personal energy consumption. For
    e-scooter, also estimates battery usage and reports feasibility.

    Never modifies any agent state. Returns an empty string when the agent
    object has no speed/energy config (e.g., a stub in unit tests).
    """
    from ..base.defs import TransportMode
    from ..entities.escooter import ScooterState

    avg_speed = getattr(dm, "avg_speed_by_mode", None)
    energy_cost = getattr(dm, "energy_cost_by_mode", None)
    if avg_speed is None or energy_cost is None:
        return ""

    pace = float(
        getattr(dm, "pace_scales", {}).get(
            getattr(dm, "pace_state", "normal"), 1.0
        )
    )
    dist = max(0.0, float(distance_m))

    lines = ["mode_estimates:"]

    # ---- Walk (always available) ----
    walk_spd = float(avg_speed.get(TransportMode.WALK, 200.0))
    walk_erate = float(energy_cost.get(TransportMode.WALK, 0.08))
    walk_m_s = walk_spd / 100.0 * pace
    walk_time_s = dist / walk_m_s if walk_m_s > 0 else float("inf")
    walk_epct = dist * walk_erate * pace
    lines.append(
        f"  walk:    time ~{_fmt_time(walk_time_s)}"
        f"  personal_energy ~{walk_epct:.1f}%"
        f"  [available]"
    )

    # ---- E-scooter (when agent has one) ----
    es = getattr(dm, "e_scooter", None)
    if es is not None:
        with_owner = bool(getattr(es, "with_owner", True))
        owner_ok = str(getattr(es, "owner_id", "")) == str(
            getattr(dm, "agent_id", "")
        )
        usable = es.state == ScooterState.USABLE

        sc_spd = float(es.avg_speed_cm_s)
        sc_erate = float(energy_cost.get(TransportMode.SCOOTER, 0.01))
        sc_batt_rate = float(getattr(dm, "scooter_batt_decay_pct_per_m", 0.04))
        sc_m_s = sc_spd / 100.0 * pace
        sc_time_s = dist / sc_m_s if sc_m_s > 0 else float("inf")
        sc_epct = dist * sc_erate * pace
        sc_batt = dist * sc_batt_rate * pace

        if usable and with_owner and owner_ok:
            avail = float(es.battery_pct)
            if sc_batt <= avail + 1e-6:
                status = "[available]"
            else:
                status = (
                    f"[insufficient battery: need {sc_batt:.1f}%,"
                    f" have {avail:.1f}%]"
                )
        elif not with_owner:
            status = "[unavailable: scooter not with you]"
        elif not owner_ok:
            status = "[unavailable: not your scooter]"
        elif es.state == ScooterState.DEPLETED:
            status = "[unavailable: battery depleted]"
        elif es.state == ScooterState.PARKED:
            status = "[unavailable: scooter parked elsewhere]"
        else:
            status = "[unavailable]"

        lines.append(
            f"  scooter: time ~{_fmt_time(sc_time_s)}"
            f"  personal_energy ~{sc_epct:.1f}%"
            f"  battery ~{sc_batt:.1f}%"
            f"  {status}"
        )

    return "\n".join(lines)
