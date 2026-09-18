# actions/navigate.py
# -*- coding: utf-8 -*-

"""
Unified query-only navigation tool.

NAVIGATE(target="<waypoint>", mode="walk"|"e-scooter"|"bus")

A single tool: it reports the **cost estimates** for the route to ``target`` —
travel time, personal energy, e-scooter battery, and (for bus) wait + fare — and
stores the route so it is drawn on the **per-step city map** (a persistent layer
*under* the agent dot, with source/destination pins). It deliberately does NOT
spell out a turn-by-turn / waypoint-id chain: the agent must still figure out
*which way to go* from its first-person views and the drawn route. Estimates are
the point — compare options and check feasibility against deadlines/energy.

  - mode="walk"      : shortest walking path on the waypoint graph (blue).
  - mode="e-scooter" : same path priced for the scooter (purple) — requires battery.
  - mode="bus"       : bus-assisted itinerary (access/bus/egress legs) — requires
                       advanced transport. Optional access_mode/egress_mode =
                       "auto"|"walk"|"scooter".

Query-only: never modifies position, clock, energy, scooter, mode, orders,
movement context, or bus state. It writes the ``[navigation]`` estimate text and
sets ``dm._nav_route_path`` / ``_nav_route_color`` / ``_nav_target_node``. That
route **persists on every later map frame** until another NAVIGATE overwrites it
or the agent reaches ``_nav_target_node`` (MOVE clears it on arrival).
"""

from typing import Any

from ..base.defs import DMAction, TransportMode
from ._nav_helpers import (
    resolve_navigation_target as _resolve_navigation_target,
    current_waypoint as _current_waypoint,
    path_distance_m as _path_distance_m,
    semantic_target_label as _semantic_target_label,
    _fmt_time,
)
from .navigate_bus import (
    compute_bus_routes as _compute_bus_routes,
    _bus_leg_nodes,
    _BUS_FARE_USD,
)
from ._visual_helpers import (
    human_name as _human_name,
    WALK_COLOR as _WALK_COLOR,
    ESCOOTER_COLOR as _ESCOOTER_COLOR,
    BUS_COLOR as _BUS_COLOR,
)

# Accept common spellings for the transport mode.
_MODE_ALIASES = {
    "walk": "walk", "foot": "walk", "on_foot": "walk", "walking": "walk",
    "e-scooter": "e-scooter", "escooter": "e-scooter", "e_scooter": "e-scooter",
    "scooter": "e-scooter",
    "bus": "bus", "transit": "bus",
}


def handle_navigate(dm: Any, act: DMAction, _allow_interrupt: bool) -> None:
    """Dispatch the unified NAVIGATE tool by ``mode``."""
    previous_ephemeral = dict(getattr(dm, "vlm_ephemeral", {}) or {})
    dm.vlm_clear_ephemeral()

    target_token = act.data.get("target") or act.data.get("to")
    if not target_token or not isinstance(target_token, str):
        dm.vlm_add_error(
            'NAVIGATE: missing target. Usage: '
            'NAVIGATE(target="<waypoint>", mode="walk"|"e-scooter"|"bus").'
        )
        dm._finish_action(success=False)
        return

    raw_mode = str(act.data.get("mode", "walk")).strip().lower()
    mode = _MODE_ALIASES.get(raw_mode)
    if mode is None:
        dm.vlm_add_error(
            f'NAVIGATE: unknown mode {raw_mode!r}. Use "walk", "e-scooter", or "bus".'
        )
        dm._finish_action(success=False)
        return

    cfg = getattr(dm, "cfg", {}) or {}
    if mode == "e-scooter" and not cfg.get("enable_battery", True):
        dm.vlm_add_error('NAVIGATE: mode="e-scooter" is unavailable in this stage.')
        dm._finish_action(success=False)
        return
    if mode == "bus" and not cfg.get("enable_advanced_transport", True):
        dm.vlm_add_error('NAVIGATE: mode="bus" is unavailable in this stage.')
        dm._finish_action(success=False)
        return

    if mode == "bus":
        _navigate_bus(dm, act, target_token)
    else:
        _navigate_ground(dm, target_token, mode, previous_ephemeral=previous_ephemeral)


def _navigate_ground(dm: Any, target_token: str, mode: str, *, previous_ephemeral=None) -> None:
    """Walk / e-scooter route: map overlay + time/energy/battery estimate."""
    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "waypoint_graph"):
        dm.vlm_add_error("NAVIGATE: map has no waypoint graph.")
        dm._finish_action(success=False)
        return

    target_node = _resolve_navigation_target(dm, target_token)
    if target_node is None:
        dm.vlm_add_error(f"NAVIGATE: cannot resolve target {target_token!r} to a waypoint.")
        dm._finish_action(success=False)
        return

    if (
        getattr(dm, "_nav_target_node", None) is target_node
        and getattr(dm, "_nav_route_path", None)
        and (previous_ephemeral or {}).get("navigation")
    ):
        dm.vlm_ephemeral.update(previous_ephemeral or {})
        active_msg = (
            "message: route already active; continue with MOVE(direction=...) "
            "using the highlighted route."
        )
        nav_lines = [
            ln for ln in str(previous_ephemeral["navigation"]).splitlines()
            if not ln.strip().startswith("message: route already active")
        ]
        nav_lines.append(active_msg)
        dm.vlm_add_ephemeral("navigation", "\n".join(nav_lines))
        dm._finish_action(success=True)
        return

    start_node = _current_waypoint(dm)
    if start_node is None:
        dm.vlm_add_error("NAVIGATE: cannot determine current waypoint.")
        dm._finish_action(success=False)
        return

    from_label = _human_name(start_node)
    to_label = _human_name(target_node)
    sem = _semantic_target_label(dm, target_node)
    if sem and sem.lower() != to_label.lower():
        to_label = f"{to_label} ({sem})"
    color = _WALK_COLOR if mode == "walk" else _ESCOOTER_COLOR

    avg_speed = getattr(dm, "avg_speed_by_mode", None) or {}
    energy_cost = getattr(dm, "energy_cost_by_mode", None) or {}
    pace = float(getattr(dm, "pace_scales", {}).get(getattr(dm, "pace_state", "normal"), 1.0))
    tmode = TransportMode.WALK if mode == "walk" else TransportMode.SCOOTER
    spd = float(avg_speed.get(tmode, 200.0 if mode == "walk" else 600.0))
    erate = float(energy_cost.get(tmode, 0.08 if mode == "walk" else 0.01))
    m_s = spd / 100.0 * pace

    def _emit(distance_m: float, message: str, *, with_estimates: bool, success: bool = True) -> None:
        lines = [f"mode: {mode}", f"from: {from_label}", f"to: {to_label}",
                 f"distance_m: {distance_m:.1f}"]
        if with_estimates and m_s > 0 and distance_m > 0:
            lines.append(f"estimated_time: ~{_fmt_time(distance_m / m_s)}")
            lines.append(f"estimated_personal_energy: ~{distance_m * erate * pace:.1f}%")
            if mode == "e-scooter":
                batt_rate = float(getattr(dm, "scooter_batt_decay_pct_per_m", 0.04))
                lines.append(f"estimated_battery: ~{distance_m * batt_rate * pace:.1f}%")
        if message:
            lines.append(f"message: {message}")
        # NOTE: the live "next_move:" line is appended by the observation builder
        # each step (it recomputes from the agent's current waypoint + facing),
        # so it is intentionally NOT baked into this persisted block.
        dm.vlm_add_ephemeral("navigation", "\n".join(lines))
        dm._finish_action(success=success)

    if start_node is target_node:
        _emit(0.0, "you are already at the target.", with_estimates=False)
        return

    graph = city_map.waypoint_graph
    if start_node not in graph.adjacency_list or target_node not in graph.adjacency_list:
        dm.vlm_add_error("NAVIGATE: endpoint is not on the waypoint graph.")
        dm._finish_action(success=False)
        return

    path, _ = graph.shortest_path_nodes(start_node, target_node)
    if not path:
        _emit(0.0, f"no {mode} route found from {from_label} to {to_label}.",
              with_estimates=False, success=False)
        return

    distance_m = _path_distance_m(city_map, path)
    # Persist the route + endpoints so the per-step map frame draws them UNDER the
    # agent dot on every later frame, until another NAVIGATE overwrites it or the
    # agent reaches the target (MOVE clears it on arrival at _nav_target_node).
    dm._nav_route_path = list(path)
    dm._nav_route_color = color
    dm._nav_target_node = target_node

    _emit(distance_m, "", with_estimates=True)


def _navigate_bus(dm: Any, act: DMAction, target_token: str) -> None:
    """Bus-assisted itinerary: top route map overlay + cost/time/fare estimate."""
    res = _compute_bus_routes(
        dm, target_token,
        access_mode=act.data.get("access_mode", "auto"),
        egress_mode=act.data.get("egress_mode", "auto"),
        top_n=1,
    )

    if res["status"] == "error":
        dm.vlm_add_error(f"NAVIGATE: {res['error_suffix']}")
        dm._finish_action(success=False)
        return

    from_label = res["from_label"] or "?"
    to_label = res["to_label"] or "?"

    if res["status"] != "ok" or not res["top"]:
        dm.vlm_add_ephemeral(
            "navigation",
            f"mode: bus\nfrom: {from_label}\nto: {to_label}\nmessage: {res['message']}",
        )
        dm._finish_action(success=True)
        return

    opt = res["top"][0]
    access_nodes = res["access_path"].get(opt["A_node"], [])
    egress_nodes = res["egress_path"].get(opt["B_node"], [])
    bus_nodes = _bus_leg_nodes(res, opt)

    # Persist the full access+bus+egress chain so the per-step map frame draws it
    # under the agent dot until another NAVIGATE or arrival at the final node.
    dm._nav_route_path = list(access_nodes) + list(bus_nodes) + list(egress_nodes)
    dm._nav_route_color = _BUS_COLOR
    dm._nav_target_node = (egress_nodes[-1] if egress_nodes else opt["B_node"])

    msg = f"top bus-assisted route to {to_label} drawn on the map."
    lines = [
        "mode: bus",
        f"from: {from_label}",
        f"to: {to_label}",
        f"total_estimated_time: ~{_fmt_time(opt['estimated_total_time_s'])}",
        f"access: {opt['access_mode']}, {opt['access_dist_m']:.0f}m, ~{_fmt_time(opt['access_time_s'])}",
        f"board_stop: {opt['board_at']}",
        f"alight_stop: {opt['alight_at']}",
        f"wait_time: ~{_fmt_time(opt['wait_time_after_reaching_boarding_s'])}",
        f"bus_ride_time: ~{_fmt_time(opt['bus_travel_time_s'])}",
        f"egress: {opt['egress_mode']}, {opt['egress_dist_m']:.0f}m, ~{_fmt_time(opt['egress_time_s'])}",
        f"fare: ${_BUS_FARE_USD:.2f}",
        f"message: {msg}",
    ]
    dm.vlm_add_ephemeral("navigation", "\n".join(lines))
    dm._finish_action(success=True)
