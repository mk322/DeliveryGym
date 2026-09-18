# actions/navigate_bus.py
# -*- coding: utf-8 -*-

"""
Query-only bus-assisted navigation.

NAVIGATE_BUS(target="<waypoint>", access_mode="auto", egress_mode="auto")
returns complete door-to-door itineraries:

    current location
      --(access leg: walk or e-scooter)-->  boarding bus station
      --(bus leg: ride on one canonical route)-->  alighting bus station
      --(egress leg: walk or e-scooter)-->  target

For every feasible (route, direction, boarding-stop, alighting-stop) pair and
every feasible access/egress mode choice it estimates:

    estimated_total_time =  access_time
                          +  wait_time_after_reaching_boarding_stop
                          +  bus_travel_time (dwell at boarding + ride +
                                              intermediate-stop dwells)
                          +  egress_time

The wait for the bus is computed **after** the agent reaches the boarding
stop (``now + access_time``), not from "now". Itineraries are ranked by
``estimated_total_time`` and the top 3 are returned.

Design constraints (query-only):
  - Reads only canonical, non-mutated route geometry from
    ``dm._bus_manager.routes[*]`` and the immutable ``_birth_sim_time``
    captured by ``BusManager.init_bus_system``. Never reads live bus state
    (``bus.x/y/state/current_stop_index/...``) and never calls
    ``bus.update()``.
  - Does not modify ``dm.x``, ``dm.y``, ``dm.clock``, ``dm.energy_pct``,
    ``dm.e_scooter``, ``dm._bus_ctx``, ``dm.mode``, ``dm._move_ctx`` or any
    bus-manager state. Deterministic and stateless per call.

The rendered ``[navigate_bus]`` block is human-readable: access/egress legs
use the same turn-by-turn, segment-merged directions as NAVIGATE_DIRECTIONS,
and no raw waypoint ids (``int_N`` / ``dock_N``) are exposed anywhere.

Approximation note (inherited from VIEW_BUS_OPTIONS): the schedule assumes a
back-and-forth cycle of period ``2 * (sum(inter_ride) + sum(dwells))`` and
ignores the simulator's terminal-reversal quirks (~2% slower per cycle). This
is acceptable for planning.

Scooter "implicit carry" assumption: ``actions/board_bus.py`` leaves the
agent's e-scooter with its owner (it never parks it), so the scooter is
treated as available again at the alighting stop. Combined access+egress
scooter routes therefore check the *combined* battery drain.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from ..base.defs import DMAction, TransportMode
from ..entities.escooter import ScooterState
from ._nav_helpers import (
    resolve_navigation_target as _resolve_navigation_target,
    current_waypoint as _current_waypoint,
    path_distance_m as _path_distance_m,
    semantic_target_label as _semantic_target_label,
)

# Number of itineraries returned, ranked by estimated_total_time.
_TOP_N = 3

# Bus fare. Mirrors the hardcoded value in actions/board_bus.py (~line 70).
_BUS_FARE_USD = 1.0


# ---------------------------------------------------------------------------
# Pure schedule forecaster.
#
# These three helpers (_compute_schedule, _next_arrival_times,
# _travel_time_A_to_B) were originally derived from the now-removed
# VIEW_BUS_OPTIONS helper and are the single live copy of the bus-schedule
# forecast used by NAVIGATE(mode="bus"). If the bus scheduling model changes,
# update them here, or promote them into _nav_helpers.py once the runtime tool
# design stabilizes.
# ---------------------------------------------------------------------------

def _compute_schedule(route: Any) -> Optional[Dict[str, Any]]:
    """
    Compute per-stop, per-direction in-cycle offsets and the cycle period.

    ``t_rel = 0`` marks the bus's forward arrival at ``stop[0]``
    (= bus_birth_time + initial_approach, in absolute sim time).

    Returns ``None`` for degenerate routes (<2 stops, non-positive speed, or
    empty path_points).
    """
    N = len(route.stops)
    speed = float(getattr(route, "speed_cm_s", 0.0))
    pp = getattr(route, "path_points", None) or []
    if N < 2 or speed <= 0.0 or len(pp) < 2:
        return None

    waits = [float(s.wait_time_s) for s in route.stops]
    initial_approach = math.hypot(pp[1][0] - pp[0][0], pp[1][1] - pp[0][1]) / speed

    # Inter-stop ride times (forward direction). Stop[i] lives at path[2i+1].
    inter: List[float] = []
    for i in range(N - 1):
        d = 0.0
        for k in (2 * i + 1, 2 * i + 2):
            if k + 1 < len(pp):
                d += math.hypot(pp[k + 1][0] - pp[k][0], pp[k + 1][1] - pp[k][1])
        inter.append(d / speed)

    arrival_fwd: Dict[int, float] = {0: 0.0}
    for i in range(1, N):
        arrival_fwd[i] = arrival_fwd[i - 1] + waits[i - 1] + inter[i - 1]

    arrival_rev: Dict[int, float] = {N - 1: arrival_fwd[N - 1]}
    t = arrival_fwd[N - 1] + waits[N - 1]
    for i in range(N - 2, -1, -1):
        t += inter[i]
        arrival_rev[i] = t
        if i > 0:
            t += waits[i]

    period = 2.0 * (sum(inter) + sum(waits))

    return {
        "N": N,
        "waits": waits,
        "inter": inter,
        "arrival_fwd": arrival_fwd,
        "arrival_rev": arrival_rev,
        "period": period,
        "initial_approach": initial_approach,
    }


def _next_arrival_times(
    bus_birth: float,
    schedule: Dict[str, Any],
    stop_idx: int,
    direction: int,
    now_sim: float,
    K: int = 2,
) -> List[float]:
    """
    Next ``K`` absolute sim times at which the bus is at ``stop_idx`` available
    to travel in ``direction`` (``+1`` forward, ``-1`` reverse), ``>= now_sim``.
    """
    period = float(schedule["period"])
    if period <= 0.0:
        return []

    if direction > 0:
        in_cycle = schedule["arrival_fwd"].get(stop_idx)
    else:
        in_cycle = schedule["arrival_rev"].get(stop_idx)
    if in_cycle is None:
        return []

    t0 = bus_birth + schedule["initial_approach"]
    k_target = (now_sim - t0 - in_cycle) / period
    k_min = max(0, int(math.ceil(k_target - 1e-9)))

    out: List[float] = []
    for j in range(K):
        k = k_min + j
        t = t0 + in_cycle + k * period
        if t >= now_sim - 1e-6:
            out.append(t)
    return out


def _travel_time_A_to_B(
    schedule: Dict[str, Any], A_idx: int, B_idx: int, direction: int
) -> float:
    """
    Time on the bus from **departing** A to **arriving at** B: inter-stop ride
    times plus intermediate-stop dwells. Excludes dwell at A (added by caller)
    and dwell at B. Returns ``inf`` if B is unreachable from A in ``direction``.
    """
    inter = schedule["inter"]
    waits = schedule["waits"]
    if direction > 0:
        if not (A_idx < B_idx):
            return float("inf")
        ride = sum(inter[A_idx:B_idx])
        dwells = sum(waits[A_idx + 1:B_idx])
        return ride + dwells
    else:
        if not (A_idx > B_idx):
            return float("inf")
        ride = sum(inter[B_idx:A_idx])
        dwells = sum(waits[B_idx + 1:A_idx])
        return ride + dwells


# ---------------------------------------------------------------------------
# Formatting helpers (no raw waypoint ids in agent-facing output)
# ---------------------------------------------------------------------------

def _human_name(node: Any) -> str:
    """Human-readable name only — never the raw ``int_N`` / ``dock_N`` id."""
    return (
        getattr(node, "waypoint_name", "")
        or getattr(node, "waypoint_id", "")
        or str(node)
    )


def _sanitize_steps(steps: List[str], path: List[Any]) -> List[str]:
    """
    Replace every ``"<id> (<name>)"`` waypoint label produced by
    ``_directions_from_path`` (via ``waypoint_label``) with just ``<name>``,
    so no raw ``int_N`` / ``dock_N`` id reaches the agent-facing output.

    Uses exact literal replacement keyed on the actual path nodes rather than a
    regex; this keeps the substitution robust against any punctuation that
    intersection names might contain. Replacement is done longest-label-first
    so ``"id (name)"`` is consumed before any bare ``id``.

      "At intersection int_5 (Maple St & Oak Ave), turn left"
        -> "At intersection Maple St & Oak Ave, turn left"
      "Arrive at dock_90 (24 Pine Ave)."
        -> "Arrive at 24 Pine Ave."
    """
    repls: List[Tuple[str, str]] = []
    for node in path:
        wp_id = getattr(node, "waypoint_id", "") or ""
        wp_name = getattr(node, "waypoint_name", "") or ""
        if wp_id and wp_name:
            repls.append((f"{wp_id} ({wp_name})", wp_name))
            repls.append((wp_id, wp_name))  # fallback for any bare-id mention
    # Longest patterns first so "id (name)" wins over "id".
    repls.sort(key=lambda kv: len(kv[0]), reverse=True)

    out: List[str] = []
    for s in steps:
        for old, new in repls:
            if old in s:
                s = s.replace(old, new)
        out.append(s)
    return out


def _fmt_clock(t_s: float) -> str:
    """Render an absolute sim time (seconds since episode start) as H:MM:SS."""
    t = int(max(0.0, float(t_s)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_dur(t_s: float) -> str:
    """Render a non-negative duration as ``Xm Ys`` / ``Ys`` / ``Xh Ym``."""
    t = int(max(0.0, round(float(t_s))))
    if t >= 3600:
        h, rem = divmod(t, 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h {m}m"
    if t >= 60:
        m, s = divmod(t, 60)
        return f"{m}m {s}s"
    return f"{t}s"


# ---------------------------------------------------------------------------
# Mode feasibility / timing (numeric; mirrors mode_estimates_text formulas)
# ---------------------------------------------------------------------------

def _scooter_available(dm: Any) -> bool:
    """True iff the agent's own e-scooter is usable and with the owner."""
    es = getattr(dm, "e_scooter", None)
    if es is None:
        return False
    try:
        return (
            es.state == ScooterState.USABLE
            and bool(getattr(es, "with_owner", True))
            and str(getattr(es, "owner_id", "")) == str(getattr(dm, "agent_id", ""))
        )
    except Exception:
        return False


def _pace(dm: Any) -> float:
    try:
        return float(getattr(dm, "pace_scales", {}).get(
            getattr(dm, "pace_state", "normal"), 1.0
        ))
    except Exception:
        return 1.0


def _walk_speed_cm_s(dm: Any) -> float:
    avg = getattr(dm, "avg_speed_by_mode", None) or {}
    return float(avg.get(TransportMode.WALK, 200.0))


def _scooter_speed_cm_s(dm: Any) -> float:
    es = getattr(dm, "e_scooter", None)
    if es is None:
        return 0.0
    return float(getattr(es, "avg_speed_cm_s", 0.0))


def _leg_time_s(dm: Any, dist_m: float, speed_cm_s: float) -> float:
    """Travel time in seconds for ``dist_m`` metres at ``speed_cm_s`` (pace-scaled)."""
    pace = _pace(dm)
    denom = speed_cm_s * pace
    if denom <= 0:
        return float("inf")
    return float(dist_m) * 100.0 / denom


def _scooter_batt_for(dm: Any, dist_m: float) -> float:
    """Battery percent consumed by an e-scooter leg of ``dist_m`` metres."""
    rate = float(getattr(dm, "scooter_batt_decay_pct_per_m", 0.04))
    return float(dist_m) * rate * _pace(dm)


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------

def _build_block(
    *,
    from_label: str,
    to_label: str,
    now_sim: float,
    routes: List[Dict[str, Any]],
    message: str,
) -> str:
    lines = [
        f"From: {from_label}",
        f"To  : {to_label}",
        f"Now : {_fmt_clock(now_sim)}",
    ]

    for r in routes:
        lines.append("")
        lines.append(
            f"Route {r['rank']}  (estimated {_fmt_dur(r['estimated_total_time_s'])} total)"
        )

        a = r["access_leg"]
        lines.append("")
        lines.append(
            f"  Access leg  ({a['mode']}, {a['distance_m']:.0f}m, "
            f"{_fmt_dur(a['estimated_time_s'])})"
        )
        for s in a["steps"]:
            lines.append(f"    {s}")

        b = r["bus_leg"]
        lines.append("")
        lines.append("  Bus leg")
        lines.append(f"    Route   : {b['route_id']} ({b['direction']})")
        lines.append(f"    Board   : {b['board_at']}")
        lines.append(f"    Alight  : {b['alight_at']}")
        lines.append(
            f"    Next bus at boarding  : {_fmt_clock(b['next_bus_at_boarding_s'])}"
            f"  (wait {_fmt_dur(b['wait_time_after_reaching_boarding_s'])}"
            f" after arriving at stop)"
        )
        if b.get("second_next_bus_at_boarding_s") is not None:
            lines.append(
                f"    2nd next bus          : "
                f"{_fmt_clock(b['second_next_bus_at_boarding_s'])}"
            )
        else:
            lines.append("    2nd next bus          : (none)")
        lines.append(
            f"    Est. time to alighting: "
            f"{_fmt_dur(b['estimated_time_to_alighting_stop_s'])}"
            f"  (includes dwell + ride + intermediate stops)"
        )
        lines.append(f"    Fare    : ${b['fare_usd']:.2f}")

        e = r["egress_leg"]
        lines.append("")
        lines.append(
            f"  Egress leg  ({e['mode']}, {e['distance_m']:.0f}m, "
            f"{_fmt_dur(e['estimated_time_s'])})"
        )
        for s in e["steps"]:
            lines.append(f"    {s}")

    lines.append("")
    lines.append(f"Message: {message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Computation (shared by the text tool and VISUAL_NAVIGATE_BUS)
# ---------------------------------------------------------------------------

def compute_bus_routes(
    dm: Any,
    target_token: str,
    access_mode: str = "auto",
    egress_mode: str = "auto",
    top_n: int = _TOP_N,
) -> Dict[str, Any]:
    """
    Pure, query-only computation of ranked bus-assisted itineraries.

    Performs target/current resolution, schedule forecasting, access/egress
    pathing, feasibility filtering, and ranking — but does NOT render text or
    images and does NOT touch the observation channel. Both the text tool
    (NAVIGATE_BUS) and the visual tool (VISUAL_NAVIGATE_BUS) consume the result.

    Returns a dict:
      - ``status``       : "error" | "empty" | "ok"
      - ``error_suffix`` : str (when status == "error"); the caller prepends its
                           own action name.
      - ``message``      : str (graceful message for "empty"; default summary for "ok")
      - ``from_label`` / ``to_label`` : human-readable names (no raw ids), or None
      - ``now_sim``      : float
      - ``top``          : list of ranked itinerary option dicts (<= top_n)
      - ``access_path`` / ``egress_path`` : {stop_node -> [waypoint nodes]}
      - ``idx_to_node``  : {route_id -> {stop_index -> stop_node}} (for bus-leg geometry)
    """
    result: Dict[str, Any] = {
        "status": "empty", "error_suffix": None, "message": "",
        "from_label": None, "to_label": None, "now_sim": 0.0,
        "top": [], "access_path": {}, "egress_path": {}, "idx_to_node": {},
    }

    city_map = getattr(dm, "city_map", None)
    if city_map is None or not hasattr(city_map, "waypoint_graph"):
        result["status"] = "error"
        result["error_suffix"] = "map has no waypoint graph."
        return result

    access_mode = str(access_mode or "auto").lower()
    egress_mode = str(egress_mode or "auto").lower()
    if access_mode not in ("auto", "walk", "scooter"):
        access_mode = "auto"
    if egress_mode not in ("auto", "walk", "scooter"):
        egress_mode = "auto"

    target_node = _resolve_navigation_target(dm, target_token)
    if target_node is None:
        result["status"] = "error"
        result["error_suffix"] = f"cannot resolve target {target_token!r} to a waypoint."
        return result

    cur_node = _current_waypoint(dm)
    if cur_node is None:
        result["status"] = "error"
        result["error_suffix"] = (
            "cannot determine current waypoint (no nearest waypoint found)."
        )
        return result

    result["from_label"] = _human_name(cur_node)
    result["to_label"] = _human_name(target_node)
    sem = _semantic_target_label(dm, target_node)
    if sem and sem.lower() != result["to_label"].lower():
        result["to_label"] = f"{result['to_label']} ({sem})"
    now_sim = float(dm.clock.now_sim())
    result["now_sim"] = now_sim

    bm = getattr(dm, "_bus_manager", None)
    if bm is None or not getattr(bm, "routes", None):
        result["message"] = (
            "bus system unavailable." if bm is None else "no bus routes defined."
        )
        return result

    birth_time = getattr(bm, "_birth_sim_time", None)
    if birth_time is None:
        result["message"] = "bus schedule not initialized."
        return result

    if cur_node is target_node:
        result["message"] = "current location matches target; no bus option needed."
        return result

    graph = city_map.waypoint_graph

    # ----- Step 1: per-route schedules + node->stop_index maps -----
    route_data: List[Tuple[str, Dict[str, Any], Dict[Any, int], Dict[int, str]]] = []
    candidate_nodes: set = set()
    idx_to_node: Dict[str, Dict[int, Any]] = {}
    for route_id, route in bm.routes.items():
        sched = _compute_schedule(route)
        if sched is None:
            continue
        node_to_idx: Dict[Any, int] = {}
        idx_to_name: Dict[int, str] = {}
        i2n: Dict[int, Any] = {}
        for i, s in enumerate(route.stops):
            node = (
                city_map.resolve_waypoint(s.name)
                if hasattr(city_map, "resolve_waypoint")
                else None
            )
            idx_to_name[i] = s.name
            if node is not None:
                node_to_idx[node] = i
                i2n[i] = node
                candidate_nodes.add(node)
        route_data.append((route_id, sched, node_to_idx, idx_to_name))
        idx_to_node[route_id] = i2n

    result["idx_to_node"] = idx_to_node

    if not route_data or not candidate_nodes:
        result["message"] = "no usable bus routes."
        return result

    # ----- Step 2: point-to-point access/egress paths, cached per stop node -----
    def _on_graph(n: Any) -> bool:
        try:
            return n in graph.adjacency_list
        except Exception:
            return False

    access_path: Dict[Any, List[Any]] = {}
    access_dist_m: Dict[Any, float] = {}
    egress_path: Dict[Any, List[Any]] = {}
    egress_dist_m: Dict[Any, float] = {}

    cur_on_graph = _on_graph(cur_node)
    tgt_on_graph = _on_graph(target_node)

    for S in candidate_nodes:
        if not _on_graph(S):
            continue
        if cur_on_graph:
            p, _d = graph.shortest_path_nodes(cur_node, S)
            if p:
                access_path[S] = p
                access_dist_m[S] = _path_distance_m(city_map, p)
        if tgt_on_graph:
            p, _d = graph.shortest_path_nodes(S, target_node)
            if p:
                egress_path[S] = p
                egress_dist_m[S] = _path_distance_m(city_map, p)

    result["access_path"] = access_path
    result["egress_path"] = egress_path

    # ----- Step 3: feasible per-leg modes -----
    scooter_ok = _scooter_available(dm)
    batt_pct = float(getattr(getattr(dm, "e_scooter", None), "battery_pct", 0.0)) \
        if scooter_ok else 0.0

    def _modes_for(choice: str) -> List[str]:
        if choice == "walk":
            return ["walk"]
        if choice == "scooter":
            return ["e-scooter"] if scooter_ok else []
        # auto
        return ["walk"] + (["e-scooter"] if scooter_ok else [])

    access_modes = _modes_for(access_mode)
    egress_modes = _modes_for(egress_mode)

    walk_spd = _walk_speed_cm_s(dm)
    sc_spd = _scooter_speed_cm_s(dm)

    def _leg(mode: str, dist_m: float) -> Tuple[float, float]:
        """Return (time_s, scooter_batt_pct) for a leg in the given mode."""
        if mode == "e-scooter":
            return _leg_time_s(dm, dist_m, sc_spd), _scooter_batt_for(dm, dist_m)
        return _leg_time_s(dm, dist_m, walk_spd), 0.0

    # ----- Step 4: enumerate (route, A, B, dir) x (access_mode, egress_mode) -----
    options: List[Dict[str, Any]] = []
    seen: set = set()

    for route_id, sched, node_to_idx, idx_to_name in route_data:
        boardable = [n for n in node_to_idx if n in access_path]
        alightable = [n for n in node_to_idx if n in egress_path]
        for A_node in boardable:
            A_idx = node_to_idx[A_node]
            a_dist = access_dist_m[A_node]
            for B_node in alightable:
                B_idx = node_to_idx[B_node]
                if A_idx == B_idx:
                    continue
                direction = 1 if A_idx < B_idx else -1
                bus_travel = _travel_time_A_to_B(sched, A_idx, B_idx, direction)
                if bus_travel == float("inf"):
                    continue
                e_dist = egress_dist_m[B_node]

                for am in access_modes:
                    a_time, a_batt = _leg(am, a_dist)
                    if not math.isfinite(a_time):
                        continue
                    if am == "e-scooter" and a_batt > batt_pct + 1e-6:
                        continue
                    for em in egress_modes:
                        e_time, e_batt = _leg(em, e_dist)
                        if not math.isfinite(e_time):
                            continue
                        if em == "e-scooter" and e_batt > batt_pct + 1e-6:
                            continue
                        # Combined scooter battery (implicit carry through bus).
                        if am == "e-scooter" and em == "e-scooter":
                            if a_batt + e_batt > batt_pct + 1e-6:
                                continue

                        # A zero-distance leg makes walk/scooter indistinguishable;
                        # normalize its mode so the two collapse into one itinerary
                        # instead of wasting a top-N slot on a meaningless label.
                        am_eff = am if a_dist > 1e-6 else "walk"
                        em_eff = em if e_dist > 1e-6 else "walk"
                        key = (route_id, direction, A_idx, B_idx, am_eff, em_eff)
                        if key in seen:
                            continue
                        seen.add(key)

                        arrival_at_boarding = now_sim + a_time
                        arrivals = _next_arrival_times(
                            birth_time, sched, A_idx, direction,
                            arrival_at_boarding, K=2,
                        )
                        if not arrivals:
                            continue
                        next_bus = arrivals[0]
                        second_next = arrivals[1] if len(arrivals) > 1 else None
                        wait_time = next_bus - arrival_at_boarding
                        # Time from the bus arriving at the boarding stop to it
                        # arriving at the alighting stop: dwell at boarding +
                        # ride + intermediate-stop dwells (mirrors the
                        # arrival_at_B computation in VIEW_BUS_OPTIONS).
                        dwell_A = float(sched["waits"][A_idx])
                        eta_to_alight = dwell_A + bus_travel
                        total = a_time + wait_time + eta_to_alight + e_time

                        options.append({
                            "route_id": route_id,
                            "direction": direction,
                            "A_node": A_node, "A_idx": A_idx,
                            "B_node": B_node, "B_idx": B_idx,
                            "board_at": idx_to_name[A_idx],
                            "alight_at": idx_to_name[B_idx],
                            "access_mode": am_eff, "access_dist_m": a_dist,
                            "access_time_s": a_time,
                            "egress_mode": em_eff, "egress_dist_m": e_dist,
                            "egress_time_s": e_time,
                            "next_bus_at_boarding_s": next_bus,
                            "second_next_bus_at_boarding_s": second_next,
                            "wait_time_after_reaching_boarding_s": wait_time,
                            "bus_travel_time_s": bus_travel,
                            "estimated_time_to_alighting_stop_s": eta_to_alight,
                            "estimated_total_time_s": total,
                        })

    # ----- Step 5: collapse to distinct itineraries, then rank, keep top N -----
    # A "distinct itinerary" is one transit trip: (route_id, direction, board,
    # alight). Access/egress mode is a free per-leg choice on top of it, so all
    # mode combos for the same trip are collapsed to a single entry — the best
    # one by estimated_total_time. Deterministic tie-break: prefer walk for
    # access (don't spend scooter battery when it doesn't help) and e-scooter
    # for egress (faster egress strictly lowers total, so it only ties when the
    # egress distance is zero). This stops near-identical mode variants of one
    # trip from crowding out genuinely different trips in the top 3.
    _ACCESS_PREF = {"walk": 0, "e-scooter": 1}
    _EGRESS_PREF = {"e-scooter": 0, "walk": 1}

    def _select_key(o: Dict[str, Any]):
        return (
            o["estimated_total_time_s"],
            _ACCESS_PREF.get(o["access_mode"], 9),
            _EGRESS_PREF.get(o["egress_mode"], 9),
        )

    best_by_itinerary: Dict[Tuple[Any, int, int, int], Dict[str, Any]] = {}
    for o in options:
        itin = (o["route_id"], o["direction"], o["A_idx"], o["B_idx"])
        cur = best_by_itinerary.get(itin)
        if cur is None or _select_key(o) < _select_key(cur):
            best_by_itinerary[itin] = o

    # Rank distinct itineraries by total time; deterministic order on ties.
    distinct = list(best_by_itinerary.values())
    distinct.sort(key=lambda o: (
        o["estimated_total_time_s"],
        str(o["route_id"]), o["direction"], o["A_idx"], o["B_idx"],
    ))
    top = distinct[:top_n]

    if not top:
        result["message"] = (
            "no viable bus itinerary connects your location to the target "
            "(no boardable/alightable stop pair on a common route in a "
            "reachable direction)."
        )
        return result

    result["status"] = "ok"
    result["top"] = top
    n = len(top)
    result["message"] = f"{n} distinct bus-assisted route{'' if n == 1 else 's'} found."
    return result


def _bus_leg_nodes(res: Dict[str, Any], opt: Dict[str, Any]) -> List[Any]:
    """Ordered stop nodes from boarding to alighting along the chosen route.

    Used by the unified NAVIGATE tool to draw the bus leg of the map overlay.
    """
    i2n = res["idx_to_node"].get(opt["route_id"], {})
    a, b = opt["A_idx"], opt["B_idx"]
    step = 1 if b > a else -1
    out: List[Any] = []
    for i in range(a, b + step, step):
        n = i2n.get(i)
        if n is not None:
            out.append(n)
    return out
