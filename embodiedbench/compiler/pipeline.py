"""General map -> training-env pipeline. Rule-based, automated, no AI.

Given *any* DeliveryBench map directory, this produces a training environment or
explains why it cannot. Nothing here calls a model: every decision is a
threshold on a measured graph property, so the same map always yields the same
verdict and a human can check the arithmetic.

The pipeline exists because of what Paris taught us. Paris loaded fine and then
every episode failed at the first step, because ``MOVE(direction="forward")``
assumes a cardinal street grid and only 17.4% of Paris segments are near-cardinal
(design plan §3.2). The bug was not in Paris. The bug was assuming an action
abstraction instead of deriving it from the map.

So the central rule is:

    the action abstraction is a property of the map, not a constant

``analyze`` measures the graph, ``decide_navigation`` picks the action space from
those measurements, ``validate`` gates on solvability, and ``compile_map`` runs
all three and emits a config plus a certificate.

Stages
------
1. **load**      open the map through the engine and read its own graph
2. **analyze**   degree, edge length, cardinal alignment, connectivity, docks
3. **decide**    choose the navigation mode from thresholds
4. **validate**  scripted oracle deliveries must actually complete
5. **certify**   emit env config, WorldBundle, and a grade with its evidence
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds. Every one is a declared, reviewable constant -- not a tuned magic
# number and not a model's opinion.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Thresholds:
    """Rule-based decision boundaries for the pipeline."""

    # An edge shorter than this is degenerate. design plan §3.3.1 names Paris
    # segments 186/187, both under 5 mm, as things to filter before normalizing.
    min_edge_m: float = 0.5
    # Cardinal tolerance from design plan §3.2's own measurement (within 6 degrees).
    cardinal_tolerance_deg: float = 6.0
    # Below this fraction of near-cardinal edges, directional MOVE cannot be the
    # primary action: Paris measures 0.174 and fails at the first step.
    cardinal_fraction_for_move: float = 0.60
    # A road-network junction with more neighbours than this is implausible and
    # usually indicates endpoints merged that should not have been.
    max_plausible_degree: int = 6
    # An edge needs external validation when it is long *for its own map*.
    # An absolute cut cannot work: procgen maps legitimately reach 100-141 m
    # because they contain long straight roads, so a fixed 100 m flagged all ten
    # real maps and said nothing. Measured across the ten, procgen maps have a
    # max/median edge ratio of 3.0-4.6x while Paris reaches 25.6x, so a multiple
    # of the map's own median separates "long road" from "implausible link".
    # The absolute floor stops the rule over-firing on a map whose median edge
    # is tiny.
    long_edge_p50_factor: float = 6.0
    long_edge_floor_m: float = 100.0
    # The certified region must be one dominant component, not a scatter.
    min_largest_component_fraction: float = 0.90
    # A point action's reach, as a multiple of the map's own median edge, so one
    # point step and one waypoint step cover comparable ground on any map. An
    # absolute cap cannot do that: 18 m is about one Paris edge but several
    # blocks on a dense procgen map. The band stops a pathological median from
    # producing a reach of centimetres or of kilometres.
    point_range_p50_factor: float = 1.0
    point_range_min_m: float = 5.0
    point_range_max_m: float = 40.0
    # Node the graph before analysing it. Measured on Paris, 37.2% of edges ran
    # straight past a junction without stopping there, letting an agent skip
    # intersections it never visited. Repair is on by default because an
    # un-noded graph is not a description of the map.
    repair_graph: bool = True
    # Scripted oracle deliveries that must succeed for the map to be usable.
    solvability_seeds: int = 5
    min_solvability_rate: float = 0.8


THRESHOLDS = Thresholds()


# ─────────────────────────────────────────────────────────────────────────────
# Measurements
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GraphAnalysis:
    """Everything the decision rules are allowed to look at."""

    node_count: int = 0
    edge_count: int = 0
    mean_degree: float = 0.0
    degree_histogram: dict[int, int] = field(default_factory=dict)
    over_connected_nodes: int = 0
    dock_nodes: int = 0
    junction_nodes: int = 0
    edge_length_m: dict[str, float] = field(default_factory=dict)
    degenerate_edges: int = 0
    long_edges: int = 0
    cardinal_fraction: float = 0.0
    largest_component_fraction: float = 0.0
    component_count: int = 0
    long_edge_threshold_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _long_edge_threshold(lengths: list[float], thresholds: Thresholds) -> float:
    """The per-map length above which an edge is treated as implausible."""
    if not lengths:
        return thresholds.long_edge_floor_m
    ordered = sorted(lengths)
    median = ordered[len(ordered) // 2]
    return max(thresholds.long_edge_floor_m, thresholds.long_edge_p50_factor * median)


def analyze_graph(city_map: Any, thresholds: Thresholds = THRESHOLDS) -> GraphAnalysis:
    """Measure the graph. Pure computation, no decisions."""
    graph = city_map.waypoint_graph
    adjacency = getattr(graph, "adjacency_list", {}) or {}
    nodes = list(adjacency)
    index_of = {id(node): position for position, node in enumerate(nodes)}

    degrees = [len(adjacency.get(node) or []) for node in nodes]
    kinds = Counter(
        str(getattr(node, "type", None) or getattr(node, "waypoint_kind", None)) for node in nodes
    )

    lengths: list[float] = []
    cardinal_hits = 0
    seen_pairs: set[tuple[int, int]] = set()
    for node in nodes:
        for neighbour in adjacency.get(node) or []:
            a, b = index_of.get(id(node)), index_of.get(id(neighbour))
            if a is None or b is None or a == b:
                continue
            pair = (min(a, b), max(a, b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            dx = float(neighbour.position.x) - float(node.position.x)
            dy = float(neighbour.position.y) - float(node.position.y)
            length_m = math.hypot(dx, dy) / 100.0
            lengths.append(length_m)
            # Angle to the nearest cardinal axis.
            bearing = math.degrees(math.atan2(dy, dx)) % 90.0
            if min(bearing, 90.0 - bearing) <= thresholds.cardinal_tolerance_deg:
                cardinal_hits += 1

    lengths.sort()

    def percentile(fraction: float) -> float:
        if not lengths:
            return 0.0
        return lengths[min(len(lengths) - 1, int(len(lengths) * fraction))]

    # Connectivity over the undirected graph.
    largest = 0
    components = 0
    unvisited = set(range(len(nodes)))
    neighbours_by_index: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for node in nodes:
        a = index_of[id(node)]
        for neighbour in adjacency.get(node) or []:
            b = index_of.get(id(neighbour))
            if b is not None:
                neighbours_by_index[a].append(b)
                neighbours_by_index[b].append(a)
    while unvisited:
        start = unvisited.pop()
        size = 1
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in neighbours_by_index[current]:
                if neighbour in unvisited:
                    unvisited.discard(neighbour)
                    size += 1
                    queue.append(neighbour)
        components += 1
        largest = max(largest, size)

    return GraphAnalysis(
        node_count=len(nodes),
        edge_count=len(seen_pairs),
        mean_degree=round(statistics.mean(degrees), 3) if degrees else 0.0,
        degree_histogram=dict(sorted(Counter(degrees).items())),
        over_connected_nodes=sum(1 for d in degrees if d > thresholds.max_plausible_degree),
        dock_nodes=kinds.get("dock", 0),
        junction_nodes=kinds.get("intersection", 0),
        edge_length_m={
            "min": round(lengths[0], 3) if lengths else 0.0,
            "p50": round(percentile(0.50), 2),
            "p90": round(percentile(0.90), 2),
            "p99": round(percentile(0.99), 2),
            "max": round(lengths[-1], 2) if lengths else 0.0,
        },
        degenerate_edges=sum(1 for length in lengths if length < thresholds.min_edge_m),
        long_edges=sum(1 for length in lengths if length > _long_edge_threshold(lengths, thresholds)),
        long_edge_threshold_m=round(_long_edge_threshold(lengths, thresholds), 2),
        cardinal_fraction=round(cardinal_hits / len(lengths), 4) if lengths else 0.0,
        largest_component_fraction=round(largest / len(nodes), 4) if nodes else 0.0,
        component_count=components,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decisions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NavigationDecision:
    """Which action abstraction this map supports, and why."""

    mode: str  # "graph" | "cardinal"
    enabled_actions: list[str]
    enable_waypoint_marks: bool
    rationale: str
    cardinal_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class PointNavDecision:
    """Whether this map can host the two point-navigation modes, and how far.

    design plan §9 wants all three modes, with ``nav_waypoint`` first. The other two
    are not a property of the *map* so much as of the runtime serving it, and
    the honest decomposition is worth stating rather than collapsing:

    definable   both point modes need only camera intrinsics, a traversable
                surface to project onto, and a pose lattice. Any map with a
                usable graph has all three, so both modes are definable
                wherever ``nav_waypoint`` is.
    servable    a point action ends at a lattice pose, and design plan §9.5 forbids
                snapping in cached mode only, because that would make the two
                runtimes different transition systems. A node-keyed album has no
                image at an arbitrary lattice pose, so a cached runtime cannot
                serve the point modes even though the map defines them. That is
                a runtime limit, recorded here so nobody reads "definable" as
                "available in every runtime".

    ``depth_provider_is_geometric`` carries design plan §9.3's warning forward: where
    the range comes from a ground-plane raycast, ``nav_point_2d_depth`` is point
    selection plus known geometry rather than metric-depth reasoning, and
    results must be reported that way.
    """

    definable: bool
    servable_runtimes: list[str]
    max_range_m: float
    lattice_spacing_cm: float
    lattice_headings: int
    depth_provider: str
    depth_provider_is_geometric: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def decide_point_navigation(
    analysis: GraphAnalysis,
    *,
    thresholds: Thresholds = THRESHOLDS,
    album_is_lattice_baked: bool = False,
) -> PointNavDecision:
    """Scale point-action reach to the map, and say which runtimes can serve it."""
    from embodiedbench.embodiment.point_nav import (
        DEFAULT_LATTICE_HEADINGS,
        DEFAULT_LATTICE_SPACING_CM,
    )

    median_edge_m = float((analysis.edge_length_m or {}).get("p50") or 0.0)
    definable = analysis.node_count > 0 and analysis.edge_count > 0

    if median_edge_m <= 0:
        reach = thresholds.point_range_min_m
        reach_reason = (
            f"no measurable median edge, so the reach falls back to the "
            f"{thresholds.point_range_min_m:g} m floor"
        )
    else:
        raw = median_edge_m * thresholds.point_range_p50_factor
        reach = min(max(raw, thresholds.point_range_min_m), thresholds.point_range_max_m)
        clamped = "" if raw == reach else f" (clamped from {raw:.1f} m)"
        reach_reason = (
            f"reach {reach:.1f} m = {thresholds.point_range_p50_factor:g}x the map's own "
            f"median edge of {median_edge_m:.1f} m{clamped}"
        )

    servable = ["live"]
    if album_is_lattice_baked:
        servable.append("cached")
        serve_reason = "the album is baked on the pose lattice, so cached can serve point modes"
    else:
        serve_reason = (
            "the album is keyed on graph nodes, which have no image at an arbitrary "
            "lattice pose, so point modes are live-runtime only (design plan §9.5)"
        )

    return PointNavDecision(
        definable=definable,
        servable_runtimes=servable if definable else [],
        max_range_m=round(reach, 2),
        lattice_spacing_cm=DEFAULT_LATTICE_SPACING_CM,
        lattice_headings=DEFAULT_LATTICE_HEADINGS,
        depth_provider="ground_plane_raycast",
        # design plan §9.3: say so plainly rather than letting a reader assume the
        # mode measures depth.
        depth_provider_is_geometric=True,
        rationale=(
            f"{reach_reason}; {serve_reason}"
            if definable
            else "map has no usable graph, so no navigation mode is definable"
        ),
    )


def decide_navigation(
    analysis: GraphAnalysis, thresholds: Thresholds = THRESHOLDS
) -> NavigationDecision:
    """Pick the action space from the measured geometry.

    Graph navigation (``MOVE_TO``) is correct on every map, because a one-hop
    step to a named neighbour is defined whatever the street angles are.
    Directional ``MOVE`` is only offered when the map is actually a grid; on
    Paris it produces "blocked" in three of four directions and the first
    shortest-path step is unreachable.

    So the rule is conservative in the safe direction: graph navigation is the
    default, and cardinal MOVE is *added* only when the map earns it.
    """
    grid_like = analysis.cardinal_fraction >= thresholds.cardinal_fraction_for_move
    base = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"]
    if grid_like:
        return NavigationDecision(
            mode="cardinal+graph",
            enabled_actions=base + ["MOVE", "NAVIGATE"],
            enable_waypoint_marks=True,
            rationale=(
                f"{analysis.cardinal_fraction:.1%} of edges lie within "
                f"{thresholds.cardinal_tolerance_deg:g} degrees of a cardinal axis "
                f"(>= {thresholds.cardinal_fraction_for_move:.0%}), so directional MOVE is a "
                "valid abstraction and is offered alongside graph navigation."
            ),
            cardinal_fraction=analysis.cardinal_fraction,
        )
    return NavigationDecision(
        mode="graph",
        enabled_actions=base + ["NAVIGATE"],
        enable_waypoint_marks=True,
        rationale=(
            f"only {analysis.cardinal_fraction:.1%} of edges are near-cardinal "
            f"(< {thresholds.cardinal_fraction_for_move:.0%}), so directional MOVE would leave "
            "most shortest-path steps unreachable. Graph navigation (MOVE_TO) is the "
            "primary action, per design plan §3.3.4."
        ),
        cardinal_fraction=analysis.cardinal_fraction,
    )


def count_affordances(city_map: Any) -> dict[str, int]:
    """POI types present on this map, and how many of each.

    This is what lets one environment host many tasks: a task declares what it
    needs (restaurants, customers, chargers) and checks it against this, instead
    of hardcoding which map names it knows how to run on.
    """
    counts: dict[str, int] = {}
    for poi in getattr(city_map, "pois", []) or []:
        name = str(getattr(poi, "type", None) or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def analyze_source_geometry(
    map_dir: Path, thresholds: Thresholds = THRESHOLDS
) -> dict[str, Any]:
    """Inspect ``roads.json`` before the engine normalizes it.

    The runtime graph cannot reveal degenerate input. The engine interpolates
    waypoints, so its shortest edge is about 1 m even for a map whose source
    contains sub-millimetre segments -- measuring the graph therefore reports
    zero degenerate edges however broken the export is.

    The damage is still real: a stress map with two sub-millimetre segments
    gained 17 spurious graph edges and lost 19 points of cardinal alignment
    against an otherwise identical baseline. That is why design plan §3.3.1 says to
    filter zero-length segments *before* normalization, and why this check reads
    the source rather than the result.
    """
    roads_path = Path(map_dir) / "roads.json"
    if not roads_path.exists():
        return {"available": False}
    try:
        roads = json.loads(roads_path.read_text()).get("roads", [])
    except (OSError, ValueError) as exc:
        return {"available": False, "error": str(exc)[:200]}

    degenerate = 0
    self_loops = 0
    nonfinite = 0
    lengths: list[float] = []
    for road in roads:
        try:
            start, end = road["start"], road["end"]
            x1, y1 = float(start["x"]), float(start["y"])
            x2, y2 = float(end["x"]), float(end["y"])
        except (KeyError, TypeError, ValueError):
            nonfinite += 1
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            nonfinite += 1
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        lengths.append(length)
        if x1 == x2 and y1 == y2:
            self_loops += 1
        elif length < thresholds.min_edge_m:
            degenerate += 1

    return {
        "available": True,
        "segment_count": len(roads),
        "degenerate_segments": degenerate,
        "self_loop_segments": self_loops,
        "nonfinite_segments": nonfinite,
        "shortest_segment_m": round(min(lengths), 6) if lengths else 0.0,
        "longest_segment_m": round(max(lengths), 2) if lengths else 0.0,
    }


def source_findings(
    source: dict[str, Any], thresholds: Thresholds = THRESHOLDS
) -> list[dict[str, Any]]:
    """Flags derived from the source geometry rather than the runtime graph."""
    findings: list[dict[str, Any]] = []
    if not source.get("available"):
        return findings
    if source.get("degenerate_segments"):
        findings.append({
            "code": "degenerate_edges",
            "count": source["degenerate_segments"],
            "detail": (
                f"roads.json declares segments shorter than {thresholds.min_edge_m} m "
                f"(shortest {source['shortest_segment_m']} m). The engine interpolates them "
                "away, so they are invisible in the runtime graph, but they still add "
                "spurious edges and degrade cardinal alignment (design plan §3.3.1)"
            ),
            "stage": "source",
        })
    if source.get("self_loop_segments"):
        findings.append({
            "code": "self_loop_segments",
            "count": source["self_loop_segments"],
            "detail": "roads.json declares segments whose start equals their end",
            "stage": "source",
        })
    if source.get("nonfinite_segments"):
        findings.append({
            "code": "nonfinite_source_geometry",
            "count": source["nonfinite_segments"],
            "detail": "roads.json declares segments with missing or non-finite coordinates",
            "stage": "source",
        })
    return findings


def quality_findings(
    analysis: GraphAnalysis, thresholds: Thresholds = THRESHOLDS
) -> list[dict[str, Any]]:
    """Rule-based graph-quality flags. Advisory: they downgrade, never crash."""
    findings: list[dict[str, Any]] = []
    if analysis.degenerate_edges:
        findings.append({
            "code": "degenerate_graph_edges",
            "count": analysis.degenerate_edges,
            "detail": f"edges shorter than {thresholds.min_edge_m} m carry no direction",
        })
    if analysis.over_connected_nodes:
        findings.append({
            "code": "over_connected_nodes",
            "count": analysis.over_connected_nodes,
            "detail": (
                f"nodes with degree > {thresholds.max_plausible_degree}; a road junction "
                "rarely exceeds this, so these are probably endpoints merged in error "
                "(false intersections, design plan §3.2)"
            ),
        })
    if analysis.long_edges:
        findings.append({
            "code": "long_unvalidated_edges",
            "count": analysis.long_edges,
            "detail": (
                f"edges longer than {analysis.long_edge_threshold_m} m, this map's own outlier "
                f"threshold ({thresholds.long_edge_p50_factor}x its median edge, floored at "
                f"{thresholds.long_edge_floor_m} m). Such edges may cross non-road space and "
                "need NavMesh confirmation before certification (design plan §6.1 P2)"
            ),
            "stage": "graph",
        })
    if analysis.largest_component_fraction < thresholds.min_largest_component_fraction:
        findings.append({
            "code": "fragmented_graph",
            "count": analysis.component_count,
            "detail": (
                f"largest component holds {analysis.largest_component_fraction:.1%} of nodes, "
                f"below the {thresholds.min_largest_component_fraction:.0%} floor"
            ),
        })
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


async def _scripted_delivery(
    module: Any,
    map_name: str,
    seed: int,
    decision: NavigationDecision,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """One rule-based oracle delivery, used to prove the map is playable."""
    config = dataclasses.asdict(module.PRESETS["nav"])
    if base_dir:
        config["base_dir"] = str(base_dir)
    config.update(
        map_name=map_name,
        render_mode="text",
        max_steps=400,
        enable_waypoint_marks=decision.enable_waypoint_marks,
        enabled_actions=list(decision.enabled_actions),
    )
    env = module.DeliveryBench(config)
    try:
        await env.reset(seed=seed)
        agent = env._env.dms[0]
        city_map = agent.city_map
        # The oracle must plan on the graph the analysis certified, not the raw
        # one; otherwise solvability is measured against a different map.
        from embodiedbench.compiler.graph_repair import repair_graph as _node_graph

        _node_graph(city_map)

        async def act(action: str):
            return await env.step(json.dumps({"action": action}))

        await act("VIEW_ORDERS()")
        await act("ACCEPT_ORDER(0)")
        orders = list(getattr(agent, "active_orders", []) or [])
        if not orders:
            return {"seed": seed, "delivered": 0, "steps": 0, "failure": "no_order_accepted"}
        order = orders[0]

        steps = 0
        for leg, target in (("pickup", order.pickup_node), ("dropoff", order.dropoff_node)):
            for _ in range(200):
                current = city_map.nearest_waypoint(float(agent.x), float(agent.y))
                if current is target:
                    break
                path, _cost = city_map.waypoint_graph.shortest_path_nodes(current, target)
                if not path or len(path) < 2:
                    return {"seed": seed, "delivered": 0, "steps": steps, "failure": f"{leg}_no_path"}
                node_id = getattr(path[1], "waypoint_id", None)
                _obs, _reward, done, info = await act(f'MOVE_TO("{node_id}")')
                steps += 1
                error = (info or {}).get("action_error")
                if error:
                    return {"seed": seed, "delivered": 0, "steps": steps,
                            "failure": f"{leg}: {str(error)[:80]}"}
                if done:
                    break
            await act("PICKUP(orders=[0])" if leg == "pickup" else "DROP_OFF(oid=0)")

        delivered = len(getattr(agent, "completed_orders", []) or [])
        return {
            "seed": seed,
            "delivered": delivered,
            "steps": steps,
            "earnings": round(float(getattr(agent, "earnings_total", 0.0)), 2),
            "failure": None if delivered else "no_delivery_recorded",
        }
    finally:
        await env.close()


def validate_solvability(
    module: Any,
    map_name: str,
    decision: NavigationDecision,
    thresholds: Thresholds = THRESHOLDS,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Run scripted oracle deliveries; the map must actually be playable.

    An episode that raises is a result, not an interruption: a map that makes
    the engine throw is exactly what a stress case is probing for, and it must
    be reported rather than aborting the whole run.
    """
    results = []
    for seed in range(42, 42 + thresholds.solvability_seeds):
        try:
            results.append(asyncio.run(_scripted_delivery(module, map_name, seed, decision, base_dir)))
        except Exception as exc:  # noqa: BLE001
            results.append({
                "seed": seed, "delivered": 0, "steps": 0,
                "failure": f"{type(exc).__name__}: {exc}"[:200],
            })
    delivered = sum(1 for r in results if r["delivered"] >= 1)
    rate = delivered / len(results) if results else 0.0
    return {
        "episodes": results,
        "delivered_episodes": delivered,
        # Unrounded: the EnvSpec validator recomputes delivered/episodes and
        # demands equality to 1e-6, which any seed count that does not divide
        # 1000 would fail against a 3-decimal figure.
        "solvability_rate": rate,
        "passes": rate >= thresholds.min_solvability_rate,
        "mean_steps": round(statistics.mean([r["steps"] for r in results]), 1) if results else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


# Failures the engine raises for maps it cannot turn into a task, mapped to a
# stable reason code. Matching on message text is unpleasant, but the engine
# raises bare RuntimeError/ValueError, so the alternative is one opaque code for
# every cause -- which would tell a caller nothing about how to fix the map.
_UNUSABLE_REASONS: tuple[tuple[str, str, str], ...] = (
    ("world_nodes is empty", "no_pois",
     "the map declares no POI nodes, so no pickup or drop-off can exist"),
    ("No valid pickup or dropoff nodes", "no_usable_pois",
     "POIs exist but none were recognised; the engine keeps only a hardcoded "
     "vocabulary (restaurant/store/rest_area/hospital/car_rental/customer/building, "
     "or instance names starting with BP_Building) and silently drops the rest"),
    ("Failed to bind pickup or dropoff node", "pois_unreachable",
     "POIs could not be bound to the road graph; usually there are no roads, or "
     "no road lies near enough to a POI to anchor a door"),
    ("cannot convert float NaN to integer", "nonfinite_geometry",
     "the map contains NaN or infinite coordinates"),
    ("'NoneType' object has no attribute", "malformed_poi_record",
     "a POI record is missing a structure the engine dereferences without checking"),
)


def classify_failure(exc: Exception) -> tuple[str, str]:
    """Map an engine exception onto a stable (code, explanation)."""
    message = str(exc)
    for needle, code, explanation in _UNUSABLE_REASONS:
        if needle in message:
            return code, explanation
    return "unknown_load_failure", f"{type(exc).__name__}: {message}"[:300]


def unusable_verdict(
    map_name: str, thresholds: Thresholds, exc: Exception, *, stage: str
) -> dict[str, Any]:
    """A structured 'this map cannot become an env, and here is why'."""
    code, explanation = classify_failure(exc)
    return {
        "schema": "embodiedbench/map_pipeline/v0.1",
        "map": map_name,
        "thresholds": dataclasses.asdict(thresholds),
        "analysis": GraphAnalysis().to_dict(),
        "navigation": None,
        "quality_findings": [
            {"code": code, "count": 1, "detail": explanation, "stage": stage}
        ],
        "validation": {"passes": False, "solvability_rate": 0.0, "episodes": []},
        "grade": "fail",
        "unusable": True,
        "failure": {
            "stage": stage,
            "code": code,
            "explanation": explanation,
            "exception": f"{type(exc).__name__}: {exc}"[:300],
        },
        "env_config": None,
    }


def compile_map(
    map_name: str,
    *,
    thresholds: Thresholds = THRESHOLDS,
    run_validation: bool = True,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Run the full map -> training-env pipeline for one map."""
    from embodiedbench.baseline.compat import apply_map_compatibility_patches
    from embodiedbench.baseline.determinism import apply_deterministic_patches
    from embodiedbench.baseline.replay import load_vendor_env_module
    from embodiedbench.compiler.procgen import compile_procgen_world

    apply_deterministic_patches()
    apply_map_compatibility_patches()
    module = load_vendor_env_module()

    # ── 1. load ──────────────────────────────────────────────────────────────
    async def open_map():
        config = dataclasses.asdict(module.PRESETS["nav"])
        if base_dir:
            config["base_dir"] = str(base_dir)
        config.update(map_name=map_name, render_mode="text", max_steps=8)
        env = module.DeliveryBench(config)
        await env.reset(seed=0)
        return env

    # A map that cannot be opened is a verdict, not an exception. The pipeline's
    # contract is "any map in, a training env or a stated reason out"; crashing
    # gives the caller neither, and every unusable-map stress case lands here.
    try:
        env = asyncio.run(open_map())
    except Exception as exc:  # noqa: BLE001
        return unusable_verdict(map_name, thresholds, exc, stage="load")

    try:
        agent = env._env.dms[0]
        repair = None
        if thresholds.repair_graph:
            from embodiedbench.compiler.graph_repair import repair_graph as _node_graph

            repair = _node_graph(agent.city_map).to_dict()
        analysis = analyze_graph(agent.city_map, thresholds)
        affordances = count_affordances(agent.city_map)
        world = compile_procgen_world(
            agent.city_map, map_name=map_name, order_manager=env._env.om
        )
    except Exception as exc:  # noqa: BLE001
        return unusable_verdict(map_name, thresholds, exc, stage="analyze")
    finally:
        try:
            asyncio.run(env.close())
        except Exception:  # noqa: BLE001 - teardown must not mask the verdict
            pass

    # ── 2-3. decide ──────────────────────────────────────────────────────────
    decision = decide_navigation(analysis, thresholds)
    maps_root = Path(base_dir) if base_dir else (
        REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"
    )
    source = analyze_source_geometry(maps_root / "maps" / map_name, thresholds)
    findings = source_findings(source, thresholds) + quality_findings(analysis, thresholds)

    # ── 4. validate ──────────────────────────────────────────────────────────
    validation = (
        validate_solvability(module, map_name, decision, thresholds, base_dir)
        if run_validation
        else {"skipped": True, "passes": False}
    )

    # ── 5. certify ───────────────────────────────────────────────────────────
    if not validation.get("passes"):
        grade = "fail"
    elif findings:
        grade = "B"  # playable, with declared limitations
    else:
        grade = "A"

    env_config = {
        "map_name": map_name,
        "enabled_actions": decision.enabled_actions,
        "enable_waypoint_marks": decision.enable_waypoint_marks,
        "render_mode": "text",
    }

    return {
        "schema": "embodiedbench/map_pipeline/v0.1",
        "map": map_name,
        "thresholds": dataclasses.asdict(thresholds),
        "analysis": analysis.to_dict(),
        "graph_repair": repair,
        "affordances": affordances,
        "source_geometry": source,
        "navigation": decision.to_dict(),
        "quality_findings": findings,
        "validation": validation,
        "grade": grade,
        "env_config": env_config,
        "world_bundle_sha256": world.content_hash(),
        "world_nodes": len(world.nav_graph.nodes),
        "world_edges": len(world.nav_graph.edges),
    }
