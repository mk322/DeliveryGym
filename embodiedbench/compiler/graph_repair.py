"""Rule-based repair of an un-noded navigation graph.

Measured on CityCore Paris: 2689 of 7230 runtime edges (37.2%) pass within 1 m
of at least one other node without being split there, some skipping ten or more.
The same measurement on every procgen map returns 0%.

An unsplit edge is not cosmetic. If A-B runs straight past C, an agent can step
A->B and skip the intersection at C entirely -- a shortcut through a junction it
never visited, and, where the edge spans 640 m in a dense city, almost certainly
through buildings. It also inflates node degree (Paris mean 12.44 against
procgen's 2.25), which is the symptom that first exposed this.

The source geometry is not to blame: Paris ``roads.json`` has 190 segments with
only 10 mid-span crossings, so the density is introduced during graph
construction. The repair therefore runs on the built graph.

The operation is *noding*: replace a skipping edge with the chain of edges
through the nodes it passes. Geometry is preserved exactly -- no node moves, no
node is invented, and every point reachable before is reachable after, because
the chain connects the same endpoints along the same line. What changes is that
travel along the line now has to visit the junctions on it.

No model is involved; every decision is a distance test against a declared
tolerance.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# A node counts as lying on an edge when it is within this distance of the line.
# Chosen well below the shortest real edge (~1 m after the engine's waypoint
# interpolation) so a genuinely separate node is never absorbed.
ON_LINE_TOLERANCE_M = 1.0
# Ignore the ends: a node at t≈0 or t≈1 *is* an endpoint, not a skipped junction.
END_MARGIN = 0.02
# Edges shorter than this cannot meaningfully skip anything.
MIN_SPAN_M = 12.0


@dataclass
class RepairReport:
    """What the repair changed, in numbers a reviewer can check."""

    nodes: int = 0
    edges_before: int = 0
    edges_after: int = 0
    edges_split: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    skipped_nodes_recovered: int = 0
    mean_degree_before: float = 0.0
    mean_degree_after: float = 0.0
    max_degree_before: int = 0
    max_degree_after: int = 0
    longest_edge_before_m: float = 0.0
    longest_edge_after_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges_before": self.edges_before,
            "edges_after": self.edges_after,
            "edges_split": self.edges_split,
            "edges_added": self.edges_added,
            "edges_removed": self.edges_removed,
            "skipped_nodes_recovered": self.skipped_nodes_recovered,
            "mean_degree_before": round(self.mean_degree_before, 3),
            "mean_degree_after": round(self.mean_degree_after, 3),
            "max_degree_before": self.max_degree_before,
            "max_degree_after": self.max_degree_after,
            "longest_edge_before_m": round(self.longest_edge_before_m, 2),
            "longest_edge_after_m": round(self.longest_edge_after_m, 2),
            "notes": self.notes,
        }


def _point_on_segment(ax, ay, bx, by, px, py) -> tuple[float, float]:
    """Distance from P to segment AB, and the parameter t of the projection."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t_clamped = max(0.0, min(1.0, t))
    cx, cy = ax + t_clamped * dx, ay + t_clamped * dy
    return math.hypot(px - cx, py - cy), t


class _Grid:
    """Uniform spatial index, so noding does not become O(edges x nodes)."""

    def __init__(self, points: list[tuple[float, float]], cell_m: float):
        self.cell = cell_m
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (x, y) in enumerate(points):
            self.buckets[(int(x // cell_m), int(y // cell_m))].append(index)

    def near_segment(self, ax, ay, bx, by, pad: float) -> set[int]:
        lo_x, hi_x = min(ax, bx) - pad, max(ax, bx) + pad
        lo_y, hi_y = min(ay, by) - pad, max(ay, by) + pad
        found: set[int] = set()
        for cell_x in range(int(lo_x // self.cell), int(hi_x // self.cell) + 1):
            for cell_y in range(int(lo_y // self.cell), int(hi_y // self.cell) + 1):
                found.update(self.buckets.get((cell_x, cell_y), ()))
        return found


def find_skipped_nodes(
    adjacency: dict, *, tolerance_m: float = ON_LINE_TOLERANCE_M, min_span_m: float = MIN_SPAN_M
) -> tuple[list[tuple[Any, Any, list[Any]]], _Grid]:
    """For every edge, the nodes it passes without stopping at, ordered along it."""
    nodes = list(adjacency)
    points = [(n.position.x / 100.0, n.position.y / 100.0) for n in nodes]
    grid = _Grid(points, cell_m=max(25.0, min_span_m))

    seen: set[tuple[int, int]] = set()
    out: list[tuple[Any, Any, list[Any]]] = []
    for node in nodes:
        ax, ay = node.position.x / 100.0, node.position.y / 100.0
        for neighbour in adjacency.get(node) or []:
            pair = tuple(sorted((id(node), id(neighbour))))
            if pair in seen:
                continue
            seen.add(pair)
            bx, by = neighbour.position.x / 100.0, neighbour.position.y / 100.0
            if math.hypot(bx - ax, by - ay) < min_span_m:
                continue
            hits: list[tuple[float, Any]] = []
            for candidate_index in grid.near_segment(ax, ay, bx, by, tolerance_m):
                candidate = nodes[candidate_index]
                if candidate is node or candidate is neighbour:
                    continue
                px, py = points[candidate_index]
                distance, t = _point_on_segment(ax, ay, bx, by, px, py)
                if distance <= tolerance_m and END_MARGIN < t < 1.0 - END_MARGIN:
                    hits.append((t, candidate))
            if hits:
                hits.sort(key=lambda item: item[0])
                out.append((node, neighbour, [candidate for _t, candidate in hits]))
    return out, grid


def repair_graph(
    city_map: Any,
    *,
    tolerance_m: float = ON_LINE_TOLERANCE_M,
    min_span_m: float = MIN_SPAN_M,
    max_passes: int = 8,
) -> RepairReport:
    """Node the graph in place: split skipping edges into chains.

    Iterated to a fixpoint. One pass is not enough: splitting A-B at C creates
    A-C and C-B, and either of those can itself run past a further node that the
    original edge's ordering hid. A single pass on Paris left 125 edges still
    skipping.

    Returns a report rather than mutating silently, so a caller can refuse a
    repair that changed more than it expected.
    """
    graph = city_map.waypoint_graph
    adjacency = graph.adjacency_list

    def degrees() -> list[int]:
        return [len(v or []) for v in adjacency.values()]

    def edge_stats() -> tuple[int, float]:
        seen: set[tuple[int, int]] = set()
        longest = 0.0
        for node, neighbours in adjacency.items():
            for neighbour in neighbours or []:
                pair = tuple(sorted((id(node), id(neighbour))))
                if pair in seen:
                    continue
                seen.add(pair)
                longest = max(
                    longest,
                    math.dist(
                        (node.position.x, node.position.y),
                        (neighbour.position.x, neighbour.position.y),
                    )
                    / 100.0,
                )
        return len(seen), longest

    before_degrees = degrees()
    edges_before, longest_before = edge_stats()
    report = RepairReport(
        nodes=len(adjacency),
        edges_before=edges_before,
        mean_degree_before=sum(before_degrees) / max(1, len(before_degrees)),
        max_degree_before=max(before_degrees, default=0),
        longest_edge_before_m=longest_before,
    )

    def connected(a: Any, b: Any) -> bool:
        return b in (adjacency.get(a) or [])

    def unlink(a: Any, b: Any) -> None:
        for x, y in ((a, b), (b, a)):
            neighbours = adjacency.get(x)
            if neighbours and y in neighbours:
                neighbours.remove(y)

    def link(a: Any, b: Any) -> None:
        adjacency.setdefault(a, [])
        adjacency.setdefault(b, [])
        if b not in adjacency[a]:
            adjacency[a].append(b)
        if a not in adjacency[b]:
            adjacency[b].append(a)

    for pass_index in range(max_passes):
        skipping, _grid = find_skipped_nodes(
            adjacency, tolerance_m=tolerance_m, min_span_m=min_span_m
        )
        if not skipping:
            report.notes.append(f"reached a fixpoint after {pass_index} pass(es)")
            break
        for start, end, middles in skipping:
            chain = [start, *middles, end]
            # Build the chain first, then drop the shortcut. In this order the
            # graph is never disconnected part-way through, so an interrupted
            # repair degrades to a graph with extra edges rather than one
            # missing them.
            added = 0
            for left, right in zip(chain, chain[1:]):
                if left is right:
                    continue
                if not connected(left, right):
                    link(left, right)
                    added += 1
            if connected(start, end):
                unlink(start, end)
                report.edges_removed += 1
            report.edges_added += added
            report.edges_split += 1
            report.skipped_nodes_recovered += len(middles)
    else:
        remaining, _grid = find_skipped_nodes(
            adjacency, tolerance_m=tolerance_m, min_span_m=min_span_m
        )
        if remaining:
            # Say so rather than reporting a clean repair: a graph that will not
            # converge is itself a finding about the map.
            report.notes.append(
                f"did not converge in {max_passes} passes; {len(remaining)} edges still skip"
            )

    after_degrees = degrees()
    edges_after, longest_after = edge_stats()
    report.edges_after = edges_after
    report.mean_degree_after = sum(after_degrees) / max(1, len(after_degrees))
    report.max_degree_after = max(after_degrees, default=0)
    report.longest_edge_after_m = longest_after
    return report


def verify_repair(city_map: Any, *, tolerance_m: float = ON_LINE_TOLERANCE_M) -> dict[str, Any]:
    """Re-measure after a repair: no edge should still skip a node."""
    adjacency = city_map.waypoint_graph.adjacency_list
    remaining, _grid = find_skipped_nodes(adjacency, tolerance_m=tolerance_m)
    return {
        "edges_still_skipping": len(remaining),
        "worst_skip": max((len(m) for _a, _b, m in remaining), default=0),
    }
