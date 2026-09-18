"""Runtime determinism patches for the vendored DeliveryBench engine.

the design plan M0 requires episodes to replay to identical transition, terminal-state,
and score hashes, and M4 requires byte-identical ``EpisodeSpec`` from two clean
processes. The vendored engine cannot satisfy either, for one reason:

    vlm_delivery/base/graph.py:339,354,399,417
        pq = [(0.0, id(start), start)]
        heapq.heappush(pq, (alt, id(v), v))

``id(v)`` is a CPython memory address. It is the heap's tie-breaker, so whenever
two routes have equal cost — routine on a street grid, where blocks are equal
length — the chosen route depends on where the allocator happened to place the
node objects. That varies between processes and between episodes inside one
process, so route choice, and everything downstream of it (distance, deadline,
energy, earnings, arrival order), is not reproducible.

These patches replace the address tie-breaker with a stable key derived from
node identity. They are applied at runtime to the imported module. ``vendor/``
is never modified: the design plan M0 requires that existing repositories are left
untouched, and a patch we own is auditable in a way an edited vendored file is
not.

Every patch records the sha256 of the function source it replaced, so a report
can state exactly what was changed and a later vendor bump fails loudly instead
of silently reverting the fix.
"""

from __future__ import annotations

import heapq
import inspect
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.artifacts.hashing import sha256_bytes


@dataclass
class PatchRecord:
    """What a single determinism patch replaced, and why."""

    target: str
    reason: str
    original_source_sha256: str
    applied: bool
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "target": self.target,
            "reason": self.reason,
            "original_source_sha256": self.original_source_sha256,
            "applied": self.applied,
        }
        if self.skipped_reason:
            out["skipped_reason"] = self.skipped_reason
        return out


@dataclass
class PatchSet:
    records: list[PatchRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_count": len(self.records),
            "applied_count": sum(1 for r in self.records if r.applied),
            "patches": [r.to_dict() for r in self.records],
        }


TIE_BREAK_REASON = (
    "Dijkstra used id(node) — a CPython memory address — to break equal-cost "
    "ties in the priority queue. Equal-cost routes are common on a street grid, "
    "so route choice varied per process and per episode, breaking replay "
    "determinism (the design plan M0) and cross-process EpisodeSpec equality (the design plan "
    "M4). Replaced with a stable key: the node's waypoint id when present, "
    "otherwise its rounded position. Path *costs* are unchanged; only which of "
    "several equally short routes is returned becomes deterministic."
)

# Positions are floats in centimetres. Rounding to 1e-3 cm keeps distinct nodes
# distinct (the closest authored Paris nodes are metres apart) while making the
# key insensitive to representation noise.
_POSITION_QUANTUM = 3


def stable_node_key(node: Any) -> tuple:
    """A total, address-independent ordering key for a graph node."""
    waypoint_id = getattr(node, "waypoint_id", None)
    position = getattr(node, "position", None)
    coords: tuple[float, ...] = ()
    if position is not None:
        coords = tuple(
            round(float(getattr(position, axis, 0.0) or 0.0), _POSITION_QUANTUM)
            for axis in ("x", "y", "z")
        )
    # str() rather than the raw value so a None id sorts against a str id
    # without a TypeError mid-Dijkstra.
    return (str(waypoint_id) if waypoint_id is not None else "", coords)


def _deterministic_shortest_path_nodes(self, start, target):
    """Address-independent replacement for ``Graph.shortest_path_nodes``.

    Same algorithm and same weights as the vendored version; only the heap
    tie-breaker changes.
    """
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, stable_node_key(start), start)]
    seen = set()

    while pq:
        du, _key, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == target:
            break
        for v in self.adjacency_list[u]:
            alt = du + self._w(u, v)
            if (v not in dist) or (alt < dist[v] - 1e-6):
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, stable_node_key(v), v))

    if target not in dist:
        return [], float("inf")

    path = []
    cur = target
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path, float(dist[target])


def _deterministic_shortest_path_xy_to_node(self, x, y, target):
    """Address-independent replacement for ``Graph.shortest_path_xy_to_node``.

    Snapping, weights, the two-source seeding, the ``start_choice`` rule, and
    the ``cur in prev`` path reconstruction are all reproduced from the vendored
    implementation. Only the heap tie-breaker differs.
    """
    snap = self.snap_to_nearest_edge(x, y)
    if snap is None:
        return [], float("inf"), None
    a, b = snap["a"], snap["b"]
    da, db = snap["da"], snap["db"]

    dist = {a: da, b: db}
    prev = {}
    pq = [(da, stable_node_key(a), a), (db, stable_node_key(b), b)]
    seen = set()
    start_choice = None

    while pq:
        du, _key, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if start_choice is None and (u is a or u is b):
            start_choice = u
        if u == target:
            break
        for v in self.adjacency_list[u]:
            alt = du + self._w(u, v)
            if (v not in dist) or (alt < dist[v] - 1e-6):
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, stable_node_key(v), v))

    if target not in dist:
        return [], float("inf"), start_choice

    path = []
    cur = target
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, float(dist[target]), start_choice


def apply_deterministic_patches() -> PatchSet:
    """Apply every determinism patch. Idempotent.

    Puts the vendored checkout on ``sys.path`` itself rather than assuming the
    caller already imported the env module: patching must not depend on import
    order, or a caller that patches first would silently no-op.
    """
    from embodiedbench.baseline.replay import _ensure_vendor_on_path

    _ensure_vendor_on_path()
    from vagen.envs.deliverybench.vlm_delivery.base import graph as graph_module

    patches = PatchSet()

    if getattr(graph_module, "_embodiedbench_determinism_applied", False):
        return getattr(graph_module, "_embodiedbench_patchset")

    for name, replacement in (
        ("shortest_path_nodes", _deterministic_shortest_path_nodes),
        ("shortest_path_xy_to_node", _deterministic_shortest_path_xy_to_node),
    ):
        original = getattr(graph_module.Graph, name)
        source = inspect.getsource(original)
        record = PatchRecord(
            target=f"vlm_delivery.base.graph.Graph.{name}",
            reason=TIE_BREAK_REASON,
            original_source_sha256=sha256_bytes(source.encode()),
            applied=False,
        )
        if "id(" not in source:
            # The vendored implementation no longer uses an address tie-breaker.
            # Applying our replacement anyway would be an unreviewed behavior
            # change rather than a fix, so refuse and say so.
            record.skipped_reason = (
                f"vendored {name} no longer contains an id() tie-breaker; "
                "re-review this patch against the new implementation before applying"
            )
        else:
            setattr(graph_module.Graph, name, replacement)
            record.applied = True
        patches.records.append(record)

    graph_module._embodiedbench_determinism_applied = True
    graph_module._embodiedbench_patchset = patches
    return patches
