# base/graph.py
# -*- coding: utf-8 -*-

import math
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List, Set

from .types import Vector, Node, Edge


# =========================================================
# Geometry helpers
# =========================================================
def _proj_with_raw(p: Vector, a: Vector, b: Vector):
    """
    Compute the projection of point p onto the infinite line through segment (a, b),
    and also the clamped projection on the segment itself.

    Returns:
        (proj_raw, t_raw, proj_seg, t, da, db, seg_len)
            proj_raw : projection of p on the infinite line AB
            t_raw    : scalar parameter for proj_raw on AB (unclamped)
            proj_seg : projection clamped onto the segment [A, B]
            t        : clamped parameter in [0, 1] for proj_seg on AB
            da       : distance from A to proj_seg along the segment
            db       : distance from B to proj_seg along the segment
            seg_len  : length of segment AB
    """
    abx, aby = (b.x - a.x), (b.y - a.y)
    apx, apy = (p.x - a.x), (p.y - a.y)
    ab_len2 = abx * abx + aby * aby
    if ab_len2 <= 1e-9:
        proj_raw = Vector(a.x, a.y)
        return proj_raw, 0.0, proj_raw, 0.0, 0.0, 0.0, 0.0
    t_raw = (apx * abx + apy * aby) / ab_len2
    proj_raw = Vector(a.x + abx * t_raw, a.y + aby * t_raw)
    t = max(0.0, min(1.0, t_raw))
    proj_seg = Vector(a.x + abx * t, a.y + aby * t)
    da = math.hypot(proj_seg.x - a.x, proj_seg.y - a.y)
    db = math.hypot(proj_seg.x - b.x, proj_seg.y - b.y)
    seg_len = math.sqrt(ab_len2)
    return proj_raw, t_raw, proj_seg, t, da, db, seg_len


def project_point_to_segment(p: Vector, a: Vector, b: Vector):
    """
    Thin wrapper used by Map code.

    Returns:
        (proj_raw, da, db, seg_len)
            proj_raw : projection of p on the infinite line AB
            da       : distance from A to the clamped projection along the segment
            db       : distance from B to the clamped projection along the segment
            seg_len  : length of segment AB

    Note:
        da/db are computed using the clamped foot-of-perpendicular on the segment.
        The Map side primarily cares about proj_raw for snapping.
    """
    proj_raw, _t_raw, proj_seg, _t, da, db, seg_len = _proj_with_raw(p, a, b)
    # da/db are geometric distances based on the clamped foot on the segment.
    return proj_raw, da, db, seg_len


# =========================================================
# Graph structure over map nodes/edges
# =========================================================
class Graph:
    """
    Undirected graph over map nodes and edges.

    Responsibilities:
        - Maintain adjacency between Node objects.
        - Store per-edge metadata (e.g., kind: road / crosswalk / endcap).
        - Provide geometric snapping utilities to road edges.
        - Support shortest-path queries (node-to-node and XY-to-node).
        - Allow safe copying of the graph structure (nodes are reused).

    All distances are in centimeters, consistent with the map representation.
    """

    def __init__(self):
        # Core graph containers
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()
        self.adjacency_list: Dict[Node, List[Node]] = defaultdict(list)

        # Edge metadata keyed by an order-invariant (node_a, node_b) string pair
        self.edge_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # -----------------------------------------------------
    # Basic graph construction
    # -----------------------------------------------------
    def add_node(self, node: Node):
        """Register a node in the graph."""
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        """
        Register an undirected edge in the graph and update adjacency.

        Self-loops are ignored.
        """
        if edge.node1 is edge.node2:
            return
        self.edges.add(edge)
        if edge.node2 not in self.adjacency_list[edge.node1]:
            self.adjacency_list[edge.node1].append(edge.node2)
        if edge.node1 not in self.adjacency_list[edge.node2]:
            self.adjacency_list[edge.node2].append(edge.node1)

    def connected(self, a: Node, b: Node) -> bool:
        """Return True if b is directly adjacent to a."""
        return b in self.adjacency_list[a]

    def _edge_key(self, a: Node, b: Node) -> Tuple[str, str]:
        """
        Build a stable key for edge metadata.

        Uses the string representation of two nodes and sorts them so that
        (a, b) and (b, a) share the same key.
        """
        ka, kb = str(a), str(b)
        return (ka, kb) if ka < kb else (kb, ka)

    def set_edge_meta(self, a: Node, b: Node, meta: Dict[str, Any]):
        """Attach a shallow copy of meta to edge (a, b)."""
        self.edge_meta[self._edge_key(a, b)] = dict(meta)

    def get_edge_meta(self, a: Node, b: Node) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for edge (a, b), if any."""
        return self.edge_meta.get(self._edge_key(a, b))

    # -----------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------
    def find_existing_node_at(self, pos: Vector, eps_cm: float = 80.0) -> Optional[Node]:
        """
        Find an existing node near the given position.

        Args:
            pos: Target position in world coordinates.
            eps_cm: Maximum radius (in cm) within which a node is considered a match.

        Returns:
            The closest node within eps_cm, or None if no node is close enough.
        """
        px, py = float(pos.x), float(pos.y)
        thr2 = eps_cm * eps_cm
        best = None
        best_d2 = None
        for nd in self.nodes:
            dx = float(nd.position.x) - px
            dy = float(nd.position.y) - py
            d2 = dx * dx + dy * dy
            if d2 <= thr2 and (best is None or d2 < best_d2):
                best, best_d2 = nd, d2
        return best

    def _iter_road_edges_only(self):
        """
        Iterate over edges that represent road-like segments.

        Filters out:
            - any edge whose meta["kind"] is not one of ("road", "crosswalk", "endcap")
            - edges whose endpoints are of type "dock" (to avoid treating dock_link
              endpoints as normal roads)
        """
        for e in list(self.edges):
            meta = self.get_edge_meta(e.node1, e.node2) or {}
            if meta.get("kind") not in ("road", "crosswalk", "endcap"):
                continue
            # Avoid treating dock_link endpoints as normal roads.
            if getattr(e.node1, "type", "") == "dock" or getattr(e.node2, "type", "") == "dock":
                continue
            yield e

    # -----------------------------------------------------
    # Snapping to road geometry
    # -----------------------------------------------------
    def snap_to_nearest_edge(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        """
        Snap a world position to the nearest road-like edge.

        Only considers edges treated as roads (see _iter_road_edges_only) and
        uses the clamped projection onto the segment for distance.

        Args:
            x, y: World coordinates (cm).

        Returns:
            A dictionary describing the best snap candidate, or None if no
            road-like edge is found. The dictionary contains:

                {
                    "edge"    : Edge,
                    "a"       : Node,      # edge start
                    "b"       : Node,      # edge end
                    "proj"    : Vector,    # clamped projection on segment
                    "proj_raw": Vector,    # projection on infinite line
                    "t"       : float,     # clamped parameter in [0, 1]
                    "t_raw"   : float,     # raw parameter (unclamped)
                    "da"      : float,     # distance from A to proj_seg
                    "db"      : float,     # distance from B to proj_seg
                    "dist"    : float,     # distance from (x, y) to proj_seg
                }
        """
        p = Vector(x, y)
        best = None
        for e in self._iter_road_edges_only():
            a, b = e.node1.position, e.node2.position
            proj_raw, t_raw, proj_seg, t, da, db, _ = _proj_with_raw(p, a, b)
            d = p.distance(proj_seg)  # distance to the clamped projection
            if (best is None) or (d < best["dist"] - 1e-6):
                best = {
                    "edge": e,
                    "a": e.node1,
                    "b": e.node2,
                    "proj": proj_seg,       # clamped projection
                    "proj_raw": proj_raw,   # projection on infinite line
                    "t": t,
                    "t_raw": t_raw,
                    "da": da,
                    "db": db,
                    "dist": d,
                }
        return best

    # -----------------------------------------------------
    # Insert node on edge (split or short extension)
    # -----------------------------------------------------
    def ensure_node_on_edge_at(
        self,
        edge: Edge,
        proj_raw: Vector,
        snap_cm: float = 100.0,
        max_extend_cm: float = 1500.0,
    ) -> Node:
        """
        Ensure there is a graph node at the given projection on an edge.

        Behavior is split into two cases:

        1) Projection lies outside the segment (extension region):
           - Create a new "dock" node at proj_raw.
           - Connect it to whichever endpoint (a or b) is closer in the
             extension direction via a "dock_link" edge.

        2) Projection lies inside the segment:
           - Create a new "normal" node at proj_raw.
           - Remove the original edge (a, b) and its adjacency.
           - Add two new edges (a, new_node) and (new_node, b).
           - Copy the original edge metadata to both new edges.

        Args:
            edge:
                The original edge to insert on.
            proj_raw:
                Raw projection position (Vector) on the infinite line AB.
            snap_cm:
                Snap tolerance (currently unused in the logic, kept for API compatibility).
            max_extend_cm:
                Maximum allowed extension length (currently unused in the logic,
                kept for future constraints / checks).

        Returns:
            The Node instance corresponding to the inserted (or created) node.
        """
        a, b = edge.node1, edge.node2
        ax, ay = a.position.x, a.position.y
        bx, by = b.position.x, b.position.y
        px, py = proj_raw.x, proj_raw.y

        old_meta = self.get_edge_meta(a, b) or {}

        # Compute t_raw to determine whether proj_raw lies on the segment or its extension.
        abx, aby = (bx - ax), (by - ay)
        apx, apy = (px - ax), (py - ay)
        ab_len2 = abx * abx + aby * aby
        t_raw = (apx * abx + apy * aby) / (ab_len2 if ab_len2 > 1e-9 else 1.0)

        # ---- Extension region: only allow "short-circuit" as dock + dock_link.
        if t_raw <= 0.0 or t_raw >= 1.0:
            from_node = a if t_raw <= 0.0 else b
            extend_len = math.hypot(px - from_node.position.x, py - from_node.position.y)
            new_node = Node(Vector(px, py), "dock")
            self.add_node(new_node)
            self.add_edge(Edge(from_node, new_node))
            self.set_edge_meta(from_node, new_node, {"kind": "dock_link", "name": ""})
            return new_node

        # ---- Inside segment: split edge into two segments via a new node.
        new_node = Node(Vector(px, py), "normal")
        self.add_node(new_node)

        # Remove original edge and its adjacency entries.
        while edge in self.edges:
            self.edges.remove(edge)
        while b in self.adjacency_list.get(a, []):
            self.adjacency_list[a].remove(b)
        while a in self.adjacency_list.get(b, []):
            self.adjacency_list[b].remove(a)
        if old_meta:
            self.edge_meta.pop(self._edge_key(a, b), None)

        # Add two new edges and inherit metadata.
        self.add_edge(Edge(a, new_node))
        self.add_edge(Edge(new_node, b))
        if old_meta:
            self.set_edge_meta(a, new_node, old_meta)
            self.set_edge_meta(new_node, b, old_meta)

        return new_node

    # -----------------------------------------------------
    # Shortest path (Dijkstra)
    # -----------------------------------------------------
    def _w(self, u: Node, v: Node) -> float:
        """Edge weight function: Euclidean distance between node positions."""
        return u.position.distance(v.position)

    def shortest_path_nodes(self, start: Node, target: Node):
        """
        Compute the shortest path between two nodes using Dijkstra's algorithm.

        Args:
            start: Source node.
            target: Destination node.

        Returns:
            (path, dist)
                path: List[Node] from start to target (inclusive).
                      Empty list if target is unreachable.
                dist: Total distance (float) or +inf if unreachable.
        """
        import heapq
        dist: Dict[Node, float] = {start: 0.0}
        prev: Dict[Node, Node] = {}
        pq = [(0.0, id(start), start)]
        seen = set()

        while pq:
            du, _, u = heapq.heappop(pq)
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
                    heapq.heappush(pq, (alt, id(v), v))

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

    def shortest_path_xy_to_node(self, x: float, y: float, target: Node):
        """
        Compute a shortest path from an arbitrary position (x, y) to a target node.

        The algorithm:
            - Snaps (x, y) to the nearest road edge, yielding two candidate
              endpoints (a, b) with distances da and db along the edge.
            - Runs a multi-source Dijkstra from {a, b} with initial distances
              {da, db}.
            - Tracks which endpoint was effectively used as the starting node.

        Args:
            x, y: World coordinates (cm) of the starting position.
            target: Destination node.

        Returns:
            (path, dist, start_choice)
                path        : List[Node] from chosen start node to target.
                dist        : Total distance (float) or +inf if unreachable.
                start_choice: The node (a or b) chosen as the effective start,
                              or None if no path was found.
        """
        snap = self.snap_to_nearest_edge(x, y)
        if snap is None:
            return [], float("inf"), None
        a, b = snap["a"], snap["b"]
        da, db = snap["da"], snap["db"]

        import heapq
        dist: Dict[Node, float] = {a: da, b: db}
        prev: Dict[Node, Node] = {}
        pq = [(da, id(a), a), (db, id(b), b)]
        seen = set()
        start_choice = None

        while pq:
            du, _, u = heapq.heappop(pq)
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
                    heapq.heappush(pq, (alt, id(v), v))

        if target not in dist:
            return [], float("inf"), start_choice

        path = []
        cur = target
        while cur in prev:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, float(dist[target]), start_choice

    # -----------------------------------------------------
    # Copy
    # -----------------------------------------------------
    def copy(self) -> "Graph":
        """
        Create a shallow structural copy of the graph.

        Notes:
            - Node objects are reused (shared) between the original and copy.
              This is safe because pathfinding does not mutate nodes.
            - Edge objects and adjacency lists are re-created.
            - Edge metadata is shallow-copied.
        """
        g = Graph()
        # Original nodes can be reused; pathfinding does not modify them.
        for n in self.nodes:
            g.add_node(n)
        for e in self.edges:
            a, b = e.node1, e.node2
            ee = Edge(a, b)
            g.add_edge(ee)
            meta = self.get_edge_meta(a, b)
            if meta:
                g.set_edge_meta(a, b, meta)
        return g