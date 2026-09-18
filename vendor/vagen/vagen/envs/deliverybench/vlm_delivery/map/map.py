# map/map.py
# -*- coding: utf-8 -*-

import json
import math
import os
import random
import re
import zlib
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple, Set

from ..base.types import Vector, Node, Edge, Road
from ..base.graph import Graph, project_point_to_segment  # geometric projection
# Runtime traffic-light *state* is decided by the single TrafficController
# (utils/hazards.py), built from the FPV manifest — the map no longer imports the
# old per-edge edge_signal_check runtime path. ``load_traffic_lights`` is kept
# only to expose the world-JSON light *nodes* as render data for the offline
# capture tools (`tools/render_*`); it is not consulted at runtime.
from ..utils.traffic_lights import load_traffic_lights


# ---------------------------------------------------------------------------
# Street naming
# ---------------------------------------------------------------------------
# Roads in the scenario are renamed at import time to look like real US street
# names ("Maple St", "Church Ave", ...) rather than the legacy "Nth road
# (left/right)" placeholders. The two sidewalks of a single carriageway share
# one user-visible name; sides are distinguished by house-number parity
# (right = even, left = odd) — the same convention used by US postal
# addressing.
#
# Internally the per-sidewalk edge `name` meta keeps a `" (left)"` / `" (right)"`
# suffix because the skeleton / polyline / projection code still groups by it.
# Every observation builder strips the suffix via `Map._display_road_name`
# before the string reaches the agent.
_STREET_NAME_POOL: Tuple[str, ...] = (
    # Trees — extremely common across US towns and cities.
    "Oak", "Maple", "Elm", "Pine", "Cedar", "Birch", "Willow", "Walnut",
    "Cherry", "Chestnut", "Hickory", "Sycamore", "Spruce", "Ash", "Aspen",
    "Magnolia", "Poplar", "Juniper", "Beech", "Linden",
    # Civic / geographic — also very common.
    "Main", "Park", "Church", "Spring", "Highland", "Mill", "Center", "School",
    "Court", "Market", "Union", "State", "Lake", "River", "Hill", "Forest",
    "Sunset", "Bay", "Grove", "Vine",
)

_SIDE_RE = re.compile(r"\s*\((?:left|right)\)\s*$")

# =============== Angle utilities ===============
def _angle_deg(vx: float, vy: float) -> float:
    ang = math.degrees(math.atan2(vy, vx))
    ang = (ang + 360.0) % 360.0
    if ang >= 180.0:
        ang -= 180.0
    return ang


def _fmt_xy_m(x_cm: float, y_cm: float, decimals: int = 2) -> str:
    return f"({float(x_cm)/100.0:.{decimals}f}m, {float(y_cm)/100.0:.{decimals}f}m)"


class Map:
    NEAR_EPS = 300.0  # 3m radius for “near” checks (values are in cm)
    DOOR_YAW_OFFSET_DEG = 90.0

    def __init__(self, cfg: Dict[str, Any]):
        # Three graphs:
        # - graph_full: full sidewalk/pedestrian graph
        # - graph_skel: skeleton graph (merged road/crosswalk segments)
        # - graph_drive: driving lanes graph
        self.graph_full = Graph()
        self.graph_skel = Graph()
        self.graph_drive = Graph()

        # Semantic POIs / orders
        self.pois: List[Node] = []
        # Each entry:
        # {
        #   "node": <poi Node>,
        #   "dock_node": <Node>,
        #   "door_node": <Node|None>,
        #   "road_name": <str|None>,
        #   "center": (x, y),
        #   "building_box": {x,y,w,h,yaw,poi_type} | None
        # }
        self.poi_meta: List[Dict[str, Any]] = []
        self._dock_nodes = set()
        self.order_meta: List[Dict[str, Any]] = []

        self._road_id = 0
        self._xwalk_id = 0

        # Map from door node to its parent POI node
        self._door2poi: Dict[Node, Node] = {}

        # Stable per-type indexing (according to JSON load order)
        self._poi_seq_counter: Dict[str, int] = {}

        # Human-readable address book (populated by _build_addresses()).
        # Maps an address string (e.g. "124 Pine St") or a display name
        # (e.g. "restaurant 3") to the POI's dock_node so callers can
        # resolve tokens without knowing internal node references.
        self.address_book: Dict[str, Node] = {}

        # Waypoint graph (populated by _build_waypoint_graph()).
        # Nodes = intersections ∪ POI docks. Adjacency captures the
        # "next named place you reach if you keep walking down the
        # current road" — i.e. real-world block-by-block movement.
        self.waypoint_graph: Graph = Graph()
        # Stable ID -> waypoint Node (e.g. "int_5", "dock_7").
        self.waypoints_by_id: Dict[str, Node] = {}
        # Reverse: waypoint Node -> its public id.
        self._waypoint_id_by_node: Dict[Node, str] = {}
        # World-JSON pedestrian-light *nodes*, exposed as render data for the
        # offline FPV capture tools. NOT used for runtime red/green decisions —
        # that is TrafficController (utils/hazards.py), built from the FPV manifest.
        self.traffic_lights: List[Dict[str, Any]] = []

        # Driving lane offset (relative to the road centerline, in cm),
        # and spacing between lane waypoints (in cm).

        self.cfg = cfg
        
        self.DRIVE_LANE_OFFSET = self.cfg.get("drive_lane_offset_cm", 300.0)          # e.g., 3m
        self.DRIVE_WAYPOINT_SPACING_CM = self.cfg.get("drive_waypoint_spacing_cm", 5000)  # 50m
        self.agent_yaw = 0  
        self._last_agent_pos = None  

    # ---------------- Compatibility layer (forward to graph_full) ----------------
    @property
    def nodes(self):
        return self.graph_full.nodes

    @property
    def edges(self):
        return self.graph_full.edges

    @property
    def adjacency_list(self):
        return self.graph_full.adjacency_list

    @property
    def edge_meta(self):
        return self.graph_full.edge_meta

    def add_node(self, node: Node):
        self.graph_full.add_node(node)

    def add_edge(self, edge: Edge):
        self.graph_full.add_edge(edge)

    def _connected(self, a: Node, b: Node) -> bool:
        return self.graph_full.connected(a, b)

    def _edge_key(self, a: Node, b: Node) -> Tuple[str, str]:
        return self.graph_full._edge_key(a, b)

    def _set_edge_meta(self, a: Node, b: Node, meta: Dict[str, Any]):
        self.graph_full.set_edge_meta(a, b, meta)

    def _get_edge_meta(self, a: Node, b: Node) -> Optional[Dict[str, Any]]:
        return self.graph_full.get_edge_meta(a, b)

    # ---------------- Geometry helpers ----------------
    @staticmethod
    def _project_point_to_segment(p: Vector, a: Vector, b: Vector) -> Tuple[Vector, float]:
        """
        Project point p onto the segment a->b.

        Returns
        -------
        proj : Vector
            Clamped projection on the segment (t is clamped into [0, 1]).
        t : float
            Unclamped parametric coordinate along the infinite line a->b.
            This is useful to avoid misclassification when a projected point
            lies on the line extension instead of the actual segment.
        """
        ax, ay = float(a.x), float(a.y)
        bx, by = float(b.x), float(b.y)
        px, py = float(p.x), float(p.y)

        vx, vy = (bx - ax), (by - ay)
        den = vx * vx + vy * vy
        if den <= 1e-12:
            # Degenerate segment: a == b, return the endpoint directly.
            return Vector(ax, ay), 0.0

        t = ((px - ax) * vx + (py - ay) * vy) / den
        # Clamp to segment
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        proj = Vector(ax + vx * t, ay + vy * t)
        return proj, t

    @staticmethod
    def _rotate(vec: Vector, ang_rad: float) -> Vector:
        c, s = math.cos(ang_rad), math.sin(ang_rad)
        return Vector(vec.x * c - vec.y * s, vec.x * s + vec.y * c)

    # ---------------- Bounding box / snapping ----------------
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [float(n.position.x) for n in self.nodes]
        ys = [float(n.position.y) for n in self.nodes]
        if not xs or not ys:
            return (-1000, 1000, -1000, 1000)
        return (min(xs), max(xs), min(ys), max(ys))

    def snap_to_nearest_edge(
        self, x: float, y: float, use_segment: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Snap (x, y) to the nearest real edge (road/crosswalk/endcap), ignoring aux_* edges.

        Returns
        -------
        dict or None
            A dictionary containing:
            {
                'edge', 'a', 'b', 'proj', 'da', 'db', 't'
            }

        Snapping modes
        --------------
        - use_segment=False (default):
            primary sort key = distance to the infinite line,
            tie-breaker      = distance to the clamped segment.
        - use_segment=True:
            primary sort key = distance to the clamped segment,
            tie-breaker      = distance to the infinite line.

        The returned 'proj' follows the chosen mode (line vs. segment),
        while 't' always stores the raw parametric coordinate on the infinite line.
        """
        p = Vector(x, y)
        best = None
        best_key = None   # (primary, secondary)
        EPS = 1e-9

        for e in list(self.edges):
            a, b = e.node1, e.node2
            meta = self._get_edge_meta(a, b) or {}
            kind = (meta.get("kind") or "")
            if kind.startswith("aux_"):
                continue
            if kind not in ("road", "crosswalk", "endcap"):
                continue

            ax, ay = float(a.position.x), float(a.position.y)
            bx, by = float(b.position.x), float(b.position.y)
            px, py = float(p.x), float(p.y)

            vx, vy = (bx - ax), (by - ay)
            den = vx * vx + vy * vy
            if den <= 1e-12:
                continue  # Degenerate segment

            # Raw line parameter and projection (possibly outside the segment)
            t_raw = ((px - ax) * vx + (py - ay) * vy) / den
            proj_line = Vector(ax + vx * t_raw, ay + vy * t_raw)

            # Segment-clamped projection
            t_seg = max(0.0, min(1.0, t_raw))
            proj_seg = Vector(ax + vx * t_seg, ay + vy * t_seg)

            # Squared distances
            d_line2 = (px - proj_line.x) ** 2 + (py - proj_line.y) ** 2
            d_seg2 = (px - proj_seg.x) ** 2 + (py - proj_seg.y) ** 2

            # Primary / secondary key
            key = (d_seg2, d_line2) if use_segment else (d_line2, d_seg2)

            if (
                best_key is None
                or key[0] < best_key[0] - EPS
                or (abs(key[0] - best_key[0]) <= EPS and key[1] < best_key[1] - EPS)
            ):
                best_key = key
                chosen_proj = proj_seg if use_segment else proj_line
                best = {
                    "edge": e,
                    "a": a,
                    "b": b,
                    "proj": chosen_proj,           # projection consistent with the chosen mode
                    "da": a.position.distance(chosen_proj),
                    "db": b.position.distance(chosen_proj),
                    "t": float(t_raw),             # raw line parameter (unbounded)
                }
        return best

    def _snap_to_nearest_edge_in_graph(
        self,
        g: Graph,
        x: float,
        y: float,
        use_segment: bool = False,
        valid_kinds: Tuple[str, ...] = ("road", "crosswalk", "endcap", "drive"),
    ) -> Optional[Dict[str, Any]]:
        """
        Snap (x, y) to the nearest edge in a given graph `g`,
        optionally restricted to a set of valid edge kinds.
        """
        p = Vector(x, y)
        best, best_key = None, None
        EPS = 1e-9
        for e in list(g.edges):
            a, b = e.node1, e.node2
            meta = g.get_edge_meta(a, b) or {}
            kind = (meta.get("kind") or "")
            if kind.startswith("aux_"):
                continue
            if kind not in valid_kinds:
                continue
            ax, ay = float(a.position.x), float(a.position.y)
            bx, by = float(b.position.x), float(b.position.y)
            px, py = float(p.x), float(p.y)
            vx, vy = (bx - ax), (by - ay)
            den = vx * vx + vy * vy
            if den <= 1e-12:
                continue
            t_raw = ((px - ax) * vx + (py - ay) * vy) / den
            proj_line = Vector(ax + vx * t_raw, ay + vy * t_raw)
            t_seg = 0.0 if t_raw < 0.0 else (1.0 if t_raw > 1.0 else t_raw)
            proj_seg = Vector(ax + vx * t_seg, ay + vy * t_seg)
            d_line2 = (px - proj_line.x) ** 2 + (py - proj_line.y) ** 2
            d_seg2 = (px - proj_seg.x) ** 2 + (py - proj_seg.y) ** 2
            key = (d_seg2, d_line2) if use_segment else (d_line2, d_seg2)
            if (
                best_key is None
                or key[0] < best_key[0] - EPS
                or (abs(key[0] - best_key[0]) <= EPS and key[1] < best_key[1] - EPS)
            ):
                best_key = key
                chosen_proj = proj_seg if use_segment else proj_line
                best = {
                    "edge": e,
                    "a": a,
                    "b": b,
                    "proj": chosen_proj,
                    "da": a.position.distance(chosen_proj),
                    "db": b.position.distance(chosen_proj),
                    "t": float(t_raw),
                }
        return best

    # Keep the original public API but delegate to graph_full explicitly.
    def snap_to_nearest_edge(self, x: float, y: float, use_segment: bool = False):
        return self._snap_to_nearest_edge_in_graph(
            self.graph_full,
            x,
            y,
            use_segment,
            valid_kinds=("road", "crosswalk", "endcap"),
        )

    # ---------------- Type predicates ----------------
    @staticmethod
    def _t_of(nd: Node) -> str:
        return (getattr(nd, "type", "") or "").lower()

    def _is_roadpoint(self, nd: Node, include_docks: bool) -> bool:
        t = self._t_of(nd)
        if t in ("normal", "intersection"):
            if not include_docks and nd in self._dock_nodes:
                return False
            return True
        return False

    def _is_transparent(self, nd: Node) -> bool:
        t = self._t_of(nd)
        return (nd in self._dock_nodes) or (t in ("door", "bus_station", "charging_station"))

    # ---------------- Orientation / grouping (based on yaw) ----------------
    @staticmethod
    def _quantize_yaw_deg(yaw: float) -> float:
        """Quantize yaw into one of {0, 90, 180, 270} degrees."""
        a = ((float(yaw) % 360.0) + 360.0) % 360.0
        choices = [0.0, 90.0, 180.0, 270.0]
        return min(choices, key=lambda c: min(abs(c - a), 360.0 - abs(c - a)))

    @staticmethod
    def _unit_from_angle(angle_deg: float) -> Vector:
        """Global frame: 0→+x, 90→+y, 180/−180→−x, 270/−90→−y."""
        rad = math.radians(angle_deg)
        return Vector(math.cos(rad), math.sin(rad))

    @staticmethod
    def _edge_group(a: Node, b: Node, tol_deg: float = 6.0) -> Optional[str]:
        """
        Classify an edge into 'h' (horizontal) or 'v' (vertical).

        Returns
        -------
        'h'   : if the edge is within tol_deg of horizontal,
        'v'   : if it is within tol_deg of vertical,
        None  : otherwise.
        """
        vx, vy = (b.position.x - a.position.x), (b.position.y - a.position.y)
        ang = abs(_angle_deg(vx, vy))  # [0, 180]
        if min(ang, abs(ang - 180.0)) <= tol_deg:
            return "h"
        if abs(ang - 90.0) <= tol_deg:
            return "v"
        return None

    @staticmethod
    def _hv_group_from_yaw(yaw_q: float) -> str:
        """Quantized yaw: 0/180 → 'h', 90/270 → 'v'."""
        return "h" if yaw_q in (0.0, 180.0) else "v"

    # —— Snap to nearest edge with h/v grouping preference
    def _snap_to_nearest_edge_with_group(
        self, x: float, y: float, group: Optional[str], use_segment: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Snap (x, y) to the nearest real edge (road/crosswalk/endcap), ignoring aux_* edges.

        Behavior
        --------
        - If group is 'h' or 'v', we first restrict the search to edges that
          belong to the same group; if none is found, we fall back to all edges.
        - use_segment=False:
            primary sort key = distance to infinite line,
            tie-breaker      = distance to clamped segment.
        - use_segment=True:
            primary sort key = distance to clamped segment,
            tie-breaker      = distance to infinite line.

        Returns
        -------
        dict or None
            {
              'edge', 'a', 'b', 'proj', 'da', 'db', 't'
            }
        where 't' is the raw, unbounded line parameter t_raw, while 'proj'
        is consistent with the chosen snapping mode (line or segment).
        """

        def _best(prefer_group: Optional[str]) -> Optional[Dict[str, Any]]:
            p = Vector(x, y)
            best = None
            best_key = None  # (primary, secondary)
            EPS = 1e-9

            for e in list(self.edges):
                a, b = e.node1, e.node2
                meta = self._get_edge_meta(a, b) or {}
                kind = (meta.get("kind") or "")
                if kind.startswith("aux_"):
                    continue
                if kind not in ("road", "crosswalk", "endcap"):
                    continue

                # Filter by edge h/v group if requested
                g = self._edge_group(a, b)
                if prefer_group is not None and g != prefer_group:
                    continue

                # Compute line and segment projections
                ax, ay = float(a.position.x), float(a.position.y)
                bx, by = float(b.position.x), float(b.position.y)
                px, py = float(p.x), float(p.y)

                vx, vy = (bx - ax), (by - ay)
                den = vx * vx + vy * vy
                if den <= 1e-12:
                    continue  # degenerate

                t_raw = ((px - ax) * vx + (py - ay) * vy) / den
                proj_line = Vector(ax + vx * t_raw, ay + vy * t_raw)  # infinite line projection
                t_seg = 0.0 if t_raw < 0.0 else (1.0 if t_raw > 1.0 else t_raw)
                proj_seg = Vector(ax + vx * t_seg, ay + vy * t_seg)   # clamped segment projection

                d_line2 = (px - proj_line.x) ** 2 + (py - proj_line.y) ** 2
                d_seg2 = (px - proj_seg.x) ** 2 + (py - proj_seg.y) ** 2

                key = (d_seg2, d_line2) if use_segment else (d_line2, d_seg2)
                if (
                    best_key is None
                    or key[0] < best_key[0] - EPS
                    or (abs(key[0] - best_key[0]) <= EPS and key[1] < best_key[1] - EPS)
                ):
                    best_key = key
                    chosen_proj = proj_seg if use_segment else proj_line
                    best = {
                        "edge": e,
                        "a": a,
                        "b": b,
                        "proj": chosen_proj,
                        "da": a.position.distance(chosen_proj),
                        "db": b.position.distance(chosen_proj),
                        "t": float(t_raw),   # raw line parameter
                    }
            return best

        ret = _best(group)
        if ret is not None:
            return ret
        return _best(None)

    # —— Nearest skeleton road name (with h/v grouping preference)
    def _nearest_skeleton_road_name_by_group(self, px: float, py: float, group: Optional[str]) -> Optional[str]:
        p = Vector(px, py)
        best_name, best_d = None, None
        for e in self.graph_skel.edges:
            u, v = e.node1, e.node2
            meta = self.graph_skel.get_edge_meta(u, v) or {}
            if meta.get("kind") != "road":
                continue
            name = meta.get("name") or ""
            if not name:
                continue
            g = self._edge_group(u, v)
            if group is not None and g != group:
                continue
            proj, _ = self._project_point_to_segment(p, u.position, v.position)
            d = p.distance(proj)
            if (best_d is None) or (d < best_d - 1e-9):
                best_d, best_name = d, name
        # If no edge is found in the preferred group, fall back to all roads.
        if best_name is None and group is not None:
            return self._nearest_skeleton_road_name_by_group(px, py, None)
        return best_name

    # ---------------- Intersection helpers ----------------
    def _get_or_create_intersection(self, pos: Vector, eps_cm: float = 1.0) -> Node:
        """
        Return an existing 'intersection' node at `pos` within eps_cm,
        or create a new one if none exists.
        """
        for nd in self.nodes:
            if getattr(nd, "type", "") != "intersection":
                continue
            if nd.position.distance(pos) <= eps_cm:
                return nd
        n = Node(Vector(pos.x, pos.y), "intersection")
        self.add_node(n)
        return n

    # =========================================================
    #                Common helpers (factored out for reuse)
    # =========================================================
    def _nearest_node_at(self, xf: float, yf: float, tol_cm: float = 50.0) -> Optional[Node]:
        """
        Return the nearest node to (xf, yf) within tol_cm (Euclidean),
        based on squared distance in the full graph.
        """
        best, best_d2 = None, None
        tx, ty = float(xf), float(yf)
        for nd in self.nodes:
            dx, dy = tx - float(nd.position.x), ty - float(nd.position.y)
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2 - 1e-9:
                best_d2 = d2
                best = nd
        if best is not None and (best_d2 is not None) and best_d2 <= tol_cm * tol_cm:
            return best
        return None

    def _roles_at_xy(self, xf: float, yf: float, eps: float = 1.0) -> List[str]:
        """
        Return any order-related roles associated with the given coordinates,
        e.g., pickup/dropoff anchors for orders.
        """
        roles = []
        for rec in getattr(self, "order_meta", []):
            pu, do = rec.get("pickup_node"), rec.get("dropoff_node")
            if pu and abs(xf - pu.position.x) <= eps and abs(yf - pu.position.y) <= eps:
                roles.append(f"pick up address of order {rec['id']}")
            if do and abs(xf - do.position.x) <= eps and abs(yf - do.position.y) <= eps:
                roles.append(f"drop off address of order {rec['id']}")
        return sorted(set(roles))

    def _display_name_of(self, node: Node) -> str:
        """
        Prefer the node's stable display_name if present;
        otherwise fall back to its type.
        """
        name = getattr(node, "display_name", "").strip()
        if name:
            return name
        t = self._t_of(node) or "poi"
        return t

    def _road_name_for_node(self, nd: Node) -> str:
        """Return the node's own road_name (no inference).

        The stored ``road_name`` attribute carries the side-tagged internal
        name (e.g. ``"Maple St (left)"``); we strip the side suffix here so
        every caller of this method receives the clean user-facing form.
        """
        return self._display_road_name(getattr(nd, "road_name", "") or "")

    def _mk_item(
        self,
        kind: str,
        node: Node,
        dist_cm: float,
        x_cm: Optional[float] = None,
        y_cm: Optional[float] = None,
        force_type: Optional[str] = None,
        override_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Create a unified dictionary describing a node-like item
        (waypoint / poi / intersection in the S-set).
        """
        px = float(node.position.x) if x_cm is None else float(x_cm)
        py = float(node.position.y) if y_cm is None else float(y_cm)
        typ = force_type if force_type is not None else self._t_of(node)
        rn = self._road_name_for_node(node)
        item = {
            "kind": kind,                       # 'waypoint' / 'poi' / 'intersection' (for S)
            "id": str(node) if override_id is None else str(override_id),
            "type": typ,
            "name": self._display_name_of(node),
            "x": px,
            "y": py,
            "dist_cm": float(dist_cm),
            "is_dock": (node in self._dock_nodes),
            "roles": self._roles_at_xy(px, py),
            "label": "",
            "label_text": "",
            "road_name": rn,
            "_node": node,                      # later used to recompute distances via shortest paths
        }
        return item

    # =========================================================
    #                   1) Road network import + skeleton
    # =========================================================
    def import_roads(self, map_path: str):
        """
        Import the basic road network from roads.json:
        - Builds sidewalk intersections and roads on graph_full.
        - Builds driving lanes and their connectivity on graph_drive.
        - Connects adjacent roads and drive roads.
        - Builds the skeleton graph.
        - Interpolates waypoints and assigns road names to waypoints.
        """
        with open(map_path, 'r', encoding='utf-8') as f:
            roads_data = json.load(f)

        # Seed the per-map street-name shuffle. We derive the map name
        # from the path (e.g. ``.../maps/medium-city-22/roads.json``) so
        # the same scenario always produces the same names.
        map_dir = os.path.dirname(os.path.abspath(map_path))
        map_name = os.path.basename(map_dir) or "default"
        self._init_road_name_pool(map_name)

        roads = roads_data['roads']
        # One name per continuous street (collinear segments merged), instead of
        # a fresh name per roads.json segment.
        seg_names = self._street_names_by_segment(roads)
        for road_index, road in enumerate(roads):
            # meters → cm
            start = Vector(road['start']['x'] * 100.0, road['start']['y'] * 100.0)
            end = Vector(road['end']['x'] * 100.0, road['end']['y'] * 100.0)
            road_obj = Road(start, end)

            # Four sidewalk intersection points (two per side)
            normal_vector = Vector(road_obj.direction.y, -road_obj.direction.x)
            p1 = road_obj.start - normal_vector * self.cfg.get("sidewalk_offset_cm", 110.0) + road_obj.direction * self.cfg.get("sidewalk_offset_cm", 110.0)
            p2 = road_obj.end - normal_vector * self.cfg.get("sidewalk_offset_cm", 110.0) - road_obj.direction * self.cfg.get("sidewalk_offset_cm", 110.0)
            p3 = road_obj.end + normal_vector * self.cfg.get("sidewalk_offset_cm", 110.0) - road_obj.direction * self.cfg.get("sidewalk_offset_cm", 110.0)
            p4 = road_obj.start + normal_vector * self.cfg.get("sidewalk_offset_cm", 110.0) + road_obj.direction * self.cfg.get("sidewalk_offset_cm", 110.0)

            n1 = self._get_or_create_intersection(p1)
            n2 = self._get_or_create_intersection(p2)
            n3 = self._get_or_create_intersection(p3)
            n4 = self._get_or_create_intersection(p4)

            # === Driving lanes (separate from sidewalks), offset from the road centerline by DRIVE_LANE_OFFSET ===
            dirv = road_obj.direction
            nrm = Vector(dirv.y, -dirv.x)

            # Four lane corners (mirroring p1..p4, but with DRIVE_LANE_OFFSET)
            d1 = road_obj.start - nrm * self.DRIVE_LANE_OFFSET + dirv * self.DRIVE_LANE_OFFSET
            d2 = road_obj.end - nrm * self.DRIVE_LANE_OFFSET - dirv * self.DRIVE_LANE_OFFSET
            d3 = road_obj.end + nrm * self.DRIVE_LANE_OFFSET - dirv * self.DRIVE_LANE_OFFSET
            d4 = road_obj.start + nrm * self.DRIVE_LANE_OFFSET + dirv * self.DRIVE_LANE_OFFSET

            def _get_or_create_drive_node(pos: Vector, eps_cm: float = 1.0) -> Node:
                """
                Return an existing intersection node in the drive graph near `pos`,
                or create a new one as an 'intersection' in graph_drive.
                """
                for nd in self.graph_drive.nodes:
                    if nd.position.distance(pos) <= eps_cm:
                        return nd
                n = Node(Vector(pos.x, pos.y), "intersection")
                self.graph_drive.add_node(n)
                return n

            dn1 = _get_or_create_drive_node(d1)
            dn2 = _get_or_create_drive_node(d2)
            dn3 = _get_or_create_drive_node(d3)
            dn4 = _get_or_create_drive_node(d4)

            self._road_id += 1  # still unique per segment (internal drive-lane ids)
            display_name = seg_names.get(road_index) or (
                f"{self._pick_road_basename(self._road_id)} "
                f"{self._orientation_suffix(road_obj.start, road_obj.end)}"
            )
            # The "(left)" / "(right)" tag is an *internal* detail that
            # keeps the polyline / projection / parity logic working. The
            # agent never sees it — observation builders run every road
            # name through `_display_road_name` first.
            left_name = f"{display_name} (left)"
            right_name = f"{display_name} (right)"

            # Sidewalk edges along the road
            for (a, b, nm) in ((n1, n2, left_name), (n3, n4, right_name)):
                if a is b:
                    continue
                self.add_edge(Edge(a, b))
                self._set_edge_meta(a, b, {"name": nm, "kind": "road"})

            # Endcap edges (cross-sidewalk)
            for a, b in ((n1, n4), (n2, n3)):
                if a is b:
                    continue
                self.add_edge(Edge(a, b))
                self._set_edge_meta(a, b, {"name": "", "kind": "endcap"})

            # —— Longitudinal driving lanes: one-way
            self.graph_drive.add_edge(Edge(dn1, dn2))  # L lane: e.g., left-to-right / bottom-to-top
            self.graph_drive.set_edge_meta(
                dn1,
                dn2,
                {
                    "name": f"drive_{self._road_id}",
                    "kind": "drive",
                    "lane": "L",
                    "oneway": True,
                    "forward_only": True,  # only allow a->b
                },
            )

            self.graph_drive.add_edge(Edge(dn3, dn4))  # R lane: e.g., right-to-left / top-to-bottom
            self.graph_drive.set_edge_meta(
                dn3,
                dn4,
                {
                    "name": f"drive_{self._road_id}",
                    "kind": "drive",
                    "lane": "R",
                    "oneway": True,
                    "forward_only": True,  # only allow a->b
                },
            )

            # —— Cross-lane connectors at intersections: bidirectional
            self.graph_drive.add_edge(Edge(dn1, dn4))
            self.graph_drive.set_edge_meta(
                dn1,
                dn4,
                {
                    "name": f"drive_{self._road_id}",
                    "kind": "drive",
                    "lane": "X",
                    "oneway": False,
                },
            )

            self.graph_drive.add_edge(Edge(dn2, dn3))
            self.graph_drive.set_edge_meta(
                dn2,
                dn3,
                {
                    "name": f"drive_{self._road_id}",
                    "kind": "drive",
                    "lane": "X",
                    "oneway": False,
                },
            )

        self.connect_adjacent_roads()
        self.connect_adjacent_drive_roads()
        # Build the skeleton first, then insert waypoints, then assign waypoint road names
        # based on nearest skeleton roads.
        self._build_skeleton()
        self.interpolate_waypoints(spacing_cm=5000)  # 50m spacing (normal nodes only)
        self._assign_waypoint_road_names_from_skeleton()

        self._interpolate_drive_waypoints(spacing_cm=self.DRIVE_WAYPOINT_SPACING_CM)

    def connect_adjacent_roads(self, threshold_cm: Optional[float] = None):
        """
        Connect sidewalk intersections into crosswalks when they are close enough.
        """
        if threshold_cm is None:
            threshold_cm = self.cfg.get("sidewalk_offset_cm", 110.0) * 2.0 + 100.0
        inters = [n for n in self.nodes if getattr(n, "type", "") == "intersection"]
        n = len(inters)
        for i in range(n):
            ni = inters[i]
            for j in range(i + 1, n):
                nj = inters[j]
                if self._connected(ni, nj):
                    continue
                if ni.position.distance(nj.position) <= threshold_cm:
                    if ni is nj:
                        continue
                    self.add_edge(Edge(ni, nj))
                    self._xwalk_id += 1
                    self._set_edge_meta(
                        ni,
                        nj,
                        {"name": f"xwalk_{self._xwalk_id}", "kind": "crosswalk"},
                    )

    def connect_adjacent_drive_roads(self, threshold_cm: Optional[float] = None):
        """
        Connect driving-lane intersections at the same intersection (smaller threshold
        than sidewalks to avoid spurious connections).
        """
        # This builds local connectors between drive-lane endpoints around intersections.
        if threshold_cm is None:
            threshold_cm = self.cfg.get("drive_lane_offset_cm", 300.0) * 2.0 + 100.0

        inters = [n for n in self.graph_drive.nodes if getattr(n, "type", "") == "intersection"]
        n = len(inters)
        for i in range(n):
            ni = inters[i]
            for j in range(i + 1, n):
                nj = inters[j]
                if self.graph_drive.connected(ni, nj):
                    continue
                if ni.position.distance(nj.position) <= threshold_cm:
                    if ni is nj:
                        continue
                    self.graph_drive.add_edge(Edge(ni, nj))
                    self.graph_drive.set_edge_meta(
                        ni,
                        nj,
                        {"name": "", "kind": "drive", "lane": "X", "oneway": False},
                    )

    def interpolate_waypoints(self, spacing_cm: int = 1500):
        """
        Insert 'normal' nodes (waypoints) along all edges.

        Note
        ----
        Road names for these waypoints are NOT assigned here; they are later
        overwritten using nearest skeleton road names.
        """
        current_edges = list(self.edges)
        for edge in current_edges:
            a, b = edge.node1, edge.node2
            distance = getattr(edge, "weight", a.position.distance(b.position))
            num_points = int(distance // spacing_cm)
            if num_points <= 1:
                continue

            dx = b.position.x - a.position.x
            dy = b.position.y - a.position.y
            norm = (dx * dx + dy * dy) ** 0.5
            if norm <= 1e-9:
                continue
            ux, uy = dx / norm, dy / norm

            new_nodes = []
            for i in range(1, num_points):
                px = a.position.x + ux * (i * spacing_cm)
                py = a.position.y + uy * (i * spacing_cm)
                node = Node(Vector(px, py), "normal")
                self.add_node(node)
                new_nodes.append(node)

            old_meta = self._get_edge_meta(a, b)
            # Split the original edge into multiple segments
            if edge in self.edges:
                self.edges.remove(edge)
            while b in self.adjacency_list[a]:
                self.adjacency_list[a].remove(b)
            while a in self.adjacency_list[b]:
                self.adjacency_list[b].remove(a)
            if old_meta is not None:
                self.edge_meta.pop(self._edge_key(a, b), None)

            chain = [a] + new_nodes + [b]
            for u, v in zip(chain, chain[1:]):
                self.add_edge(Edge(u, v))
                if old_meta is not None:
                    self._set_edge_meta(u, v, old_meta)

    # ------ Metadata helpers ------
    @staticmethod
    def _ordinal(n: int) -> str:
        """Convert an integer into its English ordinal representation.

        Retained for back-compat / debugging; the road-name producer no
        longer calls this. See `_pick_road_basename` for the current
        US-style naming scheme.
        """
        if 10 <= (n % 100) <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    # ------ US-style street naming -------------------------------------
    def _init_road_name_pool(self, map_name: str) -> None:
        """Seed a per-map deterministic permutation of ``_STREET_NAME_POOL``.

        Called once at the top of :meth:`import_roads`. The shuffled pool
        is consumed in road-id order by :meth:`_pick_road_basename`.
        """
        seed = zlib.crc32(map_name.encode("utf-8"))
        rng = random.Random(seed)
        pool = list(_STREET_NAME_POOL)
        rng.shuffle(pool)
        self._road_name_pool: List[str] = pool

    def _pick_road_basename(self, road_id: int) -> str:
        """Return the basename ("Maple", "Church", ...) for ``road_id``.

        Cycles deterministically through the per-map shuffled pool. The
        pool currently contains 40 names; bundled scenarios top out at 30
        roads, so each road gets a unique basename in practice.
        """
        pool = getattr(self, "_road_name_pool", None) or list(_STREET_NAME_POOL)
        return pool[(road_id - 1) % len(pool)]

    def _street_names_by_segment(self, roads: List[Dict[str, Any]]) -> Dict[int, str]:
        """Assign one street name per *street*, not per road segment.

        ``roads.json`` often splits a single straight street into several
        collinear segments (e.g. at each cross-street). Naming each segment
        independently produced confusing results like "Poplar St" and
        "Church St" on the same continuous road. Here we group segments that
        share an orientation and lie on the same line (same perpendicular
        offset within a tolerance) and give the whole group ONE basename.

        Returns ``{segment_index: "<Base> St|Ave"}``.
        """
        TOL_CM = 400.0  # segments whose center offset matches within this = one street
        info: List[Tuple[bool, float]] = []
        for road in roads:
            sx = road["start"]["x"] * 100.0; sy = road["start"]["y"] * 100.0
            ex = road["end"]["x"] * 100.0;   ey = road["end"]["y"] * 100.0
            horiz = abs(ex - sx) >= abs(ey - sy)
            offset = (sy + ey) / 2.0 if horiz else (sx + ex) / 2.0
            info.append((horiz, offset))

        # Cluster collinear segments per orientation into groups.
        seg_group: Dict[int, Tuple[bool, float]] = {}
        for horiz in (True, False):
            items = sorted((off, i) for i, (h, off) in enumerate(info) if h == horiz)
            anchor: Optional[float] = None
            for off, i in items:
                if anchor is None or abs(off - anchor) > TOL_CM:
                    anchor = off
                seg_group[i] = (horiz, round(anchor, 1))

        # Deterministic basename per group, consumed in sorted group order.
        ranks = {gk: r for r, gk in enumerate(sorted(set(seg_group.values())))}
        names: Dict[int, str] = {}
        for i, (horiz, _off) in enumerate(info):
            base = self._pick_road_basename(ranks[seg_group[i]] + 1)
            names[i] = f"{base} {'St' if horiz else 'Ave'}"
        return names

    @staticmethod
    def _orientation_suffix(start: Vector, end: Vector) -> str:
        """Return ``"St"`` for horizontal-ish roads, ``"Ave"`` for vertical-ish.

        Mirrors the Manhattan / many-grid-city convention where one
        orientation is named "Streets" and the perpendicular orientation
        is named "Avenues". Falls back to "St" for perfectly diagonal
        segments (none of the bundled maps have those today).
        """
        dx = abs(end.x - start.x)
        dy = abs(end.y - start.y)
        return "St" if dx >= dy else "Ave"

    @staticmethod
    def _display_road_name(name: str) -> str:
        """Strip the trailing ``" (left)"`` / ``" (right)"`` from a road name.

        Sidewalk edges are stored internally with side-tagged names so the
        skeleton / polyline / projection logic can still group them; every
        observation builder pipes those names through this helper so the
        agent only ever sees the clean US-style form (``"Maple St"``).
        """
        if not name:
            return name
        return _SIDE_RE.sub("", name)

    # ------ Skeleton construction (intersection + merged edges) ------
    def _build_skeleton(self, include_crosswalks: bool = True):
        """
        Build a skeleton graph:
        - Nodes are deduplicated intersections;
        - Edges are merged full segments of roads/crosswalks between intersections.
        """
        inters = [nd for nd in self.nodes if getattr(nd, "type", "") == "intersection"]

        def kxy(nd: Node) -> Tuple[int, int]:
            return (int(round(nd.position.x)), int(round(nd.position.y)))

        key2canon: Dict[Tuple[int, int], Node] = {}
        for nd in inters:
            key = kxy(nd)
            if key not in key2canon:
                key2canon[key] = nd

        seen = set()
        edges_out: List[Tuple[Node, Node, str, str]] = []

        def walk(src: Node, nb: Node, kind: str, name: str) -> Optional[Tuple[Node, float]]:
            """
            Walk along edges of type `kind` (and matching `name` for roads),
            starting from src→nb, until hitting the next intersection or a dead end.

            Returns
            -------
            (terminal_intersection, length_cm) or None
            """
            prev, cur = src, nb
            length = src.position.distance(nb.position)
            while True:
                if getattr(cur, "type", "") == "intersection":
                    return cur, float(length)
                nxt = None
                step = 0.0
                for v in self.adjacency_list.get(cur, []):
                    if v is prev:
                        continue
                    meta = self._get_edge_meta(cur, v)
                    if not meta:
                        continue
                    if meta.get("kind") != kind:
                        continue
                    if kind == "road" and (meta.get("name") or "") != name:
                        continue
                    nxt = v
                    step = cur.position.distance(v.position)
                    break
                if nxt is None:
                    return None
                length += step
                prev, cur = cur, nxt

        for s in inters:
            s_c = key2canon[kxy(s)]
            for nb in self.adjacency_list.get(s, []):
                meta = self._get_edge_meta(s, nb) or {}
                kind = meta.get("kind", "")
                name = (meta.get("name") or "") if kind in ("road", "crosswalk") else ""
                if kind == "road":
                    if not name:
                        continue
                elif kind == "crosswalk":
                    if not include_crosswalks:
                        continue
                else:
                    continue

                walked = walk(s, nb, kind, name)
                if not walked:
                    continue
                t, _ = walked
                t_c = key2canon[kxy(t)]
                if s_c is t_c:
                    continue
                key = (min(id(s_c), id(t_c)), max(id(s_c), id(t_c)), kind, name)
                if key in seen:
                    continue
                seen.add(key)
                edges_out.append((s_c, t_c, kind, name))

        self.graph_skel = Graph()
        skel_nodes_seen = set()
        for u, v, kind, name in edges_out:
            if u not in skel_nodes_seen:
                self.graph_skel.add_node(u)
                skel_nodes_seen.add(u)
            if v not in skel_nodes_seen:
                self.graph_skel.add_node(v)
                skel_nodes_seen.add(v)
            self.graph_skel.add_edge(Edge(u, v))
            self.graph_skel.set_edge_meta(u, v, {"kind": kind, "name": name or ""})

    # === Waypoint naming: assign waypoint road_name from nearest skeleton road (ignore h/v) ===
    def _assign_waypoint_road_names_from_skeleton(self):
        """
        Assign road_name to 'normal' nodes (waypoints) using the nearest
        skeleton road. Dock and intersection nodes are skipped to avoid
        ambiguous or misleading names.
        """
        for nd in self.nodes:
            t = (getattr(nd, "type", "") or "").lower()
            if t != "normal":
                # Do not assign road_name to intersections, doors, etc. here.
                continue
            if nd in self._dock_nodes:
                continue
            best_name, best_d = "", None
            p = nd.position
            for e in self.graph_skel.edges:
                u, v = e.node1, e.node2
                meta = self.graph_skel.get_edge_meta(u, v) or {}
                if meta.get("kind") != "road":
                    continue
                name = meta.get("name") or ""
                if not name:
                    continue
                proj, _ = self._project_point_to_segment(p, u.position, v.position)
                d = p.distance(proj)
                if (best_d is None) or (d < best_d - 1e-9):
                    best_d, best_name = d, name
            nd.road_name = best_name or ""

    # === Dock orthogonalization and aux_* tagging ===
    def _ensure_orthogonal_dock(self, base_edge: Edge, proj: Vector, t: float) -> Node:
        """
        Ensure a valid dock node at projection `proj` onto base_edge.

        Behavior
        --------
        - If t in [0, 1]: create or reuse an on-edge node at `proj`
          (via ensure_node_on_edge_at). The resulting edges are part of
          the main graph.
        - If t < 0 or t > 1: create an extension node on the line continuation
          and connect it to the closer endpoint with an aux_ext edge.
        """
        a, b = base_edge.node1, base_edge.node2
        if 0.0 <= t <= 1.0:
            nd = self.graph_full.ensure_node_on_edge_at(base_edge, proj, snap_cm=120.0)
            return nd

        # t is outside [0, 1]: create an extended dock point
        end = a if t < 0.0 else b
        dock = Node(Vector(proj.x, proj.y), "normal")
        self.add_node(dock)
        if not self._connected(end, dock):
            self.add_edge(Edge(end, dock))
            self._set_edge_meta(end, dock, {"name": "", "kind": "aux_ext"})
        return dock

    # —— Door position on a building rectangle boundary, given outward yaw
    @staticmethod
    def _door_point_from_yaw(
        center: Vector,
        bbox_w: float,
        bbox_h: float,
        rect_yaw_deg: float,
        outward_yaw_deg: float,
    ) -> Vector:
        """
        Compute the world-space door position on a building rectangle.

        Parameters
        ----------
        center : Vector
            Center of the rectangle in world coordinates.
        bbox_w, bbox_h : float
            Rectangle width and height (world units).
        rect_yaw_deg : float
            Rectangle orientation in degrees (world frame).
        outward_yaw_deg : float
            Desired outward-facing normal direction in degrees (world frame).

        Returns
        -------
        Vector
            Door position at the intersection of the outward ray with
            the rectangle boundary.
        """
        half_w, half_h = 0.5 * bbox_w, 0.5 * bbox_h
        # World direction → local frame
        d_world = Map._unit_from_angle(outward_yaw_deg)
        yaw_rad = math.radians(rect_yaw_deg)
        c, s = math.cos(-yaw_rad), math.sin(-yaw_rad)
        dx = d_world.x * c - d_world.y * s
        dy = d_world.x * s + d_world.y * c
        eps = 1e-9
        # Determine which side is hit
        tx = (half_w / abs(dx)) if abs(dx) > eps else float("inf")
        ty = (half_h / abs(dy)) if abs(dy) > eps else float("inf")
        if tx < ty:
            # Hit left/right edges (x = ±half_w)
            x_local = math.copysign(half_w, dx)
            y_local = (dy / max(abs(dx), eps)) * x_local
        else:
            # Hit top/bottom edges (y = ±half_h)
            y_local = math.copysign(half_h, dy)
            x_local = (dx / max(abs(dy), eps)) * y_local
        # Back to world frame
        c2, s2 = math.cos(yaw_rad), math.sin(yaw_rad)
        wx = x_local * c2 - y_local * s2
        wy = x_local * s2 + y_local * c2
        return Vector(center.x + wx, center.y + wy)

    # =========================================================
    #          2) One-shot POI / building import (yaw only)
    # =========================================================
    def _find_existing_node_at(self, pos: Vector, eps_cm: float = 80.0) -> Optional[Node]:
        """Delegate to graph_full for consistency."""
        return self.graph_full.find_existing_node_at(pos, eps_cm)

    def import_pois(self, world_json: str):
        """
        Import POIs and buildings from world_json.

        Rules
        -----
        - Building-like objects (restaurant/store/rest_area/hospital/car_rental/
          customer/building/BP_Building_*) are anchored by their door position
          in world space. The door position is used for snapping and road name
          retrieval (nearest skeleton road), and buildings still respect h/v
          grouping when snapping to roads.
        - Point-like POIs (charging_station/bus_station) use their own locations
          and are connected by aux_perp to the nearest dock node.

        Stable indexing
        ---------------
        A stable display_name is assigned for each POI type, e.g., "restaurant 2".
        """
        DOOR_YAW_OFFSET_DEG = 90.0  # door outward direction offset in degrees

        with open(world_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        building_like = {
            "restaurant",
            "store",
            "rest_area",
            "hospital",
            "car_rental",
            "customer",
            "building",
        }
        point_like = {"charging_station", "bus_station"}

        nodes = data.get("nodes", [])
        # Render-only data for the offline capture tools (see note on the attr).
        self.traffic_lights = load_traffic_lights(nodes)

        for obj in nodes:
            props = obj.get("properties", {}) or {}
            inst = str(obj.get("instance_name", "") or "")
            loc = props.get("location", {}) or {}
            ori = props.get("orientation", {}) or {}
            bbox = props.get("bbox", {}) or {}
            pt = (props.get("poi_type") or props.get("type") or "").strip().lower()

            # Classify POI type
            if pt in building_like:
                tname = pt
                is_building = True
            elif inst.startswith("BP_Building"):
                tname = "building"
                is_building = True
            elif pt in point_like:
                tname = pt
                is_building = False
            else:
                continue

            cx, cy = float(loc.get("x", 0.0)), float(loc.get("y", 0.0))
            center = Vector(cx, cy)

            # ======= Building-like POIs: determine door position and road name =======
            if is_building:
                # Quantize yaw for grouping; grouping itself does not include the 90° offset.
                yaw_deg_raw = float((ori.get("yaw", 0.0) or 0.0))
                yaw_q = self._quantize_yaw_deg(yaw_deg_raw)
                group = self._hv_group_from_yaw(yaw_q)

                # Compute the door position (only the door uses the 90° outward offset).
                door_pos = None
                w = float(bbox.get("x", 0.0)) * 1.2  # slightly enlarge the rectangle
                h = float(bbox.get("y", 0.0)) * 1.2
                yaw_rect = float(ori.get("yaw", 0.0) or 0.0)
                if w > 0 and h > 0:
                    outward = (yaw_q + DOOR_YAW_OFFSET_DEG) % 360.0
                    door_pos = self._door_point_from_yaw(center, w, h, yaw_rect, outward)

                # Reference point for snapping and road_name: prefer door, fall back to center
                ax = (door_pos.x if door_pos else center.x)
                ay = (door_pos.y if door_pos else center.y)

                # Nearest skeleton road name (prefer same h/v group)
                road_name = self._nearest_skeleton_road_name_by_group(ax, ay, group)

                # POI node at building center
                poi_node = Node(center, type=tname)
                poi_node.road_name = road_name
                seq = self._poi_seq_counter.get(tname, 0) + 1
                self._poi_seq_counter[tname] = seq
                poi_node.display_name = f"{tname} {seq}"
                poi_node.name = poi_node.display_name
                self.pois.append(poi_node)

                # Snap to nearest real edge (prefer same group, prefer road-like edges)
                best = self._snap_to_nearest_edge_with_group(ax, ay, group)
                dock_node = None
                if best is not None:
                    proj = best["proj"]
                    t = best.get("t", 0.0)
                    dock_node = self._ensure_orthogonal_dock(best["edge"], proj, t)
                    self._dock_nodes.add(dock_node)
                    dock_node.road_name = road_name

                # Create door node and perpendicular link (aux_perp)
                door_node = None
                if dock_node is not None and door_pos is not None:
                    door_node = Node(door_pos, "door")
                    if door_node not in self.nodes:
                        self.add_node(door_node)
                    if not self._connected(dock_node, door_node):
                        self.add_edge(Edge(dock_node, door_node))
                        self._set_edge_meta(dock_node, door_node, {"name": "", "kind": "aux_perp"})
                    door_node.road_name = road_name

                if door_node is not None:
                    self._door2poi[door_node] = poi_node

                # Metadata for visualization (building bounding box)
                building_box = (
                    {
                        "x": cx,
                        "y": cy,
                        "w": float(bbox.get("x", 0.0)) * 1.2,
                        "h": float(bbox.get("y", 0.0)) * 1.2,
                        "yaw": float(ori.get("yaw", 0.0)),
                        "poi_type": tname,
                    }
                    if bbox
                    else None
                )

                self.poi_meta.append(
                    {
                        "node": poi_node,
                        "dock_node": dock_node,
                        "door_node": door_node,
                        "road_name": road_name,
                        "center": (cx, cy),
                        "building_box": building_box,
                    }
                )
                continue  # handled building branch

            # ======= Point-like POIs: charging/bus stations =======
            # Use nearest skeleton road name (no h/v grouping)
            road_name = self._nearest_skeleton_road_name_by_group(cx, cy, None)

            poi_node = Node(center, type=tname)
            poi_node.road_name = road_name
            seq = self._poi_seq_counter.get(tname, 0) + 1
            self._poi_seq_counter[tname] = seq
            poi_node.display_name = f"{tname} {seq}"
            poi_node.name = poi_node.display_name
            self.pois.append(poi_node)

            # Snap to nearest real edge (plain nearest)
            best = self.snap_to_nearest_edge(cx, cy)
            dock_node = None
            if best is not None:
                proj = best["proj"]
                t = best.get("t", 0.0)
                dock_node = self._ensure_orthogonal_dock(best["edge"], proj, t)
                self._dock_nodes.add(dock_node)
                dock_node.road_name = road_name

            # Connect POI itself to dock via aux_perp
            if dock_node is not None:
                self.add_node(poi_node)
                if poi_node.position.distance(dock_node.position) > 1e-6:
                    self.add_edge(Edge(dock_node, poi_node))
                    self._set_edge_meta(dock_node, poi_node, {"name": "", "kind": "aux_perp"})

            self.poi_meta.append(
                {
                    "node": poi_node,
                    "dock_node": dock_node,
                    "door_node": None,
                    "road_name": road_name,
                    "center": (cx, cy),
                    "building_box": None,
                }
            )

        # After every POI has been ingested, derive deterministic
        # meters-along-road addresses + populate self.address_book.
        self._build_addresses()

        # Build a coarse "waypoint graph" whose nodes are intersections
        # and POI docks. This is what the agent perceives when stepping
        # block-by-block via STEP_TO. MOVE(x, y) is unaffected and still
        # uses the underlying full graph.
        self._build_waypoint_graph()

    # =========================================================
    #                   2.5) Human-readable POI addresses
    # =========================================================
    def _build_road_polylines(self) -> Dict[str, List[Node]]:
        """
        For every named road, build an ordered chain of skeleton nodes
        that represents the road's polyline.

        The skeleton already collapses each road into intersection-to-
        intersection segments; here we just stitch those segments back
        together. For branched roads (rare in synthetic grid maps) we
        walk the longest simple path; remaining branches still get
        sensible — though not visually intuitive — meters-along values
        because every POI is projected onto the chosen polyline.

        Returns
        -------
        Dict[str, List[Node]]
            Map from road name to a list of skeleton nodes, ordered from
            one endpoint to the other. The endpoint with the smaller
            (x, y) is used as the start to guarantee determinism.
        """
        edges_by_road: Dict[str, List[Tuple[Node, Node]]] = defaultdict(list)
        for e in self.graph_skel.edges:
            meta = self.graph_skel.get_edge_meta(e.node1, e.node2) or {}
            if meta.get("kind") != "road":
                continue
            name = meta.get("name") or ""
            if not name:
                continue
            edges_by_road[name].append((e.node1, e.node2))

        polylines: Dict[str, List[Node]] = {}
        for name, edges in edges_by_road.items():
            adj: Dict[Node, List[Node]] = defaultdict(list)
            for u, v in edges:
                if v not in adj[u]:
                    adj[u].append(v)
                if u not in adj[v]:
                    adj[v].append(u)

            # Endpoints = degree-1 nodes; if none (cycle / branching only),
            # fall back to any node in the sub-graph.
            endpoints = [n for n, nb in adj.items() if len(nb) == 1]
            if not endpoints:
                endpoints = list(adj.keys())
            if not endpoints:
                continue

            start = min(endpoints, key=lambda n: (n.position.x, n.position.y))

            # Walk a simple path from `start`. At each step prefer an
            # unvisited neighbour; if multiple, pick the one whose
            # heading is most colinear with the current heading so the
            # polyline stays straight on grid maps. This keeps
            # meters-along monotone along the visual road.
            polyline: List[Node] = [start]
            visited: Set[Node] = {start}
            prev: Optional[Node] = None
            cur: Node = start
            while True:
                nxts = [n for n in adj[cur] if n not in visited]
                if not nxts:
                    break
                if prev is None or len(nxts) == 1:
                    nxt = nxts[0]
                else:
                    # Continue most-colinear direction.
                    px, py = prev.position.x, prev.position.y
                    cx, cy = cur.position.x, cur.position.y
                    hx, hy = (cx - px), (cy - py)
                    hn = math.hypot(hx, hy) or 1.0
                    hx, hy = hx / hn, hy / hn

                    def colinearity(n: Node) -> float:
                        dx, dy = (n.position.x - cx), (n.position.y - cy)
                        dn = math.hypot(dx, dy) or 1.0
                        return (dx * hx + dy * hy) / dn

                    nxt = max(nxts, key=colinearity)
                polyline.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt

            polylines[name] = polyline
        return polylines

    @staticmethod
    def _project_on_polyline(
        p: Vector,
        polyline: List[Node],
    ) -> Tuple[float, str, float]:
        """
        Project ``p`` onto the polyline formed by ``polyline``.

        Parameters
        ----------
        p : Vector
            World-space point to project (in cm).
        polyline : List[Node]
            Ordered chain of skeleton nodes.

        Returns
        -------
        Tuple[float, str, float]
            ``(meters_along, side, perp_dist_cm)`` where
            - ``meters_along`` is the distance, in METERS, from
              ``polyline[0]`` to the foot of perpendicular along the
              polyline,
            - ``side`` is ``"left"`` or ``"right"`` of the polyline's
              forward direction at the closest segment,
            - ``perp_dist_cm`` is the perpendicular distance from ``p``
              to the polyline (cm), useful for diagnostics.
        """
        if len(polyline) < 2:
            return 0.0, "right", 0.0

        best: Optional[Tuple[float, float, str]] = None  # (perp_cm, m_along_cm, side)
        cum_cm = 0.0
        for i in range(len(polyline) - 1):
            a = polyline[i].position
            b = polyline[i + 1].position
            abx, aby = (b.x - a.x), (b.y - a.y)
            seg_len2 = abx * abx + aby * aby
            if seg_len2 <= 1e-9:
                continue
            seg_len = math.sqrt(seg_len2)
            apx, apy = (p.x - a.x), (p.y - a.y)
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / seg_len2))
            fx, fy = (a.x + abx * t), (a.y + aby * t)
            perp_cm = math.hypot(p.x - fx, p.y - fy)
            m_along_cm = cum_cm + t * seg_len
            # Cross product (b-a) x (p-a) > 0 => p is to the LEFT of a→b.
            cross = abx * apy - aby * apx
            side = "left" if cross > 0 else "right"
            if best is None or perp_cm < best[0] - 1e-9:
                best = (perp_cm, m_along_cm, side)
            cum_cm += seg_len

        if best is None:
            return 0.0, "right", 0.0
        return best[1] / 100.0, best[2], best[0]

    def _build_addresses(self):
        """
        Assign a deterministic, human-readable address to every POI.

        For each POI we project its ``dock_node`` (the point on the road
        where the building's footpath meets the street) onto the road's
        polyline. The integer meter offset becomes the house number,
        with parity by side of the road (right=even, left=odd — a common
        US convention). Collisions are resolved by bumping by ±2 so
        parity is preserved.

        Side effects
        ------------
        - Adds keys to each ``poi_meta`` entry:
          ``address`` (str), ``meters_along`` (float, meters),
          ``road_side`` (``"left"`` | ``"right"``),
          ``kind`` (POI type string).
        - Stamps ``address`` on the POI node, dock node, and door node
          (if any).
        - Populates ``self.address_book`` so callers can resolve either
          the address string ("124 Pine St") or the legacy display name
          ("restaurant 3") back to the POI's dock node.
        """
        self.address_book = {}
        polylines = self._build_road_polylines()

        # Process POIs in a deterministic order so address assignment is
        # reproducible across runs (sort by kind then display_name).
        def _sort_key(meta: Dict[str, Any]) -> Tuple[str, str]:
            node = meta.get("node")
            kind = (getattr(node, "type", "") or "").lower()
            disp = getattr(node, "display_name", "") or ""
            return (kind, disp)

        for meta in sorted(self.poi_meta, key=_sort_key):
            dock = meta.get("dock_node")
            road_name = meta.get("road_name") or ""
            if dock is None or not road_name:
                meta["address"] = ""
                meta["kind"] = (getattr(meta.get("node"), "type", "") or "").lower()
                continue

            polyline = polylines.get(road_name)
            if not polyline or len(polyline) < 2:
                meta["address"] = ""
                meta["kind"] = (getattr(meta.get("node"), "type", "") or "").lower()
                continue

            m_along, side, _ = self._project_on_polyline(dock.position, polyline)

            # Integer house number with parity by side (right=even, left=odd).
            h = max(1, int(round(m_along)))
            desired_parity = 0 if side == "right" else 1
            if (h % 2) != desired_parity:
                h += 1

            # Use the clean user-facing road name ("Maple St") for the
            # displayed address; parity already encodes the sidewalk.
            display_road = self._display_road_name(road_name)

            address = f"{h} {display_road}"
            while address in self.address_book:
                h += 2
                address = f"{h} {display_road}"

            meta["address"] = address
            meta["meters_along"] = m_along
            meta["road_side"] = side
            meta["kind"] = (getattr(meta.get("node"), "type", "") or "").lower()

            # Stamp the address onto the POI / dock / door nodes.
            for n in (meta.get("node"), dock, meta.get("door_node")):
                if n is not None:
                    n.address = address

            self.address_book[address] = dock
            disp = getattr(meta.get("node"), "display_name", None)
            if disp:
                self.address_book[disp] = dock

    def _road_node_address(self, node: Node, polylines: Dict[str, List[Node]]) -> str:
        """Meters-along-road house number for a plain road/intersection node.

        Lets a waypoint that sits on a single street read like "120 River Ave"
        instead of the bare street name. Mirrors the dock addressing in
        :meth:`_build_addresses` (right=even / left=odd parity, collision bump)
        and registers the result in ``address_book``.
        """
        raw = ""
        for nb in self.graph_skel.adjacency_list.get(node, []):
            em = self.graph_skel.get_edge_meta(node, nb) or {}
            if em.get("kind") == "road" and em.get("name"):
                raw = em["name"]
                break
        if not raw:
            return ""
        polyline = polylines.get(raw)
        if not polyline or len(polyline) < 2:
            return ""
        m_along, side, _ = self._project_on_polyline(node.position, polyline)
        h = max(1, int(round(m_along)))
        desired_parity = 0 if side == "right" else 1
        if (h % 2) != desired_parity:
            h += 1
        disp = self._display_road_name(raw)
        address = f"{h} {disp}"
        while address in self.address_book:
            h += 2
            address = f"{h} {disp}"
        return address

    # =========================================================
    #             2.6) Waypoint graph (intersections ∪ docks)
    # =========================================================
    def _build_waypoint_graph(self):
        """
        Build ``self.waypoint_graph`` whose nodes are the named places
        an agent can stand "outside" of, in the real-world sense:

        - **Intersection waypoints** — every node in ``graph_skel`` that
          is reachable by at least one named road; given id ``int_<n>``
          and name ``"<road_a> & <road_b>"`` (sorted by road name).
        - **POI dock waypoints** — every POI's ``dock_node``; given id
          ``dock_<n>`` and name = the POI's street address.

        Two waypoints are connected by an edge in the waypoint graph
        when you can walk between them along the underlying road graph
        without passing any *other* waypoint. The edge stores the real
        walking distance (cm), the compass bearing from u to v, and
        the road name (when known).

        This graph is what the agent perceives via STEP_TO. The full
        graph (with 50 m mid-block waypoints, sidewalks, crosswalks,
        and dock-to-door perpendiculars) is left untouched and still
        used by MOVE(x, y) for the original routed movement.
        """
        self.waypoint_graph = Graph()
        self.waypoints_by_id = {}
        self._waypoint_id_by_node = {}

        # ---- 1) Collect waypoint nodes ----
        intersections: List[Node] = [
            n for n in self.graph_skel.nodes
            if (getattr(n, "type", "") or "").lower() == "intersection"
        ]
        # Some skeleton "intersections" may actually be terminal (degree 1
        # in graph_skel). We still include them — they're the start/end
        # of a road and a courier can stand there.
        docks: List[Node] = list(self._dock_nodes)

        # ---- 2) Assign IDs deterministically ----
        # Intersections: sort by (x, y) for stability across runs.
        intersections_sorted = sorted(
            intersections,
            key=lambda n: (round(n.position.x, 2), round(n.position.y, 2)),
        )
        # Polylines (per street) for meters-along house numbers on single-road nodes.
        polylines = self._build_road_polylines()
        # Build intersection names from incident skeleton road names.
        for i, n in enumerate(intersections_sorted):
            wp_id = f"int_{i}"
            road_names: Set[str] = set()
            for nb in self.graph_skel.adjacency_list.get(n, []):
                em = self.graph_skel.get_edge_meta(n, nb) or {}
                if em.get("kind") != "road":
                    continue
                rn = self._display_road_name(em.get("name") or "")
                if rn:
                    road_names.add(rn)
            if len(road_names) >= 2:
                # A genuine crossing of two+ streets: name it by the cross streets.
                wp_name = " & ".join(sorted(road_names))
            else:
                # On a single street (or an un-named node): give a real house
                # number so the agent reads "120 River Ave", not bare "River Ave".
                wp_name = (
                    self._road_node_address(n, polylines)
                    or (next(iter(road_names)) if road_names else wp_id)
                )
            n.address = wp_name
            self.waypoint_graph.add_node(n)
            n.waypoint_id = wp_id
            n.waypoint_name = wp_name
            n.waypoint_kind = "intersection"
            self.waypoints_by_id[wp_id] = n
            self._waypoint_id_by_node[n] = wp_id
            self.address_book[wp_id] = n
            self.address_book[wp_name] = n

        # Docks: sort by (address) for stability — addresses are
        # already deterministic.
        dock_to_meta: Dict[Node, Dict[str, Any]] = {
            meta["dock_node"]: meta
            for meta in self.poi_meta
            if meta.get("dock_node") is not None
        }
        docks_sorted = sorted(
            (d for d in docks if d in dock_to_meta),
            key=lambda d: dock_to_meta[d].get("address", ""),
        )
        for i, n in enumerate(docks_sorted):
            wp_id = f"dock_{i}"
            meta = dock_to_meta[n]
            wp_name = meta.get("address") or wp_id
            self.waypoint_graph.add_node(n)
            n.waypoint_id = wp_id
            n.waypoint_name = wp_name
            n.waypoint_kind = "dock"
            n.poi_kind = meta.get("kind", "")
            self.waypoints_by_id[wp_id] = n
            self._waypoint_id_by_node[n] = wp_id
            self.address_book[wp_id] = n

            # Also tag the POI's main node and door node with the
            # dock's waypoint id so callers (e.g. Order.to_text) can
            # render "{addr} [{dock_id}]" without knowing about the
            # door↔dock relationship.
            for shadow in (meta.get("node"), meta.get("door_node")):
                if shadow is not None and shadow is not n:
                    if not getattr(shadow, "waypoint_id", ""):
                        shadow.waypoint_id = wp_id
                    if not getattr(shadow, "waypoint_kind", ""):
                        shadow.waypoint_kind = "dock"
                    if not getattr(shadow, "poi_kind", ""):
                        shadow.poi_kind = meta.get("kind", "")

        waypoint_set: Set[Node] = set(self.waypoints_by_id.values())

        # ---- 3) Collapse graph_full edges into adjacencies ----
        # From each waypoint, walk the road graph along all real
        # ground-level edges, skipping intermediate interpolated
        # (non-waypoint) nodes, until we hit the next waypoint. The
        # following edge kinds are traversable:
        #   - "road"      : street segment between waypoints
        #   - "endcap"    : short bridge between adjacent intersections
        #                   that share a corner of two perpendicular roads
        #   - "crosswalk" : crossing the street at an intersection
        #   - "aux_perp"  : dock → door / dock → point-POI (dead-end,
        #                   harmless: door isn't a waypoint so the walk
        #                   along this branch just returns None)
        #   - "aux_ext"   : off-edge dock extension (same dead-end logic)
        VALID_KINDS = {"road", "endcap", "crosswalk", "aux_perp", "aux_ext"}

        def _walk_until_waypoint(
            src: Node, first_step: Node
        ) -> Optional[Tuple[Node, float, str, Set[str]]]:
            """
            Walk along graph_full starting with src→first_step. Skip
            non-waypoint nodes. Return (terminal_waypoint, dist_cm,
            road_name) or None if no terminal is found.
            """
            prev: Node = src
            cur: Node = first_step
            total = src.position.distance(first_step.position)
            road_name = ""
            crossing_kinds: Set[str] = set()
            first_meta = self._get_edge_meta(src, first_step) or {}
            if first_meta.get("kind") == "road":
                road_name = first_meta.get("name") or ""
            if first_meta.get("kind") in {"endcap", "crosswalk"}:
                crossing_kinds.add(str(first_meta.get("kind")))

            visited = {src, first_step}
            while True:
                if cur in waypoint_set:
                    return cur, float(total), road_name, crossing_kinds
                # Choose the next graph_full neighbour to continue.
                nxt = None
                step_d = 0.0
                step_road = ""
                for v in self.adjacency_list.get(cur, []):
                    if v is prev or v in visited:
                        continue
                    em = self._get_edge_meta(cur, v) or {}
                    k = em.get("kind", "")
                    if k not in VALID_KINDS:
                        continue
                    nxt = v
                    step_d = cur.position.distance(v.position)
                    if k == "road":
                        step_road = em.get("name") or step_road
                    if k in {"endcap", "crosswalk"}:
                        crossing_kinds.add(str(k))
                    break
                if nxt is None:
                    return None
                total += step_d
                if step_road and not road_name:
                    road_name = step_road
                visited.add(nxt)
                prev, cur = cur, nxt

        # Collect edges, deduped by (a, b, road_name) — keep the shortest
        # walk between a pair (different paths along the same road can
        # appear when the road forks at intersections).
        edge_best: Dict[Tuple[int, int, str], Tuple[Node, Node, float, Set[str]]] = {}
        for w in self.waypoints_by_id.values():
            for nb in self.adjacency_list.get(w, []):
                em = self._get_edge_meta(w, nb) or {}
                if em.get("kind") not in VALID_KINDS:
                    continue
                walked = _walk_until_waypoint(w, nb)
                if walked is None:
                    continue
                end, dist_cm, road_name, crossing_kinds = walked
                if end is w:
                    continue
                key = (
                    min(id(w), id(end)),
                    max(id(w), id(end)),
                    road_name or "",
                )
                cur = edge_best.get(key)
                if cur is None or dist_cm < cur[2] - 1e-6:
                    edge_best[key] = (w, end, dist_cm, set(crossing_kinds))

        for (w, e, dist_cm, crossing_kinds) in edge_best.values():
            self.waypoint_graph.add_edge(Edge(w, e))
            bearing = self._bearing_deg(w.position, e.position)
            legal_move = self._is_cardinal_waypoint_bearing(bearing)
            # Pure geometry: does this MOVE-legal edge cross a vehicle road? The
            # red/green *state* is decided at runtime by TrafficController, not
            # here, so the map carries no signal dict.
            crosses_vehicle_road = bool(crossing_kinds and legal_move)
            self.waypoint_graph.set_edge_meta(
                w,
                e,
                {
                    "dist_cm": float(dist_cm),
                    "bearing_deg": float(bearing),
                    "road_name": self._infer_edge_road_name(w, e),
                    "legal_move": legal_move,
                    "crosses_vehicle_road": crosses_vehicle_road,
                    "crossing_kinds": sorted(crossing_kinds) if crosses_vehicle_road else [],
                    "traffic_signal": None,
                },
            )

    # ---- helpers used by the waypoint graph ----
    @staticmethod
    def _bearing_deg(a: Vector, b: Vector) -> float:
        """Compass-style bearing in degrees (0=N, 90=E, ...) from a→b."""
        dx, dy = (b.x - a.x), (b.y - a.y)
        # Screen-y typically points down; expose math-like angle anyway
        # but rotate so 0° = north (positive y). This matches the
        # convention used by `compass_from_bearing` below.
        ang = math.degrees(math.atan2(dx, dy))
        return (ang + 360.0) % 360.0

    @staticmethod
    def compass_from_bearing(bearing_deg: float) -> str:
        """Convert a bearing (0–360°) into one of 8 compass directions."""
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int(((bearing_deg + 22.5) % 360.0) // 45.0)
        return dirs[idx]

    @staticmethod
    def _is_cardinal_waypoint_bearing(bearing_deg: float, tol_deg: float = 30.0) -> bool:
        """True for MOVE-reachable cardinal waypoint edges, false for diagonals."""
        b = float(bearing_deg) % 360.0
        return any(min(abs(b - c), 360.0 - abs(b - c)) <= tol_deg for c in (0.0, 90.0, 180.0, 270.0))

    def _infer_edge_road_name(self, u: Node, v: Node) -> str:
        """Best-effort road name for an adjacency edge.

        Names attached to nodes carry the internal side suffix; we strip
        it here so the user-facing waypoint-graph edges expose the clean
        US-style form (``"Maple St"``, not ``"Maple St (left)"``).
        """
        return self._display_road_name(
            getattr(u, "road_name", "")
            or getattr(v, "road_name", "")
            or ""
        )

    # ---- public helpers consumed by the env / prompt builder ----
    def adjacents(self, node: Node) -> List[Dict[str, Any]]:
        """
        Return the waypoint-graph neighbours of ``node`` as a list of
        small dicts suitable for embedding in an observation:

        ``{"id", "name", "kind", "poi_kind", "dist_m", "bearing_deg", "compass", "road_name", "node"}``.

        ``node`` must already be a waypoint (intersection or dock); if
        not, returns ``[]``. To find the nearest waypoint to an
        arbitrary (x, y), use :meth:`nearest_waypoint`.
        """
        if node is None or node not in self.waypoint_graph.adjacency_list:
            return []
        out: List[Dict[str, Any]] = []
        for nb in self.waypoint_graph.adjacency_list.get(node, []):
            em = self.waypoint_graph.get_edge_meta(node, nb) or {}
            dist_cm = float(em.get("dist_cm", node.position.distance(nb.position)))
            bearing = float(self._bearing_deg(node.position, nb.position))
            legal_move = bool(em.get("legal_move", self._is_cardinal_waypoint_bearing(bearing)))
            crosses_vehicle_road = bool(em.get("crosses_vehicle_road"))
            # Runtime light state is not exposed here (vision-only; decided by
            # TrafficController). The map only reports the crossing geometry.
            signal = None
            out.append({
                "id": self._waypoint_id_by_node.get(nb, str(nb)),
                "name": getattr(nb, "waypoint_name", ""),
                "kind": getattr(nb, "waypoint_kind", ""),
                "poi_kind": getattr(nb, "poi_kind", ""),
                "dist_m": dist_cm / 100.0,
                "bearing_deg": bearing,
                "compass": self.compass_from_bearing(bearing),
                "road_name": self._display_road_name(em.get("road_name", "")),
                "legal_move": legal_move,
                "crosses_vehicle_road": crosses_vehicle_road,
                "crossing_kinds": list(em.get("crossing_kinds", [])),
                "traffic_signal": signal,
                "node": nb,
            })
        # Sort by id so observations are stable across calls.
        out.sort(key=lambda d: d["id"])
        return out

    def resolve_waypoint(self, token: str) -> Optional[Node]:
        """
        Resolve a free-form token to a waypoint Node.

        Accepts:
        - a waypoint id (``"int_3"``, ``"dock_7"``),
        - a POI address (``"124 Maple St"``),
        - a POI display name (``"restaurant 3"``),
        - an intersection name (``"Maple St & Oak Ave"``).
        Returns ``None`` if no match is found.
        """
        if not token:
            return None
        t = token.strip()
        n = self.address_book.get(t)
        if n is not None:
            return n
        # Try a relaxed case-insensitive match.
        low = t.lower()
        for k, v in self.address_book.items():
            if k.lower() == low:
                return v
        return None

    def nearest_waypoint(
        self, x_cm: float, y_cm: float
    ) -> Optional[Node]:
        """Return the closest waypoint to an (x, y) position in cm."""
        best: Optional[Tuple[float, Node]] = None
        for n in self.waypoint_graph.adjacency_list.keys():
            d = math.hypot(x_cm - n.position.x, y_cm - n.position.y)
            if best is None or d < best[0]:
                best = (d, n)
        return best[1] if best else None

    # =========================================================
    #                   3) Orders (reuse existing POIs only)
    # =========================================================
    def _pick_meta(self, tname: str, hint_xy: Optional[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        """
        Pick a POI meta record by type name and an optional spatial hint.
        If hint_xy is provided, choose the closest POI of that type.
        """
        tname = (tname or "").strip().lower()
        cands = [
            m
            for m in self.poi_meta
            if self._t_of(m["node"]) == tname or (tname == "building" and self._t_of(m["node"]) == "building")
        ]
        if not cands:
            return None
        if hint_xy is None:
            return cands[0]
        hx, hy = hint_xy
        return min(
            cands,
            key=lambda m: (m.get("center", (0, 0))[0] - hx) ** 2
            + (m.get("center", (0, 0))[1] - hy) ** 2,
        )

    def set_active_orders(
        self,
        orders: List[Dict[str, Any]],
        world_json_path: Optional[str] = None,
        eps_cm: float = 80.0,
    ) -> List[Dict[str, Any]]:
        """
        Match pickup/dropoff endpoints for orders against the existing POI set
        (by type + nearest to hint position).

        - Only uses already-imported POIs.
        - Copies matched building boxes into rec['*building'], so that the viewer
          can render building outlines even when building-link mode is disabled.
        """
        self.order_meta = []

        def _xy_from_hint(h):
            if not isinstance(h, dict):
                return None
            return (float(h.get("x", 0.0)), float(h.get("y", 0.0)))

        results = []
        for od in orders:
            oid = str(od.get("id", ""))
            pt = (od.get("pickup_type") or "restaurant")
            dt = (od.get("dropoff_type") or "building")
            phxy = _xy_from_hint(od.get("pickup_hint"))
            dhxy = _xy_from_hint(od.get("dropoff_hint"))

            pm = self._pick_meta(pt, phxy)
            dm = self._pick_meta(dt, dhxy)

            rec = {
                "id": oid,
                "pickup_node": pm.get("door_node") or pm.get("dock_node") if pm else None,
                "dropoff_node": dm.get("door_node") or dm.get("dock_node") if dm else None,
                "pickup_building": (pm.get("building_box") if pm else None),
                "dropoff_building": (dm.get("building_box") if dm else None),
            }
            self.order_meta.append(rec)
            results.append(rec)
        return results

    def clear_active_orders(self):
        """Reset the active order metadata."""
        self.order_meta = []

    # =========================================================
    #                   4) Agent package + reachability
    # =========================================================
    def _intersection_cluster(self, seed: Optional[Node]) -> set:
        """
        Return the connected component (cluster) of intersections reachable
        from `seed`, only traversing intersection-type nodes.
        """
        if seed is None or self._t_of(seed) != "intersection":
            return set()
        vis = {seed}
        dq = deque([seed])
        while dq:
            x = dq.popleft()
            for nb in self.adjacency_list[x]:
                if self._t_of(nb) == "intersection" and nb not in vis:
                    vis.add(nb)
                    dq.append(nb)
        return vis

    def _adjacent_best_counting(
        self,
        seed: Node,
        agent_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[Node], Optional[float]]:
        """
        Starting from seed, find adjacent counting nodes (normal/intersection,
        excluding docks) without passing through any other counting nodes.

        Behavior
        --------
        - If agent_xy is None:
            Return the first reachable counting neighbor (neighbor, dist_from_seed),
            preserving the original behavior.
        - If agent_xy is provided:
            Enumerate all adjacent counting nodes and use the shortest path
            distance from agent_xy to each candidate (via shortest_path_xy_to_node)
            to choose the best neighbor. Return the chosen neighbor and its
            distance from the seed (neighbor, dist_from_seed).

        Returns
        -------
        (neighbor_node, dist_from_seed) or (None, None) if none found.
        """

        def _is_seed_candidate(nd: Node) -> bool:
            t = (getattr(nd, "type", "") or "").lower()
            return t in ("normal", "intersection") and (nd not in self._dock_nodes)

        visited = {seed}
        q = deque()

        # First consider seed's neighbors.
        # If a direct neighbor is a counting node and agent_xy is None,
        # we can immediately return it.
        for v in self.adjacency_list.get(seed, []):
            step = seed.position.distance(v.position)
            if _is_seed_candidate(v):
                if agent_xy is None:
                    return v, step
                else:
                    # With agent_xy, we collect all candidates instead of early exit.
                    q.clear()
            else:
                visited.add(v)
                q.append((v, step))

        # BFS over non-counting nodes only, collecting all adjacent counting nodes
        neighbors: List[Tuple[Node, float]] = []

        # Re-check direct neighbors if agent_xy != None (they must be included too).
        for v in self.adjacency_list.get(seed, []):
            step = seed.position.distance(v.position)
            if _is_seed_candidate(v):
                neighbors.append((v, step))

        while q:
            u, dacc = q.popleft()
            for v in self.adjacency_list.get(u, []):
                if v in visited:
                    continue
                step = u.position.distance(v.position)
                if _is_seed_candidate(v):
                    if agent_xy is None:
                        return v, dacc + step
                    neighbors.append((v, dacc + step))
                else:
                    visited.add(v)
                    q.append((v, dacc + step))

        if not neighbors:
            return None, None

        if agent_xy is None:
            # Preserve legacy behavior: return the first collected neighbor.
            return neighbors[0]

        ax, ay = agent_xy
        best = None
        best_total = None

        for nb, d_seed_nb in neighbors:
            # Use the shortest path distance from agent to neighbor as selection metric.
            _path, total_cm, _ = self.shortest_path_xy_to_node(ax, ay, nb)
            if not math.isfinite(total_cm):
                continue
            if best is None or total_cm < best_total - 1e-6:
                best, best_total = (nb, d_seed_nb), total_cm

        return best if best is not None else (None, None)

    def get_reachable_set_xy(
        self,
        x: float,
        y: float,
        include_docks: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute reachable nearby waypoints and intersections from (x, y).

        Distances are consistently measured via shortest_path_xy_to_node(x, y, node)
        plus an optional off-graph straight-line segment if the agent is not
        currently on a graph node.

        When the position is off-graph, the distance from (x, y) to the nearest
        real edge projection is added as a fixed prefix to all entries, so that
        off-graph positions do not artificially appear closer (e.g., 0.0m).
        """
        res = {"next_hop": [], "next_intersections": []}
        from collections import deque

        # ---------- Local helpers ----------
        def _t(nd):
            return (getattr(nd, "type", "") or "").lower()

        def _is_dock(nd):
            return nd in self._dock_nodes

        def _is_counting(nd):
            # Counting nodes: normal/intersection, optionally including docks
            return _t(nd) in ("normal", "intersection") and (include_docks or not _is_dock(nd))

        def _is_seed_candidate(nd):
            # Candidate for waypoint seeds (docks are always excluded)
            return _t(nd) in ("normal", "intersection") and (not _is_dock(nd))

        # Is the current position near an existing node?
        node0 = self._nearest_node_at(x, y, tol_cm=50.0)

        # Off-graph straight-line distance prefix (if not on a node)
        off_graph_cm = 0.0
        if node0 is None:
            snap = self.snap_to_nearest_edge(x, y, use_segment=True)
            if snap is not None:
                px, py = float(snap["proj"].x), float(snap["proj"].y)
                off_graph_cm = float(Vector(x, y).distance(Vector(px, py)))

        # Distance cache (includes off_graph_cm)
        _dist_cache: Dict[int, float] = {}

        def _dist_from_agent(nd: Node) -> float:
            nid = id(nd)
            if nid in _dist_cache:
                return _dist_cache[nid]
            _path, dist_cm, _ = self.shortest_path_xy_to_node(x, y, nd)
            dist_total = float(dist_cm) + float(off_graph_cm)
            _dist_cache[nid] = dist_total
            return dist_total

        items_by_key: Dict[tuple, Dict[str, Any]] = {}

        def _put_item(it):
            # Use either stable id, or (type,x,y) as a key.
            key = ("id", it["id"]) if it.get("id") else (
                "xy",
                it["type"],
                int(round(it["x"])),
                int(round(it["y"])),
            )
            if key not in items_by_key:
                items_by_key[key] = it

        # From a starting node, collect adjacent counting nodes,
        # expanding only through non-counting nodes.
        def _adjacent_countings(start):
            out: List[Node] = []
            visited = {start}
            q = deque()

            # First consider direct neighbors.
            for v in self.adjacency_list.get(start, []):
                if _is_seed_candidate(v):
                    out.append(v)
                else:
                    visited.add(v)
                    q.append(v)

            # Expand through non-counting nodes only.
            while q:
                u = q.popleft()
                for v in self.adjacency_list.get(u, []):
                    if v in visited:
                        continue
                    if _is_seed_candidate(v):
                        out.append(v)
                    else:
                        visited.add(v)
                        q.append(v)

            # Deduplicate by node id.
            by_id: Dict[int, Node] = {}
            for nd in out:
                k = id(nd)
                if k not in by_id:
                    by_id[k] = nd
            return list(by_id.values())
        
        

        # From a non-counting/POI node, find up to K nearest counting nodes in hop distance.
        def _nearest_k_countings_by_hop(start, K=2):
            found: Dict[int, Tuple[Node, int]] = {}
            visited = set([start])
            q = deque([(start, 0)])
            while q and len(found) < K:
                u, hops = q.popleft()
                for v in self.adjacency_list.get(u, []):
                    if v in visited:
                        continue
                    if _is_seed_candidate(v):
                        vid = id(v)
                        if (vid not in found) or (hops + 1 < found[vid][1]):
                            found[vid] = (v, hops + 1)
                    else:
                        visited.add(v)
                        q.append((v, hops + 1))
            arr = sorted(found.values(), key=lambda t: t[1])[:K]
            return [nd for (nd, _h) in arr]

        # BFS from a seed counting node, collecting POIs until the next counting node.
        door2poi = getattr(self, "_door2poi", {}) or {}

        def _bfs_until_next_counting(side, node0_ref):
            q = deque([side])
            visited = {side}
            while q:
                u = q.popleft()

                # Stop expansion at the next counting node, but continue other branches.
                if (u is not side) and _is_counting(u):
                    continue

                ut = self._t_of(u)
                roles_here = self._roles_at_xy(float(u.position.x), float(u.position.y))
                parent = door2poi.get(u) if ut == "door" else None
                parent_type = (getattr(parent, "type", "") or "").lower() if parent else None

                # Collect POIs: doors / charging / bus stations.
                # Docks are not added to N.
                is_poi = (
                    (u is not side and u is not node0_ref)
                    and (
                        ut in ("charging_station", "bus_station")
                        or (ut == "door" and (parent_type != "building" or len(roles_here) != 0))
                    )
                )
                if is_poi:
                    if parent is not None:
                        it = self._mk_item(
                            "poi",
                            parent,
                            0.0,
                            x_cm=float(u.position.x),
                            y_cm=float(u.position.y),
                            force_type=self._t_of(parent),
                            override_id=parent,
                        )
                        # Distance should be measured to the door node.
                        it["_node"] = u
                    else:
                        it = self._mk_item("poi", u, 0.0)
                    _put_item(it)

                # Expand neighbors, prioritizing POI-like nodes so they are visited earlier.
                poi_first, others = [], []
                for v in self.adjacency_list.get(u, []):
                    if v in visited:
                        continue
                    vt = self._t_of(v)
                    (poi_first if vt in ("door", "charging_station", "bus_station") or (v in self._dock_nodes) else others).append(
                        v
                    )
                for v in poi_first + others:
                    visited.add(v)
                    q.append(v)

        # ---------- Seed selection behavior ----------
        seeds_waypoints: List[Node] = []
        seeds_bfs: List[Node] = []

        if node0 is not None:
            # Starting exactly at a node.
            if _is_seed_candidate(node0):
                # At a counting node: do not add node0 itself into N; use adjacent
                # counting nodes as next_hop (waypoints), and BFS from both node0
                # and those neighbors.
                adj = _adjacent_countings(node0)
                for nd in adj:
                    if _is_seed_candidate(nd):
                        seeds_waypoints.append(nd)
                seeds_bfs = ([node0] + seeds_waypoints)
            else:
                # At a POI / non-counting node: find up to two nearest counting seeds.
                seeds_waypoints = _nearest_k_countings_by_hop(node0, K=2)
                seeds_bfs = list(seeds_waypoints)
        else:
            # Off-graph starting point: project to nearest edge, then use its endpoints
            # to find counting seeds.
            snap = self.snap_to_nearest_edge(x, y, use_segment=True)
            if snap is None:
                return res
            edge, proj = snap["edge"], snap["proj"]
            a, b = edge.node1, edge.node2

            def _find_nearest_counting(start):
                if _is_seed_candidate(start):
                    return start
                q = deque([start])
                visited = {start}
                while q:
                    u = q.popleft()
                    for v in self.adjacency_list.get(u, []):
                        if v in visited:
                            continue
                        if _is_seed_candidate(v):
                            return v
                        visited.add(v)
                        q.append(v)
                return None

            seedA = _find_nearest_counting(a)
            seedB = _find_nearest_counting(b)
            seeds = []
            if seedA is not None:
                seeds.append(seedA)
            if seedB is not None:
                seeds.append(seedB)

            if seeds:
                if len(seeds) == 1 or (len(seeds) >= 2 and seeds[0] is seeds[1]):
                    only_seed = seeds[0]
                    # For a single seed, choose the neighbor that is closest to the agent.
                    neighbor, _ = self._adjacent_best_counting(only_seed, agent_xy=(x, y))
                    if neighbor is not None:
                        seeds_waypoints = [only_seed, neighbor]
                    else:
                        seeds_waypoints = [only_seed]
                else:
                    seeds_waypoints = seeds[:2]
            seeds_bfs = list(seeds_waypoints)

        # ---------- Write waypoints (excluding the current counting node itself) ----------
        seen_ids = set()
        for nd in seeds_waypoints:
            if id(nd) in seen_ids:
                continue
            seen_ids.add(id(nd))
            if _is_seed_candidate(nd):  # docks are excluded
                _put_item(self._mk_item("waypoint", nd, 0.0))

        # ---------- BFS collection of POIs ----------
        node0_ref = node0
        for nd in seeds_bfs:
            _bfs_until_next_counting(nd, node0_ref)

        # ---------- Update distances via shortest paths and sort (Next Hops) ----------
        for it in items_by_key.values():
            nd = it.get("_node", None)
            if isinstance(nd, Node):
                it["dist_cm"] = float(_dist_from_agent(nd))
            else:
                nd2 = self._nearest_node_at(it["x"], it["y"], tol_cm=60.0)
                if nd2 is not None:
                    it["_node"] = nd2
                    it["dist_cm"] = float(_dist_from_agent(nd2))
                else:
                    it["dist_cm"] = float("inf")

        next_all = sorted(items_by_key.values(), key=lambda d: d["dist_cm"])
        for i, it in enumerate(next_all, 1):
            # Waypoint that is actually an intersection is shown as 'intersection'.
            if it["kind"] == "waypoint" and it["type"] == "intersection":
                shown_name = "intersection"
            elif it["kind"] == "waypoint":
                shown_name = "waypoint"
            else:
                shown_name = it.get("name") or (it.get("type") or "poi")

            dist_m = round(float(it["dist_cm"]) / 100.0, 1)
            rn = it.get("road_name") or ""
            # Intersections do not show road names.
            rn_tail = (f" • {rn}" if (rn and it.get("type") != "intersection") else "")
            roles = it.get("roles") or []
            roles_txt = (" / " + " / ".join(roles)) if roles else ""
            it["label"] = f"N{i}"
            it["label_text"] = (
                f"N{i}: {shown_name}{roles_txt} at {_fmt_xy_m(it['x'], it['y'])} • {dist_m}m{rn_tail}"
            )

        # =============== Compute S-set (next_intersections) ===============
        # Helper: lookup nearest node by coordinates
        def _node_by_xy(xx, yy, tol_cm=50.0):
            return self._nearest_node_at(xx, yy, tol_cm=tol_cm)

        # From a starting node, traverse through non-intersection nodes and
        # record the first layer of intersections encountered (do not expand from them).
        def _first_intersections_from(start):
            hits = set()
            seen = {start}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in self.adjacency_list.get(u, []):
                    if v in seen:
                        continue
                    meta = (self._get_edge_meta(u, v) or {})
                    kind = (meta.get("kind") or "")
                    if kind.startswith("aux_"):
                        continue
                    if self._t_of(v) == "intersection":
                        hits.add(v)     # record and stop; no further expansion from this intersection
                        continue
                    seen.add(v)
                    q.append(v)
            return hits

        # 1) Collect explicit intersection seeds (current intersection + intersections in next_hop).
        seeds_S = set()
        if node0 is not None and self._t_of(node0) == "intersection":
            seeds_S.add(node0)
        for it in next_all:
            if it["kind"] == "waypoint" and it["type"] == "intersection":
                nd = _node_by_xy(it["x"], it["y"], tol_cm=60.0)
                if nd is not None:
                    seeds_S.add(nd)

        S_items: List[Dict[str, Any]] = []

        if seeds_S:
            # --- We have explicit intersection seeds: form a cluster and then its frontiers ---
            cluster = set()
            for s in seeds_S:
                cluster |= self._intersection_cluster(s)

            def _frontiers_from_cluster(cluster_set):
                """
                Return frontiers: intersections just beyond the given cluster.
                """
                front = set()
                visited = set(cluster_set)
                q = deque()
                # Start from neighbors of each intersection in the cluster
                for u in list(cluster_set):
                    for v in self.adjacency_list.get(u, []):
                        meta = (self._get_edge_meta(u, v) or {})
                        kind = (meta.get("kind") or "")
                        if kind.startswith("aux_"):
                            continue
                        if v in visited:
                            continue
                        if self._t_of(v) == "intersection":
                            if v not in cluster_set:
                                front.add(v)
                            continue
                        visited.add(v)
                        q.append(v)
                # Expand through non-intersection nodes only
                while q:
                    u = q.popleft()
                    for v in self.adjacency_list.get(u, []):
                        if v in visited:
                            continue
                        meta = (self._get_edge_meta(u, v) or {})
                        kind = (meta.get("kind") or "")
                        if kind.startswith("aux_"):
                            continue
                        if self._t_of(v) == "intersection":
                            if v not in cluster_set:
                                front.add(v)
                            continue
                        visited.add(v)
                        q.append(v)
                return list(front)

            frontiers = _frontiers_from_cluster(cluster)

            # Build S_items using shortest-path distance from (x, y) plus off-graph prefix.
            for nd in frontiers:
                dist_cm = float(_dist_from_agent(nd))
                item = self._mk_item("intersection", nd, dist_cm, force_type="intersection")
                S_items.append(item)

        else:
            # --- No explicit intersection seeds: for each normal waypoint, take its first-layer intersections ---
            cand_S = set()
            for it in next_all:
                if it["kind"] != "waypoint":
                    continue
                # Only consider normal waypoints (intersection waypoints are handled above).
                if it["type"] == "intersection":
                    continue
                nd = _node_by_xy(it["x"], it["y"], tol_cm=60.0)
                if nd is None:
                    continue
                cand_S |= _first_intersections_from(nd)

            for nd in cand_S:
                dist_cm = float(_dist_from_agent(nd))
                item = self._mk_item("intersection", nd, dist_cm, force_type="intersection")
                S_items.append(item)

        # Sort and label S-items; intersections do not show road names.
        S_items.sort(key=lambda d: d["dist_cm"])
        for i, it in enumerate(S_items, 1):
            dist_m = round(float(it["dist_cm"]) / 100.0, 1)
            it["label"] = f"S{i}"
            it["label_text"] = f"S{i}: intersection at {_fmt_xy_m(it['x'], it['y'])} • {dist_m}m"

        # Output
        res["next_hop"] = next_all
        res["next_intersections"] = S_items
        return res

    # -------- Shortest-path helpers --------
    def shortest_path_xy_to_node(self, x: float, y: float, target: Node):
        """Delegate to graph_full for shortest path from arbitrary XY to a node."""
        return self.graph_full.shortest_path_xy_to_node(x, y, target)

    def shortest_path_nodes(self, start: Node, target: Node):
        """Delegate to graph_full for shortest path between nodes."""
        return self.graph_full.shortest_path_nodes(start, target)

    # -------- POIs sorted by shortest path distance --------
    def list_direct_reachable_pois_xy(self, x: float, y: float) -> List[Dict[str, Any]]:
        """
        List all POIs reachable from (x, y), sorted by shortest path distance.
        Buildings are included (for dropoff display), with the anchor being
        their door/dock if available.
        """
        building_like = {
            "restaurant",
            "store",
            "rest_area",
            "hospital",
            "car_rental",
            "customer",
            "building",
        }
        out = []
        for meta in self.poi_meta:
            pnode = meta["node"]
            ptype = self._t_of(pnode)
            if ptype in building_like:
                anchor = meta.get("door_node") or meta.get("dock_node")
            else:
                anchor = pnode or meta.get("dock_node")
            if anchor is None:
                continue
            _path, dist_cm, _ = self.shortest_path_xy_to_node(x, y, anchor)
            if not math.isfinite(dist_cm):
                continue
            out.append(
                {
                    "id": str(pnode),
                    "type": ptype,
                    "name": self._display_name_of(pnode),
                    "x": anchor.position.x,
                    "y": anchor.position.y,
                    "dist_cm": float(dist_cm),
                    "road_name": self._road_name_for_node(anchor),
                }
            )
        out.sort(key=lambda d: d["dist_cm"])
        return out

    def agent_info_package_xy(
        self,
        x: float,
        y: float,
        include_docks: bool = False,
        limit_next: int = 50,
        limit_s: int = 50,
        limit_poi: int = 200,
        *,
        active_orders: Optional[List[Any]] = None,
        help_orders: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an information package about the local road network / POIs / order endpoints
        around an agent at position (x, y).

        Behavior
        --------
        - If active_orders / help_orders are provided, they are used as the order source.
        - Otherwise it falls back to self.order_meta for backward compatibility.

        Returns
        -------
        dict
            {
              "agent_xy": {"x", "y"},
              "reachable": {...},       # next_hop + next_intersections
              "pois": [...],            # all POIs (including buildings), enriched with order roles
              "orders": [...],          # order endpoints with shortest path distances
              "text": str,              # human-readable summary
            }
        """
        import math

        reachable = self.get_reachable_set_xy(x, y, include_docks=include_docks)
        next_hop = list(reachable.get("next_hop", []))[:limit_next]
        next_intersections = list(reachable.get("next_intersections", []))[:limit_s]

        # Detect whether the current position is on top of a POI (door / point POI)
        poi_here: Optional[str] = None
        road_here: str = ""
        roles_here: List[str] = []
        ref = Vector(x, y)

        best = None
        for meta in self.poi_meta:
            door = meta.get("door_node")
            pnode = meta.get("node")
            if door is not None:
                d = ref.distance(door.position)
                if d <= self.NEAR_EPS:
                    cand = (0, d, "door", door, meta)
                    if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                        best = cand
            if pnode is not None:
                t = (getattr(pnode, "type", "") or "").lower()
                if t in ("bus_station", "charging_station"):
                    d = ref.distance(pnode.position)
                    if d <= self.NEAR_EPS:
                        cand = (1, d, t, pnode, meta)
                        if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                            best = cand

        if best is not None:
            _, _, kind, nd, meta = best
            if kind == "door":
                poi_here = self._display_name_of(meta.get("node"))
            else:
                poi_here = self._display_name_of(nd)
            road_here = (
                self._road_name_for_node(nd)
                or self._display_road_name(meta.get("road_name") or "")
            )
            roles_here = self._roles_at_xy(float(nd.position.x), float(nd.position.y), eps=1.0)

        # All POIs (including buildings; later we filter pure buildings without order roles)
        poi_list = self.list_direct_reachable_pois_xy(x, y)[:limit_poi]

        # ——————————————————————————
        # Unified order source: prefer explicit active_orders/help_orders, otherwise self.order_meta
        # ——————————————————————————
        def iter_order_records():
            if active_orders is not None or help_orders is not None:
                # Use provided order objects with attributes.
                orders = []
                if active_orders:
                    orders.extend(active_orders)
                if help_orders:
                    orders.extend(help_orders)
                for o in orders:
                    yield {
                        "id": getattr(o, "id", None),
                        "pickup_node": getattr(o, "pickup_node", None),
                        "dropoff_node": getattr(o, "dropoff_node", None),
                    }
            else:
                # Legacy path: use self.order_meta (dict-like)
                for rec in getattr(self, "order_meta", []):
                    yield {
                        "id": rec.get("id", None),
                        "pickup_node": rec.get("pickup_node", None),
                        "dropoff_node": rec.get("dropoff_node", None),
                    }

        # Build a lookup from POI coordinates to order-related role tags.
        poi_xy2tags: Dict[tuple, List[str]] = {}
        for rec in iter_order_records():
            oid = str(rec.get("id", ""))
            pu = rec.get("pickup_node")
            do = rec.get("dropoff_node")
            if pu is not None and hasattr(pu, "position"):
                k = (int(round(pu.position.x)), int(round(pu.position.y)))
                poi_xy2tags.setdefault(k, []).append(f"pick up address of order {oid}")
            if do is not None and hasattr(do, "position"):
                k = (int(round(do.position.x)), int(round(do.position.y)))
                poi_xy2tags.setdefault(k, []).append(f"drop off address of order {oid}")

        # Formatting helpers
        def m(cm):
            return f"{round(float(cm) / 100.0, 1)}m"

        def fmt_xy(xx, yy):
            return _fmt_xy_m(xx, yy, decimals=2)

        lines: List[str] = []
        pos_line = f"Agent position: {fmt_xy(x, y)}"
        if poi_here:
            roles_tail = (" • " + " / ".join(roles_here)) if roles_here else ""
            pos_line += f" • {poi_here}{roles_tail}" + (f" • {road_here}" if road_here else "")
        lines.append(pos_line)

        lines.append(
            "The following are nearby locations and POIs with their coordinates for your reference. "
            "You should decide where to move based on your current delivery needs."
        )
        lines.append("\nNext hops:")
        for it in next_hop:
            # Waypoint that is actually an intersection.
            if it.get("kind") == "waypoint" and it.get("type") == "intersection":
                shown_name = "intersection"
            elif it.get("kind") == "waypoint":
                shown_name = "waypoint"
            else:
                shown_name = it.get("name") or (it.get("type") or "poi")
            rn = it.get("road_name") or ""
            # Intersections do not display road names here.
            rn_tail = f" • {rn}" if (rn and it.get("type") != "intersection") else ""
            roles = it.get("roles") or []
            roles_txt = (" / " + " / ".join(roles)) if roles else ""
            lines.append(
                f"{it.get('label','N?')}: {shown_name}{roles_txt} at "
                f"{fmt_xy(it.get('x',0), it.get('y',0))} • {m(it.get('dist_cm',0))}{rn_tail}"
            )

        lines.append("\nNext intersections:")
        for it in next_intersections:
            lines.append(
                f"{it.get('label','S?')}: intersection at "
                f"{fmt_xy(it.get('x',0), it.get('y',0))} • {m(it.get('dist_cm',0))}"
            )

        # Enrich POIs with order roles; still sorted purely by distance.
        enriched_pois = []
        for p in poi_list:
            k = (int(round(p.get("x", 0))), int(round(p.get("y", 0))))
            role_tags = sorted(set(poi_xy2tags.get(k, [])))
            has_order_role = len(role_tags) > 0
            enriched_pois.append(
                {
                    **p,
                    "roles": role_tags,
                    "has_order_role": has_order_role,
                }
            )

        # For printing: drop pure buildings with no order role.
        _skip_poi = getattr(self, '_skip_poi_types', set()) 
        enriched_pois_for_print = [
            p
            for p in enriched_pois
            if not (p.get("type") == "building" and not p.get("has_order_role"))
            and p.get("type") not in _skip_poi
            and p.get("name", "").split()[0] not in _skip_poi  # "charging_station 4" → "charging_station"
        ]

        lines.append("\nAll POIs by shortest-path distance:")
        for p in enriched_pois_for_print:
            name = p.get("name") or (p.get("type") or "poi")
            rn = p.get("road_name") or ""
            roles = p.get("roles") or []
            title = f"{name}" if not roles else f"{name} / {' / '.join(roles)}"
            road_tail = f" {rn}" if rn else ""
            lines.append(
                f"{title}: at {fmt_xy(p.get('x', 0), p.get('y', 0))} • {m(p.get('dist_cm', 0))}{road_tail}"
            )

        # Order endpoints (shortest-path), consistently using the unified order source.
        orders_out = []
        for rec in iter_order_records():
            oid = str(rec.get("id", ""))
            for kind_key, tag in (("pickup_node", "pickup"), ("dropoff_node", "dropoff")):
                nd = rec.get(kind_key)
                if nd is None or not hasattr(nd, "position"):
                    continue
                try:
                    _path, dist_cm, _ = self.shortest_path_xy_to_node(x, y, nd)
                except Exception:
                    dist_cm = float("inf")
                if not (isinstance(dist_cm, (int, float)) and math.isfinite(dist_cm)):
                    # Fallback: straight-line distance
                    try:
                        dx = float(nd.position.x) - x
                        dy = float(nd.position.y) - y
                        dist_cm = (dx * dx + dy * dy) ** 0.5
                    except Exception:
                        continue
                orders_out.append(
                    {
                        "order_id": oid,
                        "kind": tag,
                        "x": float(nd.position.x),
                        "y": float(nd.position.y),
                        "dist_cm": float(dist_cm),
                        "road_name": self._road_name_for_node(nd),
                    }
                )

        lines.append("\nOrder endpoints (shortest-path):")
        for o in orders_out:
            tag = "pick up" if o["kind"] == "pickup" else "drop off"
            lines.append(
                f"{tag} of order {o['order_id']}: at "
                f"{fmt_xy(o['x'], o['y'])} • {m(o['dist_cm'])} {o['road_name']}"
            )

        return {
            "agent_xy": {"x": float(x), "y": float(y)},
            "reachable": reachable,
            "pois": enriched_pois,   # full list (including buildings), upper layers may filter further
            "orders": orders_out,
            "text": "\n".join(lines),
        }

    # =========================================================
    #                   5) Export / visualization helpers
    # =========================================================
    def export_agent_graph(
        self,
        include_crosswalks: bool = True,
        print_preview: bool = True,
    ) -> Dict[str, Any]:
        """
        Export the skeleton graph for agent-level visualization or debugging.

        Returns
        -------
        {
          "nodes": [{"id","type","x","y","roles"}, ...],
          "edges": [{"u","v","dist_cm","kind","name"}, ...]
        }
        """
        nodes_out = [
            {
                "id": str(n),
                "type": "intersection",
                "x": float(n.position.x),
                "y": float(n.position.y),
                "roles": [],
            }
            for n in self.graph_skel.nodes
        ]

        edges_out = []
        for e in self.graph_skel.edges:
            u, v = e.node1, e.node2
            meta = self.graph_skel.get_edge_meta(u, v) or {}
            kind = meta.get("kind", "")
            if kind == "crosswalk" and not include_crosswalks:
                continue
            name = meta.get("name", "")
            edges_out.append(
                {
                    "u": str(u),
                    "v": str(v),
                    "dist_cm": float(u.position.distance(v.position)),
                    "kind": kind,
                    "name": (name or ""),
                }
            )
        if print_preview:
            print("Nodes:", len(nodes_out))
            print("Edges:", len(edges_out))
        return {"nodes": nodes_out, "edges": edges_out}

    def get_building_link_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], str]]:
        """
        Return all aux_* link segments for building connections.

        Returns
        -------
        list of ((x1, y1), (x2, y2), kind)
        """
        segs = []
        seen = set()
        for a in self.nodes:
            for b in self.adjacency_list.get(a, []):
                key = self._edge_key(a, b)
                if key in seen:
                    continue
                seen.add(key)
                meta = self._get_edge_meta(a, b) or {}
                kind = (meta.get("kind") or "")
                if kind.startswith("aux_"):
                    segs.append(
                        (
                            (float(a.position.x), float(a.position.y)),
                            (float(b.position.x), float(b.position.y)),
                            kind,
                        )
                    )
        return segs

    # ---------------- Generic XY→XY routing (on a copied graph) ----------------
    def route_xy_to_xy(self, ax: float, ay: float, tx: float, ty: float, snap_cm: float = 120.0):
        """
        Compute a polyline from (ax, ay) to (tx, ty) walking on the sidewalk graph.

        Notes
        -----
        - Works on a copy of graph_full, so intermediate node insertions do not
          affect the original graph.
        - The endpoints are snapped to nearest edges, and we walk along
          shortest paths between those snapped nodes.
        """
        if not self.graph_full.edges:
            return [(float(ax), float(ay)), (float(tx), float(ty))]

        g = self.graph_full.copy()

        snapA = g.snap_to_nearest_edge(ax, ay)
        snapB = g.snap_to_nearest_edge(tx, ty)
        if snapA is None or snapB is None:
            return [(float(ax), float(ay)), (float(tx), float(ty))]

        nA = g.ensure_node_on_edge_at(snapA["edge"], snapA["proj"], snap_cm=snap_cm)
        nB = g.ensure_node_on_edge_at(snapB["edge"], snapB["proj"], snap_cm=snap_cm)

        def _anchor(snap, n):
            if n is snap["a"] or n is snap["b"]:
                return float(n.position.x), float(n.position.y)
            return float(snap["proj"].x), float(snap["proj"].y)

        ancA = _anchor(snapA, nA)
        ancB = _anchor(snapB, nB)
        path_nodes, _ = g.shortest_path_nodes(nA, nB)

        route = [(float(ax), float(ay))]
        if route[-1] != ancA:
            route.append(ancA)
        for nd in path_nodes:
            pt = (float(nd.position.x), float(nd.position.y))
            if route[-1] != pt:
                route.append(pt)
        if route[-1] != ancB:
            route.append(ancB)
        end_pt = (float(tx), float(ty))
        if route[-1] != end_pt:
            route.append(end_pt)

        # Remove consecutive duplicates
        cleaned = []
        for xy in route:
            if not cleaned or cleaned[-1] != xy:
                cleaned.append(xy)
        return cleaned

    def route_xy_to_xy_mode(
        self,
        ax: float,
        ay: float,
        tx: float,
        ty: float,
        snap_cm: float = 120.0,
        mode: str = "walk",
    ):
        """
        Dispatch routing based on mode.

        - mode == "walk": use sidewalk graph.
        - mode == "car" : walk → drive graph → walk.
        """
        if mode not in ("car",):
            return self.route_xy_to_xy(ax, ay, tx, ty, snap_cm=snap_cm)

        # 1) Walk from starting point to nearest drive-lane anchor.
        snapS = self._snap_to_nearest_edge_in_graph(
            self.graph_drive,
            ax,
            ay,
            use_segment=True,
            valid_kinds=("drive",),
        )
        if snapS is None:
            # No drive lanes available, fall back to walking.
            return self.route_xy_to_xy(ax, ay, tx, ty, snap_cm=snap_cm)
        start_lane_xy = (float(snapS["proj"].x), float(snapS["proj"].y))
        seg1 = self.route_xy_to_xy(ax, ay, start_lane_xy[0], start_lane_xy[1], snap_cm=snap_cm)

        # 2) On the drive graph: shortest route between drive-lane anchors.
        snapT = self._snap_to_nearest_edge_in_graph(
            self.graph_drive,
            tx,
            ty,
            use_segment=True,
            valid_kinds=("drive",),
        )
        end_lane_xy = (float(snapT["proj"].x), float(snapT["proj"].y))
        seg2 = self._route_xy_to_xy_in_graph(
            self.graph_drive,
            start_lane_xy[0],
            start_lane_xy[1],
            end_lane_xy[0],
            end_lane_xy[1],
            valid_kinds=("drive",),
            snap_cm=snap_cm,
        )

        # 3) Walk from destination drive-lane anchor to the true destination.
        seg3 = self.route_xy_to_xy(end_lane_xy[0], end_lane_xy[1], tx, ty, snap_cm=snap_cm)

        # Merge segments and remove duplicates.
        out = []

        def _append(seg):
            nonlocal out
            for xy in seg:
                if not out or out[-1] != xy:
                    out.append(xy)

        _append(seg1)
        _append(seg2[1:])  # seg2 already includes its start; skip its first point
        _append(seg3[1:])
        return out

    def _route_xy_to_xy_in_graph(
        self,
        g: Graph,
        ax: float,
        ay: float,
        tx: float,
        ty: float,
        valid_kinds: Tuple[str, ...],
        snap_cm: float = 120.0,
    ):
        """
        Generic XY→XY routing within a given graph `g`, constrained to valid edge types.

        This operates on a copy of `g` and applies drive-direction constraints
        (one-way edges) when needed.
        """
        if not g.edges:
            return [(float(ax), float(ay)), (float(tx), float(ty))]
        g2 = g.copy()

        # Apply drive directionality to adjacency list on the copied graph.
        def _apply_drive_directions(g_dir: Graph):
            # Iterate over edges to enforce allowed directions in adjacency list.
            for e in list(g_dir.edges):
                a, b = e.node1, e.node2
                meta = g_dir.get_edge_meta(a, b) or {}
                if (meta.get("kind") != "drive"):
                    continue
                oneway = bool(meta.get("oneway", False))
                if not oneway:
                    # Bidirectional: leave adjacency as is.
                    continue
                # One-way edge.
                if meta.get("forward_only", False):
                    # Allow a->b, remove b->a.
                    if a in g_dir.adjacency_list and b in g_dir.adjacency_list.get(a, []):
                        pass  # keep a->b
                    if b in g_dir.adjacency_list:
                        while a in g_dir.adjacency_list[b]:
                            g_dir.adjacency_list[b].remove(a)
                else:
                    # If we ever support reverse-only edges, swap the direction here.
                    if b in g_dir.adjacency_list and a in g_dir.adjacency_list.get(b, []):
                        pass
                    if a in g_dir.adjacency_list:
                        while b in g_dir.adjacency_list[a]:
                            g_dir.adjacency_list[a].remove(b)

        # Re-copy and apply direction rules before snapping/ensuring new nodes.
        g2 = g.copy()
        _apply_drive_directions(g2)

        snapA = self._snap_to_nearest_edge_in_graph(
            g2, ax, ay, use_segment=True, valid_kinds=valid_kinds
        )
        snapB = self._snap_to_nearest_edge_in_graph(
            g2, tx, ty, use_segment=True, valid_kinds=valid_kinds
        )
        if snapA is None or snapB is None:
            return [(float(ax), float(ay)), (float(tx), float(ty))]
        nA = g2.ensure_node_on_edge_at(snapA["edge"], snapA["proj"], snap_cm=snap_cm)
        nB = g2.ensure_node_on_edge_at(snapB["edge"], snapB["proj"], snap_cm=snap_cm)

        def _anchor(snap, n):
            if n is snap["a"] or n is snap["b"]:
                return float(n.position.x), float(n.position.y)
            return float(snap["proj"].x), float(snap["proj"].y)

        ancA = _anchor(snapA, nA)
        ancB = _anchor(snapB, nB)
        path_nodes, _ = g2.shortest_path_nodes(nA, nB)

        route = [(float(ax), float(ay))]
        if route[-1] != ancA:
            route.append(ancA)
        for nd in path_nodes:
            pt = (float(nd.position.x), float(nd.position.y))
            if route[-1] != pt:
                route.append(pt)
        if route[-1] != ancB:
            route.append(ancB)
        end_pt = (float(tx), float(ty))
        if route[-1] != end_pt:
            route.append(end_pt)

        cleaned = []
        for xy in route:
            if not cleaned or cleaned[-1] != xy:
                cleaned.append(xy)
        return cleaned

    def _interpolate_drive_waypoints(self, spacing_cm: int):
        """
        Insert 'normal' nodes (waypoints) along all driving-lane edges
        in graph_drive, with the given spacing.
        """
        current_edges = list(self.graph_drive.edges)
        for edge in current_edges:
            a, b = edge.node1, edge.node2
            distance = getattr(edge, "weight", a.position.distance(b.position))
            num_points = int(distance // spacing_cm)
            if num_points <= 1:
                continue
            dx = b.position.x - a.position.x
            dy = b.position.y - a.position.y
            norm = (dx * dx + dy * dy) ** 0.5
            if norm <= 1e-9:
                continue
            ux, uy = dx / norm, dy / norm
            new_nodes = []
            for i in range(1, num_points):
                px = a.position.x + ux * (i * spacing_cm)
                py = a.position.y + uy * (i * spacing_cm)
                node = Node(Vector(px, py), "normal")
                self.graph_drive.add_node(node)
                new_nodes.append(node)

            old_meta = self.graph_drive.get_edge_meta(a, b)
            if edge in self.graph_drive.edges:
                self.graph_drive.edges.remove(edge)
            while b in self.graph_drive.adjacency_list[a]:
                self.graph_drive.adjacency_list[a].remove(b)
            while a in self.graph_drive.adjacency_list[b]:
                self.graph_drive.adjacency_list[b].remove(a)
            if old_meta is not None:
                self.graph_drive.edge_meta.pop(self.graph_drive._edge_key(a, b), None)

            chain = [a] + new_nodes + [b]
            for u, v in zip(chain, chain[1:]):
                self.graph_drive.add_edge(Edge(u, v))
                if old_meta is not None:
                    self.graph_drive.set_edge_meta(u, v, old_meta)

    def build_union_subgraph_from_xy(
        self,
        agent_xy: Tuple[float, float],
        seeds_xy: List[Tuple[float, float]],
        *,
        snap_cm: float = 120.0,
        include_aux: bool = False,
        return_edges_list: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a union subgraph given an agent position and a set of endpoint coordinates.

        For each seed in seeds_xy, we compute the shortest-path polyline from
        agent_xy to the seed (using route_xy_to_xy). Then we:

        - Approximate the union of these polylines as a set of original edges
          by snapping each segment midpoint to the nearest real edge.
        - Optionally filter out aux_* edges (include_aux=False).
        - Accumulate nodes that belong to the chosen edges.
        - Return both the polylines and the compressed edge/node set.

        Parameters
        ----------
        agent_xy : (ax, ay) in cm
        seeds_xy : list of (x, y) in cm
            Endpoint positions, e.g. PU/DO anchors, handoff points, etc.
        snap_cm : float
            Projection/insert precision used by route_xy_to_xy.
        include_aux : bool
            If True, keep aux_* edges as well.
        return_edges_list : bool
            If True, also return a list of raw coordinate segments.

        Returns
        -------
        dict
            {
              "agent_xy": (ax, ay),
              "seeds_xy": [(x, y), ...],
              "polylines": [ [(x,y),...], ... ],
              "edges": set(Edge),
              "nodes": set(Node),
              "total_len_cm": float,
              # Optional:
              "edges_list": [ ((x1,y1),(x2,y2)), ... ]
            }
        """
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        seeds_xy = [(float(x), float(y)) for (x, y) in (seeds_xy or [])]

        if not seeds_xy:
            ret: Dict[str, Any] = {
                "agent_xy": (ax, ay),
                "seeds_xy": [],
                "polylines": [],
                "edges": set(),
                "nodes": set(),
                "total_len_cm": 0.0,
            }
            if return_edges_list:
                ret["edges_list"] = []
            return ret

        # 1) Shortest-path polylines
        polylines: List[List[Tuple[float, float]]] = []
        total_len_cm = 0.0
        for (tx, ty) in seeds_xy:
            try:
                poly = self.route_xy_to_xy(ax, ay, tx, ty, snap_cm=snap_cm)
            except Exception:
                poly = []
            if len(poly) >= 2:
                polylines.append(poly)
                for i in range(len(poly) - 1):
                    x0, y0 = poly[i]
                    x1, y1 = poly[i + 1]
                    total_len_cm += math.hypot(x1 - x0, y1 - y0)

        if not polylines:
            ret = {
                "agent_xy": (ax, ay),
                "seeds_xy": seeds_xy,
                "polylines": [],
                "edges": set(),
                "nodes": set(),
                "total_len_cm": float(total_len_cm),
            }
            if return_edges_list:
                ret["edges_list"] = []
            return ret

        # 2) For each polyline segment, snap its midpoint to the nearest real edge.
        chosen_edges: Set["Edge"] = set()
        edges_list: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        for poly in polylines:
            for i in range(len(poly) - 1):
                mx = 0.5 * (poly[i][0] + poly[i + 1][0])
                my = 0.5 * (poly[i][1] + poly[i + 1][1])
                try:
                    snap = self.snap_to_nearest_edge(mx, my, use_segment=True)
                except Exception:
                    snap = None
                if not snap:
                    continue
                e = snap.get("edge")
                if e is None:
                    continue
                if not include_aux:
                    try:
                        meta = self._get_edge_meta(e.node1, e.node2) or {}
                        if str(meta.get("kind") or "").startswith("aux_"):
                            continue
                    except Exception:
                        pass
                if e not in chosen_edges:
                    chosen_edges.add(e)
                    if return_edges_list:
                        try:
                            a = (float(e.node1.position.x), float(e.node1.position.y))
                            b = (float(e.node2.position.x), float(e.node2.position.y))
                            edges_list.append((a, b))
                        except Exception:
                            pass

        # 3) Derive node set from chosen edges
        chosen_nodes: Set["Node"] = set()
        for e in chosen_edges:
            try:
                chosen_nodes.add(e.node1)
                chosen_nodes.add(e.node2)
            except Exception:
                pass

        ret: Dict[str, Any] = {
            "agent_xy": (ax, ay),
            "seeds_xy": seeds_xy,
            "polylines": polylines,
            "edges": chosen_edges,
            "nodes": chosen_nodes,
            "total_len_cm": float(total_len_cm),
        }
        if return_edges_list:
            ret["edges_list"] = edges_list
        return ret
