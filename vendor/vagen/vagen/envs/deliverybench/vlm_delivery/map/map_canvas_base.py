# map/map_canvas_base.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Optional, Tuple
import json, math

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QGraphicsRectItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

# ================= Styles / constants =================
COLOR_BG         = "#F7FBFF"
COLOR_EDGE       = "#9EA7B3"
COLOR_NODE       = "#5B9BD5"
COLOR_HOP_WP     = "#1F7AFF"
COLOR_NEXT_INT   = "#0066CC"
COLOR_AGENT      = "#000000"

COLOR_PICKUP     = "#FD9488"
COLOR_DROPOFF    = "#2ECC71"

COLOR_CHG        = "#27AE60"
COLOR_BUS        = "#F39C12"
COLOR_BUS_ROUTE  = "#FFA500"

COLOR_AUX_EXT    = "#C7CDD4"
COLOR_AUX_PERP   = "#A2ACB3"

COLOR_BUILDING_PLAIN  = "#B0BEC5"
PLAIN_FILL_ALPHA      = 80
COLOR_PLAIN_BORDER    = "#000000"

UI_NODE_SIZE            = 10
UI_NODE_SMALL_SIZE      = 10
UI_HOP_SIZE             = 12
UI_NINT_SIZE            = 14
UI_POI_SIZE             = 9
UI_AGENT_SIZE           = 26
UI_STAR_SIZE            = 36
UI_LABEL_PX             = 16
UI_BUILDING_ABBR_PX     = 13
UI_EDGE_WIDTH           = 2.2
UI_STAR_BORDER_W        = 1.6

# Label placement / collision avoidance
LABEL_BASE_N     = 200.0
LABEL_STEP_N     = 70.0
LABEL_BASE_PUDO  = 2600.0
LABEL_STEP_PUDO  = 300.0
LABEL_MAX_TRIES  = 24
R_TEXT_BLACK     = 1400.0
R_TEXT_COLOR     = 1200.0
R_POINT_AVOID    = 900.0

# Road name (side labels)
ROAD_NAME_OFFSET_CM  = 140.0
ROAD_NAME_GRID_CM    = 1200.0
ROAD_NAME_PAD_SCALE  = 10.0
ROAD_NAME_TRIES      = 24
ROAD_NAME_TSHIFT_CM  = 500.0

# Fonts
UI_FONT_FAMILY              = "Arial"
UI_LETTER_SPACING_PCT       = 92
EXP_FONT_FAMILY             = "Arial"
EXP_LETTER_SPACING_PCT      = 92

COLOR_DRIVE_EDGE      = "#7C8A99"   # Drive lane edge color
COLOR_DRIVE_NODE      = "#DF4E2D"   # Drive lane node color
UI_DRIVE_EDGE_WIDTH   = 1.8
UI_DRIVE_NODE_SIZE    = 7

# Building rendering
BUILDING_TYPES = {"restaurant","store","rest_area","hospital","car_rental"}
COLOR_BUILDING = {
    "restaurant": "#E74C3C","store": "#3498DB","rest_area": "#9B59B6",
    "hospital": "#E91E63","car_rental": "#1ABC9C",
}
ABBR = {"restaurant":"R","store":"S","rest_area":"A","hospital":"H","car_rental":"C"}

def _node_xy(node) -> Tuple[float, float]:
    return float(node.position.x), float(node.position.y)

def _is_road_node(nd) -> bool:
    t = getattr(nd, "type", "")
    return t in ("normal", "intersection")

# ================= Base canvas =================
class MapCanvasBase(QMainWindow):
    """
    Base map canvas.

    Responsibilities:
      * Render roads / nodes / buildings / POIs / pickup–dropoff markers / road labels.
      * Highlight frontiers and paths.
      * Render the agent marker.

    UI controls such as export / animation buttons are intentionally kept out of this class
    (they should live in MapDebugViewer or higher-level wrappers).
    """
    def __init__(self, title: str = "Map Canvas"):
        super().__init__()
        self.setWindowTitle(title)

        root = QWidget(); self.setCentralWidget(root)
        self.vbox = QVBoxLayout(root)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(COLOR_BG)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.vbox.addWidget(self.plot)

        self.info = QTextEdit(); self.info.setReadOnly(True); self.info.setFixedHeight(140)
        self.vbox.addWidget(self.info)

        # State
        self._map = None
        self._world_path: Optional[str] = None
        self._world_nodes: List[dict] = []
        self._bus_paths: List[List[Tuple[float,float]]] = []

        self._agent_item = None
        self._agent_xy: Optional[Tuple[float,float]] = None
        self._taken: List[Tuple[float,float,float]] = []
        self._last_reachable: Optional[Dict[str, List[Dict[str, Any]]]] = None

        self._label_font = QFont(); self._label_font.setBold(True); self._label_font.setPixelSize(UI_LABEL_PX)
        self._bld_font   = QFont(self._label_font); self._bld_font.setPixelSize(UI_BUILDING_ABBR_PX)

        self._road_name_fmt: str = "{name}"

        self._path_item = None
        self._path_nodes_item = None

        # Auxiliary line items currently in the scene (for clearing on refresh)
        self._aux_items: List[Any] = []

    def _make_font(self, px: int, *, bold: bool = True, export: bool = False) -> QFont:
        f = QFont()
        f.setFamily(EXP_FONT_FAMILY if export else UI_FONT_FAMILY)
        f.setPixelSize(int(px))
        f.setBold(bold)
        f.setKerning(True)
        try:
            f.setLetterSpacing(QFont.PercentageSpacing,
                               EXP_LETTER_SPACING_PCT if export else UI_LETTER_SPACING_PCT)
        except Exception:
            pass
        return f

    # ---------- Context / clearing ----------
    def set_context(self, map_obj, world_json_path: Optional[str]):
        self._map = map_obj
        self._world_path = world_json_path

    def reset_scene(self):
        self.plot.clear()
        self._agent_item = None
        self._taken.clear()
        self._last_reachable = None
        self._path_item = None
        self._path_nodes_item = None
        if self._aux_items:
            for it in self._aux_items:
                try: self.plot.removeItem(it)
                except Exception: pass
            self._aux_items.clear()

    # ---------- Main drawing entry ----------
    def draw_map(self, map_obj, world_json_path: Optional[str]=None,
                 show_bus=True, show_docks=False, show_building_links: bool=False,
                 show_road_names: bool=False, road_name_fmt: Optional[str]=None,
                 plain_mode: str = "none", show_drive: bool = False):
        """
        Draw the map into the current plot.

        Args:
            map_obj: Map object with nodes / edges / metadata.
            world_json_path: Optional world JSON path for building / bus rendering.
            show_bus: If True, draw bus routes from world JSON (if available).
            show_docks: If True, highlight dock nodes separately from normal nodes.
            show_building_links: If True, draw auxiliary links for POIs/buildings to docks
                (only in the window view; not used for exports).
            show_road_names: If True, draw road side labels (e.g., "12L", "12R").
            road_name_fmt: Optional format string for road names (currently unused except
                for storage as _road_name_fmt).
            plain_mode: Controls gray building boxes:
                * "none": do not draw generic gray buildings.
                * "pudo": draw gray boxes only for pickup/dropoff buildings.
                * "all" : draw all generic buildings as gray boxes.
            show_drive: If True, overlay the drive-lane debug layer.

        Notes:
            * Auxiliary links (POI→dock and building→dock) are drawn in the window view
              only when show_building_links is True AND plain_mode == "all".
              They are never drawn in export flows (controlled in MapDebugViewer).
        """
        self.set_context(map_obj, world_json_path)
        if road_name_fmt: self._road_name_fmt = str(road_name_fmt)
        self.reset_scene()

        if world_json_path:
            self._load_world(world_json_path)
            if show_bus: self._draw_bus_routes(self.plot)
            self._draw_buildings(self.plot)
            if plain_mode == "all":
                self._draw_all_building_boxes(self.plot)
            elif plain_mode == "pudo":
                self._draw_pudo_building_boxes(self.plot)

        # Road segments (including endcap / crosswalk, i.e., short terminal extensions)
        self._draw_roads_only(self.plot, map_obj)

        # Nodes
        dock_set = getattr(map_obj, "_dock_nodes", set())
        normals, docks, inters = [], [], []
        for n in map_obj.nodes:
            t = getattr(n, "type", "")
            if t == "normal": (docks if n in dock_set else normals).append(_node_xy(n))
            elif t == "intersection": inters.append(_node_xy(n))
        if normals:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=normals, size=UI_NODE_SMALL_SIZE,
                brush=pg.mkBrush(COLOR_NODE), pen=pg.mkPen("w", width=0.3),
                symbol="o", antialias=True))
        if show_docks and docks:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=docks, size=UI_NODE_SMALL_SIZE,
                brush=pg.mkBrush("#4C6EF5"), pen=pg.mkPen("w", width=0.3),
                symbol="o", antialias=True))
        if inters:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=inters, size=UI_NODE_SIZE,
                brush=pg.mkBrush(COLOR_NODE), pen=pg.mkPen("w", width=0.5),
                symbol="o", antialias=True))

        # Point-style POIs (visible in the window view)
        if getattr(map_obj, "pois", None):
            chg, bus = [], []
            for p in map_obj.pois:
                t = (p.type or "").lower()
                if t == "charging_station": chg.append(_node_xy(p))
                elif t == "bus_station":   bus.append(_node_xy(p))
            if chg:
                self.plot.addItem(pg.ScatterPlotItem(
                    pos=chg, size=UI_POI_SIZE,
                    pen=pg.mkPen("w", width=1), brush=pg.mkBrush(COLOR_CHG),
                    symbol="o", antialias=True))
            if bus:
                self.plot.addItem(pg.ScatterPlotItem(
                    pos=bus, size=UI_POI_SIZE+1,
                    pen=pg.mkPen("#8E5A0A", width=1), brush=pg.mkBrush(COLOR_BUS),
                    symbol="t", antialias=True))

        # Window-only auxiliary links, drawn only when show_building_links and plain_mode=="all"
        if show_building_links and plain_mode == "all":
            self._draw_aux_links(self.plot, line_width=1.4)
        # Road side labels
        if show_road_names:
            self._draw_road_names(self.plot, px_per_cm=None, export=False, size_px=12)

        # Drive-lane debug layer (draw only when explicitly enabled)
        if show_drive:
            print("Drawing drive layer...")
            self._draw_drive_layer(self.plot, map_obj)

        self.info.setText("Map drawn.")

        # If agent coordinates were set earlier, re-draw agent marker on top
        if self._agent_xy is not None:
            ax, ay = self._agent_xy
            self._agent_item = pg.ScatterPlotItem(
                pos=[(ax, ay)], size=UI_AGENT_SIZE,
                brush=pg.mkBrush(COLOR_AGENT), pen=pg.mkPen("w", width=1.4),
                symbol="o", antialias=True)
            self.plot.addItem(self._agent_item)

    def _draw_roads_only(self, target_plot, map_obj):
        drawn = set()
        for a, nbs in map_obj.adjacency_list.items():
            for b in nbs:
                key = tuple(sorted([(id(a),), (id(b),)]))
                if key in drawn: continue
                drawn.add(key)
                get_meta = getattr(map_obj, "_get_edge_meta", None)
                meta = get_meta(a, b) if callable(get_meta) else {}
                kind = (meta.get("kind") or "") if isinstance(meta, dict) else ""
                if (not kind) or kind.startswith("aux_"): continue
                if kind not in ("road", "crosswalk", "endcap"): continue
                if not _is_road_node(a) or not _is_road_node(b): continue
                ax, ay = _node_xy(a); bx, by = _node_xy(b)
                target_plot.addItem(pg.PlotDataItem(
                    x=[ax, bx], y=[ay, by],
                    pen=pg.mkPen(COLOR_EDGE, width=UI_EDGE_WIDTH),
                    antialias=True
                ))

    # ---------- Orders / frontier / path highlighting ----------
    def mark_orders(self, map_obj):
        """
        Draw pickup / dropoff building boxes and star markers for each order.
        """
        for rec in getattr(map_obj, "order_meta", []):
            oid = str(rec.get("id", ""))
            for key in ("pickup_building", "dropoff_building"):
                b = rec.get(key)
                if b and all(k in b for k in ("x", "y", "w", "h", "yaw")):
                    x = float(b["x"]); y = float(b["y"])
                    w = float(b["w"]); h = float(b["h"]); yaw = float(b["yaw"])
                    rect = QGraphicsRectItem(x - w/2.0, y - h/2.0, w, h)
                    rect.setTransformOriginPoint(x, y)
                    rect.setRotation(yaw); rect.setZValue(-4)
                    rect.setPen(pg.mkPen(COLOR_PLAIN_BORDER, width=1.2))
                    fill_q = QColor(COLOR_BUILDING_PLAIN); fill_q.setAlpha(PLAIN_FILL_ALPHA)
                    rect.setBrush(pg.mkBrush(fill_q))
                    self.plot.addItem(rect)
            self._star_with_text(rec.get("pickup_node"),  f"P{oid}", COLOR_PICKUP)
            self._star_with_text(rec.get("dropoff_node"), f"D{oid}", COLOR_DROPOFF)

    def highlight_frontier(self, reachable: Dict[str,List[Dict[str,Any]]]):
        """
        Highlight reachable frontier nodes (next hops / intersections) and their labels.
        """
        self._last_reachable = reachable
        for it in reachable.get("next_hop", []):
            self._taken.append((float(it["x"]), float(it["y"]), R_POINT_AVOID))
        for it in reachable.get("next_intersections", []):
            self._taken.append((float(it["x"]), float(it["y"]), R_POINT_AVOID))
        if self._agent_xy:
            self._taken.append((self._agent_xy[0], self._agent_xy[1], R_POINT_AVOID))

        hops_all = reachable.get("next_hop", [])
        hops_wp  = [it for it in hops_all if it.get("kind")=="waypoint" and not it.get("is_dock", False)]
        if hops_wp:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=[(it["x"],it["y"]) for it in hops_wp], size=UI_HOP_SIZE,
                brush=pg.mkBrush(COLOR_HOP_WP), pen=pg.mkPen("w", width=0.8),
                symbol="o", antialias=True))
        for it in hops_all:
            lbl = it.get("label")
            if not lbl: continue
            lx, ly = self._place_label_live((float(it["x"]), float(it["y"])), True, LABEL_BASE_N, LABEL_STEP_N)
            self._add_text(lbl, lx, ly, "#1F2D3D")

        ins = reachable.get("next_intersections", [])
        if ins:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=[(it["x"],it["y"]) for it in ins], size=UI_NINT_SIZE,
                brush=pg.mkBrush(COLOR_NEXT_INT), pen=pg.mkPen("w", width=1.2),
                symbol="d", antialias=True))
            for it in ins:
                lbl = it.get("label")
                if not lbl: continue
                lx, ly = self._place_label_live((float(it["x"]), float(it["y"])), True, LABEL_BASE_N, LABEL_STEP_N)
                self._add_text(lbl, lx, ly, "#1F2D3D")

    def clear_path_highlight(self):
        """
        Remove any previously drawn path and path node markers.
        """
        if self._path_item is not None:
            try: self.plot.removeItem(self._path_item)
            except Exception: pass
            self._path_item = None
        if self._path_nodes_item is not None:
            try: self.plot.removeItem(self._path_nodes_item)
            except Exception: pass
            self._path_nodes_item = None

    def highlight_path(self, pts: List[Tuple[float, float]]):
        """
        Highlight a given path polyline and its node markers.
        """
        if not pts or len(pts) < 2: return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        self._path_item = pg.PlotDataItem(
            x=xs, y=ys, pen=pg.mkPen("#FF4757", width=3), antialias=True)
        self._path_item.setZValue(8); self.plot.addItem(self._path_item)
        self._path_nodes_item = pg.ScatterPlotItem(
            pos=list(zip(xs, ys)), size=5, brush=pg.mkBrush("#FF4757"),
            pen=pg.mkPen("w", width=0.6), symbol="o", antialias=True)
        self._path_nodes_item.setZValue(9); self.plot.addItem(self._path_nodes_item)

    # ---------- Agent marker ----------
    def set_agent_xy(self, x: float, y: float):
        """
        Set the agent's current position and update its marker.
        """
        self._agent_xy = (x, y)
        if self._agent_item is None:
            self._agent_item = pg.ScatterPlotItem(
                pos=[(x,y)], size=UI_AGENT_SIZE,
                brush=pg.mkBrush(COLOR_AGENT), pen=pg.mkPen("w", width=1.4),
                symbol="o", antialias=True)
            self.plot.addItem(self._agent_item)
        else:
            self._agent_item.setData(pos=[(x,y)])

    # ---------- Text / world / road names ----------
    def print_info(self, text: str):
        """
        Print info text to the bottom text widget.
        """
        self.info.setText(text if isinstance(text, str) and text.strip() else "(no text)")

    def _place_label_live(self, anchor: Tuple[float,float], black_text: bool, base: float, step: float):
        """
        Place a label near an anchor point while avoiding collisions with existing labels.

        Returns:
            (x, y) coordinates for the label.
        """
        ax, ay = anchor; rlab = R_TEXT_BLACK if black_text else R_TEXT_COLOR
        dirs = [(1,1),(-1,1),(1,-1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)]
        for k in range(LABEL_MAX_TRIES):
            d = base + k*step
            for dx,dy in dirs:
                x = ax + dx*d; y = ay + dy*d
                if not self._collide_any(x,y,rlab):
                    self._taken.append((x,y,rlab)); return x, y
        self._taken.append((ax+base, ay+base, rlab))
        return ax+base, ay+base

    def _collide_any(self, x: float, y: float, r: float) -> bool:
        """
        Check whether a circle at (x, y) with radius r intersects any taken region.
        """
        for cx,cy,cr in self._taken:
            if (x-cx)**2 + (y-cy)**2 < (r+cr)**2: return True
        return False

    def _add_text(self, text: str, x: float, y: float, color: str):
        """
        Add a text item to the plot.
        """
        ti = pg.TextItem(text=text, color=color, anchor=(0.5,0.5))
        ti.setFont(self._label_font); ti.setPos(x,y); ti.setZValue(12)
        self.plot.addItem(ti)

    def _star_with_text(self, node, text: str, color: str, size_px: int=UI_STAR_SIZE, border_w: float=UI_STAR_BORDER_W):
        """
        Draw a star marker at the node and place a colored label next to it.
        """
        if not node: return
        x, y = _node_xy(node)
        star = pg.ScatterPlotItem(
            pos=[(x,y)], size=size_px, brush=pg.mkBrush(color),
            pen=pg.mkPen("w", width=border_w), symbol="star", antialias=True)
        star.setZValue(10); self.plot.addItem(star)
        lx, ly = self._place_label_live((x,y), False, LABEL_BASE_PUDO, LABEL_STEP_PUDO)
        self._add_text(text, lx, ly, color)

    # ---- PU/DO gray building boxes (window view only) ----
    def _draw_pudo_building_boxes(self, target_plot):
        """
        Draw gray building boxes only for pickup/dropoff buildings (window view).
        """
        if not getattr(self._map, "order_meta", None): return
        for rec in self._map.order_meta:
            for key in ("pickup_building", "dropoff_building"):
                b = rec.get(key)
                if not b or not all(k in b for k in ("x","y","w","h","yaw")): continue
                x = float(b["x"]); y = float(b["y"])
                w = float(b["w"]); h = float(b["h"]); yaw = float(b["yaw"])
                rect = QGraphicsRectItem(x - w/2.0, y - h/2.0, w, h)
                rect.setTransformOriginPoint(x, y); rect.setRotation(yaw); rect.setZValue(-4)
                rect.setPen(pg.mkPen(COLOR_PLAIN_BORDER, width=1.2))
                fill_q = QColor(COLOR_BUILDING_PLAIN); fill_q.setAlpha(PLAIN_FILL_ALPHA)
                rect.setBrush(pg.mkBrush(fill_q)); target_plot.addItem(rect)

    # ---- World / bus routes / buildings ----
    def _load_world(self, path: str):
        """
        Load world JSON (nodes, bus routes) from disk.
        """
        try:
            with open(path,"r",encoding="utf-8") as f: data = json.load(f)
            self._world_nodes = data.get("nodes", [])
            self._bus_paths = []
            for r in data.get("bus_routes", []):
                pth = r.get("path", [])
                if len(pth) >= 2:
                    pts = [(float(p.get("x",0))*100, float(p.get("y",0))*100) for p in pth]
                    self._bus_paths.append(pts)
        except Exception:
            self._world_nodes = []; self._bus_paths = []

    def _draw_bus_routes(self, target_plot):
        """
        Draw all bus routes loaded from the world JSON.
        """
        for pts in self._bus_paths:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            target_plot.addItem(pg.PlotDataItem(
                x=xs, y=ys, pen=pg.mkPen(COLOR_BUS_ROUTE, width=2.0), antialias=True))

    def _draw_buildings(self, target_plot):
        """
        Draw colored building rectangles and their abbreviations based on world nodes.
        """
        if not self._world_nodes: return
        font_px = max(8, int(UI_BUILDING_ABBR_PX * 0.55))
        name_lookup = self._build_poi_name_lookup_qt()
        for n in self._world_nodes:
            props = n.get("properties", {}) or {}
            poi   = (props.get("poi_type") or props.get("type") or "").strip().lower()
            if poi not in BUILDING_TYPES: continue
            loc  = props.get("location", {}) or {}
            ori  = props.get("orientation", {}) or {}
            bbox = props.get("bbox", {}) or {}
            x = float(loc.get("x",0.0)); y = float(loc.get("y",0.0))
            yaw = float(ori.get("yaw",0.0))
            w = float(bbox.get("x",0.0)) or 600.0
            h = float(bbox.get("y",0.0)) or 600.0

            rect = QGraphicsRectItem(x-w/2.0, y-h/2.0, w, h)
            rect.setTransformOriginPoint(x,y); rect.setRotation(yaw); rect.setZValue(-3)
            rect.setPen(pg.mkPen("k", width=1))
            rect.setBrush(pg.mkBrush(COLOR_BUILDING.get(poi, "#7F8C8D")))
            target_plot.addItem(rect)

            label = name_lookup.get((round(x), round(y))) or ABBR.get(poi,"?")
            ti = pg.TextItem(text=label, color="w", anchor=(0.5,0.5))
            f = QFont(self._bld_font); f.setPixelSize(font_px); ti.setFont(f)
            ti.setPos(x,y); ti.setZValue(-2); target_plot.addItem(ti)

    def _build_poi_name_lookup_qt(self):
        out = {}
        poi_meta = getattr(self._map, "poi_meta", None)
        if not poi_meta:
            return out
        for meta in poi_meta:
            node = meta.get("node")
            if node is None:
                continue
            disp = getattr(node, "display_name", None)
            if not disp:
                continue
            pos = getattr(node, "position", None)
            if pos is None:
                continue
            try:
                out[(round(float(pos.x)), round(float(pos.y)))] = str(disp)
            except Exception:
                continue
        return out

    def _draw_all_building_boxes(self, target_plot):
        """
        Draw generic gray building boxes for all plain buildings in the world data.
        """
        if not self._world_nodes: return
        for n in self._world_nodes:
            props = n.get("properties", {}) or {}
            poi   = (props.get("poi_type") or props.get("type") or "").strip().lower()
            inst  = str(n.get("instance_name", "") or "")
            is_plain = (poi == "building") or inst.startswith("BP_Building")
            if not is_plain: continue

            loc  = props.get("location", {}) or {}
            ori  = props.get("orientation", {}) or {}
            bbox = props.get("bbox", {}) or {}
            x = float(loc.get("x",0.0)); y = float(loc.get("y",0.0))
            yaw = float(ori.get("yaw",0.0))
            w = float(bbox.get("x",0.0)) or 600.0
            h = float(bbox.get("y",0.0)) or 600.0

            rect = QGraphicsRectItem(x-w/2.0, y-h/2.0, w, h)
            rect.setTransformOriginPoint(x,y); rect.setRotation(yaw); rect.setZValue(-4)
            pen = pg.mkPen(COLOR_PLAIN_BORDER, width=1.2)
            fill_q = QColor(COLOR_BUILDING_PLAIN); fill_q.setAlpha(PLAIN_FILL_ALPHA)
            rect.setPen(pen); rect.setBrush(pg.mkBrush(fill_q))
            target_plot.addItem(rect)

    # ---- Road names (<number><L/R>) ----
    def _split_name_side(self, meta: dict) -> Tuple[str, Optional[str]]:
        """
        Split road name metadata into (base name, side).

        Side is one of {"left", "right"} if specified in meta["side"] or within the
        parentheses of the name string; otherwise None.
        """
        import re
        raw_name = str(meta.get("name") or "").strip()
        side_raw = str(meta.get("side") or "").strip().lower()
        side = side_raw if side_raw in ("left", "right") else None
        if side is None:
            m = re.search(r"\((left|right)\)", raw_name, flags=re.I)
            if m: side = m.group(1).lower()
        base = re.sub(r"\([^)]*\)", "", raw_name).strip()
        base = re.sub(r"\b(road|rd|street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln|way|place|pl|court|ct|terrace|ter)\b\.?",
                      "", base, flags=re.I).strip()
        return base, side

    def _lr_label(self, base: str, side: Optional[str]) -> str:
        """
        Convert a base road name and side into a compact label like '12L' or '12R'.
        """
        import re
        m = re.search(r"(\d+)", base or "")
        num = m.group(1) if m else (re.sub(r"[^A-Za-z]", "", base)[:4] or "?").upper()
        suf = "L" if side == "left" else ("R" if side == "right" else "?")
        return f"{num}{suf}"

    def _draw_road_names(self, pw: pg.PlotWidget, px_per_cm: Optional[float],
                         export: bool, size_px: Optional[int] = None,
                         taken_rects: Optional[List[Tuple[float,float,float,float]]] = None):
        """
        Draw compact road-side labels (e.g., '12L' / '12R') along aggregated road segments.

        Args:
            pw: Target PlotWidget.
            px_per_cm: Pixels-per-centimeter scale (used mainly for export).
            export: Whether this is used for export (controls label placement pooling).
            size_px: Font size in pixels.
            taken_rects: Optional rectangle pool used to avoid overlaps in export mode.
        """
        if self._map is None: return
        get_meta = getattr(self._map, "_get_edge_meta", None)

        agg: Dict[Tuple[str,str], Dict[str, float]] = {}
        seen_pairs = set()
        for a, nbs in self._map.adjacency_list.items():
            for b in nbs:
                keyp = tuple(sorted((id(a), id(b))))
                if keyp in seen_pairs: continue
                seen_pairs.add(keyp)

                if not _is_road_node(a) or not _is_road_node(b): continue
                meta = get_meta(a, b) if callable(get_meta) else {}
                if not isinstance(meta, dict) or (meta.get("kind") != "road"): continue

                base, side = self._split_name_side(meta)
                if not base or side not in ("left", "right"):
                    continue

                ax, ay = _node_xy(a); bx, by = _node_xy(b)
                dx, dy = bx - ax, by - ay
                L = math.hypot(dx, dy)
                if L < 1e-6: continue
                mx, my = (ax + bx) / 2.0, (ay + by) / 2.0

                rec = agg.setdefault((base, side), dict(sumL=0.0, cx=0.0, cy=0.0, vx=0.0, vy=0.0))
                rec["sumL"] += L
                rec["cx"]   += mx * L
                rec["cy"]   += my * L
                rec["vx"]   += dx
                rec["vy"]   += dy

        if not agg: return

        grid_seen = set()
        pool = taken_rects if (export and px_per_cm and taken_rects is not None) else []

        def _text_box_cm(text: str) -> Tuple[float, float]:
            px = size_px if size_px is not None else 12
            w_px = max(1.0, len(text) * px * 0.6)
            h_px = px * 1.2
            if not px_per_cm: return 1.0, 1.0
            return (w_px / px_per_cm) * ROAD_NAME_PAD_SCALE, (h_px / px_per_cm) * ROAD_NAME_PAD_SCALE

        def _collide(rect, pool2):
            if not pool2: return False
            x0, x1, y0, y1 = rect
            for a0, a1, b0, b1 in pool2:
                if not (x1 <= a0 or a1 <= x0 or y1 <= b0 or b1 <= y0):
                    return True
            return False

        for (base, side), rec in agg.items():
            sumL = rec["sumL"]; 
            if sumL <= 0: continue
            mx = rec["cx"] / sumL; my = rec["cy"] / sumL
            vx, vy = rec["vx"], rec["vy"]
            vlen = math.hypot(vx, vy); 
            tx, ty = ((vx / vlen, vy / vlen) if vlen > 1e-6 else (1.0, 0.0))
            nx, ny = -ty, tx

            label = self._lr_label(base, side)

            sgn = -1 if side == "left" else 1
            offsets = [ROAD_NAME_OFFSET_CM * k for k in (1.0, 1.4, 1.8, 2.2)]
            slides  = [0.0, ROAD_NAME_TSHIFT_CM, -ROAD_NAME_TSHIFT_CM, 2*ROAD_NAME_TSHIFT_CM, -2*ROAD_NAME_TSHIFT_CM]

            w_cm, h_cm = _text_box_cm(label)
            placed = False; tries = 0
            for off in offsets:
                for sl in slides:
                    px_ = mx + sgn * nx * off + tx * sl
                    py_ = my + sgn * ny * off + ty * sl
                    rect = (px_ - w_cm/2, px_ + w_cm/2, py_ - h_cm/2, py_ + h_cm/2)
                    if not _collide(rect, pool):
                        if pool is not None: pool.append(rect)
                        placed = True; break
                    tries += 1
                    if tries >= ROAD_NAME_TRIES: break
                if placed or (tries >= ROAD_NAME_TRIES): break
            if not placed:
                px_ = mx + sgn * nx * offsets[0]
                py_ = my + sgn * ny * offsets[0]

            ti = pg.TextItem(text=label, color="#000000", anchor=(0.5,0.5))
            px_use = size_px if size_px is not None else 12
            ti.setFont(self._make_font(px_use, bold=True, export=export)); ti.setPos(px_, py_); ti.setZValue(11)
            pw.addItem(ti)

    # ---------- Auxiliary links (for window view; not used in export) ----------
    def _draw_aux_links(self, target_plot, *, line_width: float = 1.4):
        """
        Draw POI→dock and building→dock auxiliary link segments.

        Only used in the interactive window; not included in exported figures.
        """
        if self._map is None: return
        items: List[Any] = []

        # 1) POI→dock links (e.g., charging stations / bus stops)
        get_poi = getattr(self._map, "get_poi_link_segments", None)
        if callable(get_poi):
            for (p1, p2, kind) in (get_poi() or []):
                xs = [p1[0], p2[0]]; ys = [p1[1], p2[1]]
                color = COLOR_AUX_EXT if kind == "aux_ext" else COLOR_AUX_PERP
                it = pg.PlotDataItem(x=xs, y=ys,
                                     pen=pg.mkPen(color, width=line_width, style=Qt.SolidLine),
                                     antialias=True)
                items.append(it); target_plot.addItem(it)

        # 2) building→dock links (all building connectors)
        get_bld = getattr(self._map, "get_building_link_segments", None)
        if callable(get_bld):
            for (p1, p2, kind, *_) in (get_bld() or []):
                xs = [p1[0], p2[0]]; ys = [p1[1], p2[1]]
                color = COLOR_AUX_EXT if kind == "aux_ext" else COLOR_AUX_PERP
                it = pg.PlotDataItem(x=xs, y=ys,
                                     pen=pg.mkPen(color, width=line_width, style=Qt.SolidLine),
                                     antialias=True)
                items.append(it); target_plot.addItem(it)

        if target_plot is self.plot:
            self._aux_items.extend(items)

    def _draw_drive_layer(self, target_plot, map_obj):
        """
        Overlay drive-lane graph (graph_drive): drive edges and drive nodes (waypoints / intersections).
        """
        g = getattr(map_obj, "graph_drive", None)
        if g is None or not getattr(g, "edges", None):
            return

        # --- Drive edges ---
        seen = set()
        for e in g.edges:
            a, b = e.node1, e.node2
            # Only draw when meta.kind == 'drive'
            meta = g.get_edge_meta(a, b) if hasattr(g, "get_edge_meta") else {}
            if not isinstance(meta, dict) or (meta.get("kind") != "drive"):
                continue
            key = tuple(sorted((id(a), id(b))))
            if key in seen: 
                continue
            seen.add(key)
            ax, ay = float(a.position.x), float(a.position.y)
            bx, by = float(b.position.x), float(b.position.y)
            target_plot.addItem(pg.PlotDataItem(
                x=[ax, bx], y=[ay, by],
                pen=pg.mkPen(COLOR_DRIVE_EDGE, width=UI_DRIVE_EDGE_WIDTH, style=Qt.SolidLine),
                antialias=True, name="drive_edge"
            ))

        # --- Drive nodes (waypoints / intersections) ---
        normals, inters = [], []
        for n in g.nodes:
            t = getattr(n, "type", "")
            xy = (float(n.position.x), float(n.position.y))
            if t == "normal":
                normals.append(xy)
            elif t == "intersection":
                inters.append(xy)

        if normals:
            target_plot.addItem(pg.ScatterPlotItem(
                pos=normals, size=UI_DRIVE_NODE_SIZE,
                brush=pg.mkBrush(COLOR_DRIVE_NODE), pen=pg.mkPen("w", width=0.5),
                symbol="o", antialias=True, name="drive_waypoints"
            ))
        if inters:
            target_plot.addItem(pg.ScatterPlotItem(
                pos=inters, size=UI_DRIVE_NODE_SIZE+1,
                brush=pg.mkBrush(COLOR_DRIVE_NODE), pen=pg.mkPen("w", width=0.8),
                symbol="d", antialias=True, name="drive_intersections"
            ))