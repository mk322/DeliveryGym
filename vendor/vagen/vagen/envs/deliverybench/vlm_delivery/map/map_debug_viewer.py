# map/map_debug_viewer.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Optional, Tuple, Callable
import os, time, math, tempfile

from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QGraphicsRectItem
)
from PyQt5.QtCore import QTimer, QCoreApplication
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

# Reuse base drawing logic and shared colors/constants
from .map_canvas_base import (
    MapCanvasBase, _node_xy, _is_road_node,
    COLOR_BG, COLOR_EDGE, COLOR_NODE, COLOR_HOP_WP, COLOR_NEXT_INT, COLOR_AGENT,
    COLOR_PICKUP, COLOR_DROPOFF, COLOR_CHG, COLOR_BUS,
    COLOR_BUILDING_PLAIN, PLAIN_FILL_ALPHA, COLOR_PLAIN_BORDER,
    UI_NODE_SIZE, UI_NODE_SMALL_SIZE, UI_HOP_SIZE, UI_NINT_SIZE, UI_POI_SIZE, UI_EDGE_WIDTH,
    LABEL_BASE_N, LABEL_STEP_N, LABEL_BASE_PUDO, LABEL_STEP_PUDO, LABEL_MAX_TRIES,
)

# ========================= Export appearance parameters =========================
EXP_EDGE_WIDTH_GLOBAL = 8.0
EXP_EDGE_WIDTH_LOCAL  = 5.0
EXP_BUS_WIDTH_GLOBAL  = 8.0
EXP_BUS_WIDTH_LOCAL   = 5.0

EXP_NODE_SIZE_GLOBAL        = 5
EXP_NODE_SMALL_SIZE_GLOBAL  = 5
EXP_NODE_SIZE_LOCAL         = 10
EXP_NODE_SMALL_SIZE_LOCAL   = 10

EXP_HOP_SIZE_GLOBAL   = 12
EXP_HOP_SIZE_LOCAL    = 20
EXP_NINT_SIZE_GLOBAL  = 10
EXP_NINT_SIZE_LOCAL   = 20
EXP_POI_SIZE_GLOBAL   = 6
EXP_POI_SIZE_LOCAL    = 15

EXP_AGENT_SIZE_GLOBAL   = 12
EXP_AGENT_SIZE_LOCAL    = 22
EXP_LABEL_PX_GLOBAL     = 10
EXP_LABEL_PX_LOCAL      = 13
EXP_ROADNAME_PX_GLOBAL  = 12
EXP_ROADNAME_PX_LOCAL   = 16
EXP_STAR_SIZE_GLOBAL    = 18
EXP_STAR_SIZE_LOCAL     = 36
EXP_STAR_BORDER_W       = 1.6

TARGET_PX_PER_M_GLOBAL  = 3.2
TARGET_PX_PER_M_LOCAL   = 6.0
MIN_EXPORT_WIDTH_PX     = 1800
MAX_EXPORT_WIDTH_PX     = 5200

AXIS_UNIT        = "m"
CM_PER_UNIT      = 100.0
TICK_STEP_UNIT   = 100.0
AXIS_FONT_PX     = 14
AXIS_LINE_WIDTH  = 2
AXIS_MARGINS_PX  = (8, 8, 4, 18)

LOCAL_MARGIN_CM  = 3500.0
GLOBAL_PAD_CM    = 2500.0

DEBUG_EDGE_WIDTH = 4.0

# Fallback font configuration (used if parent class does not override)
UI_FONT_FAMILY         = "Arial"
UI_LETTER_SPACING_PCT  = 92
EXP_FONT_FAMILY        = "Arial"
EXP_LETTER_SPACING_PCT = 92


# ------- Dynamic-layer collector proxy: intercept addItem to track removable items -------
class _CollectorProxy:
    """
    Thin wrapper around a PlotWidget that records any items added via addItem
    into an external bucket, so they can be cleared later as a dynamic layer.
    """
    def __init__(self, pw: pg.PlotWidget, bucket: List[Any]):
        self._pw = pw
        self._bucket = bucket

    def addItem(self, item):
        self._bucket.append(item)
        self._pw.addItem(item)

    @property
    def plotItem(self):
        return self._pw.plotItem

    def getPlotItem(self):
        return self._pw.getPlotItem()


class MapDebugViewer(MapCanvasBase):
    """
    Debug-oriented viewer built on MapCanvasBase.

    Features:
      * Top control bar (buttons + inputs).
      * Simple animation of agent along a path.
      * PNG export of global and local views (static base + dynamic overlay).
      * Interactive inspection of nodes and edges.
      * Optional subgraph highlighting around active orders.

    Conventions:
      * Window view: whether to draw auxiliary links is controlled by flags passed
        to MapCanvasBase; MapDebugViewer does not change that behavior.
      * Export views: auxiliary connectors (POI→dock / building→dock) are never drawn.
        Roads / crosswalks / endcaps are still rendered.
      * Export base caching:
          - prepare_export_base() draws the static background once (off-screen).
          - generate_images() only updates dynamic layers (agent / frontier / labels).
    """
    def __init__(self, title="Map Viewer"):
        super().__init__(title)

        # === Top bar ===
        top = QHBoxLayout()
        self.title_label = QLabel(title)
        top.addWidget(self.title_label)
        top.addStretch(1)
        self.btn_reinit = QPushButton("Reinit")
        self.btn_save   = QPushButton("Save PNGs")
        self.btn_start  = QPushButton("Start")
        self.btn_pause  = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        for b in (self.btn_reinit, self.btn_save, self.btn_start, self.btn_pause):
            top.addWidget(b)
        self.btn_highlight = QPushButton("Highlight Subgraph")
        top.addWidget(self.btn_highlight)

        self.input_x = QLineEdit()
        self.input_x.setPlaceholderText("Target X (cm)")
        self.input_y = QLineEdit()
        self.input_y.setPlaceholderText("Target Y (cm)")
        self.input_x.setFixedWidth(120)
        self.input_y.setFixedWidth(120)
        self.btn_go = QPushButton("Go to (x, y)")
        for w in (self.input_x, self.input_y, self.btn_go):
            top.addWidget(w)

        self.inspect_x = QLineEdit()
        self.inspect_x.setPlaceholderText("Inspect X (cm)")
        self.inspect_y = QLineEdit()
        self.inspect_y.setPlaceholderText("Inspect Y (cm)")
        self.inspect_tol = QLineEdit()
        self.inspect_tol.setPlaceholderText("Tol (cm)")
        self.inspect_x.setFixedWidth(120)
        self.inspect_y.setFixedWidth(120)
        self.inspect_tol.setFixedWidth(90)
        self.btn_inspect = QPushButton("Inspect")
        for w in (self.inspect_x, self.inspect_y, self.inspect_tol, self.btn_inspect):
            top.addWidget(w)

        self.vbox.insertLayout(0, top)

        # Animation state
        self._anim_pts: List[Tuple[float, float]] = []
        self._anim_speed = 10000.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)
        self._i = 0
        self._done = 0.0

        # Inspect overlay items
        self._inspect_items: List[Any] = []

        # External callbacks
        self._on_reinit: Optional[Callable[[], None]] = None
        self._on_go: Optional[Callable[[float, float], None]] = None
        self._anim_done_cb: Optional[Callable[[float, float], None]] = None

        self._last_show_building_links: bool = False

        # ===== Export base canvases + dynamic-layer buckets + status =====
        self._exp_pw_g: Optional[pg.PlotWidget] = None
        self._exp_pw_l: Optional[pg.PlotWidget] = None
        self._exp_dyn_g: List[Any] = []
        self._exp_dyn_l: List[Any] = []
        # Only True after prepare_export_base() succeeds
        self._export_ready: bool = False

        # Button bindings
        self.btn_reinit.clicked.connect(self._handle_reinit)
        self.btn_save.clicked.connect(self._handle_save)
        self.btn_start.clicked.connect(self.start_animation)
        self.btn_pause.clicked.connect(self.pause_animation)
        self.btn_go.clicked.connect(self._handle_go_clicked)
        self.btn_inspect.clicked.connect(self._handle_inspect_clicked)
        self.btn_highlight.clicked.connect(self._handle_highlight_clicked)

        # Overlay items for highlighted subgraph (e.g., red union-graph approximation)
        self._subgraph_items: List[Any] = []

    # ---- Font fallback (used if parent class does not provide _make_font) ----
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

    # Override: expose plain_mode in the public signature
    def draw_map(self, map_obj, world_json_path: Optional[str] = None,
                 show_bus=True, show_docks=False, show_building_links: bool = False,
                 show_road_names: bool = False, road_name_fmt: Optional[str] = None,
                 plain_mode: Optional[str] = None, show_drive: bool = False):
        """
        Wrapper around MapCanvasBase.draw_map that also remembers the last
        show_building_links setting and supports an explicit plain_mode argument.
        """
        self._last_show_building_links = bool(show_building_links)
        print("show_drive =", show_drive)
        super().draw_map(
            map_obj,
            world_json_path,
            show_bus=show_bus,
            show_docks=show_docks,
            show_building_links=show_building_links,
            show_road_names=show_road_names,
            road_name_fmt=road_name_fmt,
            plain_mode=(plain_mode or "none"),
            show_drive=show_drive,
        )
        # Redrawing the on-screen window does not affect the off-screen export bases.
        # To refresh export bases, callers must explicitly invoke prepare_export_base().

    # ============== Export base initialization / lifecycle ==============
    def invalidate_export_base(self):
        """
        Mark export bases as invalid and dispose of the off-screen canvases.

        This does NOT rebuild them automatically; callers must explicitly call
        prepare_export_base() again if they want to export fresh images.
        """
        self._export_ready = False
        for pw in (self._exp_pw_g, self._exp_pw_l):
            if pw is None:
                continue
            try:
                scene = pw.plotItem.scene()
                if scene:
                    for it in list(scene.items()):
                        try:
                            pw.removeItem(it)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                pw.deleteLater()
            except Exception:
                pass
        self._exp_pw_g = None
        self._exp_pw_l = None
        self._exp_dyn_g.clear()
        self._exp_dyn_l.clear()

    def prepare_export_base(self, map_obj: Optional[Any] = None, world_json_path: Optional[str] = None):
        """
        Build the off-screen static base for export.

        This must be called manually. It:
          * Updates self._map / self._world_path if arguments are given.
          * Clears any previous export bases.
          * Creates two off-screen PlotWidgets (global / local).
          * Draws static layers only: roads, nodes, POIs, bus routes, buildings.
            Dynamic overlays (agent / frontier / labels) are drawn later in generate_images().
        """
        if map_obj is not None:
            self._map = map_obj
        if world_json_path is not None:
            self._world_path = world_json_path
        if self._map is None:
            raise RuntimeError("prepare_export_base: map_obj is None; please pass a valid map_obj.")

        # Clear any previous export bases
        self.invalidate_export_base()

        # Create two off-screen canvases for export
        self._exp_pw_g = self._new_canvas_export(aspect_locked=True)
        self._exp_pw_l = self._new_canvas_export(aspect_locked=True)

        # Draw static base layers
        self._draw_base(self._exp_pw_g, draw_bus=True,  draw_buildings=True, export=True, profile="global")
        self._draw_base(self._exp_pw_l, draw_bus=False, draw_buildings=True, export=True, profile="local")

        self._export_ready = True

    @property
    def export_base_ready(self) -> bool:
        """
        Whether the export bases are ready to be used by generate_images().
        """
        return bool(self._export_ready and self._exp_pw_g and self._exp_pw_l)

    # ============== Animation control ==============
    def set_reinit_callback(self, fn: Callable[[], None]):
        self._on_reinit = fn

    def set_go_callback(self, fn: Callable[[float, float], None]):
        self._on_go = fn

    def set_anim_done_callback(self, fn: Callable[[float, float], None]):
        self._anim_done_cb = fn

    def get_agent_xy(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Return the agent's current coordinates, or (None, None) if not set.
        """
        return self._agent_xy if self._agent_xy else (None, None)

    def prepare_animation(self, pts: List[Tuple[float, float]], speed_cm_s: float = 10000.0):
        """
        Initialize animation trajectory and speed.

        Args:
            pts: List of waypoints in centimeters.
            speed_cm_s: Agent speed in cm/s.
        """
        self._anim_pts = list(pts or [])
        self._anim_speed = float(speed_cm_s)
        self._i = 0
        self._done = 0.0
        if self._anim_pts:
            x0, y0 = self._anim_pts[0]
            self.set_agent_xy(x0, y0)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def start_animation(self):
        """
        Start or resume the agent animation.
        """
        if len(self._anim_pts) < 2:
            return
        self._timer.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def pause_animation(self):
        """
        Pause the agent animation.
        """
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def _on_tick(self):
        """
        Per-frame animation tick handler.
        """
        if self._i >= len(self._anim_pts) - 1:
            self.pause_animation()
            if self._anim_pts:
                fx, fy = self._anim_pts[-1]
                self.set_agent_xy(fx, fy)
                if self._anim_done_cb:
                    self._anim_done_cb(fx, fy)
            return
        p0 = self._anim_pts[self._i]
        p1 = self._anim_pts[self._i + 1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            self._i += 1
            self._done = 0.0
            return
        dt = self._timer.interval() / 1000.0
        self._done += self._anim_speed * dt
        if self._done >= L:
            self.set_agent_xy(p1[0], p1[1])
            self._i += 1
            self._done = 0.0
        else:
            t = self._done / L
            self.set_agent_xy(p0[0] + dx * t, p0[1] + dy * t)

    # ============== Export (global / local) ==============
    def _draw_orders_export(
        self,
        pw: pg.PlotWidget,
        px_per_cm: float,
        taken_rects: List[Tuple[float, float, float, float]],
        star_size_px: int,
        star_border_w: float,
        label_px: int,
    ):
        """
        Draw PU/DO star markers and labels on the export canvas, and record their
        bounding boxes into taken_rects for collision avoidance.
        """
        if not getattr(self._map, "order_meta", None):
            return

        def push_circle_rect(cx: float, cy: float, size_px: float, scale: float = 0.6):
            r_cm = (size_px * scale) / max(1.0, px_per_cm)
            taken_rects.append((cx - r_cm, cx + r_cm, cy - r_cm, cy + r_cm))

        def text_rect(text: str) -> Tuple[float, float]:
            w_px = max(1.0, len(text) * label_px * 0.6)
            h_px = label_px * 1.2
            PAD = 4.0
            return (w_px / px_per_cm) * PAD, (h_px / px_per_cm) * PAD

        def place_bb(
            ax: float,
            ay: float,
            text: str,
            base: float = LABEL_BASE_PUDO,
            step: float = LABEL_STEP_PUDO,
        ) -> Tuple[float, float]:
            """
            Place label bounding box around anchor (ax, ay) while avoiding taken_rects.
            Returns the final (cx, cy) center of the label.
            """
            w_cm, h_cm = text_rect(text)
            dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)]
            for k in range(LABEL_MAX_TRIES):
                d = base + k * step
                for dx, dy in dirs:
                    cx = ax + dx * d
                    cy = ay + dy * d
                    rect = (cx - w_cm / 2, cx + w_cm / 2, cy - h_cm / 2, cy + h_cm / 2)
                    if all(not self._overlap_rect(rect, r) for r in taken_rects):
                        taken_rects.append(rect)
                        return cx, cy
            # Fallback: single offset
            cx = ax + base
            cy = ay + base
            rect = (cx - w_cm / 2, cx + w_cm / 2, cy - h_cm / 2, cy + h_cm / 2)
            taken_rects.append(rect)
            return cx, cy

        def draw_one(node, label_text: str, color: str):
            if not node:
                return
            x = float(node.position.x)
            y = float(node.position.y)
            pw.addItem(
                pg.ScatterPlotItem(
                    pos=[(x, y)],
                    size=star_size_px,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen("w", width=star_border_w),
                    symbol="star",
                    antialias=True,
                )
            )
            push_circle_rect(x, y, star_size_px)
            tx, ty = place_bb(x, y, label_text)
            ti = pg.TextItem(text=label_text, color=color, anchor=(0.5, 0.5))
            ti.setFont(self._make_font(label_px, bold=True, export=True))
            ti.setPos(tx, ty)
            ti.setZValue(12)
            pw.addItem(ti)

        for rec in self._map.order_meta:
            oid = str(rec.get("id", "")) or ""
            draw_one(rec.get("pickup_node"),  f"P{oid}", COLOR_PICKUP)
            draw_one(rec.get("dropoff_node"), f"D{oid}", COLOR_DROPOFF)

    # --- Export-only bus renderer (configurable line width; avoids name clash with parent) ---
    def _draw_bus_routes_export(self, target_plot, bus_width: float):
        for pts in self._bus_paths:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            target_plot.addItem(
                pg.PlotDataItem(
                    x=xs,
                    y=ys,
                    pen=pg.mkPen(COLOR_BUS, width=bus_width),
                    antialias=True,
                )
            )

    def generate_images(
        self,
        reachable: Dict[str, List[Dict[str, Any]]],
        show_road_names: bool = False,
        agent_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Generate two PNG images (global view and local zoom).

        Requirements:
            * prepare_export_base() must be called successfully beforehand.

        Args:
            reachable: Frontier info with "next_hop" / "next_intersections".
            show_road_names: Whether to render road-side labels.
            agent_xy: Optional override of the agent position; if None, use self._agent_xy.

        Returns:
            (global_png_bytes, local_png_bytes)
        """
        if not self.export_base_ready:
            raise RuntimeError("Export base not prepared. Call prepare_export_base(...) first.")

        # Resolve agent position (None means: do not draw agent)
        agent_pos = agent_xy if agent_xy is not None else getattr(self, "_agent_xy", None)

        # ---------- Global view ----------
        gxmin, gxmax, gymin, gymax = self._global_bounds()
        span_x_g = max(1.0, (gxmax - gxmin))
        px_per_cm_g = TARGET_PX_PER_M_GLOBAL / 100.0
        width_px_g = int(span_x_g * px_per_cm_g)
        width_px_g = max(MIN_EXPORT_WIDTH_PX, min(MAX_EXPORT_WIDTH_PX, width_px_g))
        self._exp_pw_g.plotItem.vb.setRange(xRange=(gxmin, gxmax), yRange=(gymin, gymax), padding=0)
        QCoreApplication.processEvents()

        self._clear_dynamic_items("g")
        taken_global: List[Tuple[float, float, float, float]] = []
        proxy_g = _CollectorProxy(self._exp_pw_g, self._exp_dyn_g)

        # Draw agent on global view (if present)
        if agent_pos is not None:
            ax, ay = float(agent_pos[0]), float(agent_pos[1])
            dot = pg.ScatterPlotItem(
                pos=[(ax, ay)],
                size=EXP_AGENT_SIZE_GLOBAL,
                brush=pg.mkBrush(COLOR_AGENT),
                pen=pg.mkPen("w", width=1.0),
                symbol="o",
                antialias=True,
            )
            dot.setZValue(11)
            self._exp_pw_g.addItem(dot)
            self._exp_dyn_g.append(dot)
            # Reserve a small label bubble around the agent for collision avoidance
            r_cm = (EXP_AGENT_SIZE_GLOBAL * 0.6) / max(1.0, (width_px_g / span_x_g))
            taken_global.append((ax - r_cm, ax + r_cm, ay - r_cm, ay + r_cm))

        # Orders & optional road names (global)
        self._draw_orders_export(
            proxy_g,
            px_per_cm=width_px_g / span_x_g,
            taken_rects=taken_global,
            star_size_px=EXP_STAR_SIZE_GLOBAL,
            star_border_w=EXP_STAR_BORDER_W,
            label_px=EXP_LABEL_PX_GLOBAL,
        )
        if show_road_names:
            self._draw_road_names(
                proxy_g,
                px_per_cm=width_px_g / span_x_g,
                export=True,
                size_px=EXP_ROADNAME_PX_GLOBAL,
                taken_rects=taken_global,
            )

        self._apply_sparse_axes(self._exp_pw_g, (gxmin, gxmax, gymin, gymax))
        g_bytes = self._export(
            self._exp_pw_g,
            None,
            (gxmin, gxmax, gymin, gymax),
            target_long_side_px=width_px_g,
            return_bytes=True,
        )

        # ---------- Local view ----------
        lxmin, lxmax, lymin, lymax = self._local_bounds(reachable)
        span_x_l = max(1.0, (lxmax - lxmin))
        px_per_cm_l = TARGET_PX_PER_M_LOCAL / 100.0
        width_px_l = int(span_x_l * px_per_cm_l)
        width_px_l = max(MIN_EXPORT_WIDTH_PX, min(MAX_EXPORT_WIDTH_PX, width_px_l))
        self._exp_pw_l.plotItem.vb.setRange(xRange=(lxmin, lxmax), yRange=(lymin, lymax), padding=0)
        QCoreApplication.processEvents()

        self._clear_dynamic_items("l")
        taken_local: List[Tuple[float, float, float, float]] = []
        proxy_l = _CollectorProxy(self._exp_pw_l, self._exp_dyn_l)

        def push_circle_rect_l(cx: float, cy: float, size_px: float, scale: float = 0.6) -> None:
            """
            Reserve a circular area (in cm) around (cx, cy) to avoid label overlaps.
            """
            r_cm = (size_px * scale) / max(1.0, (width_px_l / span_x_l))
            taken_local.append((cx - r_cm, cx + r_cm, cy - r_cm, cy + r_cm))

        # Reserve around next-hop / next-intersection waypoints
        for it in reachable.get("next_hop", []):
            ax, ay = float(it["x"]), float(it["y"])
            push_circle_rect_l(ax, ay, 700.0)
        for it in reachable.get("next_intersections", []):
            ax, ay = float(it["x"]), float(it["y"])
            push_circle_rect_l(ax, ay, 700.0)

        # Draw agent on local view (if present)
        if agent_pos is not None:
            ax, ay = float(agent_pos[0]), float(agent_pos[1])
            push_circle_rect_l(ax, ay, 900.0)
            dot_l = pg.ScatterPlotItem(
                pos=[(ax, ay)],
                size=EXP_AGENT_SIZE_LOCAL,
                brush=pg.mkBrush(COLOR_AGENT),
                pen=pg.mkPen("w", width=1.4),
                symbol="o",
                antialias=True,
            )
            dot_l.setZValue(11)
            self._exp_pw_l.addItem(dot_l)
            self._exp_dyn_l.append(dot_l)

        # Draw reachable network symbols & labels (local)
        self._draw_ns(
            proxy_l,
            reachable,
            with_labels=True,
            draw_poi_symbol=False,
            for_export=True,
            px_per_cm=width_px_l / span_x_l,
            taken_rects=taken_local,
            label_px=EXP_LABEL_PX_LOCAL,
            size_hop=EXP_HOP_SIZE_LOCAL,
            size_nint=EXP_NINT_SIZE_LOCAL,
        )

        # Orders / buildings (local)
        self._draw_orders_export(
            proxy_l,
            px_per_cm=width_px_l / span_x_l,
            taken_rects=taken_local,
            star_size_px=EXP_STAR_SIZE_LOCAL,
            star_border_w=EXP_STAR_BORDER_W,
            label_px=EXP_LABEL_PX_LOCAL,
        )
        self._draw_order_building_boxes(proxy_l)

        if show_road_names:
            self._draw_road_names(
                proxy_l,
                px_per_cm=width_px_l / span_x_l,
                export=True,
                size_px=EXP_ROADNAME_PX_LOCAL,
                taken_rects=taken_local,
            )

        self._apply_sparse_axes(self._exp_pw_l, (lxmin, lxmax, lymin, lymax))
        l_bytes = self._export(
            self._exp_pw_l,
            None,
            (lxmin, lxmax, lymin, lymax),
            target_long_side_px=width_px_l,
            return_bytes=True,
        )

        return g_bytes, l_bytes

    def save_images(
        self,
        reachable: Dict[str, List[Dict[str, Any]]],
        global_path: Optional[str] = "global_map.png",
        local_path: Optional[str] = "local_zoom.png",
        show_road_names: bool = False,
    ) -> None:
        """
        Convenience wrapper around generate_images() that writes PNGs to disk
        and reports file paths in the info panel.
        """
        g_bytes, l_bytes = self.generate_images(reachable, show_road_names)
        if global_path:
            with open(global_path, "wb") as f:
                f.write(g_bytes or b"")
            self.info.append(f"Saved: {os.path.abspath(global_path)}")
        else:
            self.info.append("Saved: (global skipped)")
        if local_path:
            with open(local_path, "wb") as f:
                f.write(l_bytes or b"")
            self.info.append(f"Saved: {os.path.abspath(local_path)}")
        else:
            self.info.append("Saved: (local skipped)")

    # ====== Export / layout internals ======
    def _new_canvas_export(self, aspect_locked: bool = True) -> pg.PlotWidget:
        """
        Create a PlotWidget configured for off-screen export.
        """
        w = pg.PlotWidget()
        w.setBackground(COLOR_BG)
        w.setAspectLocked(aspect_locked)
        w.showGrid(x=True, y=True, alpha=0.15)
        for side in ("left", "right", "top", "bottom"):
            w.getPlotItem().hideAxis(side)
        try:
            w.plotItem.layout.setContentsMargins(0, 0, 0, 0)
            w.plotItem.vb.setDefaultPadding(0.0)
        except Exception:
            pass
        return w

    def _apply_sparse_axes(self, pw: pg.PlotWidget, bounds: Tuple[float, float, float, float]):
        """
        Show sparse axes with labeled ticks on all sides of the given PlotWidget.
        """
        xmin, xmax, ymin, ymax = bounds
        pi = pw.getPlotItem()
        for side in ("left", "right", "top", "bottom"):
            pi.showAxis(side)
            ax = pi.getAxis(side)
            try:
                ax.setPen(pg.mkPen("k", width=AXIS_LINE_WIDTH))
                f = QFont()
                f.setPixelSize(AXIS_FONT_PX)
                ax.setStyle(tickTextOffset=6, tickFont=f)
                ax.setTextPen(pg.mkPen("k"))
            except Exception:
                pass
        try:
            pi.layout.setContentsMargins(*AXIS_MARGINS_PX)
        except Exception:
            pass

        step_cm = TICK_STEP_UNIT * CM_PER_UNIT

        def gen_ticks(a_cm: float, b_cm: float) -> List[Tuple[float, str]]:
            if step_cm <= 0:
                return []
            k0 = math.ceil(a_cm / step_cm)
            k1 = math.floor(b_cm / step_cm)
            ticks = []
            for k in range(int(k0), int(k1) + 1):
                pos_cm = k * step_cm
                val_unit = pos_cm / CM_PER_UNIT
                lab = (
                    f"{val_unit:.0f}"
                    if abs(val_unit - round(val_unit)) < 1e-6
                    else f"{val_unit:.1f}"
                )
                ticks.append((pos_cm, lab))
            return ticks

        ticks_x = gen_ticks(xmin, xmax)
        ticks_y = gen_ticks(ymin, ymax)
        try:
            pi.getAxis("bottom").setTicks([ticks_x])
            pi.getAxis("top").setTicks([ticks_x])
            pi.getAxis("left").setTicks([ticks_y])
            pi.getAxis("right").setTicks([ticks_y])
            pi.setLabel("bottom", f"x ({AXIS_UNIT})")
            pi.setLabel("left",   f"y ({AXIS_UNIT})")
        except Exception:
            pass

    def _draw_base(
        self,
        pw: pg.PlotWidget,
        draw_bus: bool,
        draw_buildings: bool,
        export: bool = False,
        profile: str = "global",
    ):
        """
        Draw static base layers: roads, nodes, POIs, optional bus routes and buildings.

        This is called once per export base in prepare_export_base().
        """
        if self._world_path:
            self._load_world(self._world_path)
            if draw_bus:
                bus_w = (
                    EXP_BUS_WIDTH_GLOBAL if profile == "global" else EXP_BUS_WIDTH_LOCAL
                ) if export else 2.0
                # Use this class's export-only method to avoid conflicting with parent signatures
                self._draw_bus_routes_export(pw, bus_width=bus_w)
            if draw_buildings:
                self._draw_buildings(pw)

        # ---- Roads ----
        edge_w = (
            EXP_EDGE_WIDTH_GLOBAL if profile == "global" else EXP_EDGE_WIDTH_LOCAL
        ) if export else UI_EDGE_WIDTH
        drawn = set()
        for a, nbs in self._map.adjacency_list.items():
            for b in nbs:
                key = tuple(sorted([(id(a),), (id(b),)]))
                if key in drawn:
                    continue
                drawn.add(key)
                get_meta = getattr(self._map, "_get_edge_meta", None)
                meta = get_meta(a, b) if callable(get_meta) else {}
                kind = (meta.get("kind") or "") if isinstance(meta, dict) else ""
                if (not kind) or kind.startswith("aux_"):
                    continue
                if kind not in ("road", "crosswalk", "endcap"):
                    continue
                if not _is_road_node(a) or not _is_road_node(b):
                    continue
                ax, ay = _node_xy(a)
                bx, by = _node_xy(b)
                pw.addItem(
                    pg.PlotDataItem(
                        x=[ax, bx],
                        y=[ay, by],
                        pen=pg.mkPen(COLOR_EDGE, width=edge_w),
                        antialias=True,
                    )
                )

        # ---- Nodes ----
        if export:
            node_small = (
                EXP_NODE_SMALL_SIZE_GLOBAL if profile == "global" else EXP_NODE_SMALL_SIZE_LOCAL
            )
            node_main = (
                EXP_NODE_SIZE_GLOBAL if profile == "global" else EXP_NODE_SIZE_LOCAL
            )
            poi_sz = (
                EXP_POI_SIZE_GLOBAL if profile == "global" else EXP_POI_SIZE_LOCAL
            )
        else:
            node_small = UI_NODE_SMALL_SIZE
            node_main = UI_NODE_SIZE
            poi_sz = UI_POI_SIZE

        normals, inters = [], []
        for n in self._map.nodes:
            t = getattr(n, "type", "")
            if t == "normal":
                normals.append(_node_xy(n))
            elif t == "intersection":
                inters.append(_node_xy(n))
        if normals:
            pw.addItem(
                pg.ScatterPlotItem(
                    pos=normals,
                    size=node_small,
                    brush=pg.mkBrush(COLOR_NODE),
                    pen=pg.mkPen("w", width=0.3),
                    symbol="o",
                    antialias=True,
                )
            )
        if inters:
            pw.addItem(
                pg.ScatterPlotItem(
                    pos=inters,
                    size=node_main,
                    brush=pg.mkBrush(COLOR_NODE),
                    pen=pg.mkPen("w", width=0.5),
                    symbol="o",
                    antialias=True,
                )
            )

        # ---- POIs (point symbols can be kept in export) ----
        if getattr(self._map, "pois", None):
            chg, bus = [], []
            for p in self._map.pois:
                t = (p.type or "").lower()
                if t == "charging_station":
                    chg.append(_node_xy(p))
                elif t == "bus_station":
                    bus.append(_node_xy(p))
            if chg:
                pw.addItem(
                    pg.ScatterPlotItem(
                        pos=chg,
                        size=poi_sz,
                        pen=pg.mkPen("w", width=1),
                        brush=pg.mkBrush(COLOR_CHG),
                        symbol="o",
                        antialias=True,
                    )
                )
            if bus:
                pw.addItem(
                    pg.ScatterPlotItem(
                        pos=bus,
                        size=poi_sz + 1,
                        pen=pg.mkPen("#8E5A0A", width=1),
                        brush=pg.mkBrush(COLOR_BUS),
                        symbol="t",
                        antialias=True,
                    )
                )

    def _draw_order_building_boxes(self, pw: pg.PlotWidget):
        """
        Draw plain gray building boxes for orders (dynamic layer driven by order_meta).
        """
        if not getattr(self._map, "order_meta", None):
            return
        for rec in self._map.order_meta:
            for key in ("pickup_building", "dropoff_building"):
                b = rec.get(key)
                if not b or not all(k in b for k in ("x", "y", "w", "h", "yaw")):
                    continue
                x = float(b["x"])
                y = float(b["y"])
                w = float(b["w"])
                h = float(b["h"])
                yaw = float(b["yaw"])
                rect = QGraphicsRectItem(x - w / 2.0, y - h / 2.0, w, h)
                rect.setTransformOriginPoint(x, y)
                rect.setRotation(yaw)
                rect.setZValue(-4)
                rect.setPen(pg.mkPen(COLOR_PLAIN_BORDER, width=1.2))
                fill_q = QColor(COLOR_BUILDING_PLAIN)
                fill_q.setAlpha(PLAIN_FILL_ALPHA)
                rect.setBrush(pg.mkBrush(fill_q))
                pw.addItem(rect)

    # ---------- Text size & rectangle collision helpers ----------
    def _text_rect_cm(self, text: str, px_per_cm: float, px: int) -> Tuple[float, float]:
        """
        Approximate a padded label bounding box for given text in centimeters.
        """
        w_px = max(1.0, len(text) * px * 0.6)
        h_px = px * 1.2
        PAD = 4.0
        return (w_px / px_per_cm) * PAD, (h_px / px_per_cm) * PAD

    def _overlap_rect(self, a, b) -> bool:
        """
        Axis-aligned rectangle overlap test.
        """
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    # ---------- N/S (frontier nodes; with label placement; export-friendly) ----------
    def _draw_ns(
        self,
        pw: pg.PlotWidget,
        reachable: Dict[str, List[Dict[str, Any]]],
        with_labels: bool,
        draw_poi_symbol: bool,
        for_export: bool = False,
        px_per_cm: float = 1.0,
        taken_rects: Optional[List[Tuple[float, float, float, float]]] = None,
        label_px: int = EXP_LABEL_PX_GLOBAL,
        size_hop: Optional[int] = None,
        size_nint: Optional[int] = None,
    ):
        """
        Draw next-hop / next-intersection markers and (optionally) labels.

        Args:
            pw: Target PlotWidget-like.
            reachable: Frontier information dict.
            with_labels: If True, draw labels for all frontier entries.
            draw_poi_symbol: If True, highlight hops with kind == "poi".
            for_export: If True, use export-oriented sizing and collision handling.
            px_per_cm: Pixels per cm, when in export mode.
            taken_rects: List of existing bounding boxes to avoid for labels (export).
            label_px: Label font size in pixels.
            size_hop: Optional hop marker size override.
            size_nint: Optional intersection marker size override.
        """
        hops_all = reachable.get("next_hop", []) or []
        hops_wp = [it for it in hops_all if it.get("kind") == "waypoint" and not it.get("is_dock", False)]

        sz_h = size_hop if size_hop is not None else (EXP_HOP_SIZE_GLOBAL if for_export else UI_HOP_SIZE)
        sz_s = size_nint if size_nint is not None else (EXP_NINT_SIZE_GLOBAL if for_export else UI_NINT_SIZE)

        if hops_wp:
            pw.addItem(
                pg.ScatterPlotItem(
                    pos=[(it["x"], it["y"]) for it in hops_wp],
                    size=sz_h,
                    brush=pg.mkBrush(COLOR_HOP_WP),
                    pen=pg.mkPen("w", width=0.8),
                    symbol="o",
                    antialias=True,
                )
            )
        if draw_poi_symbol:
            hops_poi = [it for it in hops_all if it.get("kind") == "poi"]
            if hops_poi:
                pw.addItem(
                    pg.ScatterPlotItem(
                        pos=[(it["x"], it["y"]) for it in hops_poi],
                        size=sz_h,
                        brush=pg.mkBrush("#20BF6B"),
                        pen=pg.mkPen("w", width=0.8),
                        symbol="o",
                        antialias=True,
                    )
                )
        ins = reachable.get("next_intersections", []) or []
        if ins:
            pw.addItem(
                pg.ScatterPlotItem(
                    pos=[(it["x"], it["y"]) for it in ins],
                    size=sz_s,
                    brush=pg.mkBrush(COLOR_NEXT_INT),
                    pen=pg.mkPen("w", width=1.2),
                    symbol="d",
                    antialias=True,
                )
            )

        if not with_labels:
            return

        # On-screen labels: simply center them at the node
        if not for_export:
            for it in hops_all + ins:
                lbl = it.get("label")
                if not lbl:
                    continue
                cx, cy = float(it["x"]), float(it["y"])
                ti = pg.TextItem(text=lbl, color="#1F2D3D", anchor=(0.5, 0.5))
                f = QFont()
                f.setPixelSize(label_px)
                f.setBold(True)
                ti.setFont(f)
                ti.setPos(cx, cy)
                ti.setZValue(12)
                pw.addItem(ti)
            return

        # Export labels: avoid overlaps using taken_rects
        if taken_rects is None:
            taken_rects = []

        def place_bb(
            ax: float,
            ay: float,
            text: str,
            base: float,
            step: float,
        ) -> Tuple[float, float, Tuple[float, float, float, float]]:
            """
            Place label bounding box around anchor while avoiding collisions.
            """
            w_cm, h_cm = self._text_rect_cm(text, px_per_cm, label_px)
            dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)]
            for k in range(LABEL_MAX_TRIES):
                d = base + k * step
                for dx, dy in dirs:
                    cx = ax + dx * d
                    cy = ay + dy * d
                    rect = (cx - w_cm / 2, cx + w_cm / 2, cy - h_cm / 2, cy + h_cm / 2)
                    if all(not self._overlap_rect(rect, r) for r in taken_rects):
                        taken_rects.append(rect)
                        return cx, cy, rect
            # Fallback
            cx = ax + base
            cy = ay + base
            rect = (cx - w_cm / 2, cx + w_cm / 2, cy - h_cm / 2, cy + h_cm / 2)
            taken_rects.append(rect)
            return cx, cy, rect

        def add_text(lbl: str, x: float, y: float):
            ti = pg.TextItem(text=lbl, color="#1F2D3D", anchor=(0.5, 0.5))
            ti.setFont(self._make_font(label_px, bold=True, export=True))
            ti.setPos(x, y)
            ti.setZValue(12)
            pw.addItem(ti)

        for it in hops_all + ins:
            lbl = it.get("label")
            if not lbl:
                continue
            cx, cy, _ = place_bb(float(it["x"]), float(it["y"]), lbl, LABEL_BASE_N, LABEL_STEP_N)
            add_text(lbl, cx, cy)

    # ---------- Road names (prefer parent impl; silently ignore if missing) ----------
    def _draw_road_names(
        self,
        pw: pg.PlotWidget,
        px_per_cm: Optional[float],
        export: bool,
        size_px: Optional[int] = None,
        taken_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    ):
        """
        Delegate to the parent class's _draw_road_names if available.
        """
        try:
            return super()._draw_road_names(pw, px_per_cm, export, size_px, taken_rects)
        except AttributeError:
            return

    # ---------- Bounds helpers ----------
    def _global_bounds(self):
        """
        Compute a global bounding box covering:
          * All map nodes
          * All building boxes of specific poi types
          * All bus route points (if any)
        """
        xs, ys = [], []
        for n in self._map.nodes:
            xs.append(float(n.position.x))
            ys.append(float(n.position.y))
        for props in self._world_nodes:
            p = props.get("properties", {}) if isinstance(props, dict) else {}
            poi = (p.get("poi_type") or p.get("type") or "").lower()
            if poi not in {"restaurant", "store", "rest_area", "hospital", "car_rental", "building"}:
                continue
            loc = p.get("location", {}) or {}
            bbox = p.get("bbox", {}) or {}
            x = float(loc.get("x", 0.0))
            y = float(loc.get("y", 0.0))
            w = float(bbox.get("x", 0.0))
            h = float(bbox.get("y", 0.0))
            if w > 0 and h > 0:
                xs += [x - w / 2, x + w / 2]
                ys += [y - h / 2, y + h / 2]
        for path in self._bus_paths:
            for x, y in path:
                xs.append(x)
                ys.append(y)
        if not xs:
            return -1000, 1000, -1000, 1000
        return (
            min(xs) - GLOBAL_PAD_CM,
            max(xs) + GLOBAL_PAD_CM,
            min(ys) - GLOBAL_PAD_CM,
            max(ys) + GLOBAL_PAD_CM,
        )

    def _local_bounds(self, reachable):
        """
        Local bounding box around reachable frontier and agent position.
        """
        pts = []
        for it in reachable.get("next_hop", []):
            pts.append((float(it["x"]), float(it["y"])))
        for it in reachable.get("next_intersections", []):
            pts.append((float(it["x"]), float(it["y"])))
        if self._agent_xy:
            pts.append(self._agent_xy)
        if not pts:
            return -1000, 1000, -1000, 1000
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (
            min(xs) - LOCAL_MARGIN_CM,
            max(xs) + LOCAL_MARGIN_CM,
            min(ys) - LOCAL_MARGIN_CM,
            max(ys) + LOCAL_MARGIN_CM,
        )

    def _export(
        self,
        pw: pg.PlotWidget,
        path: Optional[str],
        bounds,
        target_long_side_px: int = 3200,
        return_bytes: bool = False,
    ) -> Optional[bytes]:
        """
        Unified PNG export helper.

        Args:
            pw: PlotWidget to export.
            path: Output path; if None or return_bytes=True, a temporary file is used.
            bounds: (xmin, xmax, ymin, ymax).
            target_long_side_px: Target length of the longer side in pixels.
            return_bytes: If True, read the exported file and return its bytes.

        Returns:
            PNG bytes if return_bytes=True, otherwise None.
        """
        xmin, xmax, ymin, ymax = bounds
        pw.plotItem.vb.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0)
        QCoreApplication.processEvents()

        span_x = max(1.0, xmax - xmin)
        span_y = max(1.0, ymax - ymin)
        aspect = span_x / span_y
        width_px = int(target_long_side_px if aspect >= 1.0 else target_long_side_px * aspect)

        ex = ImageExporter(pw.plotItem)
        ex.parameters()["width"] = max(800, width_px)
        ex.parameters()["antialias"] = True

        if return_bytes or path is None:
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                ex.export(tmp)
                with open(tmp, "rb") as f:
                    data = f.read()
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            return data

        ex.export(path)
        return None

    # ------ Dynamic-layer utilities ------
    def _clear_dynamic_items(self, which: str):
        """
        Clear only dynamic overlay items from the export canvases,
        without touching the static base.
        """
        if which == "g":
            pw, bucket = self._exp_pw_g, self._exp_dyn_g
        else:
            pw, bucket = self._exp_pw_l, self._exp_dyn_l
        if not pw or not bucket:
            if bucket:
                bucket.clear()
            return
        for it in bucket:
            try:
                pw.removeItem(it)
            except Exception:
                pass
        bucket.clear()

    # ===== Top-bar button handlers / debugging =====
    def _handle_reinit(self):
        """
        Handle 'Reinit' button: stop animation, clear window scene,
        invalidate export bases, and invoke reinit callback if set.
        """
        self.pause_animation()
        self.reset_scene()
        self.invalidate_export_base()
        if self._on_reinit:
            self._on_reinit()

    def _handle_save(self):
        """
        Handle 'Save PNGs' button: export global/local images with a timestamp.
        """
        if self._last_reachable:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.save_images(self._last_reachable, f"global_{ts}.png", f"local_{ts}.png")

    def _handle_go_clicked(self):
        """
        Handle 'Go to (x, y)' button: parse target coordinates and call go callback.
        """
        try:
            tx = float(self.input_x.text().strip())
            ty = float(self.input_y.text().strip())
        except Exception:
            self.info.append("[Go] invalid target xy")
            return
        if self._on_go:
            self._on_go(tx, ty)

    def _clear_inspect_overlay(self):
        """
        Remove any inspection overlay items from the window plot.
        """
        if hasattr(self, "_inspect_items") and self._inspect_items:
            for it in self._inspect_items:
                try:
                    self.plot.removeItem(it)
                except Exception:
                    pass
            self._inspect_items.clear()

    def _clear_subgraph_overlay(self):
        """
        Remove all highlighted subgraph overlay items from the window plot.
        """
        if hasattr(self, "_subgraph_items") and self._subgraph_items:
            for it in self._subgraph_items:
                try:
                    self.plot.removeItem(it)
                except Exception:
                    pass
            self._subgraph_items.clear()

    def _handle_inspect_clicked(self):
        """
        Handle 'Inspect' button: locate the nearest node and show its local neighborhood.
        """
        if self._map is None:
            self.info.append("[Inspect] map is None")
            return
        try:
            x = float(self.inspect_x.text().strip())
            y = float(self.inspect_y.text().strip())
            tol = float(self.inspect_tol.text().strip()) if self.inspect_tol.text().strip() else 200.0
        except Exception:
            self.info.append("[Inspect] invalid inputs")
            return

        nd = self._find_node_near(x, y, tol)
        self._clear_inspect_overlay()

        if nd is None:
            self.info.append(f"[Inspect] no node within {tol:.0f} cm of ({int(x)}, {int(y)})")
            return

        nx, ny = _node_xy(nd)
        node_item = pg.ScatterPlotItem(
            pos=[(nx, ny)],
            size=UI_NODE_SIZE * 1.6,
            brush=pg.mkBrush("#00B8D9"),
            pen=pg.mkPen("w", width=1.2),
            symbol="o",
            antialias=True,
        )
        node_item.setZValue(15)
        self.plot.addItem(node_item)
        self._inspect_items.append(node_item)

        lines: List[str] = []
        tname = getattr(nd, "type", "")
        lines.append(f"[Inspect] node at ({int(nx)}, {int(ny)}) type={tname} id={id(nd)}")

        seen_nbs = set()
        for v in self._map.adjacency_list.get(nd, []):
            if v in seen_nbs:
                continue
            seen_nbs.add(v)
            vx, vy = _node_xy(v)
            meta = {}
            get_meta = getattr(self._map, "_get_edge_meta", None)
            if callable(get_meta):
                meta = get_meta(nd, v) or {}
            kind = (meta.get("kind") or "")
            name = (meta.get("name") or "")
            L = math.hypot(vx - nx, vy - ny)
            Lm = L / 100.0

            edge_item = pg.PlotDataItem(
                x=[nx, vx],
                y=[ny, vy],
                pen=pg.mkPen("#C2185B", width=DEBUG_EDGE_WIDTH),
                antialias=True,
            )
            edge_item.setZValue(14)
            self.plot.addItem(edge_item)
            self._inspect_items.append(edge_item)

            lines.append(
                f"  -> ({int(vx)}, {int(vy)})  len={Lm:.1f}m  kind='{kind}'  name='{name}'"
                f"  nb_type={getattr(v, 'type', '')}"
            )

        xs = [nx]
        ys = [ny]
        for v in seen_nbs:
            vx, vy = _node_xy(v)
            xs.append(vx)
            ys.append(vy)
        if xs and ys:
            pad = 800.0
            xmin, xmax = min(xs) - pad, max(xs) + pad
            ymin, ymax = min(ys) - pad, max(ys) + pad
            self.plot.plotItem.vb.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0)

        self.info.append("\n".join(lines))

    def _collect_order_seeds_xy(self) -> List[Tuple[float, float]]:
        """
        Collect unique PU/DO node coordinates from self._map.order_meta.

        Returns:
            List of (x, y) coordinates in centimeters.
        """
        seeds: List[Tuple[float, float]] = []
        if not getattr(self._map, "order_meta", None):
            return seeds
        seen = set()
        for rec in self._map.order_meta:
            for key in ("pickup_node", "dropoff_node"):
                nd = rec.get(key)
                if nd is None:
                    continue
                try:
                    x = float(nd.position.x)
                    y = float(nd.position.y)
                except Exception:
                    continue
                k = (int(x), int(y))  # simple integer-based deduplication
                if k in seen:
                    continue
                seen.add(k)
                seeds.append((x, y))
        return seeds

    def _handle_highlight_clicked(self):
        """
        Handle 'Highlight Subgraph' button.

        Uses a map-level helper (build_union_subgraph_from_xy) to build a small
        union subgraph that connects the agent to order endpoints, then renders
        this subgraph as an overlay (red nodes and polylines).
        """
        if self._map is None:
            self.info.append("[Highlight] map is None")
            return
        if not self._agent_xy:
            self.info.append("[Highlight] agent position is not set")
            return

        self._clear_subgraph_overlay()

        ax, ay = self._agent_xy
        seeds_xy = self._collect_order_seeds_xy()
        if not seeds_xy:
            self.info.append("[Highlight] no pickup/dropoff nodes found in active orders")
            return

        # Preferred: pure coordinate-based union-subgraph builder
        build_fn = getattr(self._map, "build_union_subgraph_from_xy", None)

        # Compatibility note: older Map implementations may expose legacy names;
        # to avoid ambiguity, we do not fall back to them automatically here.
        if build_fn is None:
            self.info.append("[Highlight] Map missing build_union_subgraph_from_xy(agent_xy, seeds_xy)")
            return

        try:
            info = build_fn((ax, ay), seeds_xy, snap_cm=120.0, include_aux=False)
        except Exception as e:
            self.info.append(f"[Highlight] build subgraph failed: {e}")
            return

        polylines = info.get("polylines", [])
        total_len_m = float(info.get("total_len_cm", 0.0)) / 100.0

        # Seeds: red circles around endpoints
        try:
            s = pg.ScatterPlotItem(
                pos=seeds_xy,
                size=18,
                brush=pg.mkBrush("#E53935"),
                pen=pg.mkPen("w", width=1.2),
                symbol="o",
                antialias=True,
            )
            s.setZValue(16)
            self.plot.addItem(s)
            self._subgraph_items.append(s)
        except Exception:
            pass

        # Agent: red dot
        dot = pg.ScatterPlotItem(
            pos=[(ax, ay)],
            size=22,
            brush=pg.mkBrush("#E53935"),
            pen=pg.mkPen("w", width=1.4),
            symbol="o",
            antialias=True,
        )
        dot.setZValue(16)
        self.plot.addItem(dot)
        self._subgraph_items.append(dot)

        # Polylines: red thick lines
        pen = pg.mkPen("#E53935", width=5.0)
        for poly in polylines:
            if len(poly) < 2:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            ln = pg.PlotDataItem(x=xs, y=ys, pen=pen, antialias=True)
            ln.setZValue(15)
            self.plot.addItem(ln)
            self._subgraph_items.append(ln)

        ne = len(info.get("edges", []))
        nn = len(info.get("nodes", []))
        self.info.append(
            f"[Highlight] union of {len(seeds_xy)} endpoints • ~{total_len_m:.1f} m • edges={ne} nodes={nn}"
        )

    def _find_node_near(self, x: float, y: float, tol_cm: float = 200.0):
        """
        Find the nearest node within tol_cm of (x, y) in centimeters.

        Returns:
            Node object or None if not found.
        """
        if self._map is None:
            return None
        best = None
        best_d2 = None
        tx, ty = float(x), float(y)
        thr2 = float(tol_cm) * float(tol_cm)
        for n in self._map.nodes:
            nx, ny = _node_xy(n)
            dx, dy = nx - tx, ny - ty
            d2 = dx * dx + dy * dy
            if d2 <= thr2 and (best is None or d2 < (best_d2 or 1e99) - 1e-9):
                best = n
                best_d2 = d2
        return best


# Backward-compatible alias
MapViewer = MapDebugViewer