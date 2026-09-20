"""
Pillow-based map exporter (no Qt) – rewritten to match Qt MapDebugViewer export output.

Exports two PNG images (bytes):
- Global view: whole map with orders, agent, POIs, buildings, axes
- Local view: cropped window around the agent with navigation markers

Visual output is designed to be pixel-level equivalent to the Qt-based MapExportor.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ========================= Colors (match Qt exactly) =========================
COLOR_BG = (247, 251, 255)  # #F7FBFF
COLOR_EDGE = (158, 167, 179)  # #9EA7B3
COLOR_NODE = (91, 155, 213)  # #5B9BD5
COLOR_HOP_WP = (31, 122, 255)  # #1F7AFF
COLOR_NEXT_INT = (0, 102, 204)  # #0066CC
COLOR_AGENT = (0, 0, 0)  # #000000
COLOR_AGENT_BORDER = (255, 255, 255)

COLOR_PICKUP = (253, 148, 136)  # #FD9488
COLOR_DROPOFF = (46, 204, 113)  # #2ECC71

COLOR_CHG = (39, 174, 96)  # #27AE60
COLOR_BUS = (243, 156, 18)  # #F39C12
COLOR_BUS_ROUTE = (255, 165, 0)  # #FFA500

# Building colors (match Qt)
BUILDING_TYPES = {"restaurant", "store", "rest_area", "hospital", "car_rental"}
COLOR_BUILDING = {
    "restaurant": (231, 76, 60),  # #E74C3C
    "store": (52, 152, 219),  # #3498DB
    "rest_area": (155, 89, 182),  # #9B59B6
    "hospital": (233, 30, 99),  # #E91E63
    "car_rental": (26, 188, 156),  # #1ABC9C
}
COLOR_BUILDING_DEFAULT = (127, 140, 141)  # #7F8C8D (Qt default)
COLOR_BUILDING_PLAIN = (176, 190, 197)  # #B0BEC5
PLAIN_FILL_ALPHA = 80
COLOR_PLAIN_BORDER = (0, 0, 0)

# Grid: Qt uses showGrid(alpha=0.15). Pre-blend gray (128,128,128) at 15% over BG.
# R=247*0.85+128*0.15≈229, G=251*0.85+128*0.15≈232, B=255*0.85+128*0.15≈236
COLOR_GRID = (229, 232, 236)

# Axis colors
COLOR_AXIS = (0, 0, 0)

# Text colors
COLOR_TEXT_LABEL = (31, 45, 61)  # #1F2D3D – same as Qt label color

# Building abbreviations (match Qt)
ABBR = {"restaurant": "R", "store": "S", "rest_area": "A", "hospital": "H", "car_rental": "C"}

# ========================= Export appearance (match map_debug_viewer.py) =======
# These are Qt "widget-pixel" sizes. The actual pixel size in the exported image
# is multiplied by px_scale = max(img_w, img_h) / REFERENCE_WIDGET_PX.
# REFERENCE_WIDGET_PX approximates the default off-screen QWidget viewport size
# used by pyqtgraph's ImageExporter.
REFERENCE_WIDGET_PX = 1500.0

# Edge widths (Qt widget pixels)
EXP_EDGE_WIDTH_GLOBAL = 8.0
EXP_EDGE_WIDTH_LOCAL = 5.0
EXP_BUS_WIDTH_GLOBAL = 8.0
EXP_BUS_WIDTH_LOCAL = 5.0

# Node sizes (Qt widget pixels) – 2x larger
EXP_NODE_SIZE_GLOBAL = 10
EXP_NODE_SIZE_LOCAL = 20

# Frontier marker sizes – 2x larger
EXP_HOP_SIZE_LOCAL = 40
EXP_NINT_SIZE_LOCAL = 40

# POI sizes – 2x larger
EXP_POI_SIZE_GLOBAL = 12
EXP_POI_SIZE_LOCAL = 30

# Agent sizes – 4x larger (black dot)
EXP_AGENT_SIZE_GLOBAL = 48
EXP_AGENT_SIZE_LOCAL = 88

# Label font sizes (Qt widget pixels) – scaled 4x for readability
EXP_LABEL_PX_GLOBAL = 30
EXP_LABEL_PX_LOCAL = 39
EXP_ROADNAME_PX_GLOBAL = 36
EXP_ROADNAME_PX_LOCAL = 48
EXP_BUILDING_ABBR_PX = 39

# Star (order marker) sizes – 2x larger
EXP_STAR_SIZE_GLOBAL = 36
EXP_STAR_SIZE_LOCAL = 72
EXP_STAR_BORDER_W = 1.6

# Image sizing
TARGET_PX_PER_M_GLOBAL = 3.2
TARGET_PX_PER_M_LOCAL = 6.0
MIN_EXPORT_WIDTH_PX = 1800
MAX_EXPORT_WIDTH_PX = 5200

# Axis
AXIS_UNIT = "m"
CM_PER_UNIT = 100.0
TICK_STEP_UNIT = 100.0  # 100m per major tick
AXIS_FONT_PX = 28
AXIS_LINE_WIDTH = 2

LOCAL_MARGIN_CM = 3500.0
GLOBAL_PAD_CM = 2500.0

# Label placement / collision avoidance (match Qt)
LABEL_BASE_N = 200.0
LABEL_STEP_N = 70.0
LABEL_BASE_PUDO = 2600.0
LABEL_STEP_PUDO = 300.0
LABEL_MAX_TRIES = 24

# Road name placement
ROAD_NAME_OFFSET_CM = 140.0
ROAD_NAME_PAD_SCALE = 10.0
ROAD_NAME_TSHIFT_CM = 500.0
ROAD_NAME_TRIES = 24


# ========================= Helpers =========================
def _node_xy(node) -> Tuple[float, float]:
    return float(node.position.x), float(node.position.y)


def _is_road_node(nd) -> bool:
    t = getattr(nd, "type", "")
    return t in ("normal", "intersection")


def _xy_of(obj) -> Optional[Tuple[float, float]]:
    if obj is None:
        return None
    try:
        return float(obj.position.x), float(obj.position.y)
    except Exception:
        pass
    try:
        return float(obj.x), float(obj.y)
    except Exception:
        pass
    try:
        return float(obj["x"]), float(obj["y"])
    except Exception:
        return None


def _try_load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try to load a font, fallback to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf",
        "Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ========================= View transform =========================
@dataclass
class _View:
    """Maps world coordinates (cm) to pixel coordinates in the output image."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    w: int  # drawable width (px)
    h: int  # drawable height (px)
    ox: int = 0  # pixel offset x (for axis margins)
    oy: int = 0  # pixel offset y

    def scale(self) -> float:
        dx = max(1.0, float(self.xmax - self.xmin))
        dy = max(1.0, float(self.ymax - self.ymin))
        sx = self.w / dx
        sy = self.h / dy
        return float(min(sx, sy))

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        s = self.scale()
        dx = max(1.0, float(self.xmax - self.xmin))
        dy = max(1.0, float(self.ymax - self.ymin))
        used_w = dx * s
        used_h = dy * s
        pad_x = (self.w - used_w) / 2.0
        pad_y = (self.h - used_h) / 2.0
        px = (float(x) - float(self.xmin)) * s + pad_x + self.ox
        py = (float(self.ymax) - float(y)) * s + pad_y + self.oy
        return px, py

    @property
    def px_per_cm(self) -> float:
        return self.scale()


def _px_scale(img_w: int, img_h: int) -> float:
    """Compute the UI scale factor for drawing sizes.

    Qt's ImageExporter scales all pen widths / scatter sizes by
    (export_pixels / widget_viewport_pixels).  We replicate this by
    dividing the actual image long-side by an empirical reference that
    approximates the default off-screen widget viewport.
    """
    return max(img_w, img_h) / REFERENCE_WIDGET_PX


def _s(base: float, scale: float) -> int:
    """Scale a base size and return as int, minimum 1."""
    return max(1, int(round(base * scale)))


def _sf(base: float, scale: float) -> float:
    """Scale a base size and return as float."""
    return max(1.0, base * scale)


# ========================= Drawing primitives =========================
def _draw_star(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    fill: Tuple[int, ...],
    outline: Tuple[int, ...] = (255, 255, 255),
    border_w: float = 1.6,
) -> None:
    """Draw a 5-pointed star (matching Qt's 'star' symbol)."""
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = size / 2.0 if i % 2 == 0 else size * 0.2
        px = cx + r * math.cos(angle)
        py = cy - r * math.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=fill, outline=outline)


def _draw_diamond(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    fill: Tuple[int, ...],
    outline: Tuple[int, ...] = (255, 255, 255),
) -> None:
    """Draw a diamond (matching Qt's 'd' symbol)."""
    r = size / 2.0
    points = [
        (cx, cy - r),
        (cx + r, cy),
        (cx, cy + r),
        (cx - r, cy),
    ]
    draw.polygon(points, fill=fill, outline=outline)


def _draw_triangle(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    fill: Tuple[int, ...],
    outline: Tuple[int, ...] = (255, 255, 255),
) -> None:
    """Draw a downward-pointing triangle (matching pyqtgraph's 't' symbol)."""
    r = size / 2.0
    points = [
        (cx, cy + r),                     # bottom point
        (cx - r * 0.866, cy - r * 0.5),   # top-left
        (cx + r * 0.866, cy - r * 0.5),   # top-right
    ]
    draw.polygon(points, fill=fill, outline=outline)


def _draw_rotated_rect(
    draw: ImageDraw.ImageDraw,
    view: "_View",
    cx_cm: float,
    cy_cm: float,
    w_cm: float,
    h_cm: float,
    yaw_deg: float,
    fill: Tuple[int, ...],
    outline: Tuple[int, ...],
    outline_width: int = 1,
) -> None:
    """Draw a rotated rectangle in world coordinates."""
    rad = math.radians(yaw_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    hw, hh = w_cm / 2.0, h_cm / 2.0
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    corners_px = []
    for lx, ly in corners_local:
        wx = cx_cm + lx * cos_a - ly * sin_a
        wy = cy_cm + lx * sin_a + ly * cos_a
        corners_px.append(view.to_px(wx, wy))
    draw.polygon(corners_px, fill=fill, outline=outline)


# ========================= Collision avoidance =========================
def _overlap_rect(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _text_rect_cm(text: str, px_per_cm: float, px: int) -> Tuple[float, float]:
    w_px = max(1.0, len(text) * px * 0.6)
    h_px = px * 1.2
    PAD = 4.0
    return (w_px / max(0.001, px_per_cm)) * PAD, (h_px / max(0.001, px_per_cm)) * PAD


def _place_label_bb(
    ax: float,
    ay: float,
    text: str,
    px_per_cm: float,
    label_px: int,
    taken: List[Tuple[float, float, float, float]],
    base: float = LABEL_BASE_N,
    step: float = LABEL_STEP_N,
) -> Tuple[float, float]:
    w_cm, h_cm = _text_rect_cm(text, px_per_cm, label_px)
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)]
    for k in range(LABEL_MAX_TRIES):
        d = base + k * step
        for dx, dy in dirs:
            cx = ax + dx * d
            cy = ay + dy * d
            rect = (cx - w_cm / 2, cx + w_cm / 2, cy - h_cm / 2, cy + h_cm / 2)
            if all(not _overlap_rect(rect, r) for r in taken):
                taken.append(rect)
                return cx, cy
    cx, cy = ax + base, ay + base
    rect = (cx - w_cm / 2, cx + w_cm / 2, cy - h_cm / 2, cy + h_cm / 2)
    taken.append(rect)
    return cx, cy


def _push_circle_rect(
    taken: List[Tuple[float, float, float, float]],
    cx: float,
    cy: float,
    size_px: float,
    px_per_cm: float,
    scale: float = 0.6,
) -> None:
    r_cm = (size_px * scale) / max(1.0, px_per_cm)
    taken.append((cx - r_cm, cx + r_cm, cy - r_cm, cy + r_cm))


# ========================= Road name helpers (match Qt) =========================
def _split_name_side(meta: dict) -> Tuple[str, Optional[str]]:
    raw_name = str(meta.get("name") or "").strip()
    side_raw = str(meta.get("side") or "").strip().lower()
    side = side_raw if side_raw in ("left", "right") else None
    if side is None:
        m = re.search(r"\((left|right)\)", raw_name, flags=re.I)
        if m:
            side = m.group(1).lower()
    base = re.sub(r"\([^)]*\)", "", raw_name).strip()
    base = re.sub(
        r"\b(road|rd|street|st|avenue|ave|boulevard|blvd|drive|dr|lane|ln|way|place|pl|court|ct|terrace|ter)\b\.?",
        "",
        base,
        flags=re.I,
    ).strip()
    return base, side


def _lr_label(base: str, side: Optional[str]) -> str:
    m = re.search(r"(\d+)", base or "")
    num = m.group(1) if m else (re.sub(r"[^A-Za-z]", "", base)[:4] or "?").upper()
    suf = "L" if side == "left" else ("R" if side == "right" else "?")
    return f"{num}{suf}"


# =========================== Main exporter class ===========================
class MapExportorPil:
    """
    PIL-based map exporter that produces images visually matching Qt MapExportor.
    Thread-safe, no Qt dependencies.
    """

    def __init__(
        self,
        *,
        map_obj: Any,
        world_json_path: Optional[str] = None,
        show_road_names: bool = True,
        global_size: Optional[Tuple[int, int]] = None,
        local_size: Optional[Tuple[int, int]] = None,
        local_radius_cm: float = 60000.0,
    ):
        self.map = map_obj
        self.world_json_path = world_json_path
        self.show_road_names = bool(show_road_names)

        self._world_nodes: List[dict] = []
        self._bus_paths: List[List[Tuple[float, float]]] = []

        # Font cache – keyed by (pixel_size, bold)
        self._fonts: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def _font(self, size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = _try_load_font(size, bold)
        return self._fonts[key]

    # ---- World JSON loading (same as Qt _load_world) ----
    def _load_world(self) -> None:
        if not self.world_json_path:
            return
        try:
            with open(self.world_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._world_nodes = data.get("nodes", [])
            self._bus_paths = []
            for r in data.get("bus_routes", []):
                pth = r.get("path", [])
                if len(pth) >= 2:
                    pts = [
                        (float(p.get("x", 0)) * 100, float(p.get("y", 0)) * 100)
                        for p in pth
                    ]
                    self._bus_paths.append(pts)
        except Exception:
            self._world_nodes = []
            self._bus_paths = []

    def prepare_base(self) -> None:
        self._load_world()

    # ---- Main export entry ----
    def export(
        self, *, agent_xy: Tuple[float, float], orders: List[Any]
    ) -> Tuple[bytes, bytes]:
        if not self._world_nodes and self.world_json_path:
            self._load_world()

        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        order_meta = self._build_order_meta(orders)

        had_attr = hasattr(self.map, "order_meta")
        prev_val = getattr(self.map, "order_meta", None)
        self.map.order_meta = order_meta

        try:
            get_reach = getattr(self.map, "get_reachable_set_xy", None)
            reachable = (
                get_reach(ax, ay)
                if callable(get_reach)
                else {"next_hop": [], "next_intersections": []}
            )
            g = self._render_global(ax, ay, reachable)
            l = self._render_local(ax, ay, reachable)
            return g, l
        finally:
            if had_attr:
                self.map.order_meta = prev_val
            else:
                try:
                    delattr(self.map, "order_meta")
                except Exception:
                    pass

    # ---- Order metadata conversion ----
    def _build_order_meta(self, orders: List[Any]) -> List[Dict[str, Any]]:
        metas: List[Dict[str, Any]] = []
        for o in orders or []:
            try:
                oid = getattr(o, "id", None)
                pu_node = getattr(o, "pickup_node", None)
                do_node = getattr(o, "dropoff_node", None)

                if pu_node is None:
                    pu_xy = _xy_of(getattr(o, "pickup_address", None))
                    if pu_xy:
                        pu_node = _NodeStub(*pu_xy)
                if do_node is None:
                    do_xy = _xy_of(getattr(o, "delivery_address", None))
                    if do_xy:
                        do_node = _NodeStub(*do_xy)

                if pu_node is None and isinstance(o, dict) and "pickup_xy" in o:
                    pu_node = _NodeStub(*o["pickup_xy"])
                if do_node is None and isinstance(o, dict) and "dropoff_xy" in o:
                    do_node = _NodeStub(*o["dropoff_xy"])

                meta: Dict[str, Any] = {"id": oid}
                if pu_node is not None:
                    meta["pickup_node"] = pu_node
                if do_node is not None:
                    meta["dropoff_node"] = do_node

                pu_bld = getattr(o, "pickup_building", None)
                do_bld = getattr(o, "dropoff_building", None)
                if pu_bld:
                    meta["pickup_building"] = pu_bld
                if do_bld:
                    meta["dropoff_building"] = do_bld
                metas.append(meta)
            except Exception:
                continue
        return metas

    # ==================================================================
    # Bounds computation (match Qt exactly)
    # ==================================================================
    def _global_bounds(self) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for n in self.map.nodes:
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

    def _local_bounds(
        self, ax: float, ay: float, reachable: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[float, float, float, float]:
        pts: List[Tuple[float, float]] = [(ax, ay)]
        for it in reachable.get("next_hop", []):
            pts.append((float(it["x"]), float(it["y"])))
        for it in reachable.get("next_intersections", []):
            pts.append((float(it["x"]), float(it["y"])))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (
            min(xs) - LOCAL_MARGIN_CM,
            max(xs) + LOCAL_MARGIN_CM,
            min(ys) - LOCAL_MARGIN_CM,
            max(ys) + LOCAL_MARGIN_CM,
        )

    # ==================================================================
    # Image size computation (match Qt)
    # ==================================================================
    def _compute_image_size(
        self,
        bounds: Tuple[float, float, float, float],
        target_px_per_m: float,
    ) -> Tuple[int, int]:
        """Match Qt's sizing: target is computed from span_x, then adjusted by aspect."""
        xmin, xmax, ymin, ymax = bounds
        span_x = max(1.0, xmax - xmin)
        span_y = max(1.0, ymax - ymin)
        px_per_cm = target_px_per_m / 100.0
        # Qt computes target from span_x (not max of both spans)
        target_px = int(span_x * px_per_cm)
        target_px = max(MIN_EXPORT_WIDTH_PX, min(MAX_EXPORT_WIDTH_PX, target_px))

        aspect = span_x / span_y
        if aspect >= 1.0:
            # Landscape: width = target, height from aspect
            w = target_px
            h = max(800, int(target_px / aspect))
        else:
            # Portrait: width = target * aspect, height = target
            w = max(800, int(target_px * aspect))
            h = target_px
        return w, h

    # ==================================================================
    # Global View
    # ==================================================================
    def _render_global(
        self,
        ax: float,
        ay: float,
        reachable: Dict[str, List[Dict[str, Any]]],
    ) -> bytes:
        bounds = self._global_bounds()
        img_w, img_h = self._compute_image_size(bounds, TARGET_PX_PER_M_GLOBAL)
        sc = _px_scale(img_w, img_h)

        # Scaled axis margins – large enough for 4x font
        margin_l = _s(120, sc)
        margin_r = _s(120, sc)
        margin_t = _s(60, sc)
        margin_b = _s(100, sc)
        draw_w = img_w - margin_l - margin_r
        draw_h = img_h - margin_t - margin_b

        view = _View(
            xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
            w=draw_w, h=draw_h, ox=margin_l, oy=margin_t,
        )

        im = Image.new("RGB", (img_w, img_h), COLOR_BG)
        draw = ImageDraw.Draw(im)
        taken: List[Tuple[float, float, float, float]] = []

        # ---- Static base layers ----
        self._draw_grid(draw, view, sc)
        self._draw_bus_routes(draw, view, width=_sf(EXP_BUS_WIDTH_GLOBAL, sc))
        self._draw_buildings_world(draw, view, sc)
        self._draw_edges(draw, view, width=_sf(EXP_EDGE_WIDTH_GLOBAL, sc))
        self._draw_nodes(draw, view, node_size=_sf(EXP_NODE_SIZE_GLOBAL, sc), sc=sc)
        self._draw_pois(draw, view, poi_size=_sf(EXP_POI_SIZE_GLOBAL, sc))

        # ---- Dynamic layers ----
        agent_sz = _sf(EXP_AGENT_SIZE_GLOBAL, sc)
        self._draw_agent(draw, view, ax, ay, agent_sz, sc)
        _push_circle_rect(taken, ax, ay, agent_sz, view.px_per_cm)

        star_sz = _sf(EXP_STAR_SIZE_GLOBAL, sc)
        label_px = _s(EXP_LABEL_PX_GLOBAL, sc)
        self._draw_orders_export(draw, view, star_size=star_sz, label_px=label_px, taken=taken, sc=sc)

        if self.show_road_names:
            rn_px = _s(EXP_ROADNAME_PX_GLOBAL, sc)
            self._draw_road_names(draw, view, size_px=rn_px, taken=taken)

        self._draw_axes(draw, view, img_w, img_h, sc)
        return self._to_png_bytes(self._down_resize(im))

    # ==================================================================
    # Local View
    # ==================================================================
    def _render_local(
        self,
        ax: float,
        ay: float,
        reachable: Dict[str, List[Dict[str, Any]]],
    ) -> bytes:
        bounds = self._local_bounds(ax, ay, reachable)
        img_w, img_h = self._compute_image_size(bounds, TARGET_PX_PER_M_LOCAL)
        sc = _px_scale(img_w, img_h)

        margin_l = _s(120, sc)
        margin_r = _s(120, sc)
        margin_t = _s(60, sc)
        margin_b = _s(100, sc)
        draw_w = img_w - margin_l - margin_r
        draw_h = img_h - margin_t - margin_b

        view = _View(
            xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
            w=draw_w, h=draw_h, ox=margin_l, oy=margin_t,
        )

        im = Image.new("RGB", (img_w, img_h), COLOR_BG)
        draw = ImageDraw.Draw(im)
        taken: List[Tuple[float, float, float, float]] = []

        for it in reachable.get("next_hop", []):
            _push_circle_rect(taken, float(it["x"]), float(it["y"]), 700.0, view.px_per_cm)
        for it in reachable.get("next_intersections", []):
            _push_circle_rect(taken, float(it["x"]), float(it["y"]), 700.0, view.px_per_cm)

        # ---- Static base layers ----
        self._draw_grid(draw, view, sc)
        self._draw_buildings_world(draw, view, sc)
        self._draw_edges(draw, view, width=_sf(EXP_EDGE_WIDTH_LOCAL, sc))
        self._draw_nodes(draw, view, node_size=_sf(EXP_NODE_SIZE_LOCAL, sc), sc=sc)
        self._draw_pois(draw, view, poi_size=_sf(EXP_POI_SIZE_LOCAL, sc))

        # ---- Dynamic layers ----
        agent_sz = _sf(EXP_AGENT_SIZE_LOCAL, sc)
        _push_circle_rect(taken, ax, ay, 900.0, view.px_per_cm)
        self._draw_agent(draw, view, ax, ay, agent_sz, sc)

        hop_sz = _sf(EXP_HOP_SIZE_LOCAL, sc)
        nint_sz = _sf(EXP_NINT_SIZE_LOCAL, sc)
        label_px = _s(EXP_LABEL_PX_LOCAL, sc)
        self._draw_frontier(draw, view, reachable, size_hop=hop_sz, size_nint=nint_sz,
                            label_px=label_px, taken=taken, sc=sc)

        star_sz = _sf(EXP_STAR_SIZE_LOCAL, sc)
        self._draw_orders_export(draw, view, star_size=star_sz, label_px=label_px, taken=taken, sc=sc)
        self._draw_order_building_boxes(draw, view)

        if self.show_road_names:
            rn_px = _s(EXP_ROADNAME_PX_LOCAL, sc)
            self._draw_road_names(draw, view, size_px=rn_px, taken=taken)

        self._draw_axes(draw, view, img_w, img_h, sc)
        return self._to_png_bytes(self._down_resize(im))

    # ==================================================================
    # Drawing: Grid (match Qt showGrid alpha=0.15)
    # ==================================================================
    def _draw_grid(self, draw: ImageDraw.ImageDraw, view: _View, sc: float) -> None:
        step_cm = TICK_STEP_UNIT * CM_PER_UNIT
        grid_w = _s(1, sc)
        x = math.floor(view.xmin / step_cm) * step_cm
        while x <= view.xmax:
            px, _ = view.to_px(x, view.ymin)
            _, py_top = view.to_px(x, view.ymax)
            _, py_bot = view.to_px(x, view.ymin)
            draw.line([(px, py_top), (px, py_bot)], fill=COLOR_GRID, width=grid_w)
            x += step_cm
        y = math.floor(view.ymin / step_cm) * step_cm
        while y <= view.ymax:
            px_l, py = view.to_px(view.xmin, y)
            px_r, _ = view.to_px(view.xmax, y)
            draw.line([(px_l, py), (px_r, py)], fill=COLOR_GRID, width=grid_w)
            y += step_cm

    # ==================================================================
    # Drawing: Axes
    # ==================================================================
    def _draw_axes(
        self, draw: ImageDraw.ImageDraw, view: _View, img_w: int, img_h: int, sc: float
    ) -> None:
        font_sz = _s(AXIS_FONT_PX, sc)
        font = self._font(font_sz, bold=False)
        label_font = self._font(font_sz, bold=True)
        line_w = _s(AXIS_LINE_WIDTH, sc)
        tick_len = _s(5, sc)

        x0, y0 = view.to_px(view.xmin, view.ymax)
        x1, y1 = view.to_px(view.xmax, view.ymin)
        draw.rectangle([(x0, y0), (x1, y1)], outline=COLOR_AXIS, width=line_w)

        step_cm = TICK_STEP_UNIT * CM_PER_UNIT

        # X ticks
        k0 = math.ceil(view.xmin / step_cm)
        k1 = math.floor(view.xmax / step_cm)
        for k in range(int(k0), int(k1) + 1):
            pos_cm = k * step_cm
            val_unit = pos_cm / CM_PER_UNIT
            label = f"{val_unit:.0f}" if abs(val_unit - round(val_unit)) < 1e-6 else f"{val_unit:.1f}"
            px, _ = view.to_px(pos_cm, view.ymin)
            draw.line([(px, y1), (px, y1 + tick_len)], fill=COLOR_AXIS, width=1)
            draw.text((px, y1 + tick_len + 2), label, fill=COLOR_AXIS, font=font, anchor="mt")
            draw.line([(px, y0 - tick_len), (px, y0)], fill=COLOR_AXIS, width=1)
            draw.text((px, y0 - tick_len - 2), label, fill=COLOR_AXIS, font=font, anchor="mb")

        # Y ticks
        k0 = math.ceil(view.ymin / step_cm)
        k1 = math.floor(view.ymax / step_cm)
        for k in range(int(k0), int(k1) + 1):
            pos_cm = k * step_cm
            val_unit = pos_cm / CM_PER_UNIT
            label = f"{val_unit:.0f}" if abs(val_unit - round(val_unit)) < 1e-6 else f"{val_unit:.1f}"
            _, py = view.to_px(view.xmin, pos_cm)
            draw.line([(x0 - tick_len, py), (x0, py)], fill=COLOR_AXIS, width=1)
            draw.text((x0 - tick_len - 2, py), label, fill=COLOR_AXIS, font=font, anchor="rm")
            draw.line([(x1, py), (x1 + tick_len, py)], fill=COLOR_AXIS, width=1)
            draw.text((x1 + tick_len + 2, py), label, fill=COLOR_AXIS, font=font, anchor="lm")

        mid_x = (x0 + x1) / 2
        draw.text((mid_x, img_h - 2), f"x ({AXIS_UNIT})", fill=COLOR_AXIS, font=label_font, anchor="mb")
        draw.text((4, (y0 + y1) / 2), f"y ({AXIS_UNIT})", fill=COLOR_AXIS, font=label_font, anchor="lm")

    # ==================================================================
    # Drawing: Edges
    # ==================================================================
    def _draw_edges(self, draw: ImageDraw.ImageDraw, view: _View, width: float) -> None:
        get_meta = getattr(self.map, "_get_edge_meta", None)
        drawn = set()
        w = max(1, int(round(width)))
        for a, nbs in self.map.adjacency_list.items():
            for b in nbs:
                key = tuple(sorted([(id(a),), (id(b),)]))
                if key in drawn:
                    continue
                drawn.add(key)
                meta = get_meta(a, b) if callable(get_meta) else {}
                kind = (meta.get("kind") or "") if isinstance(meta, dict) else ""
                if (not kind) or kind.startswith("aux_"):
                    continue
                if kind not in ("road", "crosswalk", "endcap"):
                    continue
                if not _is_road_node(a) or not _is_road_node(b):
                    continue
                p1 = view.to_px(*_node_xy(a))
                p2 = view.to_px(*_node_xy(b))
                draw.line([p1, p2], fill=COLOR_EDGE, width=w)

    # ==================================================================
    # Drawing: Nodes
    # ==================================================================
    def _draw_nodes(self, draw: ImageDraw.ImageDraw, view: _View, node_size: float, sc: float) -> None:
        pen_w_normal = max(1.0, 0.3 * sc)
        pen_w_inter = max(1.0, 0.5 * sc)
        for n in self.map.nodes:
            t = getattr(n, "type", "")
            if t not in ("normal", "intersection"):
                continue
            px, py = view.to_px(*_node_xy(n))
            r = node_size / 2.0
            pw = pen_w_inter if t == "intersection" else pen_w_normal
            # White outline ring
            draw.ellipse(
                (px - r - pw, py - r - pw, px + r + pw, py + r + pw),
                fill=(255, 255, 255),
            )
            # Filled node
            draw.ellipse((px - r, py - r, px + r, py + r), fill=COLOR_NODE)

    # ==================================================================
    # Drawing: Bus routes
    # ==================================================================
    def _draw_bus_routes(self, draw: ImageDraw.ImageDraw, view: _View, width: float) -> None:
        w = max(1, int(round(width)))
        for pts in self._bus_paths:
            px_pts = [view.to_px(x, y) for x, y in pts]
            if len(px_pts) >= 2:
                draw.line(px_pts, fill=COLOR_BUS_ROUTE, width=w)

    # ==================================================================
    # Drawing: Buildings from world JSON
    # ==================================================================
    def _draw_buildings_world(self, draw: ImageDraw.ImageDraw, view: _View, sc: float) -> None:
        if not self._world_nodes:
            return
        bld_font_sz = _s(int(EXP_BUILDING_ABBR_PX * 0.55), sc)
        font = self._font(bld_font_sz, bold=True)
        outline_w = _s(1, sc)
        name_lookup = self._build_poi_name_lookup()

        for n in self._world_nodes:
            props = n.get("properties", {}) or {}
            poi = (props.get("poi_type") or props.get("type") or "").strip().lower()
            if poi not in BUILDING_TYPES:
                continue
            loc = props.get("location", {}) or {}
            ori = props.get("orientation", {}) or {}
            bbox = props.get("bbox", {}) or {}
            x = float(loc.get("x", 0.0))
            y = float(loc.get("y", 0.0))
            yaw = float(ori.get("yaw", 0.0))
            w = float(bbox.get("x", 0.0)) or 600.0
            h = float(bbox.get("y", 0.0)) or 600.0

            fill = COLOR_BUILDING.get(poi, COLOR_BUILDING_DEFAULT)
            _draw_rotated_rect(draw, view, x, y, w, h, yaw, fill=fill, outline=(0, 0, 0), outline_width=outline_w)

            label = name_lookup.get((round(x), round(y))) or ABBR.get(poi, "?")
            cpx, cpy = view.to_px(x, y)
            draw.text((cpx, cpy), label, fill=(255, 255, 255), font=font, anchor="mm")

    def _build_poi_name_lookup(self) -> Dict[Tuple[int, int], str]:
        out: Dict[Tuple[int, int], str] = {}
        poi_meta = getattr(self.map, "poi_meta", None)
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

    # ==================================================================
    # Drawing: POIs
    # ==================================================================
    def _draw_pois(self, draw: ImageDraw.ImageDraw, view: _View, poi_size: float) -> None:
        pois = getattr(self.map, "pois", None)
        if not pois:
            return
        for p in pois:
            t = (getattr(p, "type", None) or "").lower()
            if t not in ("charging_station", "bus_station"):
                continue
            px, py = view.to_px(*_node_xy(p))
            if t == "charging_station":
                r = poi_size / 2.0
                draw.ellipse((px - r, py - r, px + r, py + r), fill=COLOR_CHG, outline=(255, 255, 255))
            elif t == "bus_station":
                _draw_triangle(draw, px, py, poi_size + 1, fill=COLOR_BUS, outline=(142, 90, 10))

    # ==================================================================
    # Drawing: Agent
    # ==================================================================
    def _draw_agent(
        self, draw: ImageDraw.ImageDraw, view: _View,
        ax: float, ay: float, size: float, sc: float,
    ) -> None:
        px, py = view.to_px(ax, ay)
        r = size / 2.0
        border_w = max(1.0, 1.4 * sc)
        draw.ellipse(
            (px - r - border_w, py - r - border_w, px + r + border_w, py + r + border_w),
            fill=COLOR_AGENT_BORDER,
        )
        draw.ellipse((px - r, py - r, px + r, py + r), fill=COLOR_AGENT)

    # ==================================================================
    # Drawing: Orders as stars
    # ==================================================================
    def _draw_orders_export(
        self, draw: ImageDraw.ImageDraw, view: _View,
        star_size: float, label_px: int,
        taken: List[Tuple[float, float, float, float]], sc: float,
    ) -> None:
        order_meta = getattr(self.map, "order_meta", None)
        if not order_meta:
            return
        font = self._font(label_px, bold=True)

        for rec in order_meta:
            oid = str(rec.get("id", "") or "")
            for key, prefix, color in [
                ("pickup_node", "P", COLOR_PICKUP),
                ("dropoff_node", "D", COLOR_DROPOFF),
            ]:
                node = rec.get(key)
                if not node:
                    continue
                xy = _xy_of(node)
                if not xy:
                    continue
                x, y = xy
                px, py = view.to_px(x, y)

                border_w = max(1.0, EXP_STAR_BORDER_W * sc)
                _draw_star(draw, px, py, star_size, fill=color, outline=(255, 255, 255), border_w=border_w)
                _push_circle_rect(taken, x, y, star_size, view.px_per_cm)

                label = f"{prefix}{oid}"
                lx, ly = _place_label_bb(
                    x, y, label, view.px_per_cm, label_px, taken,
                    base=LABEL_BASE_PUDO, step=LABEL_STEP_PUDO,
                )
                lpx, lpy = view.to_px(lx, ly)
                draw.text((lpx, lpy), label, fill=color, font=font, anchor="mm")

    # ==================================================================
    # Drawing: Order building boxes (gray)
    # ==================================================================
    def _draw_order_building_boxes(self, draw: ImageDraw.ImageDraw, view: _View) -> None:
        order_meta = getattr(self.map, "order_meta", None)
        if not order_meta:
            return
        for rec in order_meta:
            for key in ("pickup_building", "dropoff_building"):
                b = rec.get(key)
                if not b or not all(k in b for k in ("x", "y", "w", "h", "yaw")):
                    continue
                x, y = float(b["x"]), float(b["y"])
                w, h = float(b["w"]), float(b["h"])
                yaw = float(b["yaw"])
                fill_rgba = COLOR_BUILDING_PLAIN + (PLAIN_FILL_ALPHA,)
                _draw_rotated_rect(draw, view, x, y, w, h, yaw, fill=fill_rgba, outline=COLOR_PLAIN_BORDER)

    # ==================================================================
    # Drawing: Frontier markers
    # ==================================================================
    def _draw_frontier(
        self, draw: ImageDraw.ImageDraw, view: _View,
        reachable: Dict[str, List[Dict[str, Any]]],
        size_hop: float, size_nint: float, label_px: int,
        taken: List[Tuple[float, float, float, float]], sc: float,
    ) -> None:
        font = self._font(label_px, bold=True)

        hops_all = reachable.get("next_hop", []) or []
        hops_wp = [it for it in hops_all if it.get("kind") == "waypoint" and not it.get("is_dock", False)]

        pen_w = max(1.0, 0.8 * sc)
        for it in hops_wp:
            px, py = view.to_px(float(it["x"]), float(it["y"]))
            r = size_hop / 2.0
            draw.ellipse((px - r - pen_w, py - r - pen_w, px + r + pen_w, py + r + pen_w), fill=(255, 255, 255))
            draw.ellipse((px - r, py - r, px + r, py + r), fill=COLOR_HOP_WP)

        ins = reachable.get("next_intersections", []) or []
        for it in ins:
            px, py = view.to_px(float(it["x"]), float(it["y"]))
            _draw_diamond(draw, px, py, size_nint, fill=COLOR_NEXT_INT, outline=(255, 255, 255))

        for it in hops_all + ins:
            lbl = it.get("label")
            if not lbl:
                continue
            x, y = float(it["x"]), float(it["y"])
            lx, ly = _place_label_bb(x, y, lbl, view.px_per_cm, label_px, taken,
                                     base=LABEL_BASE_N, step=LABEL_STEP_N)
            lpx, lpy = view.to_px(lx, ly)
            draw.text((lpx, lpy), lbl, fill=COLOR_TEXT_LABEL, font=font, anchor="mm")

    # ==================================================================
    # Drawing: Road names
    # ==================================================================
    def _draw_road_names(
        self, draw: ImageDraw.ImageDraw, view: _View,
        size_px: int, taken: List[Tuple[float, float, float, float]],
    ) -> None:
        if self.map is None:
            return
        get_meta = getattr(self.map, "_get_edge_meta", None)
        font = self._font(size_px, bold=True)

        agg: Dict[Tuple[str, str], Dict[str, float]] = {}
        seen_pairs: set = set()

        for a, nbs in self.map.adjacency_list.items():
            for b in nbs:
                keyp = tuple(sorted((id(a), id(b))))
                if keyp in seen_pairs:
                    continue
                seen_pairs.add(keyp)
                if not _is_road_node(a) or not _is_road_node(b):
                    continue
                meta = get_meta(a, b) if callable(get_meta) else {}
                if not isinstance(meta, dict) or meta.get("kind") != "road":
                    continue
                base, side = _split_name_side(meta)
                if not base or side not in ("left", "right"):
                    continue
                ax_c, ay_c = _node_xy(a)
                bx_c, by_c = _node_xy(b)
                dx, dy = bx_c - ax_c, by_c - ay_c
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    continue
                mx, my = (ax_c + bx_c) / 2.0, (ay_c + by_c) / 2.0
                rec = agg.setdefault((base, side), dict(sumL=0.0, cx=0.0, cy=0.0, vx=0.0, vy=0.0))
                rec["sumL"] += L
                rec["cx"] += mx * L
                rec["cy"] += my * L
                rec["vx"] += dx
                rec["vy"] += dy

        if not agg:
            return

        offsets = [ROAD_NAME_OFFSET_CM * k for k in (1.0, 1.4, 1.8, 2.2)]
        slides = [0.0, ROAD_NAME_TSHIFT_CM, -ROAD_NAME_TSHIFT_CM, 2 * ROAD_NAME_TSHIFT_CM, -2 * ROAD_NAME_TSHIFT_CM]

        for (base, side), rec in agg.items():
            sumL = rec["sumL"]
            if sumL <= 0:
                continue
            mx = rec["cx"] / sumL
            my = rec["cy"] / sumL
            vx, vy = rec["vx"], rec["vy"]
            vlen = math.hypot(vx, vy)
            tx, ty = (vx / vlen, vy / vlen) if vlen > 1e-6 else (1.0, 0.0)
            nx, ny = -ty, tx

            label = _lr_label(base, side)
            sgn = -1 if side == "left" else 1
            w_cm, h_cm = _text_rect_cm(label, view.px_per_cm, size_px)

            placed = False
            tries = 0
            px_, py_ = mx, my
            for off in offsets:
                for sl in slides:
                    px_ = mx + sgn * nx * off + tx * sl
                    py_ = my + sgn * ny * off + ty * sl
                    rect = (px_ - w_cm / 2, px_ + w_cm / 2, py_ - h_cm / 2, py_ + h_cm / 2)
                    if all(not _overlap_rect(rect, r) for r in taken):
                        taken.append(rect)
                        placed = True
                        break
                    tries += 1
                    if tries >= ROAD_NAME_TRIES:
                        break
                    if placed or tries >= ROAD_NAME_TRIES:
                        break
            if not placed:
                px_ = mx + sgn * nx * offsets[0]
                py_ = my + sgn * ny * offsets[0]

            lpx, lpy = view.to_px(px_, py_)
            draw.text((lpx, lpy), label, fill=COLOR_AXIS, font=font, anchor="mm")

    # ==================================================================
    # Utility
    # ==================================================================
    def _down_resize(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        resampling = getattr(Image, "Resampling", Image)
        return im.resize((max(1, w // 8), max(1, h // 8)), resampling.LANCZOS)

    def _to_png_bytes(self, im: Image.Image) -> bytes:
        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


# ---- Stub node for order_meta conversion ----
class _Pos:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)


class _NodeStub:
    def __init__(self, x: float, y: float):
        self.position = _Pos(x, y)
        self.type = "normal"