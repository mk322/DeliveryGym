"""
Offline Google-Maps-style PNG renderer for a DeliveryBench scenario.

Reads a scenario folder (roads.json + progen_world_enriched.json) and writes a
single PNG styled like Google Maps day mode: cream-beige background with a
slightly greener city interior, road casings and inner fills with synthesized
street names along centerlines, building polygons with subtle 3D extrusion,
teardrop POI pins with vector icons (fork+knife, shopping bag, hospital cross,
tree, car), green pill markers with lightning bolts for EV charging stations
and orange pills with bus icons for bus stations, optional agent dot, plus a
compass and scale bar.

Pure Pillow. No new dependencies. Intended for offline batch rendering — not
wired into the env's per-step observation loop.

Usage:
    python3 -m vagen.envs.deliverybench.tools.render_gmaps \\
        <scenario_dir> [-o <out.png>] [--agent X Y]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------- Style tokens -----------------------------------

STYLE = dict(
    bg="#E8E1CC",                    # outer "outside the city" tint
    city_bg="#E4E8D6",               # inside the road network area (slight green = "land")
    park_fill="#CFE3B5",
    park_border="#A8C68A",
    parking_fill="#D8D3C5",
    water_fill="#A6CFE0",
    road_casing="#8A8F95",
    road_casing_hwy="#C9A24E",
    road_fill_local="#FFFFFF",
    road_fill_hwy="#FED14B",
    lane_stripe="#FFFFFF",
    crosswalk="#FFFFFF",
    bus_route="#FB8C00",
    building_fill="#D8D1BC",         # wall (slightly darker)
    building_fill_roof="#ECE6D2",    # roof (lighter)
    building_border="#9C947D",
    building_shadow=(0, 0, 0, 70),
    building_poi=dict(
        restaurant="#F9AB00",   # yellow/amber (was red)
        store="#4285F4",
        rest_area="#8E44AD",
        hospital="#E91E63",
        car_rental="#00897B",
    ),
    poi_abbr=dict(
        restaurant="R", store="S", rest_area="A", hospital="H", car_rental="C",
    ),
    poi_full=dict(
        restaurant="Restaurant", store="Store", rest_area="Rest Area",
        hospital="Hospital", car_rental="Car Rental",
    ),
    poi_charging="#1E8E3E",
    poi_bus="#FB8C00",
    agent="#1A73E8",
    agent_halo=(26, 115, 232, 70),
    agent_facing="#FF6D00",   # vivid orange facing wedge (high contrast vs blue dot)
    pickup="#EA4335",
    dropoff="#9AA0A6",        # gray (was green)
    label_dark="#202124",
    label_light="#FFFFFF",
    label_road="#5F6368",
    label_road_outline="#F1ECDF",
    shield_bg="#FFFFFF",
    shield_border="#B0B6BD",
)

FONT_CANDIDATES = [
    # Linux (this is where rollouts run) — DejaVu ships with most distros.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    # macOS fallbacks.
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

# Output size and margins. The long side is targeted; the short side is sized
# from the data's aspect ratio so we don't waste half the image on cream borders.
OUT_LONG = 2400
OUT_MIN = 1400
MARGIN_PX = 90

# Visual constants computed at render time from the view scale; see _stroke_widths.
DEFAULT_STREET_WIDTH_M = 7.5    # ~24ft, typical local street
HIGHWAY_WIDTH_M = 12.0
BUS_ROUTE_WIDTH_M = 1.5
BUILDING_SHADOW_OFFSET_PX = (3, 4)

# Pin/marker visuals (screen px).
PIN_RADIUS_PX = 18
AGENT_DOT_R = 14
AGENT_HALO_R = 40

# Road-name placement.
MIN_LABEL_ROAD_LEN_PX = 90    # only label roads at least this long on screen
ROAD_LABEL_SIZE_PX = 38


# ----------------------------- Data loading -----------------------------------

@dataclass
class World:
    roads: List[Tuple[Tuple[float, float], Tuple[float, float], bool]]
    buildings: List[dict]
    poi_buildings: List[dict]
    point_pois: List[dict]
    bus_routes: List[List[Tuple[float, float]]]
    bounds: Tuple[float, float, float, float]   # xmin xmax ymin ymax cm
    road_bounds: Tuple[float, float, float, float]


def load_world(scenario_dir: Path) -> World:
    with (scenario_dir / "roads.json").open() as f:
        rd = json.load(f)
    with (scenario_dir / "progen_world_enriched.json").open() as f:
        wd = json.load(f)

    roads = []
    for r in rd.get("roads", []):
        s = r["start"]; e = r["end"]
        roads.append((
            (float(s["x"]) * 100.0, float(s["y"]) * 100.0),
            (float(e["x"]) * 100.0, float(e["y"]) * 100.0),
            bool(r.get("is_highway", False)),
        ))

    building_like = {"restaurant", "store", "rest_area", "hospital", "car_rental",
                     "customer", "building"}
    point_like = {"charging_station", "bus_station"}
    poi_buildings: List[dict] = []
    plain_buildings: List[dict] = []
    point_pois: List[dict] = []

    for n in wd.get("nodes", []):
        props = n.get("properties", {}) or {}
        inst = str(n.get("instance_name", "") or "")
        loc = props.get("location", {}) or {}
        ori = props.get("orientation", {}) or {}
        bbox = props.get("bbox", {}) or {}
        pt = (props.get("poi_type") or props.get("type") or "").strip().lower()

        rec = dict(
            x=float(loc.get("x", 0.0)),
            y=float(loc.get("y", 0.0)),
            yaw=float(ori.get("yaw", 0.0)),
            w=float(bbox.get("x", 0.0)) or 600.0,
            h=float(bbox.get("y", 0.0)) or 600.0,
            poi_type=pt,
            inst=inst,
        )

        if pt in building_like and pt != "building":
            poi_buildings.append(rec)
        elif pt == "building" or inst.startswith("BP_Building"):
            plain_buildings.append(rec)
        elif pt in point_like:
            point_pois.append(rec)

    bus_routes: List[List[Tuple[float, float]]] = []
    for br in wd.get("bus_routes", []):
        pts = [(float(p.get("x", 0.0)) * 100.0, float(p.get("y", 0.0)) * 100.0)
               for p in br.get("path", [])]
        if len(pts) >= 2:
            bus_routes.append(pts)

    # Bounds — use ROAD network as primary, then expand only to include nearby
    # buildings (we don't want one stray building 800m out to dominate the frame).
    rxs: List[float] = []
    rys: List[float] = []
    for (a, b, _) in roads:
        rxs += [a[0], b[0]]; rys += [a[1], b[1]]
    if not rxs:
        rxs = [0.0, 1.0]; rys = [0.0, 1.0]
    road_bounds = (min(rxs), max(rxs), min(rys), max(rys))

    pad = 0.08 * max(road_bounds[1] - road_bounds[0], road_bounds[3] - road_bounds[2])
    rbx_min = road_bounds[0] - pad
    rbx_max = road_bounds[1] + pad
    rby_min = road_bounds[2] - pad
    rby_max = road_bounds[3] + pad

    xs = list(rxs); ys = list(rys)
    for rec in plain_buildings + poi_buildings:
        half = max(rec["w"], rec["h"])
        cx, cy = rec["x"], rec["y"]
        if rbx_min - half <= cx <= rbx_max + half and rby_min - half <= cy <= rby_max + half:
            xs += [cx - half, cx + half]
            ys += [cy - half, cy + half]
    for pts in bus_routes:
        for x, y in pts:
            if rbx_min <= x <= rbx_max and rby_min <= y <= rby_max:
                xs.append(x); ys.append(y)

    # Use road_bounds + padding as the view extent so the frame is always centred
    # on the road network.  Expanding by nearby-building corners (xs/ys above) can
    # place the viewport origin far outside the city when a building or the agent's
    # spawn point sits just beyond the leftmost road, making the agent dot appear
    # at the very edge of the image.
    bounds = (rbx_min, rbx_max, rby_min, rby_max)
    return World(roads, plain_buildings, poi_buildings, point_pois,
                 bus_routes, bounds, road_bounds)


# ----------------------------- Projection -------------------------------------

@dataclass
class View:
    xmin: float; xmax: float; ymin: float; ymax: float
    out_w: int; out_h: int; margin: int
    flip_y_axis: bool = True
    scale: float = 0.0
    offx: float = 0.0
    offy: float = 0.0

    def __post_init__(self) -> None:
        avail_w = self.out_w - 2 * self.margin
        avail_h = self.out_h - 2 * self.margin
        span_x = max(1.0, self.xmax - self.xmin)
        span_y = max(1.0, self.ymax - self.ymin)
        self.scale = min(avail_w / span_x, avail_h / span_y)
        used_w = span_x * self.scale
        used_h = span_y * self.scale
        self.offx = self.margin + (avail_w - used_w) / 2.0
        self.offy = self.margin + (avail_h - used_h) / 2.0

    @property
    def px_per_m(self) -> float:
        return self.scale * 100.0

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        if self.flip_y_axis:
            return (self.offx + (x - self.xmin) * self.scale,
                    self.offy + (y - self.ymin) * self.scale)
        return (self.offx + (x - self.xmin) * self.scale,
                self.offy + (self.ymax - y) * self.scale)


# ─────────────────────────────────────────────────────────────────────────────
# MAP TEXT SIZE KNOB  ←  adjust this to change the font size of text ON THE MAP
# ─────────────────────────────────────────────────────────────────────────────
# FONT_SCALE multiplies the size of every IN-MAP label: road names, POI labels
# ("Hospital", "Restaurant", "Store N"...), order pin labels, the scale bar and
# the compass. Increase it to make map text bigger, decrease to make it smaller.
# (The frame is downscaled by gmaps_out_scale afterwards, so text is drawn large
# here to stay legible.) The legend band BELOW the map is intentionally NOT
# affected by this knob — it stays a fixed size (see LEGEND_FONT_PX).
FONT_SCALE = 0.8

# Fixed pixel size of the legend text, independent of FONT_SCALE so the legend
# keeps its current size no matter how the map text is scaled.
LEGEND_FONT_PX = 52


def _font_fixed(size: int) -> ImageFont.FreeTypeFont:
    """Load a scalable TTF at an exact pixel size (FONT_SCALE NOT applied)."""
    size = max(1, int(round(size)))
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _font(size: int) -> ImageFont.FreeTypeFont:
    """In-map label font; nominal size is multiplied by FONT_SCALE."""
    return _font_fixed(size * FONT_SCALE)


def _stroke_widths(view: View) -> Dict[str, int]:
    """Auto-scale stroke widths from view pixels-per-meter.

    Target: a local road should look ~7.5 m wide on screen, but never thinner
    than 6 px (still legible at extreme zoom-out).
    """
    ppm = view.px_per_m
    local_w = max(8, int(round(DEFAULT_STREET_WIDTH_M * ppm)))
    hwy_w   = max(14, int(round(HIGHWAY_WIDTH_M * ppm)))
    bus_w   = max(3, int(round(BUS_ROUTE_WIDTH_M * ppm)))
    return dict(
        local_outer=local_w + 4,
        local_inner=local_w,
        hwy_outer=hwy_w + 6,
        hwy_inner=hwy_w,
        bus=bus_w,
    )


# ----------------------------- Names + numbering ------------------------------

def _name_roads(roads: Sequence[Tuple[Tuple[float, float], Tuple[float, float], bool]]
                ) -> List[Tuple[str, Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Group near-parallel road segments by perpendicular offset and name them.

    Horizontal-ish roads → "Main Ave", "2nd Ave"… (north-south stack).
    Vertical-ish roads   → "1st St", "2nd St"…   (east-west stack).
    Returns one (name, segment) tuple per road for label placement.
    """
    horiz, vert = [], []
    for (a, b, _) in roads:
        dx = b[0] - a[0]; dy = b[1] - a[1]
        if abs(dx) >= abs(dy):
            offset = (a[1] + b[1]) / 2.0     # y for horizontal
            horiz.append((offset, a, b))
        else:
            offset = (a[0] + b[0]) / 2.0     # x for vertical
            vert.append((offset, a, b))

    def _bin_and_name(items, base_names):
        if not items:
            return []
        items_sorted = sorted(items, key=lambda r: r[0])
        bins: List[List[Tuple]] = []
        last_off = None
        TOL_CM = 5000.0   # 50 m tolerance for "same street"
        for off, a, b in items_sorted:
            if last_off is None or abs(off - last_off) > TOL_CM:
                bins.append([])
            bins[-1].append((a, b))
            last_off = off
        named = []
        for idx, segs in enumerate(bins):
            name = base_names(idx + 1)
            for a, b in segs:
                named.append((name, (a, b)))
        return named

    def _ordinal_st(n: int) -> str:
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf} St" if n != 0 else "Main St"

    def _ave(n: int) -> str:
        if n == 1: return "Main Ave"
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf} Ave"

    named = _bin_and_name(vert, _ordinal_st) + _bin_and_name(horiz, _ave)
    return named


def _number_pois(pois: Sequence[dict]) -> Dict[Tuple[float, float], str]:
    """Assign per-type numbers to POI buildings in world-file order.

    The simulator names POIs ("restaurant 3", ...) with a per-type counter
    that follows the node order in progen_world_enriched.json — the map
    labels must use the same order or the image contradicts the text
    observations.
    """
    by_type: Dict[str, List[dict]] = {}
    for p in pois:
        by_type.setdefault(p["poi_type"], []).append(p)
    out: Dict[Tuple[float, float], str] = {}
    for pt, items in by_type.items():
        full = STYLE["poi_full"].get(pt, pt.title())
        if len(items) == 1:
            out[(items[0]["x"], items[0]["y"])] = full
        else:
            for i, rec in enumerate(items, start=1):
                out[(rec["x"], rec["y"])] = f"{full} {i}"
    return out


# ----------------------------- Drawing primitives -----------------------------

def _rotated_rect_corners(cx: float, cy: float, w: float, h: float,
                          yaw_deg: float) -> List[Tuple[float, float]]:
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    out = []
    for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return out


def _draw_building_shadow_layer(view: View, buildings: Sequence[dict],
                                size: Tuple[int, int]) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = BUILDING_SHADOW_OFFSET_PX
    for rec in buildings:
        corners = _rotated_rect_corners(rec["x"], rec["y"], rec["w"], rec["h"], rec["yaw"])
        poly = [(view.to_px(*p)[0] + ox, view.to_px(*p)[1] + oy) for p in corners]
        d.polygon(poly, fill=STYLE["building_shadow"])
    return layer.filter(ImageFilter.GaussianBlur(radius=3))


def _vary_fill(base_hex: str, seed: int) -> str:
    """Slight per-building hue variation so the city isn't a flat-color blanket."""
    h = base_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    jitter = (seed * 7919) % 13 - 6
    r = max(0, min(255, r + jitter))
    g = max(0, min(255, g + jitter))
    b = max(0, min(255, b + jitter))
    return f"#{r:02X}{g:02X}{b:02X}"


def _draw_buildings(draw: ImageDraw.ImageDraw, view: View, buildings: Sequence[dict],
                    fill: str, outline: str) -> None:
    """3D-extruded buildings: foot polygon (wall fill) → roof polygon (lighter)."""
    OFFSET_X, OFFSET_Y = -2, -3    # subtle extrusion hint — Google Maps day view is flat
    for i, rec in enumerate(buildings):
        corners = _rotated_rect_corners(rec["x"], rec["y"], rec["w"], rec["h"], rec["yaw"])
        foot = [view.to_px(x, y) for (x, y) in corners]
        roof = [(p[0] + OFFSET_X, p[1] + OFFSET_Y) for p in foot]
        # Compute the "merged silhouette" — wall fill underneath everything.
        # Build the convex-ish outline: foot polygon + roof polygon. Easier:
        # just draw foot polygon with wall fill (no outline yet), then roof polygon
        # with roof fill and an outline. The bottom/right "wall edges" become visible
        # because foot extends past roof on those sides.
        f_wall = fill
        f_roof = _vary_fill(STYLE["building_fill_roof"], i + int(rec["x"]))
        draw.polygon(foot, fill=f_wall, outline=outline)
        draw.polygon(roof, fill=f_roof, outline=outline)


def _draw_poi_buildings(draw: ImageDraw.ImageDraw, view: View,
                        poi_buildings: Sequence[dict]) -> None:
    for rec in poi_buildings:
        color = STYLE["building_poi"].get(rec["poi_type"], STYLE["building_fill"])
        corners = _rotated_rect_corners(rec["x"], rec["y"], rec["w"], rec["h"], rec["yaw"])
        poly = [view.to_px(x, y) for (x, y) in corners]
        # Lighter category-tinted fill (alpha-mix-ish via fixed pastel mapping below).
        soft = _soft_tint(color)
        draw.polygon(poly, fill=soft, outline=color, width=2)


def _soft_tint(hex_color: str) -> str:
    """Lighten a hex color toward white by ~70% (no PIL alpha-composite needed)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    mix = 0.72
    r = int(r + (255 - r) * mix)
    g = int(g + (255 - g) * mix)
    b = int(b + (255 - b) * mix)
    return f"#{r:02X}{g:02X}{b:02X}"


def _draw_roads(draw: ImageDraw.ImageDraw, view: View,
                roads: Sequence[Tuple[Tuple[float, float], Tuple[float, float], bool]],
                widths: Dict[str, int]) -> None:
    """Three-pass road draw: casing → fill → lane stripes."""
    # Sort so highways draw on top of local roads.
    locals_ = [(a, b) for (a, b, h) in roads if not h]
    hwys = [(a, b) for (a, b, h) in roads if h]

    def _stroke(pairs, casing_color, fill_color, outer_w, inner_w):
        # Casing (outer)
        for (a, b) in pairs:
            ax, ay = view.to_px(*a); bx, by = view.to_px(*b)
            draw.line([(ax, ay), (bx, by)], fill=casing_color, width=outer_w)
            for px, py in ((ax, ay), (bx, by)):
                r = outer_w // 2
                draw.ellipse([px - r, py - r, px + r, py + r], fill=casing_color)
        # Inner fill
        for (a, b) in pairs:
            ax, ay = view.to_px(*a); bx, by = view.to_px(*b)
            draw.line([(ax, ay), (bx, by)], fill=fill_color, width=inner_w)
            for px, py in ((ax, ay), (bx, by)):
                r = inner_w // 2
                draw.ellipse([px - r, py - r, px + r, py + r], fill=fill_color)

    _stroke(locals_, STYLE["road_casing"], STYLE["road_fill_local"],
            widths["local_outer"], widths["local_inner"])
    _stroke(hwys, STYLE["road_casing_hwy"], STYLE["road_fill_hwy"],
            widths["hwy_outer"], widths["hwy_inner"])

    # Lane stripe down highway centerlines (dashed white).
    stripe_w = max(2, widths["hwy_inner"] // 8)
    for (a, b) in hwys:
        ax, ay = view.to_px(*a); bx, by = view.to_px(*b)
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len < 30:
            continue
        dx, dy = (bx - ax) / seg_len, (by - ay) / seg_len
        dash = max(12, widths["hwy_inner"] // 1)
        gap = dash
        t = 0.0
        while t < seg_len:
            t_end = min(seg_len, t + dash)
            sx, sy = ax + dx * t, ay + dy * t
            ex, ey = ax + dx * t_end, ay + dy * t_end
            draw.line([(sx, sy), (ex, ey)], fill=STYLE["lane_stripe"], width=stripe_w)
            t += dash + gap


def _draw_crosswalks(draw: ImageDraw.ImageDraw, view: View,
                     roads: Sequence[Tuple[Tuple[float, float], Tuple[float, float], bool]],
                     widths: Dict[str, int]) -> None:
    """At every endpoint shared by 3+ road segments, draw 4 small white stripes."""
    EPS = 0.5  # cm
    counts: Dict[Tuple[int, int], int] = {}
    for (a, b, _) in roads:
        for p in (a, b):
            key = (round(p[0] / EPS), round(p[1] / EPS))
            counts[key] = counts.get(key, 0) + 1
    inter_pts = [k for k, c in counts.items() if c >= 3]
    if not inter_pts:
        return
    stripe_len = widths["local_inner"] * 1.2
    stripe_w = max(2, widths["local_inner"] // 6)
    gap = stripe_w * 2
    for (kx, ky) in inter_pts:
        cm_x = kx * EPS; cm_y = ky * EPS
        cx, cy = view.to_px(cm_x, cm_y)
        for direction in ("h", "v"):
            for sign in (-1, 1):
                base = widths["local_inner"] // 2 + 4
                for i in range(3):
                    off = base + i * gap + sign * (i * gap * 0.0)
                    if direction == "h":
                        x0 = cx + sign * (off)
                        x1 = x0 + sign * stripe_w
                        y0 = cy - stripe_len / 2
                        y1 = cy + stripe_len / 2
                    else:
                        y0 = cy + sign * (off)
                        y1 = y0 + sign * stripe_w
                        x0 = cx - stripe_len / 2
                        x1 = cx + stripe_len / 2
                    draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                                   fill=STYLE["crosswalk"])


def _draw_bus_routes(draw: ImageDraw.ImageDraw, view: View,
                     bus_routes: Sequence[Sequence[Tuple[float, float]]],
                     widths: Dict[str, int]) -> None:
    for pts in bus_routes:
        proj = [view.to_px(x, y) for (x, y) in pts]
        # Casing + fill so it reads as a route line.
        draw.line(proj, fill="#A3520B", width=widths["bus"] + 4)
        draw.line(proj, fill=STYLE["bus_route"], width=widths["bus"])


def _draw_text_with_halo(draw: ImageDraw.ImageDraw, xy: Tuple[float, float],
                         text: str, font: ImageFont.FreeTypeFont,
                         color: str, halo: str, anchor: str = "mm") -> None:
    """Draw text with a halo (multiple offset strokes) for legibility."""
    x, y = xy
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                   (-2, -2), (-2, 2), (2, -2), (2, 2)):
        draw.text((x + dx, y + dy), text, font=font, fill=halo, anchor=anchor)
    draw.text((x, y), text, font=font, fill=color, anchor=anchor)


def _text_size(draw: ImageDraw.ImageDraw, text: str,
               font: ImageFont.FreeTypeFont) -> Tuple[float, float]:
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]
    except Exception:
        try:
            b = font.getbbox(text)
            return b[2] - b[0], b[3] - b[1]
        except Exception:
            return len(text) * 10, 18


def _draw_order_marker_label(draw: ImageDraw.ImageDraw, x: float, y: float,
                             label: str, color: str,
                             font: ImageFont.FreeTypeFont) -> None:
    """Draw active order labels as side badges to avoid road-label collisions."""
    tw, th = _text_size(draw, label, font)
    pad_x, pad_y = 8, 4
    r = PIN_RADIUS_PX + 3
    img = getattr(draw, "_image", None)
    img_w, img_h = img.size if img is not None else (10_000, 10_000)
    badge_w = tw + pad_x * 2
    badge_h = th + pad_y * 2
    candidates = [
        (x + r * 2.35 + badge_w / 2, y),  # right, away from vertical road labels
        (x - r * 2.35 - badge_w / 2, y),
        (x, y - r * 2.45 - badge_h / 2),
        (x, y + r * 2.25 + badge_h / 2),
    ]
    cx, cy = candidates[0]
    for tx, ty in candidates:
        if (2 <= tx - badge_w / 2 and tx + badge_w / 2 <= img_w - 2
                and 2 <= ty - badge_h / 2 and ty + badge_h / 2 <= img_h - 2):
            cx, cy = tx, ty
            break
    rect = [
        cx - badge_w / 2,
        cy - badge_h / 2,
        cx + badge_w / 2,
        cy + badge_h / 2,
    ]
    # Leader line makes the badge-to-pin association explicit when address
    # labels or POI text are dense around the active order marker.
    dx, dy = cx - x, cy - y
    if abs(dx) > 1 or abs(dy) > 1:
        if abs(dx) / max(1.0, badge_w) > abs(dy) / max(1.0, badge_h):
            edge_x = rect[0] if dx > 0 else rect[2]
            edge_y = cy
        else:
            edge_x = cx
            edge_y = rect[1] if dy > 0 else rect[3]
        draw.line([(x, y), (edge_x, edge_y)], fill=(255, 255, 255, 240), width=7)
        draw.line([(x, y), (edge_x, edge_y)], fill=color, width=3)
    draw.rounded_rectangle(rect, radius=6, fill=(255, 255, 255, 238),
                           outline=color, width=2)
    draw.text((cx, cy), label, font=font, fill=color, anchor="mm")


def _draw_road_names(draw: ImageDraw.ImageDraw, view: View,
                     named: Sequence[Tuple[str, Tuple[Tuple[float, float], Tuple[float, float]]]],
                     reserved: List[Tuple[float, float, float, float]]
                     ) -> None:
    """Draw one label per (named, segment) rotated along the segment, when long enough."""
    font = _font(ROAD_LABEL_SIZE_PX)
    for name, (a, b) in named:
        ax, ay = view.to_px(*a); bx, by = view.to_px(*b)
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len < MIN_LABEL_ROAD_LEN_PX:
            continue
        mx = (ax + bx) / 2.0; my = (ay + by) / 2.0
        angle = math.degrees(math.atan2(-(by - ay), (bx - ax)))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180

        # Render label as an image, rotate, paste.
        try:
            text_bbox = font.getbbox(name)
            tw = text_bbox[2] - text_bbox[0]; th = text_bbox[3] - text_bbox[1]
        except Exception:
            tw, th = (len(name) * ROAD_LABEL_SIZE_PX // 2, ROAD_LABEL_SIZE_PX)
        bbox = (mx - tw, my - th, mx + tw, my + th)
        if any(not (bbox[2] < p[0] or bbox[0] > p[2] or bbox[3] < p[1] or bbox[1] > p[3])
               for p in reserved):
            continue
        reserved.append(bbox)

        label_img = Image.new("RGBA", (tw + 14, th + 12), (0, 0, 0, 0))
        ld = ImageDraw.Draw(label_img)
        _draw_text_with_halo(ld, (label_img.width / 2, label_img.height / 2),
                             name, font, STYLE["label_road"], STYLE["label_road_outline"])
        label_img = label_img.rotate(angle, resample=Image.BICUBIC, expand=True)
        # paste centered at (mx, my)
        cx = int(mx - label_img.width / 2)
        cy = int(my - label_img.height / 2)
        draw._image.paste(label_img, (cx, cy), label_img)


def _draw_poi_glyph(draw: ImageDraw.ImageDraw, x: float, y: float, r: float,
                    poi_type: str, color: str = "white") -> None:
    """Tiny vector glyph for a POI category, centered inside the pin head."""
    w = color
    if poi_type == "restaurant":
        # Crossed fork + knife.
        draw.line([(x - r * 0.35, y - r * 0.5), (x - r * 0.35, y + r * 0.55)], fill=w, width=2)
        draw.line([(x - r * 0.55, y - r * 0.5), (x - r * 0.55, y - r * 0.05)], fill=w, width=2)
        draw.line([(x - r * 0.15, y - r * 0.5), (x - r * 0.15, y - r * 0.05)], fill=w, width=2)
        draw.line([(x + r * 0.30, y - r * 0.5), (x + r * 0.55, y - r * 0.2)], fill=w, width=2)
        draw.line([(x + r * 0.40, y - r * 0.15), (x + r * 0.40, y + r * 0.55)], fill=w, width=2)
    elif poi_type == "store":
        # Shopping bag: rectangle body + curved handle.
        draw.rounded_rectangle([x - r * 0.55, y - r * 0.05, x + r * 0.55, y + r * 0.6],
                               radius=2, fill=w)
        draw.arc([x - r * 0.35, y - r * 0.65, x + r * 0.35, y + r * 0.05],
                 start=180, end=360, fill=w, width=2)
    elif poi_type == "hospital":
        # Plus cross.
        cw = r * 0.30
        draw.rectangle([x - cw / 2, y - r * 0.6, x + cw / 2, y + r * 0.6], fill=w)
        draw.rectangle([x - r * 0.6, y - cw / 2, x + r * 0.6, y + cw / 2], fill=w)
    elif poi_type == "rest_area":
        # Tree: triangle on a short trunk.
        draw.polygon([(x, y - r * 0.65),
                      (x - r * 0.55, y + r * 0.15),
                      (x + r * 0.55, y + r * 0.15)], fill=w)
        draw.rectangle([x - r * 0.15, y + r * 0.15, x + r * 0.15, y + r * 0.55], fill=w)
    elif poi_type == "car_rental":
        # Car silhouette: rounded rect body + smaller cabin on top.
        draw.rounded_rectangle([x - r * 0.62, y - r * 0.05, x + r * 0.62, y + r * 0.45],
                               radius=3, fill=w)
        draw.polygon([(x - r * 0.40, y - r * 0.05),
                      (x - r * 0.25, y - r * 0.45),
                      (x + r * 0.25, y - r * 0.45),
                      (x + r * 0.40, y - r * 0.05)], fill=w)


def _draw_pin(draw: ImageDraw.ImageDraw, x: float, y: float, color: str,
              radius: int = PIN_RADIUS_PX, poi_type: Optional[str] = None) -> None:
    """Google-Maps-style pin (teardrop): circle head + tip, vector icon inside."""
    tip_y = y + radius * 1.95
    p_left = (x - radius * 0.78, y + radius * 0.55)
    p_right = (x + radius * 0.78, y + radius * 0.55)
    p_tip = (x, tip_y)
    rim = radius + 3
    # White rim.
    draw.polygon([(p_left[0] - 2, p_left[1] - 1),
                  (p_right[0] + 2, p_right[1] - 1),
                  (x, tip_y + 2)], fill="white")
    draw.ellipse([x - rim, y - rim, x + rim, y + rim], fill="white")
    # Colored fill.
    draw.polygon([p_left, p_right, p_tip], fill=color)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    # Icon glyph inside.
    if poi_type:
        _draw_poi_glyph(draw, x, y - 1, radius * 0.9, poi_type)


def _is_light(hex_color: str) -> bool:
    """Perceived-luminance test so glyphs/text stay legible on light fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 150.0


def _draw_circle_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: str,
                        radius: int = PIN_RADIUS_PX, poi_type: Optional[str] = None) -> None:
    """Flat circular marker (white rim + colored disc + optional glyph).

    Used for context markers (POIs, charging/bus, route endpoints). The teardrop
    ``_draw_pin`` is reserved for the task's pickup/dropoff order markers so they
    stand out from the circular context markers.
    """
    rim = radius + 3
    draw.ellipse([x - rim, y - rim, x + rim, y + rim], fill="white")
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    if poi_type:
        # On light discs (e.g. yellow restaurant) a white glyph is invisible;
        # use a dark glyph stroke instead.
        glyph_color = STYLE["label_dark"] if _is_light(color) else "white"
        _draw_poi_glyph(draw, x, y, radius * 0.9, poi_type, color=glyph_color)


def _draw_poi_labels(draw: ImageDraw.ImageDraw, view: View,
                     poi_buildings: Sequence[dict],
                     numbering: Dict[Tuple[float, float], str],
                     reserved: List[Tuple[float, float, float, float]]) -> None:
    """Pin + label per POI building. Picks a label position from a few candidates
    to avoid overlapping previously reserved boxes."""
    font_label = _font(32)
    LABEL_PAD = 4

    def _bbox_text(text: str, font: ImageFont.FreeTypeFont) -> Tuple[float, float]:
        try:
            b = font.getbbox(text)
            return b[2] - b[0], b[3] - b[1]
        except Exception:
            return len(text) * 11, 22

    def _collides(rect: Tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = rect
        for ax0, ay0, ax1, ay1 in reserved:
            if not (x1 <= ax0 or ax1 <= x0 or y1 <= ay0 or ay1 <= y0):
                return True
        return False

    for rec in poi_buildings:
        color = STYLE["building_poi"].get(rec["poi_type"], "#808080")
        cx, cy = view.to_px(rec["x"], rec["y"])
        _draw_circle_marker(draw, cx, cy, color, poi_type=rec["poi_type"])
        # Reserve the circle head area.
        pin_box = (cx - PIN_RADIUS_PX - 4, cy - PIN_RADIUS_PX - 4,
                   cx + PIN_RADIUS_PX + 4, cy + PIN_RADIUS_PX + 4)
        reserved.append(pin_box)
        label = numbering.get((rec["x"], rec["y"]),
                              STYLE["poi_full"].get(rec["poi_type"], "?"))
        tw, th = _bbox_text(label, font_label)
        # Try below marker first, then above, then to the right.
        candidates = [
            (cx, cy + PIN_RADIUS_PX + th / 2 + LABEL_PAD + 3),        # below
            (cx, cy - PIN_RADIUS_PX - th / 2 - LABEL_PAD - 3),        # above
            (cx + PIN_RADIUS_PX + tw / 2 + LABEL_PAD * 2, cy),        # right
            (cx - PIN_RADIUS_PX - tw / 2 - LABEL_PAD * 2, cy),        # left
        ]
        for (lx, ly) in candidates:
            rect = (lx - tw / 2 - 3, ly - th / 2 - 2,
                    lx + tw / 2 + 3, ly + th / 2 + 2)
            if not _collides(rect):
                _draw_text_with_halo(draw, (lx, ly),
                                     label, font_label,
                                     STYLE["label_dark"], STYLE["bg"])
                reserved.append(rect)
                break


def _draw_lightning(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                    h: float, color: str = "white") -> None:
    """Stylized lightning bolt centered at (cx, cy), total height h."""
    w = h * 0.55
    pts = [
        (cx - w * 0.20, cy - h * 0.50),
        (cx + w * 0.45, cy - h * 0.50),
        (cx + w * 0.05, cy - h * 0.05),
        (cx + w * 0.45, cy - h * 0.05),
        (cx - w * 0.30, cy + h * 0.50),
        (cx + w * 0.05, cy + h * 0.05),
        (cx - w * 0.30, cy + h * 0.05),
    ]
    draw.polygon(pts, fill=color)


def _draw_bus_icon(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float,
                   color: str = "white") -> None:
    """Simple front-facing bus silhouette: rounded rectangle + two windows + wheels."""
    w = size * 0.78; h = size * 0.86
    body = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    draw.rounded_rectangle(body, radius=size * 0.18, fill=color)
    # Windscreen
    win_h = h * 0.30
    draw.rectangle([cx - w / 2 + 2, cy - h / 2 + 4,
                    cx + w / 2 - 2, cy - h / 2 + 4 + win_h],
                   fill=STYLE["poi_bus"])
    # Wheels
    wr = size * 0.08
    for dx in (-w / 2 + wr * 2, w / 2 - wr * 2):
        draw.ellipse([cx + dx - wr, cy + h / 2 - wr * 1.4,
                      cx + dx + wr, cy + h / 2 + wr * 0.4],
                     fill=STYLE["poi_bus"])


def _draw_point_pois(draw: ImageDraw.ImageDraw, view: View,
                     point_pois: Sequence[dict],
                     reserved: List[Tuple[float, float, float, float]]) -> None:
    """Charging stations and bus stations — both are now clearly labeled pill markers."""
    font_lbl = _font(26)
    bus = [r for r in point_pois if r["poi_type"] == "bus_station"]
    chg = [r for r in point_pois if r["poi_type"] == "charging_station"]
    bus.sort(key=lambda r: (-r["y"], r["x"]))
    chg.sort(key=lambda r: (-r["y"], r["x"]))

    def _bbox_text(text, font):
        try:
            b = font.getbbox(text)
            return b[2] - b[0], b[3] - b[1]
        except Exception:
            return len(text) * 9, 16

    def _collides(rect):
        x0, y0, x1, y1 = rect
        for ax0, ay0, ax1, ay1 in reserved:
            if not (x1 <= ax0 or ax1 <= x0 or y1 <= ay0 or ay1 <= y0):
                return True
        return False

    def _place_label(cx, cy, r, label):
        tw, th = _bbox_text(label, font_lbl)
        for (lx, ly) in [(cx, cy + r + 14), (cx, cy - r - 14),
                         (cx + r + tw / 2 + 8, cy),
                         (cx - r - tw / 2 - 8, cy)]:
            rect = (lx - tw / 2 - 3, ly - th / 2 - 2,
                    lx + tw / 2 + 3, ly + th / 2 + 2)
            if not _collides(rect):
                _draw_text_with_halo(draw, (lx, ly), label, font_lbl,
                                     STYLE["label_dark"], STYLE["city_bg"])
                reserved.append(rect)
                return

    # Charging stations: green pill with lightning bolt, plus "EV n" label.
    for i, rec in enumerate(chg, start=1):
        cx, cy = view.to_px(rec["x"], rec["y"])
        r = 12
        draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill="white")
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=STYLE["poi_charging"])
        _draw_lightning(draw, cx, cy, h=r * 1.55, color="white")
        reserved.append((cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4))
        _place_label(cx, cy, r, f"EV {i}")

    # Bus stations: orange pill with bus icon, plus "Bus n" label.
    for i, rec in enumerate(bus, start=1):
        cx, cy = view.to_px(rec["x"], rec["y"])
        r = 13
        draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill="white")
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=STYLE["poi_bus"])
        _draw_bus_icon(draw, cx, cy, size=r * 1.5)
        reserved.append((cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4))
        _place_label(cx, cy, r, f"Bus {i}")


def _draw_agent(img: Image.Image, view: View, agent_xy: Tuple[float, float],
                heading_deg: Optional[float] = None) -> None:
    ax, ay = view.to_px(*agent_xy)
    # Clamp to the visible canvas so the dot is always inside the image even when
    # the agent spawns at a position slightly outside the road-network bounds.
    pad = AGENT_DOT_R + 4
    ax = max(MARGIN_PX + pad, min(img.width  - MARGIN_PX - pad, ax))
    ay = max(MARGIN_PX + pad, min(img.height - MARGIN_PX - pad, ay))
    halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    hd.ellipse([ax - AGENT_HALO_R, ay - AGENT_HALO_R,
                ax + AGENT_HALO_R, ay + AGENT_HALO_R],
               fill=STYLE["agent_halo"])
    halo = halo.filter(ImageFilter.GaussianBlur(radius=6))
    img.alpha_composite(halo)

    d = ImageDraw.Draw(img)
    d.ellipse([ax - (AGENT_DOT_R + 4), ay - (AGENT_DOT_R + 4),
               ax + (AGENT_DOT_R + 4), ay + (AGENT_DOT_R + 4)], fill="white")
    d.ellipse([ax - AGENT_DOT_R, ay - AGENT_DOT_R,
               ax + AGENT_DOT_R, ay + AGENT_DOT_R], fill=STYLE["agent"])

    # Facing pointer: a triangle pointing in the compass heading. The map view can
    # mirror world Y to match UE captures, so derive the screen-space forward vector
    # from the View instead of assuming +Y is always up on screen.
    if heading_deg is not None:
        rad = math.radians(float(heading_deg) % 360.0)
        fx = math.sin(rad)
        fy = math.cos(rad) if view.flip_y_axis else -math.cos(rad)
        # Bigger, vivid wedge so facing reads clearly at a glance.
        tip = AGENT_DOT_R + 30
        base = AGENT_DOT_R - 2
        half = AGENT_DOT_R * 1.25
        tip_pt = (ax + fx * tip, ay + fy * tip)
        px, py = -fy, fx                                # perpendicular (right-hand)
        b1 = (ax + fx * base + px * half, ay + fy * base + py * half)
        b2 = (ax + fx * base - px * half, ay + fy * base - py * half)
        # White casing under the wedge for separation from the map, then fill.
        d.polygon([tip_pt, b1, b2], fill="white")
        ins = 0.78
        tip_i = (ax + fx * tip * ins, ay + fy * tip * ins)
        b1_i = (ax + fx * base + px * half * ins, ay + fy * base + py * half * ins)
        b2_i = (ax + fx * base - px * half * ins, ay + fy * base - py * half * ins)
        d.polygon([tip_i, b1_i, b2_i], fill=STYLE["agent_facing"])


def _draw_scale_bar(draw: ImageDraw.ImageDraw, view: View,
                    out_w: int, out_h: int, corner: str = "right") -> None:
    target_m = _pick_scale_meters(view.scale)
    length_px = target_m * 100.0 * view.scale
    if corner == "left":
        x0 = MARGIN_PX
        x1 = x0 + length_px
    else:
        x1 = out_w - MARGIN_PX
        x0 = x1 - length_px
    y = out_h - MARGIN_PX // 2
    draw.line([(x0, y), (x1, y)], fill=STYLE["label_dark"], width=4)
    tick = 8
    draw.line([(x0, y - tick), (x0, y)], fill=STYLE["label_dark"], width=4)
    draw.line([(x1, y - tick), (x1, y)], fill=STYLE["label_dark"], width=4)
    font = _font(32)
    label = f"{int(target_m)} m" if target_m < 1000 else f"{target_m/1000:.1f} km"
    tw = draw.textlength(label, font=font)
    draw.text((x0 + (length_px - tw) / 2.0, y - 36),
              label, font=font, fill=STYLE["label_dark"])


def _draw_compass(draw: ImageDraw.ImageDraw, out_w: int, flip_y_axis: bool = False) -> None:
    cx = out_w - MARGIN_PX - 30
    cy = MARGIN_PX + 30
    r = 26
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="white", outline=STYLE["label_dark"], width=2)
    if flip_y_axis:
        draw.polygon([(cx, cy + r - 6), (cx - 8, cy - 6), (cx + 8, cy - 6)],
                     fill=STYLE["pickup"])
        draw.polygon([(cx, cy - r + 6), (cx - 8, cy + 4), (cx + 8, cy + 4)],
                     fill="#808080")
    else:
        draw.polygon([(cx, cy - r + 6), (cx - 8, cy + 6), (cx + 8, cy + 6)],
                     fill=STYLE["pickup"])
        draw.polygon([(cx, cy + r - 6), (cx - 8, cy - 4), (cx + 8, cy - 4)],
                     fill="#808080")
    f = _font(24)
    if flip_y_axis:
        draw.text((cx, cy + r + 4), "N", font=f, fill=STYLE["label_dark"], anchor="mt")
    else:
        draw.text((cx, cy - r - 4), "N", font=f, fill=STYLE["label_dark"], anchor="mb")


def _pick_scale_meters(px_per_cm: float) -> float:
    target_px = 250.0
    meters = (target_px / px_per_cm) / 100.0
    for n in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        if meters <= n:
            return float(n)
    return 10000.0


# ----------------------------- Main render ------------------------------------

def _output_size(bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Pick output dims that fit the data's aspect ratio."""
    span_x = max(1.0, bounds[1] - bounds[0])
    span_y = max(1.0, bounds[3] - bounds[2])
    ratio = span_x / span_y
    if ratio >= 1.0:
        w = OUT_LONG
        h = max(OUT_MIN, int(OUT_LONG / ratio))
    else:
        h = OUT_LONG
        w = max(OUT_MIN, int(OUT_LONG * ratio))
    return w, h


def _draw_paper_texture(size: Tuple[int, int]) -> Image.Image:
    """Subtle warm noise layer applied at low opacity to break the flat cream."""
    import random
    rng = random.Random(0xCAFE)
    w, h = size
    tile_w, tile_h = w // 6, h // 6
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    px = tile.load()
    for y in range(tile_h):
        for x in range(tile_w):
            if rng.random() < 0.02:
                a = rng.randint(8, 22)
                px[x, y] = (130, 110, 80, a)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    for yy in range(0, h, tile_h):
        for xx in range(0, w, tile_w):
            layer.paste(tile, (xx, yy), tile)
    return layer.filter(ImageFilter.GaussianBlur(radius=0.6))


def _draw_highway_chevrons(draw: ImageDraw.ImageDraw, view: View,
                           roads: Sequence[Tuple[Tuple[float, float], Tuple[float, float], bool]],
                           widths: Dict[str, int]) -> None:
    """Direction arrows along highway centerlines, like Google Maps navigation."""
    for (a, b, is_hwy) in roads:
        if not is_hwy:
            continue
        ax, ay = view.to_px(*a); bx, by = view.to_px(*b)
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len < 120:
            continue
        dx = (bx - ax) / seg_len; dy = (by - ay) / seg_len
        nx, ny = -dy, dx
        spacing = max(120, int(widths["hwy_inner"] * 6))
        size = widths["hwy_inner"] * 0.35
        for t in range(spacing // 2, int(seg_len), spacing):
            cx = ax + dx * t; cy = ay + dy * t
            # Chevron pointing along (dx, dy)
            p1 = (cx + dx * size, cy + dy * size)
            p2 = (cx - dx * size + nx * size, cy - dy * size + ny * size)
            p3 = (cx - dx * size - nx * size, cy - dy * size - ny * size)
            draw.polygon([p1, p2, p3], fill="#B07A1A")


def _draw_city_area(draw: ImageDraw.ImageDraw, view: View,
                    road_bounds: Tuple[float, float, float, float]) -> None:
    """Lighter 'inside the city' background, slightly larger than the road bbox."""
    rx0, rx1, ry0, ry1 = road_bounds
    pad_cm = max((rx1 - rx0), (ry1 - ry0)) * 0.06
    x0, y0 = view.to_px(rx0 - pad_cm, ry1 + pad_cm)
    x1, y1 = view.to_px(rx1 + pad_cm, ry0 - pad_cm)
    draw.rounded_rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], radius=24,
                           fill=STYLE["city_bg"],
                           outline=None)


def _draw_parks(draw: ImageDraw.ImageDraw, view: View,
                road_bounds: Tuple[float, float, float, float],
                buildings: Sequence[dict]) -> None:
    """Synthetic 'park' patches in the larger gaps between buildings.

    Scans a coarse grid over the city area, finds cells with no building footprint,
    groups contiguous empty cells into clusters, and fills clusters above a size
    threshold with a soft green polygon (rounded rectangle approximation).
    """
    rx0, rx1, ry0, ry1 = road_bounds
    pad_cm = max((rx1 - rx0), (ry1 - ry0)) * 0.06
    x0_cm = rx0 - pad_cm; x1_cm = rx1 + pad_cm
    y0_cm = ry0 - pad_cm; y1_cm = ry1 + pad_cm

    GRID = 60
    span_x = x1_cm - x0_cm
    span_y = y1_cm - y0_cm
    cell_x = span_x / GRID
    cell_y = span_y / GRID
    occupied = [[False] * GRID for _ in range(GRID)]

    for rec in buildings:
        # Bounding rect (axis-aligned over-approximation for grid marking).
        half = max(rec["w"], rec["h"]) / 2.0 + 200.0
        bx0 = rec["x"] - half; bx1 = rec["x"] + half
        by0 = rec["y"] - half; by1 = rec["y"] + half
        cx0 = max(0, int((bx0 - x0_cm) / cell_x))
        cx1 = min(GRID - 1, int((bx1 - x0_cm) / cell_x))
        cy0 = max(0, int((y1_cm - by1) / cell_y))
        cy1 = min(GRID - 1, int((y1_cm - by0) / cell_y))
        for iy in range(cy0, cy1 + 1):
            for ix in range(cx0, cx1 + 1):
                occupied[iy][ix] = True

    # Flood fill clusters of empty cells.
    seen = [[False] * GRID for _ in range(GRID)]
    clusters: List[List[Tuple[int, int]]] = []
    for iy in range(GRID):
        for ix in range(GRID):
            if occupied[iy][ix] or seen[iy][ix]:
                continue
            stack = [(ix, iy)]; cluster = []
            while stack:
                cx, cy = stack.pop()
                if not (0 <= cx < GRID and 0 <= cy < GRID): continue
                if seen[cy][cx] or occupied[cy][cx]: continue
                seen[cy][cx] = True
                cluster.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            if cluster:
                clusters.append(cluster)

    # Draw park polygons for clusters whose bbox is "park sized" — neither too small
    # nor scenario-spanning. Limit to 5 parks max to avoid clutter.
    candidate = []
    for cl in clusters:
        xs = [c[0] for c in cl]; ys = [c[1] for c in cl]
        area = len(cl)
        bx0, bx1 = min(xs), max(xs); by0, by1 = min(ys), max(ys)
        w_cells = bx1 - bx0 + 1; h_cells = by1 - by0 + 1
        if area < 8 or area > GRID * GRID * 0.25: continue
        if w_cells > GRID * 0.5 or h_cells > GRID * 0.5: continue
        candidate.append((area, bx0, bx1, by0, by1))
    candidate.sort(reverse=True)
    for _, bx0, bx1, by0, by1 in candidate[:5]:
        cmx0 = x0_cm + bx0 * cell_x + cell_x * 0.15
        cmx1 = x0_cm + (bx1 + 1) * cell_x - cell_x * 0.15
        cmy0 = y1_cm - (by1 + 1) * cell_y + cell_y * 0.15
        cmy1 = y1_cm - by0 * cell_y - cell_y * 0.15
        px0, py0 = view.to_px(cmx0, cmy1)
        px1, py1 = view.to_px(cmx1, cmy0)
        draw.rounded_rectangle([min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1)], radius=12,
                               fill=STYLE["park_fill"],
                               outline=STYLE["park_border"], width=1)


def render_background(
    scenario_dir: Path,
    road_names: Optional[Sequence[Tuple[str, Tuple[Tuple[float, float], Tuple[float, float]]]]] = None,
    decorate: bool = True,
) -> Tuple["Image.Image", World, View]:
    """
    Render the static parts of the map (buildings, roads, labels, parks, shadows).

    road_names: optional (name, ((x1,y1),(x2,y2))) segments in cm — typically
    the simulator's own road names (Map.graph_skel edge meta) so the street
    labels match the addresses in text observations. When None, synthetic
    grid names ("1st St", "2nd Ave", ...) are generated geometrically.

    Returns the background Image, World, and View so callers can cache the result
    and call compose_frame() cheaply on every env step instead of re-rendering
    everything from scratch.
    """
    world = load_world(scenario_dir)
    out_w, out_h = _output_size(world.bounds)
    view = View(*world.bounds, out_w, out_h, MARGIN_PX)

    img = Image.new("RGBA", (out_w, out_h), STYLE["bg"])
    img.alpha_composite(_draw_paper_texture((out_w, out_h)))
    draw = ImageDraw.Draw(img)
    _draw_city_area(draw, view, world.road_bounds)
    _draw_parks(draw, view, world.road_bounds,
                list(world.buildings) + list(world.poi_buildings))

    widths = _stroke_widths(view)

    shadow = _draw_building_shadow_layer(
        view, list(world.buildings) + list(world.poi_buildings), (out_w, out_h)
    )
    img.alpha_composite(shadow)

    _draw_buildings(draw, view, world.buildings,
                    fill=STYLE["building_fill"], outline=STYLE["building_border"])
    _draw_poi_buildings(draw, view, world.poi_buildings)
    _draw_bus_routes(draw, view, world.bus_routes, widths)
    _draw_roads(draw, view, world.roads, widths)
    _draw_highway_chevrons(draw, view, world.roads, widths)
    _draw_crosswalks(draw, view, world.roads, widths)

    reserved: List[Tuple[float, float, float, float]] = []
    named = list(road_names) if road_names else _name_roads(world.roads)
    _draw_road_names(draw, view, named, reserved)
    _draw_point_pois(draw, view, world.point_pois, reserved)
    poi_numbers = _number_pois(world.poi_buildings)
    _draw_poi_labels(draw, view, world.poi_buildings, poi_numbers, reserved)

    # Static decorations (title / scale bar / compass). Skipped when the caller
    # crops dynamically and redraws decorations post-crop (see compose_frame).
    if decorate:
        title_font = _font(40)
        draw.text((MARGIN_PX, MARGIN_PX // 2),
                  scenario_dir.name, font=title_font, fill=STYLE["label_dark"])
        _draw_scale_bar(draw, view, out_w, out_h)
        _draw_compass(draw, out_w, flip_y_axis=view.flip_y_axis)

    return img, world, view


def _arrow_head(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                dx: float, dy: float, size: float, fill, outline=None) -> None:
    """A filled triangular arrowhead at (cx, cy) pointing along (dx, dy)."""
    nx, ny = -dy, dx
    tip = (cx + dx * size, cy + dy * size)
    l = (cx - dx * size * 0.55 + nx * size * 0.85, cy - dy * size * 0.55 + ny * size * 0.85)
    r = (cx - dx * size * 0.55 - nx * size * 0.85, cy - dy * size * 0.55 - ny * size * 0.85)
    draw.polygon([tip, l, r], fill=fill, outline=outline)


def _draw_route_arrows(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]],
                       color, spacing_px: float = 110.0, size: float = 14.0) -> None:
    """Place directional arrowheads along a polyline pointing toward its end."""
    if len(pts) < 2:
        return
    dist_into = spacing_px * 0.5
    for a, b in zip(pts[:-1], pts[1:]):
        ax, ay = a; bx, by = b
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-6:
            continue
        dx, dy = (bx - ax) / seg, (by - ay) / seg
        while dist_into <= seg:
            cx = ax + dx * dist_into; cy = ay + dy * dist_into
            _arrow_head(draw, cx, cy, dx, dy, size + 3, fill="white")     # casing
            _arrow_head(draw, cx, cy, dx, dy, size, fill=color)           # colored
            dist_into += spacing_px
        dist_into -= seg


_LEGEND_ITEMS = [
    ("dot_arrow", STYLE["agent"], "You (facing)"),
    ("pin", STYLE["pickup"], "Pickup"),
    ("pin", STYLE["dropoff"], "Drop-off"),
    ("circle", STYLE["building_poi"]["restaurant"], "Restaurant"),
    ("line", STYLE["agent"], "Route"),
]


def _draw_legend_icon(draw: ImageDraw.ImageDraw, kind: str, color, icx: float, cy: float) -> None:
    if kind == "dot_arrow":
        r = 9
        draw.ellipse([icx - r, cy - r, icx + r, cy + r], fill=color, outline="white", width=2)
        _arrow_head(draw, icx + r + 2, cy, 1.0, 0.0, 9, fill=STYLE["agent_facing"], outline="white")
    elif kind == "pin":
        _draw_pin(draw, icx, cy - 6, color, radius=10)
    elif kind == "circle":
        r = 11
        draw.ellipse([icx - r - 2, cy - r - 2, icx + r + 2, cy + r + 2], fill="white")
        draw.ellipse([icx - r, cy - r, icx + r, cy + r], fill=color)
    elif kind == "line":
        draw.line([(icx - 16, cy), (icx + 16, cy)], fill=color, width=6)


def _append_legend_band(img: "Image.Image") -> "Image.Image":
    """Append a white padding band BELOW the map and lay the legend out across it
    (wrapping into rows as needed) so it never overlaps the map content."""
    # Keep the legend readable after out_scale while avoiding right-edge
    # truncation on narrow navigation crops.
    font_px = min(LEGEND_FONT_PX, max(38, int(img.width / 15)))
    font = _font_fixed(font_px)   # independent of FONT_SCALE
    try:
        fb = font.getbbox("Ag\u2191")
        font_h = fb[3] - fb[1]
    except Exception:
        font_h = 26
    pad = 18
    icon_w = max(36, int(font_h * 1.4))
    icon_gap = 8
    item_gap = max(36, int(font_h * 1.2))
    row_h = int(font_h * 1.6)

    measure = ImageDraw.Draw(img)
    measured = []
    for kind, color, text in _LEGEND_ITEMS:
        try:
            tw = int(measure.textlength(text, font=font))
        except Exception:
            tw = len(text) * 13
        measured.append((kind, color, text, icon_w + icon_gap + tw))

    max_w = max(1, img.width - pad * 2)
    rows: List[List[Tuple]] = [[]]
    cur = 0
    for it in measured:
        w = it[3]
        if cur > 0 and cur + item_gap + w > max_w:
            rows.append([])
            cur = 0
        if cur > 0:
            cur += item_gap
        rows[-1].append((it, cur))
        cur += w

    band_h = pad * 2 + row_h * len(rows)
    out = Image.new("RGBA", (img.width, img.height + band_h), (255, 255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.line([(0, img.height), (img.width, img.height)], fill=STYLE["shield_border"], width=2)

    y = img.height + pad + row_h // 2
    for row in rows:
        for (it, xoff) in row:
            kind, color, text, _w = it
            icx = pad + xoff + icon_w // 2
            _draw_legend_icon(d, kind, color, icx, y)
            d.text((pad + xoff + icon_w + icon_gap, y), text,
                   font=font, fill=STYLE["label_dark"], anchor="lm")
        y += row_h
    return out


def _crop_box(view: View, agent_xy, nav_route,
              size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Pixel crop window bounded by the agent's current waypoint and the
    navigation route (current → target), with padding. Returns None when there
    is no route to zoom to (the full map is shown until NAVIGATE is called)."""
    w, h = size
    if not nav_route or len(nav_route) < 2:
        return None
    pts: List[Tuple[float, float]] = []
    if agent_xy is not None:
        pts.append(view.to_px(float(agent_xy[0]), float(agent_xy[1])))
    for n in nav_route:
        try:
            pts.append(view.to_px(float(n.position.x), float(n.position.y)))
        except Exception:
            pass
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    ppm = view.px_per_m
    # Dynamic labels are drawn before cropping, so the crop needs extra breathing
    # room for waypoint names, order badges, the agent marker, and halos.
    pad = max(ppm * 165.0, 0.42 * max(x1 - x0, y1 - y0))
    half_w = max((x1 - x0) / 2.0 + pad, ppm * 220.0)
    half_h = max((y1 - y0) / 2.0 + pad, ppm * 220.0)
    left = int(max(0, cx - half_w)); right = int(min(w, cx + half_w))
    top = int(max(0, cy - half_h)); bottom = int(min(h, cy + half_h))
    if right - left < 80 or bottom - top < 80:
        return None
    return (left, top, right, bottom)


def compose_frame(
    bg: "Image.Image",
    view: View,
    agent_xy: Optional[Tuple[float, float]] = None,
    order_markers: Optional[List[Dict]] = None,
    out_scale: float = 1.0,
    nav_route: Optional[List[Any]] = None,
    nav_color: Optional[Tuple[int, int, int, int]] = None,
    agent_heading_deg: Optional[float] = None,
    crop: bool = False,
    decorate: bool = False,
    route_waypoint_labels: bool = False,
) -> "Image.Image":
    """
    Stamp dynamic elements (nav route, agent dot, order pins) onto a copy of
    the pre-rendered background and return the composited RGB image.

    order_markers: list of dicts with keys:
        x, y   — position in cm (world coords)
        kind   — "pickup" | "dropoff"
        label  — short string drawn above the pin (e.g. "#1")
    out_scale: resize factor applied before returning (e.g. 0.5 → half resolution).
    nav_route: list of waypoint nodes (each with .position.x/.y in cm) to draw
        as a persistent route polyline.  Drawn before the agent dot so the dot
        renders on top.
    nav_color: RGBA tuple for the route line (default: blue walk colour).
    """
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    font_lbl = _font(36)

    # Draw persistent navigation route polyline + direction arrows + endpoints
    # (below order pins and the agent dot, so the agent always renders on top).
    if nav_route and len(nav_route) >= 2:
        try:
            pts = [view.to_px(float(n.position.x), float(n.position.y))
                   for n in nav_route]
            route_w = max(6, int(round(view.px_per_m * 1.6)))
            color = nav_color or (26, 115, 232, 255)
            draw.line(pts, fill=(255, 255, 255, 255), width=route_w + 4, joint="curve")
            draw.line(pts, fill=color, width=route_w, joint="curve")
            _draw_route_arrows(draw, pts, color)
            # destination = circle marker; source = green dot
            ex, ey = pts[-1]
            _draw_circle_marker(draw, ex, ey, "#EA4335", radius=PIN_RADIUS_PX)
            sx, sy = pts[0]
            sr = max(8, int(round(view.px_per_m * 1.4)))
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                         fill=(52, 168, 83, 255), outline=(255, 255, 255, 255), width=2)
            if route_waypoint_labels:
                font_wp = _font(20)
                for i, (node, (px, py)) in enumerate(zip(nav_route, pts)):
                    if i == 0:
                        label = "START"
                    elif i == len(pts) - 1:
                        label = "GOAL"
                    else:
                        label = (
                            getattr(node, "waypoint_name", None)
                            or getattr(node, "waypoint_id", None)
                            or str(i)
                        )
                    _draw_text_with_halo(
                        draw,
                        (px, py - max(16, int(round(view.px_per_m * 2.2)))),
                        str(label),
                        font_wp,
                        STYLE["label_dark"],
                        STYLE["bg"],
                    )
        except Exception:
            pass

    # Pickup/drop-off order markers stay as teardrop pins (so they're visually
    # distinct from the circular context markers).
    if order_markers:
        for m in order_markers:
            px, py = view.to_px(float(m["x"]), float(m["y"]))
            color = STYLE["pickup"] if m.get("kind") == "pickup" else STYLE["dropoff"]
            _draw_pin(draw, px, py, color, radius=PIN_RADIUS_PX + 3)
            lbl = m.get("label", "")
            if lbl:
                _draw_order_marker_label(draw, px, py, lbl, color, font_lbl)

    if agent_xy is not None:
        _draw_agent(img, view, agent_xy, heading_deg=agent_heading_deg)

    # Zoom: only after NAVIGATE is called — crop to the window bounded by the
    # current waypoint and the navigation target. Before that, the full map is
    # shown (so the agent can locate itself in the whole city).
    if crop:
        box = _crop_box(view, agent_xy, nav_route, img.size)
        if box is not None:
            img = img.crop(box)

    # Map decorations: scale bar + compass on the map; the legend goes into a
    # dedicated padding band below the map so it never overlaps map content.
    if decorate:
        d2 = ImageDraw.Draw(img)
        _draw_scale_bar(d2, view, img.width, img.height, corner="left")
        _draw_compass(d2, img.width, flip_y_axis=view.flip_y_axis)
        img = _append_legend_band(img)

    result = img.convert("RGB")
    if out_scale != 1.0:
        nw = max(1, int(result.width * out_scale))
        nh = max(1, int(result.height * out_scale))
        result = result.resize((nw, nh), Image.LANCZOS)
    return result


def render(scenario_dir: Path, out_path: Path,
           agent_xy: Optional[Tuple[float, float]] = None,
           order_markers: Optional[List[Dict]] = None,
           out_scale: float = 1.0) -> "Image.Image":
    """Render a complete frame and save to out_path. Returns the PIL Image."""
    bg, world, view = render_background(scenario_dir)
    result = compose_frame(bg, view, agent_xy, order_markers, out_scale)
    result.save(out_path, "PNG", optimize=True)
    print(f"wrote {out_path}  px_per_m={view.px_per_m:.4f}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--agent", nargs=2, type=float, default=None, metavar=("X", "Y"))
    args = ap.parse_args()
    out = args.out or Path.cwd() / f"{args.scenario_dir.name}_v2.png"
    agent = tuple(args.agent) if args.agent else None
    render(args.scenario_dir, out, agent_xy=agent)


if __name__ == "__main__":
    main()
