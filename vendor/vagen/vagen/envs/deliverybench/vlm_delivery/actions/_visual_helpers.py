# actions/_visual_helpers.py
# -*- coding: utf-8 -*-

"""
Shared rendering helpers for the visual navigation actions
(VISUAL_NAVIGATE_WALK, VISUAL_NAVIGATE_ESCOOTER, ...).

All helpers are pure reads / pure rendering — they never modify agent or
environment state. The base city map is produced by the offline renderer
(``tools/render_gmaps.py``) and its ``View`` transform is reused so a route
overlay aligns pixel-for-pixel with the map. The base render is cached per
scenario (it is static), so only the first call per scenario pays the cost.
"""

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Reuse the offline renderer's base-map drawing and coordinate transform.
from ...tools.render_gmaps import (
    render as _render_base_map,
    load_world as _load_world,
    _output_size as _gmaps_output_size,
    View as _GmapsView,
    MARGIN_PX as _GMAPS_MARGIN,
)


# Mode / marker colors (RGBA). Distinct hues so modes never read as one another.
WALK_COLOR = (26, 115, 232, 255)      # blue
ESCOOTER_COLOR = (142, 36, 170, 255)  # purple
BUS_COLOR = (245, 124, 0, 255)        # orange
START_COLOR = (52, 168, 83, 255)      # green
TARGET_COLOR = (234, 67, 53, 255)     # red
BOARD_COLOR = (245, 124, 0, 255)      # orange (matches bus leg)
ALIGHT_COLOR = (0, 188, 212, 255)     # cyan

_LEGEND_BG = (255, 255, 255, 235)
_LEGEND_BORDER = (120, 120, 120, 255)
_TEXT_DARK = (32, 33, 36, 255)

# Cache of base-map PNG bytes keyed by scenario_dir (base map is static).
_BASE_CACHE: dict = {}


def human_name(node: Any) -> str:
    """Human-readable name only — never the raw int_N / dock_N id."""
    return (
        getattr(node, "waypoint_name", "")
        or getattr(node, "waypoint_id", "")
        or str(node)
    )


def scenario_dir(dm: Any) -> Optional[str]:
    """Locate the scenario directory (holds roads.json / world json)."""
    sd = getattr(dm, "scenario_dir", None)
    if sd:
        return str(sd)
    ex = getattr(dm, "map_exportor", None)
    wjp = getattr(ex, "world_json_path", None) if ex is not None else None
    if wjp:
        return str(Path(wjp).parent)
    return None


def images_enabled(dm: Any) -> bool:
    """True when the env will actually surface action images (vision mode)."""
    return getattr(dm, "map_exportor", None) is not None


def _font(size: int):
    from PIL import ImageFont
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _base_image_and_view(scenario_path: str):
    """Return (base RGBA image, View) for a scenario, rendering once and caching."""
    sd = Path(scenario_path)
    world = _load_world(sd)
    out_w, out_h = _gmaps_output_size(world.bounds)
    view = _GmapsView(*world.bounds, out_w, out_h, _GMAPS_MARGIN)

    key = str(sd)
    if key not in _BASE_CACHE:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            _render_base_map(sd, Path(tmp), agent_xy=None)
            with open(tmp, "rb") as f:
                _BASE_CACHE[key] = f.read()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    from PIL import Image
    base = Image.open(BytesIO(_BASE_CACHE[key])).convert("RGBA")
    return base, view


# Diagonal label-anchor directions (unit-ish) so role labels fan into
# distinct quadrants and stay readable even when their dots coincide.
_ANCHOR_DIRS = {
    "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1),
}


def _draw_marker(draw, px: float, py: float, r: int, color, label: str, font,
                 anchor: str = "ne") -> None:
    """Draw a marker dot plus a label chip offset diagonally in ``anchor``.

    The label sits in a white rounded box outlined in the marker color, placed
    away from the dot in the anchor direction. Distinct anchors per role keep
    Start / Board / Alight / Target labels from overlapping each other.
    """
    draw.ellipse([px - r, py - r, px + r, py + r], fill=color,
                 outline=(255, 255, 255, 255), width=max(2, r // 4))
    if not label:
        return

    dx, dy = _ANCHOR_DIRS.get(anchor, (1, -1))
    gap = r + 6
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
    except Exception:
        bbox = (0, 0, 8 * len(label), 16)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 5
    # Anchor the text box so it grows away from the dot.
    tx = px + dx * gap if dx >= 0 else px + dx * gap - tw
    ty = py + dy * gap if dy >= 0 else py + dy * gap - th
    draw.rounded_rectangle(
        [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
        radius=6, fill=(255, 255, 255, 230), outline=color, width=2,
    )
    draw.text((tx - bbox[0], ty - bbox[1]), label, font=font, fill=_TEXT_DARK)


def _draw_legend(draw, font, route_w: int, route_color, route_label: str) -> None:
    """Small legend box, top-left under the title."""
    x0, y0 = _GMAPS_MARGIN, _GMAPS_MARGIN + 40
    pad = 12
    rows = [
        ("line", route_color, route_label),
        ("dot", START_COLOR, "Start"),
        ("dot", TARGET_COLOR, "Target"),
    ]
    line_h = 30
    box_w = 260
    box_h = pad * 2 + line_h * len(rows)
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=8,
                           fill=_LEGEND_BG, outline=_LEGEND_BORDER, width=2)
    cy = y0 + pad + line_h // 2
    for kind, color, text in rows:
        sx = x0 + pad
        if kind == "line":
            draw.line([sx, cy, sx + 34, cy], fill=color, width=max(4, route_w))
        else:
            r = 8
            draw.ellipse([sx + 9 - r, cy - r, sx + 9 + r, cy + r], fill=color,
                         outline=(255, 255, 255, 255), width=2)
        draw.text((sx + 44, cy - 9), text, font=font, fill=_TEXT_DARK)
        cy += line_h


def _draw_legend_rows(draw, font, rows: List[Tuple[str, Any, str]], route_w: int) -> None:
    """Generic legend box: each row is ("line"|"dot", color, label)."""
    x0, y0 = _GMAPS_MARGIN, _GMAPS_MARGIN + 40
    pad = 12
    line_h = 30
    box_w = 330
    box_h = pad * 2 + line_h * len(rows)
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=8,
                           fill=_LEGEND_BG, outline=_LEGEND_BORDER, width=2)
    cy = y0 + pad + line_h // 2
    for kind, color, text in rows:
        sx = x0 + pad
        if kind == "line":
            draw.line([sx, cy, sx + 34, cy], fill=color, width=max(4, route_w))
        else:
            r = 8
            draw.ellipse([sx + 9 - r, cy - r, sx + 9 + r, cy + r], fill=color,
                         outline=(255, 255, 255, 255), width=2)
        draw.text((sx + 44, cy - 9), text, font=font, fill=_TEXT_DARK)
        cy += line_h


def render_legs_overlay(
    dm: Any,
    legs: List[dict],
    markers: List[dict],
    legend_rows: List[Tuple[str, Any, str]],
):
    """
    Render the city map with multiple colored route legs and point markers.

    ``legs``    : list of ``{"nodes": [waypoint nodes], "color": rgba}``.
    ``markers`` : list of ``{"node": node, "color": rgba, "label": str}``.
    ``legend_rows`` : rows for the legend (see :func:`_draw_legend_rows`).

    Returns a PIL RGB ``Image`` or ``None``. Never raises, never mutates state.
    """
    sd = scenario_dir(dm)
    if not sd or not Path(sd).exists():
        return None
    try:
        from PIL import ImageDraw
        base, view = _base_image_and_view(sd)
        draw = ImageDraw.Draw(base)
        route_w = max(5, int(round(view.px_per_m * 1.6)))
        marker_r = max(9, int(round(view.px_per_m * 2.2)))
        font = _font(22)

        for leg in legs:
            pts = [
                view.to_px(float(n.position.x), float(n.position.y))
                for n in leg["nodes"] if n is not None
            ]
            if len(pts) >= 2:
                draw.line(pts, fill=(255, 255, 255, 255), width=route_w + 4, joint="curve")
                draw.line(pts, fill=leg["color"], width=route_w, joint="curve")

        placed: List[Tuple[float, float]] = []
        for m in markers:
            n = m.get("node")
            if n is None:
                continue
            px, py = view.to_px(float(n.position.x), float(n.position.y))
            anchor = m.get("anchor", "ne")
            dx, dy = _ANCHOR_DIRS.get(anchor, (1, -1))
            # If this dot (nearly) coincides with one already drawn, nudge it
            # diagonally in its role direction so it is not fully hidden.
            for (qx, qy) in placed:
                if abs(px - qx) <= 2 * marker_r and abs(py - qy) <= 2 * marker_r:
                    px += dx * 1.6 * marker_r
                    py += dy * 1.6 * marker_r
                    break
            placed.append((px, py))
            _draw_marker(draw, px, py, marker_r, m["color"], m.get("label", ""),
                         font, anchor=anchor)

        _draw_legend_rows(draw, font, legend_rows, route_w)
        return base.convert("RGB")
    except Exception:
        return None


def render_route_overlay(
    dm: Any,
    path: List[Any],
    route_color,
    route_label: str,
):
    """
    Render the city map with a route polyline overlaid.

    ``path`` is a list of waypoint-graph nodes (each exposing ``position.x/y``
    in centimeters). Returns a PIL ``Image`` (RGB) or ``None`` if a base map
    cannot be produced. Never raises and never mutates state.
    """
    sd = scenario_dir(dm)
    if not sd or not Path(sd).exists():
        return None
    try:
        from PIL import ImageDraw
        base, view = _base_image_and_view(sd)
        draw = ImageDraw.Draw(base)

        pts: List[Tuple[float, float]] = [
            view.to_px(float(n.position.x), float(n.position.y)) for n in path
        ]
        route_w = max(5, int(round(view.px_per_m * 1.6)))
        marker_r = max(9, int(round(view.px_per_m * 2.2)))
        font = _font(22)

        if len(pts) >= 2:
            # White casing under the route for contrast, then the colored line.
            draw.line(pts, fill=(255, 255, 255, 255), width=route_w + 4, joint="curve")
            draw.line(pts, fill=route_color, width=route_w, joint="curve")

        if pts:
            _draw_marker(draw, pts[0][0], pts[0][1], marker_r, START_COLOR, "Start",
                         font, anchor="sw")
            _draw_marker(draw, pts[-1][0], pts[-1][1], marker_r, TARGET_COLOR, "Target",
                         font, anchor="ne")

        _draw_legend(draw, font, route_w, route_color, route_label)
        return base.convert("RGB")
    except Exception:
        # Never let rendering break the (query-only) action.
        return None
