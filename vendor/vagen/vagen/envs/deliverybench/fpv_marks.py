# fpv_marks.py
# -*- coding: utf-8 -*-

"""Set-of-Marks helpers for the baked DeliveryBench FPV (enable_waypoint_marks).

Shared by the environment's FPV cross composer
(``deliverybench_env._build_fpv_cross``) and the standalone F0 tooling
(``tools/fpv_waypoint_marks.build_marked_cross``), so projection math and
marker style cannot drift between the two.

Geometry recap (empirically verified in F0; see F1_WAYPOINT_MARKS_PLAN.md):
  * each baked photo is a 90-degree pinhole: f = (W/2)/tan(45 deg); the camera
    sits 160 cm above the ground; a ground-level point at relative bearing rho
    and distance d projects to
        x = W/2 + f*tan(rho),   y = H/2 + f*160/(d*cos(rho)).
  * panel -> compass direction (relative to facing) includes the env's side
    swap: front=+0, right=+270, back=+180, left=+90.
  * markers are drawn on the FINAL composed canvas at a fixed pixel size so
    the numbers stay legible in the quarter-scale side panels.
  * candidate text must NEVER use compass bearings: the graph bearing axis is
    flipped relative to the rendered map's compass rose (F0 finding). Use
    panel language when an FPV is shown, MOVE-direction language otherwise.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

HFOV_DEG = 90.0
CAM_H_CM = 160.0
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# Very-near candidates project below the frame; clamp them this many px above
# the panel bottom. F0 shipped 12 px and dead-end scenes showed the disc
# clipping at the panel edge, so F1 raised it.
NEAR_CLAMP_PX = 30.0
# panel label -> compass dir it looks at, relative to facing (env's swap included)
PANEL_DIRS = {"front": 0.0, "right": 270.0, "back": 180.0, "left": 90.0}
_MOVE_WORDS = {0.0: "forward of you", 90.0: "to your right",
               180.0: "behind you", 270.0: "to your left"}


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def panel_label_for(bearing_deg: float, facing_deg: float) -> str:
    """Nearest FPV panel (front/right/back/left) for a compass bearing."""
    best, best_rel = "front", 1e9
    for label, doff in PANEL_DIRS.items():
        rel = abs(wrap180(float(bearing_deg) - (float(facing_deg) + doff)))
        if rel < best_rel:
            best, best_rel = label, rel
    return best


def move_words_for(bearing_deg: float, facing_deg: float) -> str:
    """MOVE-direction phrasing ("forward of you", ...) for a compass bearing.
    The map-only fallback wording for the candidate text."""
    best, best_rel = "forward of you", 1e9
    for off, words in _MOVE_WORDS.items():
        rel = abs(wrap180(float(bearing_deg) - (float(facing_deg) + off)))
        if rel < best_rel:
            best, best_rel = words, rel
    return best


def project_in_photo(rel_deg: float, dist_cm: float, w: int, h: int,
                     mirror: bool = False) -> Optional[Tuple[float, float]]:
    """Pixel position of a ground-level point at relative bearing ``rel_deg``
    (vs this photo's view axis) and distance ``dist_cm``. None if outside FOV.

    ``mirror`` flips the horizontal sign; it is uncalibrated on grid maps
    (all candidates project to panel centres there) and must be calibrated
    before non-grid maps (F2).
    """
    if abs(rel_deg) > HFOV_DEG / 2:
        return None
    f = (w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    rho = math.radians(rel_deg)
    sign = -1.0 if mirror else 1.0
    x = w / 2.0 + sign * f * math.tan(rho)
    z = max(dist_cm * math.cos(rho), 1.0)          # forward distance
    y = h / 2.0 + f * CAM_H_CM / z                  # ground point below horizon
    y = min(y, h - NEAR_CLAMP_PX)                   # clamp very-near points
    return x, y


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, label: str,
                r: int = 14) -> None:
    """Glowing numbered disc, fixed screen size (drawn on the final canvas)."""
    for rr, alpha in ((r + 8, 60), (r + 4, 110)):
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 210, 40, alpha))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 190, 20, 235),
                 outline=(20, 20, 20, 255), width=2)
    try:
        font = ImageFont.truetype(FONT_BOLD, 19)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x - tw / 2, y - th / 2 - tb[1]), label, font=font,
              fill=(10, 10, 10, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))


def overlay_marks(canvas: Image.Image,
                  geom: Dict[str, Tuple[int, int, int, int]],
                  native_size: Tuple[int, int],
                  candidates: List[Dict[str, Any]],
                  facing_deg: float,
                  *, mirror: bool = False) -> Tuple[Image.Image, int]:
    """Draw one numbered glow marker per candidate onto a composed FPV cross.

    ``geom``: panel label -> (paste_x, paste_y, panel_w, panel_h) on the
    canvas; ``native_size``: (w, h) of one full-resolution FPV photo (the
    projection frame). Each candidate is assigned to its nearest panel, so
    |relative bearing| <= 45 deg and the projection always lands. Returns the
    composited image and the number of markers drawn (callers assert it
    equals the candidate-text count).
    """
    if not candidates:
        return canvas, 0
    w, h = native_size
    facing = float(facing_deg) % 360.0
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    drawn = 0
    for c in candidates:
        bearing = float(c["bearing_deg"]) % 360.0
        label = panel_label_for(bearing, facing)
        rel = wrap180(bearing - (facing + PANEL_DIRS[label]))
        px, py, pw, ph = geom[label]
        pt = project_in_photo(rel, float(c["dist_cm"]), w, h, mirror)
        if pt is None:
            continue
        cx = px + pt[0] * (pw / w)
        cy = py + pt[1] * (ph / h)
        draw_marker(od, cx, cy, str(c["index"]))
        drawn += 1
    out = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    return out, drawn
