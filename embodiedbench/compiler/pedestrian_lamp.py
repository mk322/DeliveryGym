"""Draw a Parisian pedestrian lamp, and put one on a street view.

Why this exists rather than another bake. The map carries no lamp objects:
signalised junctions are derived from node degree, and the renderer put a
single light mesh at each of them. A junction with three ways out therefore
had three photographs of *the same lamp* from the same camera, and the
environment -- whose phase logic is per-approach and correct, one direction
red while the crossing direction is green -- was giving each of them its own
phase and charging for it. One lamp, asked to say three different things at
once. Told "the lamp for Rue de la Paix" and "the lamp for Rue Cujas", the
courier held two pictures with identical backgrounds and nothing but the
caption to tell them apart.

A real crossing has a lamp per leg, facing the person waiting to cross it. So
each approach gets its own, composited onto that approach's own street view:
the lamp belongs to one street, shows that street's phase, and is drawn at a
size that can actually be read after the harness downscales to 320 px. The
measured alternative was 43 model pixels of a lamp shared between three
streets.

It is a composite and says so. What it buys is that the phase is genuinely in
the picture and genuinely per-street, which is what the mechanic charges for.

The figure follows the French convention: a dark housing with two apertures,
the upper red standing figure, the lower green walking figure, exactly one lit.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

# Colours from the lamps themselves rather than from a palette: a lit LED
# aperture against an unlit one, and a housing dark enough to read as metal
# against a pale Haussmann facade.
RED_ON = (235, 60, 45)
GREEN_ON = (60, 210, 110)
OFF = (34, 34, 36)
HOUSING = (44, 46, 48)
HOUSING_EDGE = (22, 23, 24)
POLE = (58, 60, 62)


@dataclass(frozen=True)
class LampGeometry:
    """How big the lamp is drawn, in pixels of the frame it goes on.

    Sized from the harness, not from the scene. The frames are 1280 px wide and
    are served at 320, so anything that must survive is drawn four times larger
    than it needs to end up. A 240 px-tall housing here is 60 px served, and
    the walking figure inside it about 20 px -- read at a glance, where the
    baked lamp was 43 px of area and had to be hunted for.
    """

    # A fraction of the frame's height, not a pixel count: the albums are
    # 640x480 and 1280x960 and a lamp sized for one swamps the other. 0.30
    # leaves the street, the pavement and the far junction all visible while
    # putting about 70 px of lamp -- and 25 px of figure -- into the 320 px
    # the harness serves.
    housing_frac: float = 0.30
    margin_frac: float = 0.035
    baseline: float = 0.72    # where the foot of the housing sits

    def housing_h(self, frame_h: int) -> int:
        return max(48, int(frame_h * self.housing_frac))

    def housing_w(self, frame_h: int) -> int:
        return int(self.housing_h(frame_h) * 0.44)

    def margin(self, frame_w: int) -> int:
        return max(6, int(frame_w * self.margin_frac))


def _figure(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
            colour: tuple[int, int, int], walking: bool) -> None:
    """The little person: standing for red, mid-stride for green.

    Drawn rather than blitted from a sprite so it scales with the housing and
    stays legible at whatever size the geometry asks for.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = x0 + w // 2
    unit = max(2, h // 14)

    head_r = int(unit * 1.5)
    head_y = y0 + head_r + unit // 2
    draw.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r],
                 fill=colour)

    torso_top = head_y + head_r + unit // 2
    torso_bottom = y0 + int(h * 0.60)
    draw.line([(cx, torso_top), (cx, torso_bottom)], fill=colour, width=unit)

    if walking:
        # One leg forward, one trailing; arms opposed, which is what reads as
        # movement at small sizes far more than the exact pose does.
        draw.line([(cx, torso_bottom), (cx + int(w * 0.30), y1 - unit)],
                  fill=colour, width=unit)
        draw.line([(cx, torso_bottom), (cx - int(w * 0.22), y1 - unit)],
                  fill=colour, width=unit)
        draw.line([(cx, torso_top + unit), (cx - int(w * 0.30), torso_top + unit * 3)],
                  fill=colour, width=unit)
        draw.line([(cx, torso_top + unit), (cx + int(w * 0.26), torso_top + unit * 4)],
                  fill=colour, width=unit)
    else:
        draw.line([(cx, torso_bottom), (cx - int(w * 0.13), y1 - unit)],
                  fill=colour, width=unit)
        draw.line([(cx, torso_bottom), (cx + int(w * 0.13), y1 - unit)],
                  fill=colour, width=unit)
        draw.line([(cx, torso_top + unit), (cx - int(w * 0.17), torso_bottom - unit)],
                  fill=colour, width=unit)
        draw.line([(cx, torso_top + unit), (cx + int(w * 0.17), torso_bottom - unit)],
                  fill=colour, width=unit)


def render_lamp(state: str, frame_h: int,
                geometry: LampGeometry | None = None) -> Image.Image:
    """One lamp head, RGBA, sized for a frame this tall."""
    if state not in ("red", "green"):
        raise ValueError(f"a lamp is red or green, not {state!r}")
    g = geometry or LampGeometry()
    w, h = g.housing_w(frame_h), g.housing_h(frame_h)
    lamp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(lamp)

    radius = max(3, w // 8)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                           fill=HOUSING, outline=HOUSING_EDGE, width=max(2, w // 20))

    inset = max(4, w // 9)
    pane_h = (h - inset * 3) // 2
    panes = {
        "red": (inset, inset, w - inset, inset + pane_h),
        "green": (inset, inset * 2 + pane_h, w - inset, h - inset),
    }
    for phase, box in panes.items():
        lit = phase == state
        draw.rounded_rectangle(box, radius=max(2, radius // 2),
                               fill=OFF if not lit else (12, 12, 13))
        if lit:
            pad = max(2, w // 12)
            inner = (box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad)
            _figure(draw, inner,
                    RED_ON if phase == "red" else GREEN_ON,
                    walking=(phase == "green"))
    return lamp


def place_on(frame: Image.Image, state: str, *, on_left: bool = True,
             geometry: LampGeometry | None = None) -> Image.Image:
    """Composite a lamp onto one street view, at the kerb the courier is on.

    Placed against the frame edge at head height rather than out in the scene:
    the lamp a person reads before stepping off a kerb is the one beside them,
    and putting it there needs no depth information the album does not carry.
    """
    g = geometry or LampGeometry()
    out = frame.convert("RGBA")
    lamp = render_lamp(state, out.height, g)
    margin = g.margin(out.width)
    y = int(out.height * g.baseline) - lamp.height
    x = margin if on_left else out.width - margin - lamp.width

    # A short pole down to the pavement, so the head is mounted rather than
    # floating. It also tells the eye which frame edge the lamp belongs to.
    pole = ImageDraw.Draw(out)
    pole_w = max(3, lamp.width // 7)
    pole_x = x + lamp.width // 2 - pole_w // 2
    pole.rectangle([pole_x, y + lamp.height - 2,
                    pole_x + pole_w, int(out.height * 0.92)], fill=POLE)

    out.alpha_composite(lamp, (x, y))
    return out.convert("RGB")
