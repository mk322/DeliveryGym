"""Draw the phone's map: streets, a route on them, and where you are standing.

``navigate()`` spoke its directions and showed nothing, which is not what a
phone does. A map app draws you a picture — the streets around you, the line you
are meant to follow, a pin on the destination, your own position with the
direction you are facing — and the picture is most of why the app is useful. A
courier glances at it and knows whether the next turn is the first or the third,
which a spoken list of five legs does not tell you.

Everything here is drawn from the compiled road network and nothing else, so it
works on any map the compiler can build and needs no renderer, no engine and no
assets. That also fixes what it is: **a map, not a window**. It shows what a
survey knows — geometry, names, the route — and it cannot show a light, a
barrier, a skip or a shopfront, because a map app cannot see the street. That
line is the whole design of this environment's tool set and this image must not
cross it. What the courier sees out of its eyes stays in the photographs.

Rendered as SVG on purpose. It is text, so it costs nothing to store beside a
trajectory and can be diffed; it scales without going soft when a vision model
resizes it; and it needs no image library on the path that produces it. Callers
that want pixels rasterise once at the edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

# How much of the city the phone shows. A map app frames the route with a
# margin, and clamps how far it will zoom out so a long route stays a route
# rather than a hairline across the whole district.
MARGIN_FRACTION = 0.12
MIN_SPAN_CM = 12000.0     # never zoom in past ~120 m across: context matters
MAX_SPAN_CM = 90000.0     # never zoom out past ~900 m: the line must stay readable
# Drawing size. 4:3 to match the photographs, so a model resizing both sees them
# at the same scale.
# Back to the photographs' 4:3, and not for looks. Portrait crashed training:
# verl's get_rope_index for Qwen3-VL raised "shape mismatch: value tensor of
# shape [3, 6013] cannot be broadcast to indexing result of shape [3, 5939]",
# and the gap varied per turn -- the token count the vision grid produces
# disagreeing with the count the position-id code expects when one image in
# the batch is a different shape from the others. Evaluation never saw it
# because that path lets vLLM's server do the preprocessing; only training
# runs this function.
#
# Everything else about the redesign stays: the banner, the heading puck, the
# dark ground, the label placement.
WIDTH_PX, HEIGHT_PX = 720, 540
# Streets near the route are drawn; the rest of the city is not, or a dense map
# reads as noise. Measured in multiples of the framed span.
CONTEXT_PAD = 0.25


def _round_scale(span_cm: float) -> tuple[float, str]:
    """A scale bar length that is a round number of metres, and its label."""
    target = span_cm * 0.25 / 100.0
    for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if target <= step:
            return step * 100.0, f"{step} m"
    return 100000.0, "1 km"


@dataclass
class MapView:
    """The window on the city this drawing covers, in map centimetres."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width_px: int = WIDTH_PX
    height_px: int = HEIGHT_PX

    @property
    def span_cm(self) -> float:
        return max(self.max_x - self.min_x, self.max_y - self.min_y, 1.0)

    def to_px(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map centimetres to drawing pixels, with north up.

        **This map's north is +x and its east is +y.** Not a convention anyone
        would choose, but it is the one ``bearing_deg`` and ``compass_of``
        agree on -- ``bearing_deg((0,0),(1,0))`` is 0 degrees and
        ``compass_of(0)`` is "north" -- and the whole environment speaks it, so
        the drawing has to as well.

        The first version of this assumed the ordinary +y-is-north and flipped y
        accordingly, which drew the city rotated 90 degrees under a compass rose
        pointing the wrong way: every heading in the spoken route disagreed with
        the picture beside it. It survived its own unit test because the test was
        written from the same assumption. The test now derives the convention
        from ``bearing_deg`` instead of restating it.

        So: north (+x) goes up the drawing, east (+y) goes right.
        """
        scale = self.scale_px_per_cm()
        centre_x = (self.min_x + self.max_x) / 2.0
        centre_y = (self.min_y + self.max_y) / 2.0
        return (self.width_px / 2.0 + (point[1] - centre_y) * scale,
                self.height_px / 2.0 - (point[0] - centre_x) * scale)

    def scale_px_per_cm(self) -> float:
        """Pixels per map centimetre. One scale for both axes -- see ``to_px``:
        the drawing's width spans the map's y and its height spans the map's x.
        """
        return min(self.width_px / max(self.max_y - self.min_y, 1.0),
                   self.height_px / max(self.max_x - self.min_x, 1.0))


def frame_view(points: Iterable[tuple[float, float]], *,
               width_px: int = WIDTH_PX, height_px: int = HEIGHT_PX) -> MapView:
    """The window that holds every given point, with a margin, squared up."""
    points = list(points)
    if not points:
        return MapView(-MIN_SPAN_CM / 2, -MIN_SPAN_CM / 2,
                       MIN_SPAN_CM / 2, MIN_SPAN_CM / 2, width_px, height_px)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    centre_x, centre_y = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    # One span for both axes, then fitted to the drawing's aspect. Framing each
    # axis separately stretches the city, and a stretched map turns a right
    # angle into something the courier cannot match to the corner it is on.
    # The drawing is wider than it is tall and its width spans the map's y, so
    # the y extent is the one that gets the extra room.
    span = max(max(ys) - min(ys), (max(xs) - min(xs)) * width_px / height_px)
    span = max(MIN_SPAN_CM, min(MAX_SPAN_CM, span * (1.0 + 2 * MARGIN_FRACTION)))
    half_y = span / 2.0
    half_x = span / 2.0 * height_px / width_px
    # The clamp above bounds one axis; on a portrait frame the other is longer
    # by the aspect ratio and slipped past it, so a long route zoomed out to
    # 151 km across against a 90 km cap. Bound whichever axis ends up longer.
    widest = 2.0 * max(half_x, half_y)
    if widest > MAX_SPAN_CM:
        shrink = MAX_SPAN_CM / widest
        half_x, half_y = half_x * shrink, half_y * shrink
    narrowest = 2.0 * min(half_x, half_y)
    if narrowest < MIN_SPAN_CM:
        grow = MIN_SPAN_CM / narrowest
        half_x, half_y = half_x * grow, half_y * grow
    return MapView(centre_x - half_x, centre_y - half_y,
                   centre_x + half_x, centre_y + half_y, width_px, height_px)


def _keep_marker_visible(view: MapView, here: tuple[float, float],
                         width_px: int, height_px: int) -> MapView:
    """Slide the window until the courier sits clear of the banner and edges."""
    top, bottom = BAR_H + 70.0, height_px - 90.0
    side = 70.0
    for _ in range(4):
        x, y = view.to_px(here)
        scale = view.scale_px_per_cm()
        shift_x = shift_y = 0.0
        # Screen y runs opposite to map x -- y_screen = H/2 - (x - cx)*scale --
        # so pushing the marker down the screen means raising the window's x
        # centre, not lowering it. The sign was the other way and the fix moved
        # thirteen frames in a hundred further off the edge.
        if y < top:
            shift_y = (top - y) / scale
        elif y > bottom:
            shift_y = (bottom - y) / scale
        if x < side:
            shift_x = (x - side) / scale
        elif x > width_px - side:
            shift_x = (x - (width_px - side)) / scale
        if not shift_x and not shift_y:
            break
        # y on screen decreases as map x grows, so a downward shift of the
        # marker is a decrease of the window's x centre.
        view = MapView(view.min_x + shift_y, view.min_y + shift_x,
                       view.max_x + shift_y, view.max_y + shift_x,
                       width_px, height_px)
    return view


@dataclass
class MapDrawing:
    """One rendered map, and what went into it."""

    svg: str
    view: MapView
    streets_drawn: int = 0
    route_metres: float = 0.0
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "svg": self.svg, "streets_drawn": self.streets_drawn,
            "route_metres": round(self.route_metres, 1),
            "labels": list(self.labels),
            "span_m": round(self.view.span_cm / 100.0, 1),
        }


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def _halo_text(x: float, y: float, text: str, cls: str = "ui",
               anchor: str = "middle", rotate: float | None = None) -> str:
    """A label drawn twice: once as a thick background-coloured outline, once
    filled on top.

    The one-element version used ``paint-order="stroke"``, which is the tidy way
    to say it and is not honoured by every rasteriser -- cairosvg paints the
    stroke last, so every label on the first map came out as a smear of
    background colour. Two elements is uglier and works everywhere, and a name
    a courier cannot read is not a label.
    """
    turn = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    common = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{turn}'
    body = _esc(text)
    return (f'<text class="{cls} halo" {common}>{body}</text>'
            f'<text class="{cls}" {common}>{body}</text>')


# Sized for the frame it sits on. 150 was chosen against a 940 px portrait;
# on the 540 px frame it plus the foot bar left so little room that the label
# rules rejected every street name and the map came out unlabelled.
BAR_H = 96.0


def _text_box(x: float, y: float, half_w: float, half_h: float,
              angle_deg: float) -> tuple[float, float, float, float]:
    """The axis-aligned box a rotated label occupies, generously."""
    a = math.radians(angle_deg)
    dx = abs(half_w * math.cos(a)) + abs(half_h * math.sin(a))
    dy = abs(half_w * math.sin(a)) + abs(half_h * math.cos(a))
    return (x - dx, y - dy, x + dx, y + dy)


def _overlaps(box, placed) -> bool:
    return any(not (box[2] < b[0] or box[0] > b[2]
                    or box[3] < b[1] or box[1] > b[3]) for b in placed)


def _crosses_route(box, route_px) -> bool:
    """Does this label's box sit on the drawn route?"""
    for (ax, ay), (bx, by) in zip(route_px, route_px[1:]):
        steps = max(2, int(math.dist((ax, ay), (bx, by)) / 12))
        for i in range(steps + 1):
            t = i / steps
            px, py = ax + (bx - ax) * t, ay + (by - ay) * t
            # Wider than the route's own stroke. At 6 px a label could sit
            # eight pixels off the centreline, pass the test, and still have
            # its halo -- a 7 px stroke in the background colour -- eat a bite
            # out of an 11 px line. The route came out broken.
            if box[0] - 16 < px < box[2] + 16 and box[1] - 16 < py < box[3] + 16:
                return True
    return False


def _near_puck(box, puck, radius: float = 42.0) -> bool:
    """Keep names off the courier's own marker, which must stay findable."""
    px, py = puck
    return (box[0] - radius < px < box[2] + radius
            and box[1] - radius < py < box[3] + radius)


def _edge_safe_text(x: float, y: float, text: str, width_px: float) -> str:
    """A caption that never runs off the side, whatever it is anchored to.

    Centred text overhangs by half its width, so a marker near an edge lost
    the start or the end of its label -- "11 Rue Mouffetard" arrived as "1 Rue
    Mouffetard". Rather than clamp the centre and hope, this switches the
    anchor: hard against the left margin near the left edge, against the right
    near the right, centred in between.
    """
    half = len(text) * 13.0 / 2.0
    margin = 14.0
    # A caption wider than the screen cannot be placed, only trimmed. Losing
    # the tail with an ellipsis says so; losing it to the frame edge looks
    # like a rendering fault and hides that anything is missing.
    if 2 * half > width_px - 2 * margin:
        keep = max(6, int((width_px - 2 * margin) / 13.0) - 1)
        text = text[:keep] + "\u2026"
        half = len(text) * 13.0 / 2.0
    if x - half < margin:
        return _halo_text(margin, y, text, anchor="start")
    if x + half > width_px - margin:
        return _halo_text(width_px - margin, y, text, anchor="end")
    return _halo_text(x, y, text)


def _label_anchor(polyline: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Where to write a street's name, and at what angle, in drawing pixels.

    The middle of its longest drawn run, rotated to lie along it -- which is how
    a map labels a street and why the name is readable without a legend.
    """
    best = (0.0, polyline[0], polyline[0])
    for a, b in zip(polyline, polyline[1:]):
        length = math.dist(a, b)
        if length > best[0]:
            best = (length, a, b)
    _, a, b = best
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    # Never upside down: a name at 170 degrees reads as mirror writing.
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, angle)


def render_map(
    network: Any,
    *,
    here: tuple[float, float],
    facing_deg: float | None = None,
    route: list[tuple[float, float]] | None = None,
    destination: tuple[float, float] | None = None,
    destination_label: str = "",
    here_label: str = "you are here",
    next_street: str = "",
    next_heading: str = "",
    next_maneuver_heading: str = "",
    next_maneuver_distance_cm: float | None = None,
    blocked: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    width_px: int = WIDTH_PX,
    height_px: int = HEIGHT_PX,
) -> MapDrawing:
    """Draw the map a courier would be looking at.

    ``route`` and ``destination`` are optional: with neither, this is the "where
    am I" view a courier gets from opening the app without asking for anything.

    ``blocked`` is kept for callers that want to draw a street as shut. The
    courier environment never passes it: a rider does not file a report with the
    map app, so the phone is never told about a barrier and cannot draw one. What
    the map shows is the survey, and going round is worked out from the
    photographs.
    """
    route = route or []
    interest = [here] + list(route)
    if destination is not None:
        interest.append(destination)
    view = frame_view(interest, width_px=width_px, height_px=height_px)
    # The courier must be on the screen, and not under the banner. Framing on
    # the route alone put the marker above the top edge whenever the route ran
    # off that way, so the one thing the picture must always show -- where you
    # are -- was the thing missing from it.
    view = _keep_marker_visible(view, here, width_px, height_px)
    pad = view.span_cm * CONTEXT_PAD

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" '
        f'width="{width_px}" height="{height_px}" role="img" '
        f'aria-label="map of the streets around the courier">',
        '<defs><style>'
        # A navigation app's palette, and for the same reasons it uses one: the
        # route has to be the brightest thing on the screen, the streets have to
        # read as a network without competing with it, and the labels have to
        # sit on a ground dark enough that a halo actually separates them. The
        # old light-grey-on-cream did none of that -- the route was one blue
        # line among a dozen white ones, and every label fought the buildings.
        '.bg{fill:#1b2735}.blk{fill:#22303f;stroke:#26364a;stroke-width:.8}'
        '.st{stroke:#4a5f78;stroke-linecap:round;stroke-linejoin:round;fill:none}'
        '.nm{font:600 20px ui-sans-serif,sans-serif;fill:#c3d0e0}'
        '.rtc{stroke:#0d1720;stroke-width:17;stroke-linecap:round;'
        'stroke-linejoin:round;fill:none}'
        '.rt{stroke:#4c9bff;stroke-width:11;stroke-linecap:round;'
        'stroke-linejoin:round;fill:none}'
        '.shut{stroke:#ff5a4d;stroke-width:6;stroke-linecap:round;fill:none}'
        '.ui{font:700 23px ui-sans-serif,sans-serif;fill:#eef3fa}'
        '.pin{fill:#ff5d4e}.me{fill:#4c9bff}'
        '.halo{stroke:#141d29;stroke-width:7;stroke-linejoin:round;fill:none}'
        '.go{fill:#4c9bff;stroke:#0d1720;stroke-width:3;stroke-linejoin:round}'
        # The instruction banner, and the bar that carries the distance.
        '.band{fill:#0f7a4a;stroke:#4fd39a;stroke-width:3}'
        '.bandtext{font:800 32px ui-sans-serif,sans-serif;fill:#ffffff}'
        '.bandsub{font:700 19px ui-sans-serif,sans-serif;fill:#bff0d8}'
        '.bandpreview{font:700 17px ui-sans-serif,sans-serif;fill:#ffffff}'
        '.bar{fill:#0d1720}'
        '.bartext{font:700 26px ui-sans-serif,sans-serif;fill:#ffffff}'
        '.barsub{font:500 19px ui-sans-serif,sans-serif;fill:#8fa4bd}'
        '.puck{fill:#4c9bff;stroke:#ffffff;stroke-width:4}'
        '.puckring{fill:#4c9bff;opacity:.22}'
        '</style></defs>',
        f'<rect class="bg" width="{width_px}" height="{height_px}"/>',
    ]

    # ── the blocks between the streets ───────────────────────────────────────
    # A map is mostly this. Without them the drawing is white lines on a flat
    # ground and gives a courier nothing to match against what it can see; with
    # them, the shape of a corner on the screen is the shape of the corner it is
    # standing on. Drawn first so everything else sits on top.
    for building in getattr(network, "buildings", []):
        min_x, min_y, max_x, max_y = building.box
        if (max_x < view.min_x - pad or min_x > view.max_x + pad
                or max_y < view.min_y - pad or min_y > view.max_y + pad):
            continue
        # Top-left on screen is (max north, min east) = (max_x, min_y).
        x0, y0 = view.to_px((max_x, min_y))
        x1, y1 = view.to_px((min_x, max_y))
        if x1 - x0 < 1.5 or y1 - y0 < 1.5:
            continue
        parts.append(f'<rect class="blk" x="{x0:.1f}" y="{y0:.1f}" '
                     f'width="{x1-x0:.1f}" height="{y1-y0:.1f}" rx="1"/>')

    # ── the streets ──────────────────────────────────────────────────────────
    drawn = 0
    # Computed before the streets are labelled, because a name may not be
    # printed across the route and the route is drawn after them.
    route_px = [view.to_px(q) for q in (route or [])]
    labelled: set[str] = set()
    # Where a name has already been written, so the next one can keep clear.
    placed: list[tuple[float, float, float, float]] = []
    labels: list[str] = []
    for street in getattr(network, "streets", []):
        # Kept or dropped whole, by whether the street's extent touches the
        # window -- not by which of its *vertices* land inside it. The source
        # splines carry 120 vertices between 59 streets, so a street can cross
        # the whole drawing with both its vertices outside: filtering by vertex
        # deleted exactly those, and the map came out with the route running
        # across blank ground while every street it actually follows was
        # missing. The SVG viewBox clips the overhang for free.
        if len(street.polyline) < 2:
            continue
        xs = [p[0] for p in street.polyline]
        ys = [p[1] for p in street.polyline]
        if (max(xs) < view.min_x - pad or min(xs) > view.max_x + pad
                or max(ys) < view.min_y - pad or min(ys) > view.max_y + pad):
            continue
        points = list(street.polyline)
        pixels = [view.to_px(p) for p in points]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pixels))
        width = max(3.0, min(14.0, street.width_cm * view.scale_px_per_cm()))
        parts.append(f'<path class="st" style="stroke-width:{width:.1f}" d="{path}"/>')
        drawn += 1
        if street.name not in labelled and len(pixels) >= 2:
            # Anchored on the longest run that is actually on screen, so a
            # street entering the corner of the drawing is still named.
            visible = [q for q in pixels
                       if -20 < q[0] < width_px + 20 and -20 < q[1] < height_px + 20]
            x, y, angle = _label_anchor(visible if len(visible) >= 2 else pixels)
            # Legible type is wide type, and wide type collides. At 9 px the
            # names were unreadable but harmless; at a size that survives the
            # downscale they overprint each other and run off the edge, and two
            # names on top of one another are less use than one name alone.
            # So a name is drawn only if it fits inside the frame and clears
            # every name already placed.
            # A rotated name reaches half its own length either side of its
            # anchor, so two anchors 100 px apart can still overprint when the
            # names are long. Clearance is measured against the pair's own
            # widths rather than a constant.
            # A box, not a distance. A rotated name reaches half its length
            # either side of its anchor, so two anchors far enough apart to
            # pass a radius test still overprint when both are long and nearly
            # parallel -- which is most of a street grid. This projects each
            # name onto its own direction and rejects an overlap of the boxes.
            half_w = len(street.name) * 5.6
            angle = math.radians(angle_deg := angle)
            box = _text_box(x, y, half_w, 13.0, angle_deg)
            inset = 18.0
            fits = (box[0] > inset and box[2] < width_px - inset
                    and box[1] > BAR_H + 16
                    and box[3] < height_px - 46 - 8)
            # And off the route itself: a name printed across the blue line
            # hides the one mark the picture exists to show.
            clear = (not _overlaps(box, placed)
                     and not _near_puck(box, view.to_px(here))
                     and not _crosses_route(box, route_px))
            if fits and clear:
                placed.append(box)
                labelled.add(street.name)
                labels.append(street.name)
                parts.append(_halo_text(x, y, street.name, cls="nm", rotate=angle))

    # ── what the courier has told the phone is shut ──────────────────────────
    for a, b in (blocked or []):
        ax, ay = view.to_px(a)
        bx, by = view.to_px(b)
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        parts.append(f'<line class="shut" x1="{mx-6:.1f}" y1="{my-6:.1f}" '
                     f'x2="{mx+6:.1f}" y2="{my+6:.1f}"/>')
        parts.append(f'<line class="shut" x1="{mx-6:.1f}" y1="{my+6:.1f}" '
                     f'x2="{mx+6:.1f}" y2="{my-6:.1f}"/>')

    # ── the route ────────────────────────────────────────────────────────────
    metres = 0.0
    route_svg = ""
    first_leg_px: tuple[float, float] | None = None
    if len(route) >= 2:
        metres = sum(math.dist(a, b) for a, b in zip(route, route[1:])) / 100.0
        pixels = [view.to_px(p) for p in route]
        # Where the route goes first, in screen pixels, so the arrow below is
        # drawn from the geometry rather than from a bearing computed twice.
        first_leg_px = (pixels[1][0] - pixels[0][0], pixels[1][1] - pixels[0][1])
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pixels))
        # Held back and appended after the street names below. Drawn before
        # them, a halo could bite through it; the route is the one mark on
        # this picture that nothing is allowed to interrupt.
        route_svg = (f'<path class="rtc" d="{path}"/>'
                     f'<path class="rt" d="{path}"/>')

    parts.append(route_svg)

    # ── the destination ──────────────────────────────────────────────────────
    if destination is not None:
        x, y = view.to_px(destination)
        parts.append(
            f'<path class="pin" d="M{x:.1f},{y:.1f} l-12,-18 a14,14 0 1,1 24,0 z"/>'
            f'<circle cx="{x:.1f}" cy="{y-22:.1f}" r="5.4" fill="#eceae4"/>')
        if destination_label:
            # Clamped inside the frame: a caption that names the destination
            # is worth nothing with its first characters off the edge, and at
            # this size it overhangs easily.
            # 23 px bold runs about 12 px a character, and the label is
            # centred on its anchor, so half of it hangs either side. Clamped
            # on that, not on a guess: "Rue Mouffetard" was losing its R off
            # the left edge and "you are here" its last letter off the right.
            parts.append(_edge_safe_text(x, max(y - 34, BAR_H + 50),
                                         destination_label, width_px))

    # ── the courier ──────────────────────────────────────────────────────────
    x, y = view.to_px(here)
    if facing_deg is None:
        parts.append(f'<circle class="me" cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#ffffff"/>')
    else:
        # An arrowhead pointing the way the courier faces. The glyph is drawn
        # pointing right, bearing 0 is north and north is up, so a bearing turns
        # into a screen angle by subtracting the quarter turn between them.
        angle = facing_deg - 90.0
        parts.append(
            f'<g transform="translate({x:.1f},{y:.1f}) rotate({angle:.1f})">'
            f'<circle class="me" r="9" opacity="0.25"/>'
            f'<path class="me" d="M11,0 L-6,-6.5 L-3,0 L-6,6.5 Z"/></g>')
    # ── which way to go, said once and plainly ───────────────────────────────
    #
    # Everything else on this map is a fact about the city. This is the one
    # mark that answers the question the courier actually has, and it is drawn
    # to be read at a glance at the size the harness serves: a thick blue arrow
    # from the dot, along the first leg, outlined in white so it stands off
    # both the pale streets and the blocks between them.
    # The courier, drawn the way a navigation app draws it: a solid disc with
    # a white collar and a faint halo, so it is the most findable thing on the
    # map. It was a 12 px circle under an arrow, and in a frame full of white
    # streets and black labels it was not findable at all -- which made "where
    # am I" the hardest question the picture answered.
    heading = (math.degrees(math.atan2(first_leg_px[1], first_leg_px[0]))
               if first_leg_px else (facing_deg - 90.0 if facing_deg is not None
                                     else -90.0))
    parts.append(
        f'<circle class="puckring" cx="{x:.1f}" cy="{y:.1f}" r="34"/>'
        f'<circle class="puck" cx="{x:.1f}" cy="{y:.1f}" r="19"/>'
        f'<g transform="translate({x:.1f},{y:.1f}) rotate({heading:.1f})">'
        f'<path d="M12,0 L-7,-9 L-3,0 L-7,9 Z" fill="#ffffff"/></g>')
    # A single direction mark. There used to be a second one -- a large arrow
    # beside the puck along the same bearing -- and two arrows saying the same
    # thing at slightly different sizes read as two different claims. The
    # reference this is drawn from has one: the puck, with a chevron in it.
    label_offset = 30.0
    if first_leg_px is not None and first_leg_px[1] > 0:
        # Caption above the marker when the route heads down the page, so the
        # words never sit on the way ahead.
        label_offset = -30.0
    if here_label:
        # Offset clear of the marker, because the street name it sits on is
        # drawn along the street and the two collided.
        cy = min(max(y + label_offset, BAR_H + 26.0), height_px - 18.0)
        parts.append(_edge_safe_text(x, cy, here_label, width_px))

    # ── scale bar and north ──────────────────────────────────────────────────
    bar_cm, bar_text = _round_scale(view.span_cm)
    bar_px = bar_cm * view.scale_px_per_cm()
    bx, by = 14.0, height_px - 16.0
    parts.append(f'<line x1="{bx}" y1="{by}" x2="{bx+bar_px:.1f}" y2="{by}" '
                 f'stroke="#3a3730" stroke-width="2"/>'
                 f'<line x1="{bx}" y1="{by-4}" x2="{bx}" y2="{by+4}" '
                 f'stroke="#3a3730" stroke-width="2"/>'
                 f'<line x1="{bx+bar_px:.1f}" y1="{by-4}" x2="{bx+bar_px:.1f}" y2="{by+4}" '
                 f'stroke="#3a3730" stroke-width="2"/>'
                 + _halo_text(bx, by - 8, bar_text, anchor="start"))
    nx, ny = width_px - 22.0, 26.0
    parts.append(f'<g transform="translate({nx},{ny})">'
                 f'<path d="M0,-11 L5,7 L0,3 L-5,7 Z" fill="#3a3730"/>'
                 + _halo_text(0, 20, "N") + '</g>')
    # The bar, drawn last so nothing can overprint it. A navigation app keeps
    # the distance and the destination out of the map; here they used to be
    # written into the top-left corner where a street name also wanted to go,
    # and the two collided on most frames.
    # The instruction banner. This is the single most useful mark on a
    # navigation screen and the one this map never had: it names the street to
    # take, in words, where a bearing had to be inferred from the geometry.
    # Measured on this model: a street name is read off this map correctly ~100%
    # of the time and a direction 25%, so the banner turns the task's hardest
    # perceptual step into its easiest.
    if next_street:
        parts.append(f'<rect class="band" x="10" y="10" rx="12" '
                     f'width="{width_px - 20}" height="{BAR_H}"/>')
        # The glyph turns with the instruction. It was a fixed up-arrow over
        # the words "head down", which is a phrase about the page rather than
        # about the city and was the same on every frame whichever way the
        # route went.
        turn = {"north": 0, "north-east": 45, "east": 90, "south-east": 135,
                "south": 180, "south-west": 225, "west": 270,
                "north-west": 315}.get(next_heading, 0)
        parts.append(
            f'<g transform="translate(52,{10 + BAR_H / 2}) rotate({turn})">'
            f'<path d="M0,-24 L17,3 L7,3 L7,24 L-7,24 L-7,3 L-17,3 Z" '
            f'fill="#ffffff"/></g>')
        said = f"head {next_heading}" if next_heading else "take"
        parts.append(f'<text class="bandsub" x="92" y="{10 + 40}">'
                     f'{_esc(said)} on</text>')
        if next_maneuver_heading and next_maneuver_distance_cm is not None:
            # Dense Recast routes can put a real turn only one or two metres
            # beyond the current node.  A phone that names only that first
            # one-metre tangent gives a 3--10 m visual action no chance to stop
            # for the turn.  The preview is derived from a later point on the
            # *same* route; it neither changes nor shortcuts the blue line.
            maneuver_m = max(1, round(next_maneuver_distance_cm / 100.0))
            parts.append(
                f'<text class="bandpreview" x="{width_px - 24}" '
                f'y="{10 + 40}" text-anchor="end">'
                f'then {_esc(next_maneuver_heading)} in {maneuver_m} m</text>'
            )
        parts.append(f'<text class="bandtext" x="92" y="{10 + 78}">'
                     f'{_esc(next_street)}</text>')
    # The distance, on a bar of its own at the foot, as an app does.
    foot = 46.0
    parts.append(f'<rect class="bar" y="{height_px - foot}" '
                 f'width="{width_px}" height="{foot}"/>')
    if metres:
        parts.append(f'<text class="bartext" x="18" y="{height_px - 14}">'
                     f'{metres:.0f} m</text>')
    if destination_label:
        parts.append(f'<text class="barsub" x="{width_px - 18}" '
                     f'y="{height_px - 16}" text-anchor="end">'
                     f'to {_esc(destination_label)}</text>')
    parts.append("</svg>")

    return MapDrawing(svg="".join(parts), view=view, streets_drawn=drawn,
                      route_metres=metres, labels=labels)
