"""Build the walkable street network, its names and its addresses. Rule-based, no AI.

The graph the vendored engine ships for Paris is not a street network. Measured
against the map's own building footprints, **652 of its 1162 nodes -- 56% -- sit
inside a building**, and the album proves it: those nodes render at mean
brightness 39.6 against 88.3 for the rest, because the camera is indoors. Edges
between them cut through walls. A courier standing on one cannot see a street,
cannot name where it is, and cannot tell which way to walk. No amount of prompt
work fixes that, because the geometry itself is wrong.

The map does carry the real thing. ``roads_detailed.json`` holds 59 road
centreline splines with widths, 4441 m of carriageway, and only 12.5% of its
vertices land inside a footprint -- and those few are where a spline legitimately
passes under an overhanging upper storey. ``buildings.json`` carries 573
buildings with a centre, a bounding box, and an ``entrance_yaw_deg``: the
direction each front door faces.

So this module derives four things a delivery simulation actually needs, in the
conversion layer where they belong rather than being invented per-node at
runtime:

streets    one named street per centreline spline, resampled to walkable spacing
graph      nodes on the carriageway, edges along it, junctions where splines meet
addresses  every building projected onto its street, numbered by arc length,
           odd on one side and even on the other, the way real addresses work
doors      a delivery point per building at its entrance, snapped to the nearest
           street node, so "deliver to 42 Rue de Rivoli" means a real doorway a
           robot could stand at

Naming is deterministic: a spline's identity comes from its ``source`` actor
name, so the same map always yields the same street names, and a diff between
two compiles is a real change rather than reshuffled vocabulary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Spacing between generated street nodes. Roughly a Paris block face, and close
# to the 18.5 m median edge of the engine's own graph, so step budgets and
# point-navigation reach stay comparable to what the pipeline already measured.
DEFAULT_NODE_SPACING_CM = 1800.0
# Two spline endpoints closer than this are the same junction. Slightly over
# half the widest carriageway (10 m), so roads that visually meet are joined
# without merging genuinely separate parallel streets.
JUNCTION_MERGE_CM = 900.0
# A building further than this from any street is not addressable from it.
MAX_ADDRESS_OFFSET_CM = 6000.0
# Two street nodes closer than this are the same place. Below it an "edge" is a
# step that changes nothing but still spends a turn and a step budget.
MIN_EDGE_CM = 200.0
# A junction with at least this many arms carries a pedestrian signal. Rule-based
# and derived from the graph, so it holds on any map: a crossroads is signalised,
# a bend in a street is not. CityCore Paris ships the meshes
# (SM_PR_PedestrianTrafficLights_01/02) and an LED material whose colour can be
# driven per render, which is what lets the same view be baked red and green.
SIGNALISED_DEGREE = 3

# Paris street vocabulary. Deterministic assignment by spline index, so names are
# stable across compiles; the type prefix cycles so a map reads like a city
# rather than a list of avenues.
_STREET_TYPES = ("Rue", "Rue", "Rue", "Avenue", "Boulevard", "Rue", "Quai", "Rue")
_STREET_NAMES = (
    "de Rivoli", "Saint-Honoré", "de la Paix", "des Écoles", "du Temple",
    "de Sévigné", "Montorgueil", "Saint-Antoine", "de Vaugirard", "Oberkampf",
    "de Charonne", "des Rosiers", "Lafayette", "de Turenne", "Beaubourg",
    "du Bac", "de Grenelle", "Mouffetard", "de Belleville", "Daguerre",
    "Cler", "Crémieux", "de Bretagne", "Jacob", "Bonaparte", "de Seine",
    "Mazarine", "Dauphine", "de Buci", "Monge", "Claude Bernard", "Gay-Lussac",
    "Soufflot", "Cujas", "Saint-Jacques", "Galande", "de la Huchette",
    "des Martyrs", "Lepic", "Caulaincourt", "Ordener", "Marcadet",
    "de Flandre", "de Crimée", "Botzaris", "des Pyrénées", "Ménilmontant",
    "de Bagnolet", "Saint-Blaise", "de la Roquette", "Keller", "Léon Frot",
    "de Reuilly", "de Bercy", "Tolbiac", "Nationale", "Baudricourt",
    "de Choisy", "Bobillot",
)


def street_name(index: int, map_name: str = "citycore-paris") -> str:
    """A stable name for the n-th street of a map.

    Per-map, not merely per-index: with one vocabulary consumed from the top,
    every procgen city's first street was "Rue de Rivoli" -- the same names,
    in the same order, as Paris. Within one episode that is harmless (an
    episode is one city); across a multi-city benchmark it makes two
    different streets in two different cities indistinguishable in any log,
    transcript or model answer.

    Each map starts at its own deterministic offset in the same vocabulary.
    citycore-paris is PINNED at offset 0: its names are baked into every
    album manifest and every reported transcript, and renaming them would
    orphan the assets. (Guarded by a test.)
    """
    shifted = index
    # "" is pinned with citycore-paris: build_road_network's map_name defaults
    # to empty, and a caller that never passed it has always gotten the Paris
    # vocabulary from the top -- changing that would rename streets under the
    # only callers old enough to predate the parameter.
    if map_name not in ("", "citycore-paris"):
        import hashlib
        digest = hashlib.blake2b(map_name.encode(), digest_size=4).digest()
        shifted = index + int.from_bytes(digest, "little") % len(_STREET_NAMES)
    base = _STREET_NAMES[shifted % len(_STREET_NAMES)]
    kind = _STREET_TYPES[shifted % len(_STREET_TYPES)]
    # The disambiguating suffix counts the MAP's own streets, not the shifted
    # position in the vocabulary -- with an offset of 55, street 10 of a
    # 15-street city is still its 11th street, not "(2)".
    suffix = "" if index < len(_STREET_NAMES) else f" ({index // len(_STREET_NAMES) + 1})"
    return f"{kind} {base}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def resample(points: list[tuple[float, float]], spacing_cm: float) -> list[tuple[float, float]]:
    """Even points along a polyline, keeping both endpoints.

    The source splines are extremely sparse -- 59 chains carry 120 vertices
    between them, so a single chain is often just its two ends. Walking a 300 m
    straight with no intermediate node gives the agent nothing to stand on and
    nothing to look at, so the polyline is resampled rather than used as given.
    """
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    carry = 0.0
    for start, end in zip(points, points[1:]):
        seg = _distance(start, end)
        if seg <= 1e-6:
            continue
        travelled = spacing_cm - carry
        while travelled < seg:
            t = travelled / seg
            out.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
            travelled += spacing_cm
        carry = (carry + seg) % spacing_cm
    if _distance(out[-1], points[-1]) > spacing_cm * 0.25:
        out.append(points[-1])
    else:
        out[-1] = points[-1]
    return out


def project_to_polyline(
    point: tuple[float, float], polyline: list[tuple[float, float]]
) -> tuple[float, float, float]:
    """Closest point on a polyline: ``(offset_cm, arc_length_cm, side)``.

    ``side`` is +1 left of the direction of travel and -1 right, which is what
    lets addresses be odd on one side and even on the other.
    """
    best = (float("inf"), 0.0, 1.0)
    arc = 0.0
    for start, end in zip(polyline, polyline[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        seg_sq = dx * dx + dy * dy
        if seg_sq <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / seg_sq))
        px, py = start[0] + dx * t, start[1] + dy * t
        offset = math.hypot(point[0] - px, point[1] - py)
        if offset < best[0]:
            cross = dx * (point[1] - start[1]) - dy * (point[0] - start[0])
            best = (offset, arc + math.sqrt(seg_sq) * t, 1.0 if cross >= 0 else -1.0)
        arc += math.sqrt(seg_sq)
    return best


def extract_strokes(
    segments: list["RoadSegment"], *, weld_cm: float, max_turn_deg: float
) -> list[list[tuple[float, float]]]:
    """Group segments into streets by following each one straight on.

    This is stroke extraction, the standard rule for recovering named roads from
    unnamed segment geometry: stand at the end of a segment, and of the segments
    continuing from that point, take the one that deviates least from the
    direction you were already going. Stop when the best continuation turns by
    more than ``max_turn_deg`` -- that is a corner, and the road beyond it is a
    different street.

    It is used on every map, including the one that ships spline ids. Grouping
    Paris by its ``source`` field would have produced streets no other map could
    have, and a rule that only works where the richest export exists is not a
    rule.
    """
    endpoints: dict[tuple[int, int], list[int]] = {}

    def key(point: tuple[float, float]) -> tuple[int, int]:
        return (int(round(point[0] / weld_cm)), int(round(point[1] / weld_cm)))

    for index, segment in enumerate(segments):
        endpoints.setdefault(key(segment.start), []).append(index)
        endpoints.setdefault(key(segment.end), []).append(index)

    used: set[int] = set()
    strokes: list[list[tuple[float, float]]] = []

    def continuations(at: tuple[float, float]) -> list[int]:
        out: list[int] = []
        k = key(at)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.extend(endpoints.get((k[0] + dx, k[1] + dy), ()))
        return out

    for seed in range(len(segments)):
        if seed in used:
            continue
        used.add(seed)
        segment = segments[seed]
        chain = [segment.start, segment.end]
        # Grow forward from the end, then backward from the start.
        for forward in (True, False):
            while True:
                tip = chain[-1] if forward else chain[0]
                previous = chain[-2] if forward else chain[1]
                heading = bearing_deg(previous, tip)
                best, best_turn, best_point = None, max_turn_deg, None
                for index in continuations(tip):
                    if index in used:
                        continue
                    candidate = segments[index]
                    for near, far in ((candidate.start, candidate.end),
                                      (candidate.end, candidate.start)):
                        if _distance(near, tip) > weld_cm:
                            continue
                        turn = abs((bearing_deg(tip, far) - heading + 180.0) % 360.0 - 180.0)
                        if turn < best_turn:
                            best, best_turn, best_point = index, turn, far
                if best is None:
                    break
                used.add(best)
                if forward:
                    chain.append(best_point)
                else:
                    chain.insert(0, best_point)
        strokes.append(chain)
    return strokes


def _chain_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]], tolerance_cm: float
) -> list[list[tuple[float, float]]]:
    """Join loose segments of one street into ordered polylines.

    A spline is exported as unordered segments, so walking them in file order
    produces a zig-zag rather than a street. Endpoints within ``tolerance_cm``
    are treated as the same point and the segments are threaded end to end; a
    street that genuinely branches yields more than one polyline rather than a
    single wrong one.
    """
    remaining = list(segments)
    chains: list[list[tuple[float, float]]] = []
    while remaining:
        start, end = remaining.pop()
        chain = [start, end]
        extended = True
        while extended:
            extended = False
            for index, (a, b) in enumerate(remaining):
                if _distance(chain[-1], a) <= tolerance_cm:
                    chain.append(b)
                elif _distance(chain[-1], b) <= tolerance_cm:
                    chain.append(a)
                elif _distance(chain[0], b) <= tolerance_cm:
                    chain.insert(0, a)
                elif _distance(chain[0], a) <= tolerance_cm:
                    chain.insert(0, b)
                else:
                    continue
                remaining.pop(index)
                extended = True
                break
        chains.append(chain)
    return chains


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


# ─────────────────────────────────────────────────────────────────────────────
# Loading, across the schemas maps actually ship
# ─────────────────────────────────────────────────────────────────────────────

# Two segments whose directions differ by less than this are the same street
# continuing. Above it, the road has turned and a new street begins. 35 degrees
# is the standard "stroke" threshold from road-network generalisation: it keeps
# a curving boulevard together while splitting at a genuine corner.
STREET_CONTINUATION_DEG = 35.0
# Below this span, a file's coordinates cannot be centimetres: a city that fits
# in 200 m is not a city. Used only when a map declares no units.
CM_SPAN_THRESHOLD = 20000.0


def _scale_to_cm(span: float, declared_units: str | None) -> float:
    """Multiplier taking a file's coordinates to centimetres.

    Only ``roads_detailed.json`` and ``buildings.json`` declare units, and only
    on some maps. Where nothing is declared the scale is inferred from extent,
    which is safe because the two candidates differ by 100x: the nine procgen
    maps span 600-1200 in file units and Paris spans 716, all of which are
    metres, while the same cities in centimetres would span 60,000-120,000.
    """
    if declared_units in ("centimeters", "centimetres"):
        return 1.0
    if declared_units in ("meters", "metres"):
        return 100.0
    return 1.0 if span > CM_SPAN_THRESHOLD else 100.0


@dataclass
class RoadSegment:
    """One straight piece of carriageway, in centimetres."""

    start: tuple[float, float]
    end: tuple[float, float]
    width_cm: float = 600.0
    source: str | None = None
    is_highway: bool = False


def load_road_segments(map_dir: Path) -> tuple[list[RoadSegment], list[str]]:
    """Read the road geometry every map exports, enriched where more exists.

    ``roads.json`` is the only road file all ten maps carry, so it is the
    required input. ``roads_detailed.json`` adds per-segment width and the
    source spline id, but ships with Paris alone -- treating it as required
    would make this compiler Paris-only, which is the opposite of the point.
    """
    map_dir = Path(map_dir)
    notes: list[str] = []
    roads_path = map_dir / "roads.json"
    if not roads_path.exists():
        return [], [f"{roads_path.name} is missing: this map exports no road geometry"]

    raw = json.loads(roads_path.read_text())
    records = raw.get("roads") or []
    coords: list[float] = []
    malformed = 0
    for record in records:
        for end in ("start", "end"):
            point = record.get(end) or {}
            try:
                coords.extend([float(point.get("x", 0.0)), float(point.get("y", 0.0))])
            except (TypeError, ValueError):
                # A non-numeric coordinate is a broken row, not a broken map.
                # Raising here turned one bad record into "this map cannot be
                # compiled", when the contract is a verdict for every input.
                malformed += 1
    span = (max(coords) - min(coords)) if coords else 0.0
    scale = _scale_to_cm(span, raw.get("units"))
    notes.append(
        f"roads.json: {len(records)} segments, span {span:.0f} file units, "
        f"read as {'centimetres' if scale == 1.0 else 'metres'}"
    )
    if malformed:
        notes.append(f"skipped {malformed} segment endpoints with non-numeric coordinates")

    segments: list[RoadSegment] = []
    for record in records:
        try:
            start = (float(record["start"]["x"]) * scale, float(record["start"]["y"]) * scale)
            end = (float(record["end"]["x"]) * scale, float(record["end"]["y"]) * scale)
        except (KeyError, TypeError, ValueError):
            continue
        if _distance(start, end) < MIN_EDGE_CM:
            continue
        segments.append(RoadSegment(
            start=start, end=end, is_highway=bool(record.get("is_highway", False))
        ))

    # Exporters repeat segments -- the same carriageway written once per lane or
    # per tile. Left in, each copy seeds its own stroke, so twenty duplicates of
    # one road became twenty streets with twenty names.
    unique: dict[tuple[int, int, int, int], RoadSegment] = {}
    # A non-finite coordinate is a broken row, not a broken map: int(round())
    # would raise on it a long way from the record that carried it.
    finite = [s for s in segments
              if all(math.isfinite(v) for v in (*s.start, *s.end))]
    if len(finite) != len(segments):
        notes.append(f"dropped {len(segments) - len(finite)} segments with non-finite coordinates")
    segments = finite
    for segment in segments:
        a = (int(round(segment.start[0] / MIN_EDGE_CM)), int(round(segment.start[1] / MIN_EDGE_CM)))
        b = (int(round(segment.end[0] / MIN_EDGE_CM)), int(round(segment.end[1] / MIN_EDGE_CM)))
        unique.setdefault((*min(a, b), *max(a, b)), segment)
    if len(unique) != len(segments):
        notes.append(f"dropped {len(segments) - len(unique)} duplicate segments")
    segments = list(unique.values())

    detailed_path = map_dir / "roads_detailed.json"
    if detailed_path.exists():
        detailed = json.loads(detailed_path.read_text())
        dscale = _scale_to_cm(0.0, detailed.get("units"))
        lookup: dict[tuple[int, int, int, int], tuple[float, str | None]] = {}
        for record in detailed.get("segments") or []:
            try:
                a = (float(record["start_cm"]["x"]) * dscale, float(record["start_cm"]["y"]) * dscale)
                b = (float(record["end_cm"]["x"]) * dscale, float(record["end_cm"]["y"]) * dscale)
            except (KeyError, TypeError, ValueError):
                continue
            key = tuple(sorted((int(a[0] // 100), int(a[1] // 100), int(b[0] // 100), int(b[1] // 100))))
            lookup[key] = (float(record.get("width_cm") or 600.0), record.get("source"))
        matched = 0
        for segment in segments:
            key = tuple(sorted((
                int(segment.start[0] // 100), int(segment.start[1] // 100),
                int(segment.end[0] // 100), int(segment.end[1] // 100),
            )))
            if key in lookup:
                segment.width_cm, segment.source = lookup[key]
                matched += 1
        notes.append(f"roads_detailed.json matched width/source for {matched}/{len(segments)} segments")
    else:
        notes.append("no roads_detailed.json: using default carriageway width")
    return segments, notes


# ─────────────────────────────────────────────────────────────────────────────
# Building footprints
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Building:
    """One building, with the door a courier has to reach."""

    id: str
    centre: tuple[float, float]
    half_extent: tuple[float, float]
    entrance_yaw_deg: float | None
    poi_type: str
    navigable: bool
    nearest_road_cm: float | None

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (
            self.centre[0] - self.half_extent[0], self.centre[1] - self.half_extent[1],
            self.centre[0] + self.half_extent[0], self.centre[1] + self.half_extent[1],
        )

    def entrance_point(self) -> tuple[float, float]:
        """Where the front door is, on the facade the entrance yaw points along.

        Falls back to the centre when the export carries no entrance, which is
        honest: the caller can see that the door is unknown rather than being
        handed an invented one.
        """
        if self.entrance_yaw_deg is None:
            return self.centre
        radians = math.radians(self.entrance_yaw_deg)
        dx, dy = math.cos(radians), math.sin(radians)
        # Step from the centre to the facade along the entrance direction.
        scale = min(
            self.half_extent[0] / abs(dx) if abs(dx) > 1e-6 else float("inf"),
            self.half_extent[1] / abs(dy) if abs(dy) > 1e-6 else float("inf"),
        )
        if not math.isfinite(scale):
            return self.centre
        return (self.centre[0] + dx * scale, self.centre[1] + dy * scale)


def load_buildings(map_dir: Path) -> list[Building]:
    """Read building footprints in either schema the exporters produce.

    CityCore writes ``center_cm`` + ``bbox_cm`` in centimetres and adds an
    ``entrance_yaw_deg``; the procgen exporter writes ``bounds`` with a corner,
    width and height in metres. An earlier loader knew only the second, so on
    Paris it silently returned an empty list -- and every "clear of buildings"
    placement rule became a no-op on the one map that mattered. A schema this
    recognises neither of is an error, not an empty result.
    """
    path = Path(map_dir) / "buildings.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    records = payload.get("buildings") or []
    if not records:
        return []

    declared = payload.get("units")
    first = records[0]
    out: list[Building] = []

    if "center_cm" in first and "bbox_cm" in first:
        scale = _scale_to_cm(0.0, declared or "centimeters")
        for index, record in enumerate(records):
            centre, extent = record.get("center_cm"), record.get("bbox_cm")
            if not centre or not extent:
                continue
            yaw = record.get("entrance_yaw_deg")
            out.append(Building(
                id=str(record.get("id") or f"building_{index}"),
                centre=(float(centre["x"]) * scale, float(centre["y"]) * scale),
                half_extent=(abs(float(extent["x"])) * scale / 2.0,
                             abs(float(extent["y"])) * scale / 2.0),
                entrance_yaw_deg=float(yaw) if yaw is not None else None,
                poi_type=str(record.get("poi_type") or "building"),
                navigable=bool(record.get("deliverybench_navigable", True)),
                nearest_road_cm=(
                    float(record["nearest_road_distance_cm"])
                    if record.get("nearest_road_distance_cm") is not None else None
                ),
            ))
        return out

    if "bounds" in first:
        spans = []
        for record in records:
            bounds = record.get("bounds") or {}
            spans.extend([abs(float(bounds.get("x", 0.0))), abs(float(bounds.get("y", 0.0)))])
        scale = _scale_to_cm(max(spans) * 2 if spans else 0.0, declared)
        for index, record in enumerate(records):
            bounds = record.get("bounds") or {}
            try:
                x = float(bounds["x"]) * scale
                y = float(bounds["y"]) * scale
                width = float(bounds["width"]) * scale
                height = float(bounds["height"]) * scale
            except (KeyError, TypeError, ValueError):
                continue
            out.append(Building(
                id=str(record.get("id") or f"building_{index}"),
                centre=(x + width / 2.0, y + height / 2.0),
                half_extent=(abs(width) / 2.0, abs(height) / 2.0),
                # This exporter records no door direction. Leaving it None is
                # honest -- entrance_point() then falls back to the centre and
                # the caller can see the door is unknown rather than invented.
                entrance_yaw_deg=None,
                poi_type=str(record.get("poi_type") or record.get("type") or "building"),
                navigable=True,
                nearest_road_cm=None,
            ))
        return out

    raise ValueError(
        f"{path} uses an unrecognised footprint schema (keys: {sorted(first)[:6]}); "
        "refusing to guess, because guessing wrong returns an empty list and "
        "silently disables every rule that depends on footprints"
    )


class FootprintIndex:
    """Grid-bucketed point-in-footprint test."""

    CELL = 2000.0

    def __init__(self, buildings: Iterable[Building]):
        self.boxes = [b.box for b in buildings]
        self.grid: dict[tuple[int, int], list[int]] = {}
        for index, (min_x, min_y, max_x, max_y) in enumerate(self.boxes):
            for gx in range(int(min_x // self.CELL), int(max_x // self.CELL) + 1):
                for gy in range(int(min_y // self.CELL), int(max_y // self.CELL) + 1):
                    self.grid.setdefault((gx, gy), []).append(index)

    def contains(self, x: float, y: float) -> bool:
        for index in self.grid.get((int(x // self.CELL), int(y // self.CELL)), ()):
            min_x, min_y, max_x, max_y = self.boxes[index]
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# The street network
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Street:
    """One named street, as a polyline."""

    index: int
    name: str
    source: str
    width_cm: float
    polyline: list[tuple[float, float]]

    @property
    def length_cm(self) -> float:
        return sum(_distance(a, b) for a, b in zip(self.polyline, self.polyline[1:]))


@dataclass
class StreetNode:
    """A place on the carriageway an agent can stand."""

    id: str
    x_cm: float
    y_cm: float
    street_index: int
    arc_cm: float
    neighbours: set[str] = field(default_factory=set)
    inside_building: bool = False

    @property
    def position(self) -> tuple[float, float]:
        return (self.x_cm, self.y_cm)


@dataclass
class Address:
    """A postal address, derived rather than invented."""

    building_id: str
    street_index: int
    street_name: str
    number: int
    arc_cm: float
    offset_cm: float
    side: str
    door: tuple[float, float]
    nearest_node: str | None
    poi_type: str
    # Where a courier actually stands to deliver: the point on the carriageway
    # outside the door. Doors sit a median 13 m back from the road centreline
    # -- pavement, forecourt, building depth -- so scoring arrival at the door
    # itself is unreachable for an agent that walks on the road. A real courier
    # stops at the kerb and crosses the last few metres on foot; the kerb is the
    # position the simulation can honestly test.
    kerb_node: str | None = None
    kerb: tuple[float, float] = (0.0, 0.0)

    @property
    def text(self) -> str:
        return f"{self.number} {self.street_name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id, "address": self.text,
            "street": self.street_name, "number": self.number,
            "side": self.side, "arc_cm": round(self.arc_cm, 1),
            "offset_cm": round(self.offset_cm, 1),
            "door_x_cm": round(self.door[0], 1), "door_y_cm": round(self.door[1], 1),
            "nearest_node": self.nearest_node, "kerb_node": self.kerb_node,
            "kerb_x_cm": round(self.kerb[0], 1), "kerb_y_cm": round(self.kerb[1], 1),
            "poi_type": self.poi_type,
        }


@dataclass
class RoadNetwork:
    """The compiled street network: geometry, names and addresses together."""

    map_name: str
    streets: list[Street] = field(default_factory=list)
    nodes: dict[str, StreetNode] = field(default_factory=dict)
    addresses: list[Address] = field(default_factory=list)
    # Kept because a map is mostly the shape of the blocks between the streets.
    # They are loaded to place addresses anyway, so carrying them costs a
    # reference and saves every consumer re-reading and re-scaling the file.
    buildings: list["Building"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def signalised_nodes(self) -> set[str]:
        """Junctions where a courier must wait for a pedestrian signal.

        Derived from degree, so it needs no extra map data and generalises. The
        set is deterministic for a map, which matters because the light meshes
        are baked into the album at exactly these nodes.
        """
        return {
            node_id for node_id, node in self.nodes.items()
            if len(node.neighbours) >= SIGNALISED_DEGREE
        }

    def edges(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for node in self.nodes.values():
            for other in node.neighbours:
                out.add((node.id, other) if node.id <= other else (other, node.id))
        return out

    def components(self) -> list[int]:
        seen: set[str] = set()
        sizes: list[int] = []
        for start in sorted(self.nodes):
            if start in seen:
                continue
            stack, size = [start], 0
            seen.add(start)
            while stack:
                current = stack.pop()
                size += 1
                for neighbour in sorted(self.nodes[current].neighbours):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            sizes.append(size)
        return sorted(sizes, reverse=True)

    def summary(self) -> dict[str, Any]:
        sizes = self.components()
        inside = sum(1 for n in self.nodes.values() if n.inside_building)
        lengths = [
            _distance(self.nodes[a].position, self.nodes[b].position) / 100.0
            for a, b in self.edges()
        ]
        lengths.sort()
        q = lambda p: lengths[int(p * (len(lengths) - 1))] if lengths else 0.0
        return {
            "map": self.map_name,
            "streets": len(self.streets),
            "nodes": len(self.nodes),
            "edges": len(self.edges()),
            "nodes_inside_building": inside,
            "nodes_inside_building_pct": round(100.0 * inside / max(1, len(self.nodes)), 2),
            "components": len(sizes),
            "largest_component_fraction": round(sizes[0] / max(1, len(self.nodes)), 4) if sizes else 0.0,
            "edge_length_m": {"p10": round(q(.1), 1), "p50": round(q(.5), 1),
                              "p90": round(q(.9), 1), "max": round(lengths[-1], 1) if lengths else 0.0},
            "addresses": len(self.addresses),
            "signalised_junctions": len(self.signalised_nodes()),
            "notes": self.notes,
        }


def build_road_network(
    map_dir: Path,
    *,
    map_name: str = "",
    spacing_cm: float = DEFAULT_NODE_SPACING_CM,
    junction_merge_cm: float = JUNCTION_MERGE_CM,
) -> RoadNetwork:
    """Compile ``roads_detailed.json`` and ``buildings.json`` into a street network."""
    map_dir = Path(map_dir)
    network = RoadNetwork(map_name=map_name or map_dir.name)

    segments, load_notes = load_road_segments(map_dir)
    network.notes.extend(load_notes)
    if not segments:
        return network

    buildings = load_buildings(map_dir)
    network.buildings = buildings
    footprints = FootprintIndex(buildings)

    # ── streets ──────────────────────────────────────────────────────────────
    widths = {
        (int(round(seg.start[0])), int(round(seg.start[1]))): seg.width_cm
        for seg in segments
    }
    for stroke in extract_strokes(
        segments, weld_cm=junction_merge_cm, max_turn_deg=STREET_CONTINUATION_DEG
    ):
        if len(stroke) < 2:
            continue
        index = len(network.streets)
        network.streets.append(Street(
            index=index,
            name=street_name(index, map_name),
            source="",
            width_cm=widths.get(
                (int(round(stroke[0][0])), int(round(stroke[0][1]))), 600.0
            ),
            polyline=stroke,
        ))
    network.notes.append(
        f"{len(segments)} segments grouped into {len(network.streets)} streets "
        f"by stroke continuation (<= {STREET_CONTINUATION_DEG:g} deg turn)"
    )

    # ── nodes along each street ──────────────────────────────────────────────
    for street in network.streets:
        # Drop points that land on top of each other: a duplicated or welded
        # vertex yields a zero-length edge, which is a step the agent can take
        # that moves it nowhere and still costs a turn.
        sampled: list[tuple[float, float]] = []
        for point in resample(street.polyline, spacing_cm):
            if not sampled or _distance(point, sampled[-1]) > MIN_EDGE_CM:
                sampled.append(point)
        previous: str | None = None
        arc = 0.0
        for order, (x, y) in enumerate(sampled):
            if order:
                arc += _distance(sampled[order - 1], (x, y))
            node_id = f"s{street.index:03d}_n{order:03d}"
            network.nodes[node_id] = StreetNode(
                id=node_id, x_cm=x, y_cm=y, street_index=street.index, arc_cm=arc,
                inside_building=footprints.contains(x, y),
            )
            if previous is not None:
                network.nodes[previous].neighbours.add(node_id)
                network.nodes[node_id].neighbours.add(previous)
            previous = node_id

    # ── junctions ────────────────────────────────────────────────────────────
    # Streets meet in two distinct ways and they need different handling.
    #
    # Coincident: two nodes of different streets land on the same spot, which is
    # what a crossroads looks like once both splines have been sampled. These
    # are one place, so they are *merged*. Linking them with a sub-metre edge
    # instead gives the agent a step that moves it nowhere, and skipping them --
    # which an earlier version did, to avoid exactly that -- deleted every real
    # junction and shattered the map into 35 islands.
    #
    # Near: endpoints a few metres apart across a carriageway. These are joined
    # by a real edge.
    cell = junction_merge_cm
    buckets: dict[tuple[int, int], list[str]] = {}
    for node in network.nodes.values():
        buckets.setdefault((int(node.x_cm // cell), int(node.y_cm // cell)), []).append(node.id)

    parent: dict[str, str] = {nid: nid for nid in network.nodes}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    linked = 0
    for (gx, gy), ids in buckets.items():
        nearby: list[str] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nearby.extend(buckets.get((gx + dx, gy + dy), ()))
        for a in ids:
            for b in nearby:
                if a >= b:
                    continue
                na, nb = network.nodes[a], network.nodes[b]
                if na.street_index == nb.street_index:
                    continue
                gap = _distance(na.position, nb.position)
                if gap <= MIN_EDGE_CM:
                    union(a, b)
                elif gap <= junction_merge_cm:
                    na.neighbours.add(b)
                    nb.neighbours.add(a)
                    linked += 1

    merged = 0
    for node_id in sorted(network.nodes):
        root = find(node_id)
        if root == node_id:
            continue
        merged += 1
        survivor = network.nodes[root]
        absorbed = network.nodes.pop(node_id)
        for neighbour in absorbed.neighbours:
            if neighbour == root or neighbour not in network.nodes:
                continue
            survivor.neighbours.add(neighbour)
            network.nodes[neighbour].neighbours.add(root)
        for other in network.nodes.values():
            if node_id in other.neighbours:
                other.neighbours.discard(node_id)
                if other.id != root:
                    other.neighbours.add(root)
                    survivor.neighbours.add(other.id)
    for node in network.nodes.values():
        node.neighbours = {n for n in node.neighbours if n in network.nodes and n != node.id}

    network.notes.append(
        f"merged {merged} coincident junction nodes; linked {linked} pairs within "
        f"{junction_merge_cm:g} cm"
    )

    # ── addresses ────────────────────────────────────────────────────────────
    per_street: dict[int, list[tuple[float, float, Building, float, float]]] = {}
    for building in buildings:
        best_index, best = None, (float("inf"), 0.0, 1.0)
        for street in network.streets:
            offset, arc, side = project_to_polyline(building.centre, street.polyline)
            if offset < best[0]:
                best_index, best = street.index, (offset, arc, side)
        if best_index is None or best[0] > MAX_ADDRESS_OFFSET_CM:
            continue
        per_street.setdefault(best_index, []).append(
            (best[1], best[2], building, best[0], best[1])
        )

    for street_index, items in per_street.items():
        street = network.streets[street_index]
        # Numbers ascend with arc length, the way a real street is numbered.
        items.sort(key=lambda t: t[0])
        # One cursor for the whole street, not one per side. Two independent
        # counters each stepping by 2 keep each side monotone -- which they did,
        # 42 of 46 sides -- but they desynchronise, because the two sides carry
        # different numbers of buildings at different arc positions. The courier
        # is shown every door readable from a junction, both sides at once, so
        # what it read was the interleaving: on Rue Oberkampf
        # ``2 | 4,6 | 1,8 | 3,5,10 | 7 | 9,12`` and on Rue du Bac the entire even
        # side was a single No. 2 sitting past No. 13. 22 of 23 streets showed an
        # inversion, No. N and No. N+1 could be 12 junctions apart, and the
        # prompt's rule -- "if they are falling and you want a higher one, turn
        # around" -- walked a reviewer away from the door it was standing near.
        #
        # A shared cursor with the parity forced by the side is what a real
        # street does: numbers follow position along the street, so N and N+1 are
        # across the road from each other. A side with no building at that
        # position simply skips its number, which is also what a real street does.
        cursor = 1
        for arc, side, building, offset, _ in items:
            wants_odd = side == 1.0            # odd on the left, even on the right
            number = cursor if (cursor % 2 == 1) == wants_odd else cursor + 1
            cursor = number + 1
            door = building.entrance_point()
            nearest, nearest_d = None, float("inf")
            kerb_node, kerb_d = None, float("inf")
            for node in network.nodes.values():
                d = _distance(node.position, door)
                if d < nearest_d:
                    nearest, nearest_d = node.id, d
                # The kerb must be on the street the address is on. A corner
                # building's nearest node often belongs to the cross street,
                # which would send the courier to the wrong frontage.
                if node.street_index == street_index and d < kerb_d:
                    kerb_node, kerb_d = node.id, d
            if kerb_node is None:
                # No node on this address's own street. Falling back to the
                # nearest node anywhere produced a contradiction the agent could
                # see: standing at "1 Rue Saint-Jacques" while the sign read
                # "Quai Montorgueil". 1.4% of Paris addresses hit this. An
                # address a courier cannot stand outside on the named street is
                # not deliverable, so it is dropped rather than described wrongly.
                continue
            network.addresses.append(Address(
                building_id=building.id, street_index=street_index,
                street_name=street.name, number=number, arc_cm=arc, offset_cm=offset,
                side="odd" if side > 0 else "even", door=door,
                nearest_node=nearest, poi_type=building.poi_type,
                kerb_node=kerb_node,
                kerb=network.nodes[kerb_node].position if kerb_node else door,
            ))
    return network

