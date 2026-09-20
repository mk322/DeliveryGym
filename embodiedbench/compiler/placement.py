"""Rule-based placement of assets into a map. Deterministic, no AI.

Placing an asset badly is worse than not placing it. A charging station inside a
building is unreachable, three bus stops on one corner are pointless, and an
asset floating off the road network cannot be routed to — each produces a map
that loads, compiles, and quietly cannot support the task it was extended for.

So placement is a sequence of stated rules, each one a filter a human can check:

reachable   candidates come from the navigation graph itself, so anything placed
            is somewhere the agent can actually stand
clear       rejected if it falls inside a building footprint
spaced      rejected if within ``min_spacing_m`` of another asset of its type,
            counting assets the map already had
spread      chosen by farthest-point sampling over the surviving candidates, so
            a set of stations covers the map instead of clustering in one block
oriented    point assets face the road they sit on, derived from the direction of
            the graph edge at that node

Determinism comes from sorting every candidate list and seeding the one random
choice, so the same (map, requirement, seed) always yields the same placements —
design plan §5.2 requires every placement to record its rule and seed.

Failure is a result, not an exception. If a requirement cannot be satisfied the
engine returns what it could place and says precisely why the rest could not,
because a map that cannot host a task is something the caller must be told.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Defaults per POI type, in metres. These encode what the type is for: chargers
# and bus stops are infrastructure that should be distributed, while a hospital
# or car rental is a singleton whose spacing barely matters.
DEFAULT_SPACING_M: dict[str, float] = {
    "charging_station": 80.0,
    "bus_station": 120.0,
    "rest_area": 100.0,
    "hospital": 150.0,
    "car_rental": 150.0,
    "restaurant": 40.0,
    "store": 40.0,
}
FALLBACK_SPACING_M = 60.0
# A placement this close to a building footprint edge counts as inside it.
BUILDING_CLEARANCE_M = 2.0


@dataclass
class Candidate:
    """A position an asset could occupy."""

    x_cm: float
    y_cm: float
    node_id: str
    bearing_deg: float = 0.0


@dataclass
class Placement:
    """One placed asset, with the rule and seed that produced it."""

    poi_type: str
    node_id: str
    x_cm: float
    y_cm: float
    yaw_deg: float
    rule: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "poi_type": self.poi_type,
            "node_id": self.node_id,
            "x_cm": round(self.x_cm, 2),
            "y_cm": round(self.y_cm, 2),
            "yaw_deg": round(self.yaw_deg, 1),
            "rule": self.rule,
            "seed": self.seed,
        }


@dataclass
class PlacementResult:
    """What was placed, what was not, and why."""

    placements: list[Placement] = field(default_factory=list)
    shortfall: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    candidates_considered: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return not self.shortfall

    def to_dict(self) -> dict[str, Any]:
        return {
            "placements": [p.to_dict() for p in self.placements],
            "placed_count": len(self.placements),
            "shortfall": self.shortfall,
            "reasons": self.reasons,
            "candidates_considered": self.candidates_considered,
            "rejected": self.rejected,
            "satisfied": self.satisfied,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Map facts
# ─────────────────────────────────────────────────────────────────────────────


def load_building_footprints(map_dir: Path) -> list[tuple[float, float, float, float]]:
    """Axis-aligned building boxes in centimetres, as (min_x, min_y, max_x, max_y).

    ``buildings.json`` stores metres and a rotation. The rotation is ignored and
    the axis-aligned bound is used instead, which over-rejects slightly. That is
    the safe direction: a placement rejected for being near a building costs a
    candidate, one accepted inside a building costs the episode.
    """
    path = Path(map_dir) / "buildings.json"
    if not path.exists():
        return []
    try:
        buildings = json.loads(path.read_text()).get("buildings", [])
    except (OSError, ValueError):
        return []

    boxes: list[tuple[float, float, float, float]] = []
    for building in buildings:
        bounds = building.get("bounds") or {}
        try:
            x = float(bounds["x"]) * 100.0
            y = float(bounds["y"]) * 100.0
            width = float(bounds["width"]) * 100.0
            height = float(bounds["height"]) * 100.0
        except (KeyError, TypeError, ValueError):
            centre = building.get("center") or {}
            try:
                cx = float(centre["x"]) * 100.0
                cy = float(centre["y"]) * 100.0
            except (KeyError, TypeError, ValueError):
                continue
            half = 10.0 * 100.0
            boxes.append((cx - half, cy - half, cx + half, cy + half))
            continue
        pad = BUILDING_CLEARANCE_M * 100.0
        boxes.append((x - pad, y - pad, x + width + pad, y + height + pad))
    return boxes


def graph_candidates(city_map: Any) -> list[Candidate]:
    """Every graph node, with the bearing of an edge leaving it.

    Using the graph guarantees reachability, and taking the bearing from a real
    edge means an asset can be oriented to face the road rather than an
    arbitrary direction.
    """
    adjacency = getattr(city_map.waypoint_graph, "adjacency_list", {}) or {}
    out: list[Candidate] = []
    for node in adjacency:
        neighbours = adjacency.get(node) or []
        bearing = 0.0
        if neighbours:
            first = neighbours[0]
            bearing = math.degrees(
                math.atan2(
                    float(first.position.y) - float(node.position.y),
                    float(first.position.x) - float(node.position.x),
                )
            ) % 360.0
        out.append(
            Candidate(
                x_cm=float(node.position.x),
                y_cm=float(node.position.y),
                node_id=str(getattr(node, "waypoint_id", "") or ""),
                bearing_deg=bearing,
            )
        )
    # Sorted so the candidate order never depends on dict iteration.
    out.sort(key=lambda c: (round(c.x_cm, 2), round(c.y_cm, 2), c.node_id))
    return out


def existing_positions(world_nodes: list[dict[str, Any]], poi_type: str) -> list[tuple[float, float]]:
    """Where this POI type already sits, so new ones respect its spacing."""
    out: list[tuple[float, float]] = []
    for node in world_nodes:
        properties = node.get("properties", {}) or {}
        kind = str(properties.get("poi_type") or properties.get("type") or "").strip().lower()
        if kind != poi_type:
            continue
        location = properties.get("location") or {}
        try:
            out.append((float(location["x"]), float(location["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────────────────────


def inside_building(x_cm: float, y_cm: float, boxes: list[tuple[float, float, float, float]]) -> bool:
    for min_x, min_y, max_x, max_y in boxes:
        if min_x <= x_cm <= max_x and min_y <= y_cm <= max_y:
            return True
    return False


def far_enough(
    x_cm: float, y_cm: float, taken: list[tuple[float, float]], spacing_m: float
) -> bool:
    limit = spacing_m * 100.0
    return all(math.hypot(x_cm - tx, y_cm - ty) >= limit for tx, ty in taken)


def farthest_point_order(candidates: list[Candidate], seed: int) -> list[Candidate]:
    """Order candidates so each is as far as possible from those already chosen.

    Plain random sampling clusters; this spreads. The first pick is seeded so the
    whole ordering is deterministic but not always the same corner of the map.
    """
    if not candidates:
        return []
    rng = random.Random(seed)
    remaining = list(candidates)
    first = remaining.pop(rng.randrange(len(remaining)))
    ordered = [first]
    # Distance from each remaining candidate to the nearest chosen one.
    best = [math.hypot(c.x_cm - first.x_cm, c.y_cm - first.y_cm) for c in remaining]
    while remaining:
        index = max(range(len(remaining)), key=lambda i: (best[i], remaining[i].node_id))
        chosen = remaining.pop(index)
        best.pop(index)
        ordered.append(chosen)
        for i, candidate in enumerate(remaining):
            distance = math.hypot(candidate.x_cm - chosen.x_cm, candidate.y_cm - chosen.y_cm)
            if distance < best[i]:
                best[i] = distance
    return ordered


def place_assets(
    *,
    city_map: Any,
    world_nodes: list[dict[str, Any]],
    map_dir: Path,
    requirements: dict[str, int],
    seed: int = 0,
    spacing_overrides: dict[str, float] | None = None,
) -> PlacementResult:
    """Choose positions for the requested assets, by rule."""
    result = PlacementResult()
    candidates = graph_candidates(city_map)
    result.candidates_considered = len(candidates)
    if not candidates:
        result.shortfall = dict(requirements)
        result.reasons.append("map has no navigation graph nodes to place against")
        return result

    boxes = load_building_footprints(map_dir)
    ordered = farthest_point_order(candidates, seed)

    # Everything placed so far, regardless of type, so two different asset types
    # never land on the same node.
    occupied_nodes: set[str] = set()

    for poi_type in sorted(requirements):
        needed = requirements[poi_type]
        if needed <= 0:
            continue
        spacing = (spacing_overrides or {}).get(
            poi_type, DEFAULT_SPACING_M.get(poi_type, FALLBACK_SPACING_M)
        )
        taken = existing_positions(world_nodes, poi_type)
        placed = 0

        for attempt_spacing in (spacing, spacing / 2.0, 0.0):
            for candidate in ordered:
                if placed >= needed:
                    break
                if candidate.node_id in occupied_nodes:
                    continue
                if inside_building(candidate.x_cm, candidate.y_cm, boxes):
                    result.rejected["inside_building"] = (
                        result.rejected.get("inside_building", 0) + 1
                    )
                    continue
                if attempt_spacing and not far_enough(
                    candidate.x_cm, candidate.y_cm, taken, attempt_spacing
                ):
                    result.rejected["too_close"] = result.rejected.get("too_close", 0) + 1
                    continue

                rule = (
                    f"graph_node_farthest_point;spacing>={attempt_spacing:g}m;"
                    f"outside_building_footprint"
                )
                result.placements.append(
                    Placement(
                        poi_type=poi_type,
                        node_id=candidate.node_id,
                        x_cm=candidate.x_cm,
                        y_cm=candidate.y_cm,
                        # Face along the road, which is what a bus stop or
                        # charger beside a carriageway should do.
                        yaw_deg=candidate.bearing_deg,
                        rule=rule,
                        seed=seed,
                    )
                )
                taken.append((candidate.x_cm, candidate.y_cm))
                occupied_nodes.add(candidate.node_id)
                placed += 1
            if placed >= needed:
                break
            if attempt_spacing:
                # Relaxing spacing is a stated fallback, not a silent one.
                result.reasons.append(
                    f"{poi_type}: relaxed spacing below {attempt_spacing:g} m to place "
                    f"{needed - placed} more"
                )

        if placed < needed:
            result.shortfall[poi_type] = needed - placed
            result.reasons.append(
                f"{poi_type}: placed {placed}/{needed}; no further candidate satisfied "
                "reachability and building clearance"
            )
    return result
