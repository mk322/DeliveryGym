"""Generate adversarial maps that attack the pipeline's weak points.

A pipeline that only ever sees well-formed procgen grids and one real city has
not been tested; it has been demonstrated. These cases are built to break it,
each aimed at a specific assumption:

===========================  ==================================================
case                         the assumption it attacks
===========================  ==================================================
minimal_grid                 baseline; a trivially valid map must stay valid
single_segment               a graph can be smaller than any threshold
no_roads                     a map may contain no navigable geometry at all
disconnected_islands         connectivity is not guaranteed
degenerate_segments          zero-length edges (design plan §3.3.1, Paris 186/187)
self_loop_segments           an edge may start and end at the same point
star_hub                     degree can be far outside road-network norms
rotated_grid_37deg           cardinal alignment is a property, not a given
very_long_segments           an edge may span the whole map (Paris has 640 m)
duplicate_positions          distinct nodes may share a position
huge_grid                    the pipeline must scale, not just terminate
unknown_asset_names          POI vocabulary is hardcoded and drops silently
unicode_and_long_names       identifiers are not ASCII and not short
no_pois                      a navigable map may still be untaskable
nonfinite_coordinates        coordinates may be NaN/Inf
===========================  ==================================================

Everything is written as plain map JSON in the two files the engine actually
reads, so these are real maps, not mocks: ``roads.json`` (segments in metres)
and ``progen_world_enriched.json`` (POI nodes plus ``bus_routes``).

The workspace is assembled outside ``vendor/`` with ``vlm_delivery`` symlinked,
so generating and compiling stress maps never mutates the vendored checkout.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DELIVERYBENCH = (
    REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"
)


# ─────────────────────────────────────────────────────────────────────────────
# Map primitives (metres, matching roads.json's own units)
# ─────────────────────────────────────────────────────────────────────────────


def segment(x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
    return {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}, "is_highway": False}


def grid(rows: int, cols: int, spacing: float = 60.0, rotation_deg: float = 0.0) -> list[dict]:
    """A rows x cols street grid, optionally rotated about the origin."""
    radians = math.radians(rotation_deg)
    cos, sin = math.cos(radians), math.sin(radians)

    def place(x: float, y: float) -> tuple[float, float]:
        return (x * cos - y * sin, x * sin + y * cos)

    roads: list[dict] = []
    for row in range(rows):
        y = row * spacing
        start, end = place(0.0, y), place((cols - 1) * spacing, y)
        roads.append(segment(*start, *end))
    for col in range(cols):
        x = col * spacing
        start, end = place(x, 0.0), place(x, (rows - 1) * spacing)
        roads.append(segment(*start, *end))
    return roads


def poi(
    node_id: str,
    x: float,
    y: float,
    poi_type: str = "building",
    *,
    instance_name: str = "BP_Building_01_C",
    yaw: float = 0.0,
    bbox: tuple[float, float] = (12.0, 12.0),
) -> dict[str, Any]:
    """A POI node in the world-JSON shape the engine consumes.

    Coordinates here are centimetres: ``import_pois`` reads ``location`` as-is,
    while ``import_roads`` converts metres to centimetres. That asymmetry is the
    engine's, not ours, and getting it wrong silently places every POI 100x too
    close to the origin -- worth stating rather than rediscovering.
    """
    return {
        "id": node_id,
        "instance_name": instance_name,
        "properties": {
            "location": {"x": x * 100.0, "y": y * 100.0, "z": 0.0},
            "orientation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "bbox": {"x": bbox[0] * 100.0, "y": bbox[1] * 100.0, "z": 1000.0},
            "poi_type": poi_type,
        },
    }


def standard_pois(spacing: float = 60.0, rotation_deg: float = 0.0) -> list[dict]:
    """Enough POIs for a delivery task: restaurants, stores, and customers."""
    radians = math.radians(rotation_deg)
    cos, sin = math.cos(radians), math.sin(radians)

    def place(x: float, y: float) -> tuple[float, float]:
        return (x * cos - y * sin, x * sin + y * cos)

    nodes: list[dict] = []
    layout = [
        ("restaurant", 3, (0.35, 0.15)),
        ("store", 2, (0.65, 0.35)),
        ("customer", 4, (0.35, 0.75)),
        ("building", 4, (0.75, 0.75)),
    ]
    index = 0
    for poi_type, count, (fx, fy) in layout:
        for step in range(count):
            x = (fx + 0.12 * step) * spacing * 2
            y = (fy + 0.10 * step) * spacing * 2
            px, py = place(x, y)
            nodes.append(poi(f"{poi_type}_{index}", px, py, poi_type))
            index += 1
    return nodes


def world_json(nodes: list[dict], *, name: str, bus_routes: list | None = None) -> dict[str, Any]:
    return {
        "base_map": {"name": name, "env_bin": "stress", "width": 1000, "height": 1000},
        "nodes": nodes,
        "bus_routes": bus_routes if bus_routes is not None else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StressCase:
    """One adversarial map plus what the pipeline is expected to conclude."""

    name: str
    attacks: str
    build: Callable[[], tuple[list[dict], list[dict]]]
    # Expectations. None means "no requirement, just do not crash".
    expect_compiles: bool = True
    expect_nav_mode: str | None = None
    expect_grade: str | None = None
    expect_flags: list[str] = field(default_factory=list)
    expect_solvable: bool | None = None
    note: str = ""


def _minimal_grid() -> tuple[list[dict], list[dict]]:
    return grid(3, 3), standard_pois()


def _single_segment() -> tuple[list[dict], list[dict]]:
    return [segment(0, 0, 120, 0)], standard_pois(spacing=40.0)


def _no_roads() -> tuple[list[dict], list[dict]]:
    return [], standard_pois()


def _disconnected_islands() -> tuple[list[dict], list[dict]]:
    far = grid(3, 3)
    offset = 5000.0
    shifted = [
        segment(
            r["start"]["x"] + offset, r["start"]["y"] + offset,
            r["end"]["x"] + offset, r["end"]["y"] + offset,
        )
        for r in grid(3, 3)
    ]
    return far + shifted, standard_pois()


def _degenerate_segments() -> tuple[list[dict], list[dict]]:
    roads = grid(3, 3)
    # Sub-millimetre segments, mirroring Paris indices 186/187.
    roads.append(segment(60.0, 60.0, 60.0001, 60.0001))
    roads.append(segment(120.0, 60.0, 120.0002, 60.0))
    return roads, standard_pois()


def _self_loop_segments() -> tuple[list[dict], list[dict]]:
    roads = grid(3, 3)
    roads.append(segment(60.0, 60.0, 60.0, 60.0))
    return roads, standard_pois()


def _star_hub() -> tuple[list[dict], list[dict]]:
    roads = []
    for index in range(24):
        angle = math.radians(index * 15.0)
        roads.append(segment(0.0, 0.0, 200.0 * math.cos(angle), 200.0 * math.sin(angle)))
    return roads, standard_pois(spacing=100.0)


def _rotated_grid() -> tuple[list[dict], list[dict]]:
    return grid(4, 4, rotation_deg=37.0), standard_pois(rotation_deg=37.0)


def _very_long_segments() -> tuple[list[dict], list[dict]]:
    roads = grid(3, 3)
    roads.append(segment(0.0, 0.0, 2000.0, 0.0))
    roads.append(segment(0.0, 120.0, 0.0, 2000.0))
    return roads, standard_pois()


def _duplicate_positions() -> tuple[list[dict], list[dict]]:
    roads = grid(3, 3)
    roads.extend(grid(3, 3))  # every segment declared twice
    return roads, standard_pois()


def _huge_grid() -> tuple[list[dict], list[dict]]:
    return grid(30, 30, spacing=40.0), standard_pois(spacing=200.0)


def _unknown_asset_names() -> tuple[list[dict], list[dict]]:
    """POIs whose vocabulary the engine does not recognise.

    ``import_pois`` keeps a POI only if ``poi_type`` is in its hardcoded
    building-like/point-like sets or ``instance_name`` starts with
    ``BP_Building``. Anything else hits ``continue`` and vanishes without a
    warning -- the same silent drop design plan §3.3.11 records for
    ``pedestrian_light``. A map from a different asset pack looks fine and has
    no POIs.
    """
    nodes = [
        poi("shop_0", 40, 20, "boutique", instance_name="SM_Shop_A"),
        poi("shop_1", 60, 20, "épicerie", instance_name="SM_Shop_B"),
        poi("eat_0", 40, 80, "brasserie", instance_name="SM_Resto_A"),
        poi("home_0", 80, 80, "residence", instance_name="SM_House_A"),
        poi("light_0", 20, 20, "pedestrian_light", instance_name="RT_BP_street_light_ped"),
    ]
    return grid(3, 3), nodes


def _unicode_and_long_names() -> tuple[list[dict], list[dict]]:
    long_name = "Bâtiment_" + "très_long_" * 20
    nodes = [
        poi("rest_é", 40, 20, "restaurant", instance_name="BP_Building_Café_C"),
        poi(long_name, 60, 20, "store", instance_name="BP_Building_" + "X" * 200 + "_C"),
        poi("客户_1", 40, 80, "customer", instance_name="BP_Building_中文_C"),
        poi("cust with spaces", 80, 80, "customer", instance_name="BP_Building 01 C"),
        poi("", 20, 60, "restaurant", instance_name="BP_Building_02_C"),
    ]
    return grid(3, 3), nodes


def _no_pois() -> tuple[list[dict], list[dict]]:
    return grid(4, 4), []


def _nonfinite_coordinates() -> tuple[list[dict], list[dict]]:
    """Non-finite coordinates, written as JSON literals.

    ``json.dump`` emits bare ``NaN``/``Infinity`` tokens, which Python's own
    loader accepts. This is what a broken export actually looks like on disk.
    """
    roads = grid(3, 3)
    roads.append(segment(float("nan"), 0.0, 60.0, 0.0))
    roads.append(segment(0.0, 0.0, float("inf"), 60.0))
    return roads, standard_pois()


STRESS_CASES: list[StressCase] = [
    StressCase("minimal_grid", "baseline validity", _minimal_grid,
               expect_nav_mode="cardinal+graph", expect_solvable=True),
    StressCase("single_segment", "graphs smaller than any threshold", _single_segment),
    StressCase("no_roads", "a map with no navigable geometry", _no_roads,
               expect_solvable=False,
               note="must be rejected with a reason, not crash"),
    StressCase("disconnected_islands", "connectivity assumptions", _disconnected_islands,
               expect_flags=["fragmented_graph"]),
    StressCase("degenerate_segments", "zero-length edges (design plan §3.3.1)", _degenerate_segments,
               expect_flags=["degenerate_edges"]),
    StressCase("self_loop_segments", "start == end", _self_loop_segments),
    StressCase("star_hub", "degree far outside road norms", _star_hub,
               expect_flags=["over_connected_nodes"]),
    StressCase("rotated_grid_37deg", "cardinality is a property, not a given", _rotated_grid,
               expect_nav_mode="graph",
               note="the Paris lesson in miniature: a rotated grid must not get cardinal MOVE"),
    StressCase("very_long_segments", "map-spanning edges", _very_long_segments,
               expect_flags=["long_unvalidated_edges"]),
    StressCase("duplicate_positions", "coincident/duplicated geometry", _duplicate_positions),
    StressCase("huge_grid", "scale, not just termination", _huge_grid),
    StressCase("unknown_asset_names", "hardcoded POI vocabulary, silent drop",
               _unknown_asset_names, expect_solvable=False,
               note="a different asset pack yields zero POIs and no task"),
    StressCase("unicode_and_long_names", "non-ASCII and unbounded identifiers",
               _unicode_and_long_names),
    StressCase("no_pois", "navigable but untaskable", _no_pois, expect_solvable=False),
    StressCase("nonfinite_coordinates", "NaN/Inf in the export", _nonfinite_coordinates),
]


# ─────────────────────────────────────────────────────────────────────────────
# Workspace
# ─────────────────────────────────────────────────────────────────────────────


def write_map(maps_dir: Path, name: str, roads: list[dict], nodes: list[dict]) -> Path:
    """Write one map's two files."""
    target = maps_dir / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "roads.json").write_text(json.dumps({"roads": roads}, indent=1))
    (target / "progen_world_enriched.json").write_text(
        json.dumps(world_json(nodes, name=name), indent=1)
    )
    return target


def build_stress_workspace(root: Path, cases: list[StressCase] | None = None) -> Path:
    """Assemble a base_dir of stress maps without touching ``vendor/``.

    The engine resolves ``<base_dir>/maps/<name>/`` for map data and
    ``<base_dir>/vlm_delivery/input/`` for menus and configs, so the workspace
    carries its own maps and symlinks the vendored package for everything else.
    """
    root = Path(root).resolve()
    maps_dir = root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    link = root / "vlm_delivery"
    if not link.exists():
        link.symlink_to(VENDOR_DELIVERYBENCH / "vlm_delivery")

    for case in cases if cases is not None else STRESS_CASES:
        roads, nodes = case.build()
        write_map(maps_dir, case.name, roads, nodes)
    return root
