"""Pedestrian-light metadata helpers for DeliveryBench maps.

Each generated record represents a pedestrian signal facing one sidewalk
approach at a crossing.  The metadata is renderer-agnostic but still carries
UE Blueprint fields so a 3D placement pass can bind it to a visible face later.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

PEDESTRIAN_LIGHT_TYPE = "pedestrian_light"
PEDESTRIAN_LIGHT_INSTANCE = "RT_BP_street_light_ped"
PEDESTRIAN_LIGHT_ASSET = "/Game/RealTimeBench/Traffic/RT_BP_street_light_ped"
GENERATED_LIGHT_ID_PREFIX = "GEN_PedestrianLight_"
LEGACY_LIGHT_ID_PREFIX = "GEN_TrafficLight_"

APPROACH_OFFSET_CM = 45.0
SIDEWALK_SIDE_OFFSET_CM = 55.0
SIDEWALK_OFFSET_CM = 110.0
PEDESTRIAN_LIGHT_CORNER_OFFSET_CM = SIDEWALK_OFFSET_CM + APPROACH_OFFSET_CM

DIR_VECTORS: Dict[str, Tuple[int, int]] = {
    "east": (1, 0),
    "north": (0, 1),
    "west": (-1, 0),
    "south": (0, -1),
}

_DIR_BY_VECTOR = {v: k for k, v in DIR_VECTORS.items()}

PHASE_A = "north_south_green_east_west_red"
PHASE_B = "north_south_red_east_west_green"


def _pt_m_to_cm(pt: Mapping[str, Any]) -> Tuple[float, float]:
    return float(pt["x"]) * 100.0, float(pt["y"]) * 100.0


def _unit_direction(a: Tuple[float, float], b: Tuple[float, float]) -> str:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "north" if dy > 0 else "south"


def _yaw_for_facing(facing: str) -> float:
    # Keep the active pedestrian face aimed at the requested approach.
    return {
        "north": 0.0,
        "east": 270.0,
        "south": 180.0,
        "west": 90.0,
    }[facing]


def _opposite_direction(direction: str) -> str:
    return {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }[direction]


def _phase_alias(phase: str) -> str:
    normalized = str(phase or "").strip().lower()
    if normalized in {
        PHASE_A,
        "left_green_right_red",
        "left_walk_right_stop",
        "a",
        "north_south_green",
        "vertical_green",
    }:
        return PHASE_A
    if normalized in {
        PHASE_B,
        "left_red_right_green",
        "left_stop_right_walk",
        "b",
        "east_west_green",
        "horizontal_green",
    }:
        return PHASE_B
    raise ValueError(
        "Unknown pedestrian-light phase "
        f"{phase!r}; expected {PHASE_A} or {PHASE_B}"
    )


def _state_for_direction(direction: str, phase: str) -> str:
    phase = _phase_alias(phase)
    direction = str(direction or "").strip().lower()
    is_vertical = direction in {"north", "south"}
    if phase == PHASE_A:
        return "green" if is_vertical else "red"
    return "red" if is_vertical else "green"


def _right_side_vector(direction: str) -> Tuple[int, int]:
    """Right-hand sidewalk side for a traveller approaching the crossing."""

    dx, dy = DIR_VECTORS[direction]
    return dy, -dx


def _normalize(dx: float, dy: float) -> Tuple[float, float]:
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-9:
        return 0.0, 0.0
    return dx / norm, dy / norm


def _direction_from_vector(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "north" if dy > 0 else "south"


def _crosswalk_side_for_corner_face(sx: int, sy: int, approach_dir: str) -> str:
    """Return the intersection side crossed by this corner face.

    At a T-intersection, the side with no vehicle-road approach is just a
    same-side sidewalk connection, not a road crossing.  For example, at a
    north/east/west T, the south edge of the sidewalk square should not get a
    pedestrian-light face.
    """

    if approach_dir in {"north", "south"}:
        return "east" if sx > 0 else "west"
    return "north" if sy > 0 else "south"


def _axis_aligned(a: Tuple[float, float], b: Tuple[float, float], *, eps_cm: float = 1.0) -> bool:
    return abs(a[0] - b[0]) <= eps_cm or abs(a[1] - b[1]) <= eps_cm


def _sidewalk_points_for_road(
    road: Mapping[str, Any],
    *,
    sidewalk_offset_cm: float = SIDEWALK_OFFSET_CM,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    start = (float(road["start"]["x"]) * 100.0, float(road["start"]["y"]) * 100.0)
    end = (float(road["end"]["x"]) * 100.0, float(road["end"]["y"]) * 100.0)
    ux, uy = _normalize(end[0] - start[0], end[1] - start[1])
    nx, ny = uy, -ux
    s = float(sidewalk_offset_cm)
    p1 = (start[0] - nx * s + ux * s, start[1] - ny * s + uy * s)
    p2 = (end[0] - nx * s - ux * s, end[1] - ny * s - uy * s)
    p3 = (end[0] + nx * s - ux * s, end[1] + ny * s - uy * s)
    p4 = (start[0] + nx * s + ux * s, start[1] + ny * s + uy * s)
    return p1, p2, p3, p4


def sidewalk_crossings_from_roads(
    roads: Iterable[Mapping[str, Any]],
    *,
    sidewalk_offset_cm: float = SIDEWALK_OFFSET_CM,
) -> List[Dict[str, Any]]:
    """Return crossing segments whose endpoints are sidewalk-side points."""

    points: List[Tuple[float, float]] = []
    connected = set()
    crossings: List[Dict[str, Any]] = []

    def key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        ka = (int(round(a[0])), int(round(a[1])))
        kb = (int(round(b[0])), int(round(b[1])))
        return (ka, kb) if ka <= kb else (kb, ka)

    def add_connected(a: Tuple[float, float], b: Tuple[float, float]) -> None:
        connected.add(key(a, b))

    def add_crossing(kind: str, a: Tuple[float, float], b: Tuple[float, float]) -> None:
        if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= 1e-6:
            return
        if not _axis_aligned(a, b):
            return
        k = key(a, b)
        if k in connected:
            return
        connected.add(k)
        cx, cy = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        crossings.append(
            {
                "kind": kind,
                "a": {"x": a[0], "y": a[1]},
                "b": {"x": b[0], "y": b[1]},
                "center": {"x": cx, "y": cy},
                "axis": "south-north" if abs(b[1] - a[1]) >= abs(b[0] - a[0]) else "east-west",
            }
        )

    for road in roads:
        p1, p2, p3, p4 = _sidewalk_points_for_road(
            road,
            sidewalk_offset_cm=sidewalk_offset_cm,
        )
        points.extend([p1, p2, p3, p4])
        add_connected(p1, p2)
        add_connected(p3, p4)
        add_crossing("endcap", p1, p4)
        add_crossing("endcap", p2, p3)

    threshold_cm = float(sidewalk_offset_cm) * 2.0 + 100.0
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            if key(a, b) in connected:
                continue
            if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= threshold_cm:
                add_crossing("crosswalk", a, b)

    return crossings


def road_intersections_from_roads(roads: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return road graph junctions with degree >= 3 from ``roads.json`` records."""

    return [
        crossing
        for crossing in road_crossing_centers_from_roads(roads)
        if int(crossing["degree"]) >= 3
    ]


def road_crossing_centers_from_roads(roads: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return road graph endpoint centers that can host sidewalk crossings."""

    adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)
    for road in roads:
        start = (float(road["start"]["x"]), float(road["start"]["y"]))
        end = (float(road["end"]["x"]), float(road["end"]["y"]))
        adjacency[start].append(end)
        adjacency[end].append(start)

    crossings: List[Dict[str, Any]] = []
    for point_m, neighbors in sorted(adjacency.items()):
        approaches = sorted({_unit_direction(point_m, nb) for nb in neighbors})
        crossings.append(
            {
                "center_m": {"x": point_m[0], "y": point_m[1]},
                "center_cm": {"x": point_m[0] * 100.0, "y": point_m[1] * 100.0},
                "degree": len(neighbors),
                "approach_directions": approaches,
            }
        )
    return crossings


def build_pedestrian_light_nodes(
    roads: Iterable[Mapping[str, Any]],
    *,
    map_name: str,
    initial_phase: str = "left_green_right_red",
) -> List[Dict[str, Any]]:
    """Build four sidewalk-corner pedestrian-light poles per road crossing.

    Each road graph endpoint/crossing center gets one pole at each corner of
    the sidewalk square around it. The two-way pedestrian-light asset exposes
    two faces, so each pole controls the two crossings directly in front of
    that corner.
    """

    nodes: List[Dict[str, Any]] = []
    face_slots = ("left", "right")
    corner_specs = (
        ("south_west", -1, -1, ("south", "west")),
        ("south_east", 1, -1, ("south", "east")),
        ("north_east", 1, 1, ("north", "east")),
        ("north_west", -1, 1, ("north", "west")),
    )
    crossings = road_crossing_centers_from_roads(roads)
    for crossing_idx, crossing_center in enumerate(crossings):
        center = crossing_center["center_cm"]
        center_m = crossing_center["center_m"]
        degree = int(crossing_center["degree"])
        crossing_kind = (
            "intersection_crosswalk"
            if degree >= 3
            else "road_joint_crosswalk"
            if degree == 2
            else "road_end_crosswalk"
        )
        for corner_idx, (corner_name, sx, sy, approach_dirs) in enumerate(corner_specs):
            loc_x = float(center["x"]) + sx * PEDESTRIAN_LIGHT_CORNER_OFFSET_CM
            loc_y = float(center["y"]) + sy * PEDESTRIAN_LIGHT_CORNER_OFFSET_CM

            faces: Dict[str, Dict[str, Any]] = {}
            face_states: Dict[str, str] = {}
            controlled_crossings: List[Dict[str, Any]] = []
            for face_idx, approach_dir in enumerate(approach_dirs):
                crosswalk_side = _crosswalk_side_for_corner_face(sx, sy, approach_dir)
                if crosswalk_side not in crossing_center["approach_directions"]:
                    continue

                # The pole sits on the far sidewalk side.  Its visible face
                # points back across the road toward the pedestrian approaching
                # from the opposite side.
                face_dir = _opposite_direction(approach_dir)
                vx, vy = DIR_VECTORS[face_dir]
                face_name = face_slots[face_idx]
                state = _state_for_direction(face_dir, initial_phase)
                crossing = {
                    "center": {
                        "x": float(center["x"]),
                        "y": float(center["y"]),
                        "z": 0.0,
                    },
                    "center_m": {
                        "x": float(center_m["x"]),
                        "y": float(center_m["y"]),
                    },
                    "crossing_kind": crossing_kind,
                    "crossing_axis": (
                        "south-north"
                        if face_dir in {"north", "south"}
                        else "east-west"
                    ),
                    "controlled_approach_direction": face_dir,
                    "far_side_sidewalk_direction": approach_dir,
                    "crossing_center_index": crossing_idx,
                    "intersection_index": crossing_idx,
                    "intersection_degree": degree,
                    "intersection_approach_directions": crossing_center["approach_directions"],
                    "crosswalk_side_direction": crosswalk_side,
                    "corner": corner_name,
                }
                faces[face_name] = {
                    "component_side": "_l_" if face_name == "left" else "_r_",
                    "face_index": face_idx,
                    "facing": {"x": float(vx), "y": float(vy)},
                    "facing_direction": face_dir,
                    "far_side_sidewalk_direction": approach_dir,
                    "crosswalk_side_direction": crosswalk_side,
                    "state": state,
                    "controlled_crossing_index": crossing_idx,
                }
                face_states[face_name] = state
                controlled_crossings.append(crossing)

            if not controlled_crossings:
                continue

            primary = controlled_crossings[0]
            primary_dir = str(primary["controlled_approach_direction"])
            wait_point = {"x": loc_x, "y": loc_y, "z": 0.0}
            nodes.append(
                {
                    "id": (
                        f"{GENERATED_LIGHT_ID_PREFIX}{map_name}_"
                        f"crossing_{crossing_idx:03d}_{corner_idx:02d}"
                    ),
                    "instance_name": PEDESTRIAN_LIGHT_INSTANCE,
                    "properties": {
                        "poi_type": PEDESTRIAN_LIGHT_TYPE,
                        "type": PEDESTRIAN_LIGHT_TYPE,
                        "ue_asset_path": PEDESTRIAN_LIGHT_ASSET,
                        "location": {"x": loc_x, "y": loc_y, "z": 0.0},
                        "orientation": {
                            "pitch": 0.0,
                            "yaw": _yaw_for_facing(primary_dir),
                            "roll": 0.0,
                        },
                        "signal_phase": _phase_alias(initial_phase),
                        "signal_state": next(iter(face_states.values())),
                        "face_states": face_states,
                        "faces": faces,
                        "waiting_point": wait_point,
                        "controlled_crossing": primary,
                        "controlled_crossings": controlled_crossings,
                        "placement_model": "sidewalk_corner_square",
                        "placement_note": (
                            "Generated two-face pedestrian signal at a sidewalk "
                            "corner. Four poles form a square around each road "
                            "crossing/intersection; each pole sits on the far "
                            "side sidewalk corner and its visible faces point "
                            "back across the road toward approaching pedestrians."
                        ),
                    },
                }
            )
    return nodes


def is_pedestrian_light_node(node: Mapping[str, Any]) -> bool:
    props = node.get("properties", {}) or {}
    kind = str(props.get("poi_type") or props.get("type") or "").lower()
    inst = str(node.get("instance_name") or "").lower()
    return kind == PEDESTRIAN_LIGHT_TYPE or "street_light_ped" in inst


def pedestrian_light_nodes(world: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [node for node in world.get("nodes", []) if is_pedestrian_light_node(node)]


def upsert_generated_pedestrian_lights(
    world: MutableMapping[str, Any],
    lights: Iterable[Mapping[str, Any]],
    *,
    remove_legacy_generated: bool = True,
) -> None:
    """Replace generated pedestrian lights in ``world`` with ``lights``."""

    kept = []
    for node in world.get("nodes", []):
        node_id = str(node.get("id", ""))
        if node_id.startswith(GENERATED_LIGHT_ID_PREFIX):
            continue
        if remove_legacy_generated and node_id.startswith(LEGACY_LIGHT_ID_PREFIX):
            continue
        kept.append(node)
    kept.extend(deepcopy(list(lights)))
    world["nodes"] = kept


def phase_to_face_states(phase: str) -> Dict[str, str]:
    phase = _phase_alias(phase)
    if phase == PHASE_A:
        return {"north": "green", "south": "green", "east": "red", "west": "red"}
    return {"north": "red", "south": "red", "east": "green", "west": "green"}


def phase_for_time(seconds: float, *, period_s: float = 120.0) -> str:
    """Alternate phases by time; first half north-south green, second half east-west green."""

    if period_s <= 0:
        raise ValueError("period_s must be positive")
    return PHASE_A if float(seconds) % period_s < period_s / 2.0 else PHASE_B


def apply_pedestrian_light_phase(
    world: MutableMapping[str, Any],
    *,
    phase: Optional[str] = None,
    seconds: Optional[float] = None,
    period_s: float = 120.0,
) -> str:
    """Update all pedestrian-light nodes in-place and return the applied phase."""

    chosen = _phase_alias(phase if phase is not None else phase_for_time(float(seconds or 0.0), period_s=period_s))
    states_by_dir = phase_to_face_states(chosen)
    for node in world.get("nodes", []):
        if not is_pedestrian_light_node(node):
            continue
        props = node.setdefault("properties", {})
        props["signal_phase"] = chosen
        props["face_states"] = {}
        faces = props.setdefault("faces", {})
        for face, face_rec in faces.items():
            direction = str(face_rec.get("facing_direction") or "").lower()
            state = states_by_dir.get(direction, "red")
            face_rec["state"] = state
            props["face_states"][face] = state
        props["signal_state"] = (
            next(iter(props["face_states"].values()))
            if props["face_states"]
            else chosen
        )
    return chosen
