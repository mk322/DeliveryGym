"""World-JSON traffic-light **data helpers** (data-generation only).

This module reads pedestrian/traffic-light records out of the generated world
JSON and exposes the minute-parity signal function used by the 2D crossing
renderer. It is **not** the runtime evaluator.

Runtime red/green decisions and the signalised set live in the single
``TrafficController`` (``utils/hazards.py``), which is built from the FPV
manifest so it stays aligned with the rendered images. The old per-edge
``edge_signal_check`` / ``check_dm_edge_signal`` runtime path was removed when
the two parallel traffic-light implementations were unified — see
``OBSTACLE_TRAFFIC_DESIGN.md`` and ``TRAFFIC_LIGHT_AUDIT.md``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_CONTROL_RADIUS_CM = 650.0
DEFAULT_RED_LIGHT_PENALTY_S = 15.0
DEFAULT_RED_LIGHT_ENERGY_MULTIPLIER = 1.0

_BACK_OF = {"north": "south", "south": "north", "east": "west", "west": "east"}
_DIR_DEG = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}


def _props(node: Mapping[str, Any]) -> Mapping[str, Any]:
    return node.get("properties", {}) or {}


def is_traffic_light_node(node: Mapping[str, Any]) -> bool:
    props = _props(node)
    kind = str(props.get("poi_type") or props.get("type") or "").lower()
    inst = str(node.get("instance_name") or "").lower()
    return (
        kind in {"traffic_light", "pedestrian_light"}
        or "traffic_light" in inst
        or "street_light_ped" in inst
    )


def _node_xy_cm(node: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    props = _props(node)
    loc = props.get("location", {}) or {}
    if "x" in loc and "y" in loc:
        return float(loc["x"]), float(loc["y"])

    crossing = props.get("controlled_crossing", {}) or {}
    center = crossing.get("center") or crossing.get("center_cm")
    if isinstance(center, Mapping) and "x" in center and "y" in center:
        return float(center["x"]), float(center["y"])
    return None


def load_traffic_lights(world_nodes: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Extract light records from world-json nodes (data-gen / inspection)."""

    lights: List[Dict[str, Any]] = []
    for idx, node in enumerate(world_nodes or []):
        if not isinstance(node, Mapping) or not is_traffic_light_node(node):
            continue
        xy = _node_xy_cm(node)
        if xy is None:
            continue
        props = dict(_props(node))
        lights.append(
            {
                "id": str(node.get("id") or f"traffic_light_{idx}"),
                "kind": str(props.get("poi_type") or props.get("type") or "traffic_light"),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "properties": props,
            }
        )
    return lights


def signal_state_for_axis(seconds: float, axis: str) -> str:
    """Deterministic two-phase signal (must match TrafficController in hazards.py).

    Odd minute numbers make the south-north axis red; the perpendicular
    (east-west) axis is green. Even minutes flip. This is the same
    ``odd_minute_red_axis = NS`` convention the runtime controller uses, so the
    2D crossing renders match what the env scores.
    """

    minute = int(max(0.0, float(seconds)) // 60.0)
    sn_red = bool(minute % 2 == 1)
    axis_norm = str(axis or "").strip().lower().replace("_", "-")
    if axis_norm in {"south-north", "north-south", "vertical", "sn", "ns"}:
        return "red" if sn_red else "green"
    return "green" if sn_red else "red"


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    den = vx * vx + vy * vy
    if den <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / den
    t = max(0.0, min(1.0, t))
    qx, qy = ax + vx * t, ay + vy * t
    return math.hypot(px - qx, py - qy)


def _light_xy(light: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        return float(light["x"]), float(light["y"])
    except Exception:
        loc = (_props(light).get("location") or {})
        if "x" in loc and "y" in loc:
            return float(loc["x"]), float(loc["y"])
    return None


def _crossing_center_xy(light: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    props = _props(light)
    crossings = props.get("controlled_crossings") or []
    crossing = crossings[0] if crossings else props.get("controlled_crossing")
    if isinstance(crossing, Mapping):
        center = crossing.get("center") or crossing.get("center_cm")
        if isinstance(center, Mapping) and "x" in center and "y" in center:
            return float(center["x"]), float(center["y"])
    return None


def _movement_crosswalk_side(
    *,
    from_node: Any,
    to_node: Any,
    light: Mapping[str, Any],
) -> str:
    center = _crossing_center_xy(light)
    if center is None:
        return ""
    cx, cy = center
    mx = (float(from_node.position.x) + float(to_node.position.x)) / 2.0
    my = (float(from_node.position.y) + float(to_node.position.y)) / 2.0
    move_dir = movement_direction(from_node, to_node)
    if move_dir in {"east", "west"}:
        return "north" if my >= cy else "south"
    return "east" if mx >= cx else "west"


def _matching_face(light: Mapping[str, Any], face_direction: str, crosswalk_side: str) -> str:
    faces = (_props(light).get("faces") or {})
    for face_name, face in faces.items():
        if str(face.get("facing_direction") or "").lower() != face_direction:
            continue
        face_side = str(face.get("crosswalk_side_direction") or "").lower()
        if face_side and str(crosswalk_side or "").lower() and face_side != crosswalk_side:
            continue
        if str(face.get("facing_direction") or "").lower() == face_direction:
            return str(face_name)
    return ""


def _has_directional_faces(light: Mapping[str, Any]) -> bool:
    return bool((_props(light).get("faces") or {}))


def _select_edge_light(
    *,
    from_node: Any,
    to_node: Any,
    lights: Iterable[Mapping[str, Any]],
    control_radius_cm: float,
) -> Optional[Dict[str, Any]]:
    """Select the pedestrian signal an agent should see before crossing.

    The rendered/design convention places the visible signal on the far sidewalk
    near the destination waypoint, with the face pointing back toward the
    source-side pedestrian waiting area.
    """

    ax, ay = float(from_node.position.x), float(from_node.position.y)
    bx, by = float(to_node.position.x), float(to_node.position.y)
    move_dir = movement_direction(from_node, to_node)
    face_dir = _BACK_OF[move_dir]
    radius = max(0.0, float(control_radius_cm))

    best: Optional[Tuple[float, Dict[str, Any]]] = None
    fallback: Optional[Tuple[float, Dict[str, Any]]] = None
    for light in lights or []:
        xy = _light_xy(light)
        if xy is None:
            continue
        lx, ly = xy
        segment_dist = _dist_point_to_segment(lx, ly, ax, ay, bx, by)
        target_dist = math.hypot(lx - bx, ly - by)
        crosswalk_side = _movement_crosswalk_side(
            from_node=from_node,
            to_node=to_node,
            light=light,
        )
        face = _matching_face(light, face_dir, crosswalk_side)
        has_faces = _has_directional_faces(light)
        rec = {
            "light": light,
            "x": lx,
            "y": ly,
            "segment_dist_cm": float(segment_dist),
            "target_dist_cm": float(target_dist),
            "light_face": face,
            "light_face_direction": face_dir if face else "",
            "crosswalk_side_direction": crosswalk_side,
            "movement_direction": move_dir,
        }
        if (
            not has_faces
            and segment_dist <= radius
            and (fallback is None or segment_dist < fallback[0])
        ):
            fallback = (segment_dist, rec)
        if face and target_dist <= radius:
            score = target_dist + segment_dist * 0.05
            if best is None or score < best[0]:
                best = (score, rec)

    if best is not None:
        best[1]["selection"] = "target_side_facing_source"
        return best[1]
    if fallback is not None:
        fallback[1]["selection"] = "nearest_crossing_signal"
        return fallback[1]
    return None


def edge_signal_check(
    *,
    from_node: Any,
    to_node: Any,
    lights: Iterable[Mapping[str, Any]],
    seconds: float,
    control_radius_cm: float = DEFAULT_CONTROL_RADIUS_CM,
) -> Optional[Dict[str, Any]]:
    """Return signal metadata for a waypoint edge, or ``None`` if uncontrolled."""

    selected = _select_edge_light(
        from_node=from_node,
        to_node=to_node,
        lights=lights,
        control_radius_cm=control_radius_cm,
    )
    if selected is None:
        return None

    axis = movement_axis(from_node, to_node)
    state = signal_state_for_axis(seconds, axis)
    light = selected["light"]
    return {
        "controlled": True,
        "light_id": str(light.get("id", "")),
        "axis": axis,
        "state": state,
        "minute": int(max(0.0, float(seconds)) // 60.0),
        "distance_cm": float(selected["segment_dist_cm"]),
        "target_distance_cm": float(selected["target_dist_cm"]),
        "movement_direction": str(selected["movement_direction"]),
        "light_face": str(selected["light_face"]),
        "light_face_direction": str(selected["light_face_direction"]),
        "crosswalk_side_direction": str(selected.get("crosswalk_side_direction") or ""),
        "selection": str(selected["selection"]),
    }


def traffic_cfg(dm: Any) -> Dict[str, Any]:
    cfg = (getattr(dm, "cfg", {}) or {}).get("traffic_lights", {}) or {}
    return {
        "control_radius_cm": float(cfg.get("control_radius_cm", DEFAULT_CONTROL_RADIUS_CM)),
        "red_light_penalty_s": float(cfg.get("red_light_penalty_s", DEFAULT_RED_LIGHT_PENALTY_S)),
        "red_light_energy_multiplier": float(
            cfg.get("red_light_energy_multiplier", DEFAULT_RED_LIGHT_ENERGY_MULTIPLIER)
        ),
        "require_visible_signal_view": bool(cfg.get("require_visible_signal_view", False)),
        "fpv_yaw_offset_deg": float(cfg.get("fpv_yaw_offset_deg", 90.0)),
        "visible_signal_views": cfg.get("visible_signal_views"),
    }


def _has_visible_signal_view(dm: Any, from_node: Any, to_node: Any) -> bool:
    """Whether the source waypoint has a red/green FPV panel for this move.

    When visible-view enforcement is enabled, geometry alone is not enough: the
    agent must have received a light variant for the exact source waypoint and
    movement yaw before the MOVE can be checked.
    """

    cfg = traffic_cfg(dm)
    if not cfg["require_visible_signal_view"]:
        return True
    views = cfg.get("visible_signal_views")
    if not views:
        return False

    move_dir = movement_direction(from_node, to_node)
    compass_deg = _DIR_DEG.get(move_dir)
    if compass_deg is None:
        return False
    yaw = (float(cfg["fpv_yaw_offset_deg"]) - compass_deg) % 360.0
    key = (
        round(float(from_node.position.x), 1),
        round(float(from_node.position.y), 1),
        round(float(yaw), 1),
    )
    visible = {
        (round(float(x), 1), round(float(y), 1), round(float(view_yaw), 1))
        for x, y, view_yaw in views
    }
    return key in visible


def check_dm_edge_signal(dm: Any, from_node: Any, to_node: Any) -> Optional[Dict[str, Any]]:
    city_map = getattr(dm, "city_map", None)
    lights = getattr(city_map, "traffic_lights", []) if city_map is not None else []
    if not lights:
        return None
    cfg = traffic_cfg(dm)
    if not _has_visible_signal_view(dm, from_node, to_node):
        return None
    now_s = getattr(getattr(dm, "clock", None), "now_sim", lambda: 0.0)()
    return edge_signal_check(
        from_node=from_node,
        to_node=to_node,
        lights=lights,
        seconds=float(now_s),
        control_radius_cm=cfg["control_radius_cm"],
    )
