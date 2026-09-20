"""Render a 2D diagnostic map of waypoint road-crossing moves.

This is intentionally Pillow-only, matching ``render_gmaps.py``. It highlights
every waypoint-graph edge that VAGEN currently treats as a legal move crossing
a vehicle road, labels the source/target waypoints, and marks the
destination-side pedestrian-light pole selected for that crossing.

Example:
    python -m vagen.envs.deliverybench.tools.render_waypoint_crossings_2d \
        vagen/envs/deliverybench/maps/small-city-11
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from PIL import Image, ImageDraw, ImageFilter

from vagen.envs.deliverybench.tools.render_gmaps import (
    MARGIN_PX,
    OUT_LONG,
    OUT_MIN,
    STYLE,
    View,
    _draw_text_with_halo,
    _font,
    load_world,
)
from vagen.envs.deliverybench.tools.render_waypoint_pedestrian_light_ue import (
    DEFAULT_AGENT_FACING,
    _abs_direction_from_compass,
    _load_map,
    _relative_direction_for_abs,
    _select_visible_light_for_edge,
)
from vagen.envs.deliverybench.vlm_delivery.utils.traffic_lights import (
    signal_state_for_axis,
)


def _meters(value_cm: float) -> float:
    return float(value_cm) / 100.0


def _light_xy(light: Mapping[str, Any]) -> Tuple[float, float]:
    try:
        return float(light["x"]), float(light["y"])
    except Exception:
        loc = (light.get("properties", {}) or {}).get("location", {}) or {}
        return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    fill: str,
    width: int,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([start, end], fill=fill, width=width)
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    left = -uy, ux
    head_len = 14.0
    head_w = 8.0
    p1 = (ex, ey)
    p2 = (ex - ux * head_len + left[0] * head_w, ey - uy * head_len + left[1] * head_w)
    p3 = (ex - ux * head_len - left[0] * head_w, ey - uy * head_len - left[1] * head_w)
    draw.polygon([p1, p2, p3], fill=fill)


def _star_points(cx: float, cy: float, outer: float = 11.0, inner: float = 5.0) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        ang = -math.pi / 2.0 + i * math.pi / 5.0
        pts.append((cx + math.cos(ang) * radius, cy + math.sin(ang) * radius))
    return pts


def _multiline_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    *,
    font: Any,
    spacing: int = 1,
) -> Tuple[float, float, float, float]:
    try:
        return draw.multiline_textbbox(xy, text, font=font, spacing=spacing)
    except Exception:
        x, y = xy
        lines = text.splitlines() or [text]
        widths: List[float] = []
        heights: List[float] = []
        for line in lines:
            try:
                bbox = draw.textbbox((x, y), line, font=font)
                widths.append(float(bbox[2] - bbox[0]))
                heights.append(float(bbox[3] - bbox[1]))
            except Exception:
                w, h = draw.textsize(line, font=font)
                widths.append(float(w))
                heights.append(float(h))
        width = max(widths or [0.0])
        line_h = max(heights or [10.0])
        height = len(lines) * line_h + max(0, len(lines) - 1) * spacing
        return x, y, x + width, y + height


def _crossing_records(scenario_dir: Path, *, agent_facing: str) -> List[Dict[str, Any]]:
    city_map = _load_map(scenario_dir)
    lights_by_id = {str(light.get("id")): light for light in city_map.traffic_lights}
    records: List[Dict[str, Any]] = []

    for source_id, source in sorted(city_map.waypoints_by_id.items(), key=lambda item: item[0]):
        for adj in city_map.adjacents(source):
            target = adj.get("node")
            if target is None:
                continue
            target_id = str(adj.get("id") or getattr(target, "waypoint_id", "unknown"))
            dx = float(target.position.x) - float(source.position.x)
            dy = float(target.position.y) - float(source.position.y)
            if not bool(adj.get("legal_move", _is_legal_move_bearing(float(adj.get("bearing_deg", 0.0))))):
                continue
            signal = adj.get("traffic_signal")
            if not adj.get("crosses_vehicle_road") or not signal:
                continue
            move_abs = _abs_direction_from_compass(adj.get("compass"), dx, dy)
            try:
                action = _relative_direction_for_abs(agent_facing, move_abs)
            except ValueError:
                action = move_abs
            if action == "front":
                action = "forward"

            light_id = str(signal.get("light_id") or "")
            light_face = str(signal.get("light_face") or "")
            light_face_direction = str(signal.get("light_face_direction") or "")
            if not light_id or not light_face_direction:
                light_id, light_face, light_face_direction = _select_visible_light_for_edge(
                    city_map.traffic_lights,
                    source_x=float(source.position.x),
                    source_y=float(source.position.y),
                    target_x=float(target.position.x),
                    target_y=float(target.position.y),
                    movement_abs_direction=move_abs,
                    fallback_light_id=str(signal.get("light_id") or "unknown_light"),
                )
            light = lights_by_id.get(light_id, {})
            lx, ly = _light_xy(light)
            records.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source": source,
                    "target": target,
                    "compass": adj.get("compass"),
                    "action": action,
                    "move_abs": move_abs,
                    "axis": signal.get("axis"),
                    "runtime_light_id": signal.get("light_id"),
                    "runtime_light_face": signal.get("light_face"),
                    "runtime_light_face_direction": signal.get("light_face_direction"),
                    "traffic_selection": signal.get("selection"),
                    "selected_light_id": light_id,
                    "selected_light_face": light_face,
                    "selected_light_face_direction": light_face_direction,
                    "selected_light_xy": (lx, ly),
                }
            )
    return records


def _make_view(scenario_dir: Path) -> View:
    world = load_world(scenario_dir)
    xmin, xmax, ymin, ymax = world.bounds
    pad = max(xmax - xmin, ymax - ymin) * 0.045
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    span_x = max(1.0, xmax - xmin)
    span_y = max(1.0, ymax - ymin)
    if span_x >= span_y:
        out_w = OUT_LONG
        out_h = max(OUT_MIN, int(round(OUT_LONG * span_y / span_x)))
    else:
        out_h = OUT_LONG
        out_w = max(OUT_MIN, int(round(OUT_LONG * span_x / span_y)))
    return View(xmin, xmax, ymin, ymax, out_w, out_h, MARGIN_PX)


def _make_focus_view(records: List[Dict[str, Any]]) -> View:
    xs: List[float] = []
    ys: List[float] = []
    for rec in records:
        for node_key in ("source", "target"):
            node = rec[node_key]
            xs.append(float(node.position.x))
            ys.append(float(node.position.y))
        lx, ly = rec["selected_light_xy"]
        xs.append(float(lx))
        ys.append(float(ly))

    if not xs or not ys:
        return View(0.0, 1000.0, 0.0, 1000.0, 1500, 1100, MARGIN_PX)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 900.0)
    pad = max(450.0, span * 0.7)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half = span / 2.0 + pad
    return View(cx - half, cx + half, cy - half, cy + half, 1500, 1100, MARGIN_PX)


def _ang_diff(a: float, b: float) -> float:
    d = abs((float(a) - float(b)) % 360.0)
    return min(d, 360.0 - d)


def _is_legal_move_bearing(bearing_deg: float, tol_deg: float = 30.0) -> bool:
    return any(_ang_diff(bearing_deg, cardinal) <= tol_deg for cardinal in (0.0, 90.0, 180.0, 270.0))


def _dist_point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx, vy = bx - ax, by - ay
    den = vx * vx + vy * vy
    if den <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / den
    t = max(0.0, min(1.0, t))
    qx, qy = ax + vx * t, ay + vy * t
    return math.hypot(px - qx, py - qy)


def _face_axis(face_direction: str) -> str:
    return "south-north" if str(face_direction).lower() in {"north", "south"} else "east-west"


def _face_state_for_phase(face_direction: str, phase: str) -> str:
    seconds = 60.0 if str(phase).lower() == "odd" else 0.0
    return signal_state_for_axis(seconds, _face_axis(face_direction))


def _light_crossing_records(light: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    props = light.get("properties", {}) or {}
    records = props.get("controlled_crossings") or []
    if records:
        return [r for r in records if isinstance(r, Mapping)]
    one = props.get("controlled_crossing")
    return [one] if isinstance(one, Mapping) else []


def _light_crossing_index(light: Mapping[str, Any]) -> int | None:
    for rec in _light_crossing_records(light):
        for key in ("crossing_center_index", "intersection_index"):
            if key in rec:
                try:
                    return int(rec[key])
                except Exception:
                    pass
    return None


def _light_crossing_center(light: Mapping[str, Any]) -> Tuple[float, float] | None:
    for rec in _light_crossing_records(light):
        center = rec.get("center") or rec.get("center_cm")
        if isinstance(center, Mapping) and "x" in center and "y" in center:
            return float(center["x"]), float(center["y"])
    return None


def _group_lights_by_crossing(city_map: Any) -> Dict[int, Dict[str, Any]]:
    groups: Dict[int, Dict[str, Any]] = {}
    for light in city_map.traffic_lights:
        idx = _light_crossing_index(light)
        center = _light_crossing_center(light)
        if idx is None or center is None:
            continue
        group = groups.setdefault(idx, {"index": idx, "center": center, "lights": []})
        group["lights"].append(light)
    return groups


def _local_crossing_view(group: Mapping[str, Any], waypoints: List[Tuple[str, Any]]) -> View:
    xs = [float(group["center"][0])]
    ys = [float(group["center"][1])]
    for _, node in waypoints:
        xs.append(float(node.position.x))
        ys.append(float(node.position.y))
    for light in group.get("lights", []):
        lx, ly = _light_xy(light)
        xs.append(lx)
        ys.append(ly)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 700.0)
    pad = max(300.0, span * 0.4)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half = span / 2.0 + pad
    return View(cx - half, cx + half, cy - half, cy + half, 1400, 1100, MARGIN_PX)


def _local_waypoints_and_moves(city_map: Any, center: Tuple[float, float]) -> Tuple[List[Tuple[str, Any]], List[Dict[str, Any]]]:
    cx, cy = center
    wp_by_id: Dict[str, Any] = {}
    moves: List[Dict[str, Any]] = []
    for edge in city_map.waypoint_graph.edges:
        u = edge.node1
        v = edge.node2
        meta = city_map.waypoint_graph.get_edge_meta(u, v) or {}
        ux, uy = float(u.position.x), float(u.position.y)
        vx, vy = float(v.position.x), float(v.position.y)
        dist_cm = float(meta.get("dist_cm", u.position.distance(v.position)))
        seg_dist = _dist_point_to_segment(cx, cy, ux, uy, vx, vy)
        if dist_cm > 1000.0 or seg_dist > 800.0:
            continue
        bearing = float(meta.get("bearing_deg", city_map._bearing_deg(u.position, v.position)))
        legal_move = bool(meta.get("legal_move", _is_legal_move_bearing(bearing)))
        uid = city_map._waypoint_id_by_node.get(u, "")
        vid = city_map._waypoint_id_by_node.get(v, "")
        if uid:
            wp_by_id[uid] = u
        if vid:
            wp_by_id[vid] = v
        if not legal_move:
            continue
        moves.append(
            {
                "source_id": uid,
                "target_id": vid,
                "source": u,
                "target": v,
                "checked": bool(meta.get("crosses_vehicle_road")),
                "axis": (meta.get("traffic_signal") or {}).get("axis", ""),
                "dist_cm": dist_cm,
            }
        )

    for wp_id, node in city_map.waypoints_by_id.items():
        if math.hypot(float(node.position.x) - cx, float(node.position.y) - cy) <= 650.0:
            wp_by_id[wp_id] = node

    waypoints = sorted(wp_by_id.items(), key=lambda item: item[0])
    return waypoints, moves


def render_crossing_graph(
    scenario_dir: Path,
    *,
    crossing_index: int,
    phase: str,
    out_path: Path,
) -> Path:
    scenario_dir = scenario_dir.resolve()
    city_map = _load_map(scenario_dir)
    groups = _group_lights_by_crossing(city_map)
    if crossing_index not in groups:
        raise ValueError(f"No generated pedestrian-light crossing index {crossing_index}.")

    group = groups[crossing_index]
    waypoints, moves = _local_waypoints_and_moves(city_map, group["center"])
    view = _local_crossing_view(group, waypoints)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (view.out_w, view.out_h), STYLE["city_bg"])
    draw = ImageDraw.Draw(img)
    title_font = _font(28)
    label_font = _font(18)
    small_font = _font(14)

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [35, 35, view.out_w - 35, view.out_h - 35],
        radius=16,
        fill=(255, 255, 255, 130),
    )
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(9)))

    # Local road skeleton.
    for edge in city_map.graph_skel.edges:
        meta = city_map.graph_skel.get_edge_meta(edge.node1, edge.node2) or {}
        if meta.get("kind") != "road":
            continue
        a = view.to_px(edge.node1.position.x, edge.node1.position.y)
        b = view.to_px(edge.node2.position.x, edge.node2.position.y)
        draw.line([a, b], fill="#9aa79d", width=5)
        draw.line([a, b], fill="#f7f7f1", width=3)

    # Legal MOVE edges. Diagonal graph connectors are intentionally omitted.
    for move in moves:
        a = view.to_px(move["source"].position.x, move["source"].position.y)
        b = view.to_px(move["target"].position.x, move["target"].position.y)
        if move["checked"]:
            draw.line([a, b], fill="#ffffff", width=12)
            _arrow(draw, a, b, fill="#f57c00", width=6)
            _arrow(draw, b, a, fill="#f57c00", width=6)
        else:
            draw.line([a, b], fill="#ffffff", width=9)
            draw.line([a, b], fill="#2e7d32", width=4)

    # Waypoint locations.
    for wp_id, node in waypoints:
        x, y = view.to_px(node.position.x, node.position.y)
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill="#ffffff")
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill="#1f77b4", outline="#0b3d75")
        _draw_text_with_halo(draw, (x + 10, y - 10), wp_id, label_font, "#0b3d75", "#ffffff")

    # Crossing center.
    cx, cy = view.to_px(*group["center"])
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill="#222222")

    # Pedestrian-light poles and only the faces that actually exist.
    for light in group.get("lights", []):
        lx, ly = _light_xy(light)
        px, py = view.to_px(lx, ly)
        draw.ellipse([px - 12, py - 12, px + 12, py + 12], fill="#ffffff")
        draw.ellipse([px - 9, py - 9, px + 9, py + 9], fill="#8e24aa", outline="#4a148c")
        faces = (light.get("properties", {}) or {}).get("faces", {}) or {}
        for face_name, face in faces.items():
            face_dir = str(face.get("facing_direction") or "").lower()
            vx, vy = {
                "north": (0.0, 1.0),
                "east": (1.0, 0.0),
                "south": (0.0, -1.0),
                "west": (-1.0, 0.0),
            }.get(face_dir, (0.0, 0.0))
            if vx == 0.0 and vy == 0.0:
                continue
            state = _face_state_for_phase(face_dir, phase)
            color = "#2e7d32" if state == "green" else "#d32f2f"
            end = view.to_px(lx + vx * 360.0, ly + vy * 360.0)
            _arrow(draw, (px, py), end, fill=color, width=4)
            _draw_text_with_halo(
                draw,
                (px + vx * 32.0 + 8.0, py + vy * 32.0),
                f"{face_dir[0].upper()}/{state[0].upper()}",
                small_font,
                color,
                "#ffffff",
            )

    title = (
        f"{scenario_dir.name}: crossing_{crossing_index:03d} local MOVE graph ({phase} minute)\n"
        "green = legal MOVE, orange = legal MOVE checked by pedestrian light; "
        "diagonal connectors are not legal MOVE actions"
    )
    _draw_text_with_halo(draw, (MARGIN_PX, 40), title, title_font, "#202124", "#ffffff", anchor="la")

    legend_items = [
        ("#2e7d32", "legal cardinal MOVE edge"),
        ("#f57c00", "legal MOVE requiring pedestrian-light check"),
        ("#8e24aa", "pedestrian-light pole"),
        ("#d32f2f", "red face for this phase"),
        ("#2e7d32", "green face for this phase"),
    ]
    legend_x = MARGIN_PX
    legend_y = view.out_h - MARGIN_PX - len(legend_items) * 24
    for i, (color, text) in enumerate(legend_items):
        y = legend_y + i * 24
        draw.rounded_rectangle([legend_x, y, legend_x + 18, y + 12], radius=2, fill=color)
        draw.text((legend_x + 26, y - 4), text, font=small_font, fill="#202124")

    img.convert("RGB").save(out_path)
    return out_path


def render_all_crossing_graphs(
    scenario_dir: Path,
    *,
    out_dir: Path | None = None,
    phases: Tuple[str, ...] = ("even", "odd"),
) -> List[Path]:
    scenario_dir = scenario_dir.resolve()
    city_map = _load_map(scenario_dir)
    groups = _group_lights_by_crossing(city_map)
    out_dir = out_dir or scenario_dir / "waypoint_crossing_graphs"
    paths: List[Path] = []
    for idx in sorted(groups):
        for phase in phases:
            out_path = out_dir / f"crossing_{idx:03d}_{phase}.png"
            paths.append(
                render_crossing_graph(
                    scenario_dir,
                    crossing_index=idx,
                    phase=phase,
                    out_path=out_path,
                )
            )
    return paths


def render(
    scenario_dir: Path,
    *,
    out_path: Path | None = None,
    agent_facing: str = DEFAULT_AGENT_FACING,
    source_id: str | None = None,
    target_id: str | None = None,
) -> Path:
    scenario_dir = scenario_dir.resolve()
    city_map = _load_map(scenario_dir)
    all_records = _crossing_records(scenario_dir, agent_facing=agent_facing)
    focus_mode = bool(source_id or target_id)
    records = [
        rec for rec in all_records
        if (not source_id or str(rec["source_id"]) == str(source_id))
        and (not target_id or str(rec["target_id"]) == str(target_id))
    ]
    if focus_mode and not records:
        raise ValueError(
            "No traffic-light checked waypoint crossing matched "
            f"source_id={source_id!r}, target_id={target_id!r}."
        )

    view = _make_focus_view(records) if focus_mode else _make_view(scenario_dir)
    if out_path is None:
        if focus_mode:
            src = str(source_id or "any_source")
            dst = str(target_id or "any_target")
            out_path = scenario_dir / f"waypoint_crossing_focus_{src}_to_{dst}.png"
        else:
            out_path = scenario_dir / "waypoint_crossing_graphs" / "all_checked_crossings.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (view.out_w, view.out_h), STYLE["city_bg"])
    draw = ImageDraw.Draw(img)
    label_font = _font(17)
    small_font = _font(13)
    title_font = _font(32)

    # Soft background for the map area.
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [35, 35, view.out_w - 35, view.out_h - 35],
        radius=18,
        fill=(255, 255, 255, 120),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    img.alpha_composite(shadow)

    # Road skeleton.
    for edge in city_map.graph_skel.edges:
        meta = city_map.graph_skel.get_edge_meta(edge.node1, edge.node2) or {}
        if meta.get("kind") != "road":
            continue
        a = view.to_px(edge.node1.position.x, edge.node1.position.y)
        b = view.to_px(edge.node2.position.x, edge.node2.position.y)
        draw.line([a, b], fill="#c4c7c5", width=4)
        draw.line([a, b], fill="#f6f6f2", width=2)

    crossing_keys = {
        tuple(sorted((str(rec["source_id"]), str(rec["target_id"]))))
        for rec in records
    }

    # Context waypoint graph.
    for edge in city_map.waypoint_graph.edges:
        u = edge.node1
        v = edge.node2
        em = city_map.waypoint_graph.get_edge_meta(u, v) or {}
        uid = city_map._waypoint_id_by_node.get(u, "")
        vid = city_map._waypoint_id_by_node.get(v, "")
        key = tuple(sorted((uid, vid)))
        if key in crossing_keys:
            continue
        if not bool(em.get("legal_move", _is_legal_move_bearing(float(em.get("bearing_deg", 0.0))))):
            continue
        draw.line(
            [view.to_px(u.position.x, u.position.y), view.to_px(v.position.x, v.position.y)],
            fill=(112, 148, 112, 95),
            width=2,
        )

    # Crossing edges, one line per unique edge.
    drawn_crossings = set()
    for rec in records:
        key = tuple(sorted((str(rec["source_id"]), str(rec["target_id"]))))
        if key in drawn_crossings:
            continue
        drawn_crossings.add(key)
        a = view.to_px(rec["source"].position.x, rec["source"].position.y)
        b = view.to_px(rec["target"].position.x, rec["target"].position.y)
        draw.line([a, b], fill="#ffffff", width=9)
        draw.line([a, b], fill="#ff8c00", width=5)
        if focus_mode:
            _arrow(draw, a, b, fill="#e65100", width=7)

    # Directed crossing move labels.
    if not focus_mode:
        label_count = 0
        for rec in records:
            sx, sy = rec["source"].position.x, rec["source"].position.y
            tx, ty = rec["target"].position.x, rec["target"].position.y
            mx, my = view.to_px((sx + tx) / 2.0, (sy + ty) / 2.0)
            ox = 0 if label_count % 2 == 0 else 18
            oy = -18 if label_count % 2 == 0 else 14
            text = f"{rec['source_id']}->{rec['target_id']}\n{rec['action']} {rec['axis']}"
            box_x, box_y = mx + ox, my + oy
            bbox = _multiline_bbox(draw, (box_x, box_y), text, font=small_font, spacing=1)
            draw.rounded_rectangle(
                [bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3],
                radius=4,
                fill=(255, 248, 225, 218),
                outline="#ff8c00",
                width=1,
            )
            draw.multiline_text((box_x, box_y), text, font=small_font, fill="#7a3e00", spacing=1)
            label_count += 1

    # Waypoints.
    crossing_wp_ids = set()
    for rec in records:
        crossing_wp_ids.add(str(rec["source_id"]))
        crossing_wp_ids.add(str(rec["target_id"]))
    for wp_id, node in city_map.waypoints_by_id.items():
        x, y = view.to_px(node.position.x, node.position.y)
        if wp_id.startswith("int_"):
            fill = "#1f77b4"
            outline = "#0b3d75"
            r = 7 if wp_id in crossing_wp_ids else 5
        else:
            fill = "#8fb98f"
            outline = "#4b7a4b"
            r = 4
        draw.ellipse([x - r, y - r, x + r, y + r], fill="white")
        draw.ellipse([x - r + 1, y - r + 1, x + r - 1, y + r - 1], fill=fill, outline=outline)
        if wp_id in crossing_wp_ids:
            _draw_text_with_halo(draw, (x + 8, y - 8), wp_id, label_font, "#0b3d75", "#ffffff")

    if focus_mode:
        for rec in records:
            sx, sy = view.to_px(rec["source"].position.x, rec["source"].position.y)
            tx, ty = view.to_px(rec["target"].position.x, rec["target"].position.y)
            lx, ly = view.to_px(*rec["selected_light_xy"])
            draw.ellipse([sx - 15, sy - 15, sx + 15, sy + 15], fill="#1565c0", outline="white", width=3)
            draw.ellipse([tx - 13, ty - 13, tx + 13, ty + 13], fill="#2e7d32", outline="white", width=3)
            _draw_text_with_halo(draw, (sx + 18, sy - 22), "agent/start", label_font, "#0d47a1", "#ffffff")
            _draw_text_with_halo(draw, (tx + 18, ty + 8), "target waypoint", label_font, "#1b5e20", "#ffffff")
            draw.line([(sx, sy), (lx, ly)], fill=(142, 36, 170, 110), width=3)

    # All pedestrian-light placements and faces.
    selected_light_ids = {str(rec["selected_light_id"]) for rec in records}
    for light in city_map.traffic_lights:
        light_id = str(light.get("id") or "")
        lx, ly = _light_xy(light)
        x, y = view.to_px(lx, ly)
        selected = light_id in selected_light_ids
        radius = 7 if selected else 4
        fill = "#ba68c8" if selected else "#d7a7e0"
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="white")
        draw.ellipse(
            [x - radius + 1, y - radius + 1, x + radius - 1, y + radius - 1],
            fill=fill,
            outline="#6a1b9a",
        )
        faces = (light.get("properties", {}) or {}).get("faces", {}) or {}
        for face in faces.values():
            face_dir = str(face.get("facing_direction") or "").lower()
            vx, vy = {
                "north": (0.0, 1.0),
                "east": (1.0, 0.0),
                "south": (0.0, -1.0),
                "west": (-1.0, 0.0),
            }.get(face_dir, (0.0, 0.0))
            if vx == 0.0 and vy == 0.0:
                continue
            end = view.to_px(lx + vx * 420.0, ly + vy * 420.0)
            _arrow(draw, (x, y), end, fill="#ab47bc" if selected else "#ce93d8", width=2)

    # Selected destination-side pedestrian lights used by env checking.
    selected_lights: Dict[Tuple[str, str], Tuple[float, float, str]] = {}
    for rec in records:
        key = (str(rec["selected_light_id"]), str(rec["selected_light_face"]))
        selected_lights[key] = (
            float(rec["selected_light_xy"][0]),
            float(rec["selected_light_xy"][1]),
            str(rec["selected_light_face_direction"]),
        )
    for (light_id, face), (lx, ly, face_dir) in selected_lights.items():
        x, y = view.to_px(lx, ly)
        draw.polygon(_star_points(x, y, outer=13, inner=6), fill="#8e24aa", outline="white")
        short = light_id.replace("GEN_PedestrianLight_", "PL_")
        short = short.replace(scenario_dir.name.replace("-", "_") + "_", "")
        _draw_text_with_halo(
            draw,
            (x + 10, y + 8),
            f"{short}:{face}/{face_dir}",
            small_font,
            "#6a1b9a",
            "#ffffff",
            anchor="la",
        )

    # Small arrows from each target waypoint toward its selected target-side light.
    for rec in records:
        target = rec["target"]
        sx, sy = view.to_px(target.position.x, target.position.y)
        lx, ly = view.to_px(*rec["selected_light_xy"])
        # Shorten both ends so arrows do not cover dots/stars.
        dx, dy = lx - sx, ly - sy
        length = math.hypot(dx, dy)
        if length > 1e-6:
            ux, uy = dx / length, dy / length
            a = (sx + ux * 10.0, sy + uy * 10.0)
            b = (lx - ux * 13.0, ly - uy * 13.0)
            _arrow(draw, a, b, fill="#8e24aa", width=2)

    if focus_mode:
        rec = records[0]
        even_state = signal_state_for_axis(0.0, str(rec["axis"]))
        odd_state = signal_state_for_axis(60.0, str(rec["axis"]))
        title = (
            f"{scenario_dir.name}: focused traffic-light waypoint check\n"
            f"{rec['source_id']} -> {rec['target_id']} | MOVE({rec['action']}) | "
            f"axis={rec['axis']} | even={even_state}, odd={odd_state}\n"
            f"selected target-side light: {rec['selected_light_id']} "
            f"face={rec['selected_light_face']}/{rec['selected_light_face_direction']}"
        )
    else:
        title = (
            f"{scenario_dir.name}: traffic-light checked waypoint crossings\n"
            f"{len(crossing_keys)} unique crossing edges, {len(records)} directed moves, "
            f"{len(city_map.traffic_lights)} pedestrian-light poles"
        )
    _draw_text_with_halo(draw, (MARGIN_PX, 38), title, title_font, "#202124", "#ffffff", anchor="la")

    legend_items = [
        ("#ff8c00", "waypoint edge checked by traffic-light rule"),
        ("#8e24aa", "selected far-side light for checked move"),
        ("#d7a7e0", "all pedestrian-light poles + face arrows"),
        ("#1f77b4", "intersection waypoint"),
        ("#8fb98f", "dock waypoint / context"),
    ]
    legend_x = MARGIN_PX
    legend_y = view.out_h - MARGIN_PX - len(legend_items) * 24
    for i, (color, text) in enumerate(legend_items):
        y = legend_y + i * 24
        draw.rounded_rectangle([legend_x, y, legend_x + 18, y + 12], radius=2, fill=color)
        draw.text((legend_x + 26, y - 3), text, font=small_font, fill="#202124")

    img.convert("RGB").save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--target-id", default=None)
    parser.add_argument(
        "--all-crossing-graphs",
        action="store_true",
        help="Render one local legal-MOVE graph per generated crossing.",
    )
    parser.add_argument(
        "--crossing-graphs-dir",
        type=Path,
        default=None,
        help="Output directory for --all-crossing-graphs.",
    )
    parser.add_argument(
        "--phase",
        default="both",
        choices=("even", "odd", "both"),
        help="Minute phase for local crossing graphs.",
    )
    parser.add_argument(
        "--agent-facing",
        default=DEFAULT_AGENT_FACING,
        choices=("north", "east", "south", "west"),
    )
    args = parser.parse_args()
    if args.all_crossing_graphs:
        phases = ("even", "odd") if args.phase == "both" else (args.phase,)
        paths = render_all_crossing_graphs(
            args.scenario_dir,
            out_dir=args.crossing_graphs_dir,
            phases=phases,
        )
        print(f"wrote {len(paths)} local crossing graph(s)")
        for path in paths:
            print(path)
        return

    out = render(
        args.scenario_dir,
        out_path=args.out,
        agent_facing=args.agent_facing,
        source_id=args.source_id,
        target_id=args.target_id,
    )
    print(f"wrote waypoint crossing diagnostic map: {out}")


if __name__ == "__main__":
    main()
