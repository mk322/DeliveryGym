"""Render a 2D preview for a prepared DeliveryBench FPV render manifest.

The preview is intentionally tied to the generated ``manifest.jsonl`` so it
shows the waypoint set that will be used by UE rendering, overlaid on the road
and building assets from the map JSON files.

Example:
    python -m vagen.envs.deliverybench.tools.render_fpv_prep_map_2d \
        vagen/envs/deliverybench/maps/small-city-13 \
        --manifest vagen/envs/deliverybench/deliverybench_fpv/small-city-13/prep_3d_ue_dataset/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw

from vagen.envs.deliverybench.tools.render_gmaps import (
    MARGIN_PX,
    OUT_LONG,
    OUT_MIN,
    STYLE,
    View,
    _draw_building_shadow_layer,
    _draw_buildings,
    _draw_bus_routes,
    _draw_city_area,
    _draw_compass,
    _draw_crosswalks,
    _draw_poi_buildings,
    _draw_point_pois,
    _draw_roads,
    _draw_scale_bar,
    _draw_text_with_halo,
    _font,
    _font_fixed,
    _stroke_widths,
    load_world,
)


WAYPOINT_COLORS = {
    "intersection": "#1A73E8",
    "dock": "#5F6368",
}
TRAFFIC_COLOR = "#D93025"
OBSTACLE_COLOR = "#7B1FA2"


def _iter_manifest(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_waypoints(manifest_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    waypoints: Dict[str, Dict[str, Any]] = {}
    render_counts: Dict[str, int] = {}
    for row in _iter_manifest(manifest_path):
        if row.get("status") != "ok":
            continue
        wp_id = str(row.get("waypoint_id") or "")
        if not wp_id:
            continue
        render_kind = str(row.get("render_kind") or "")
        render_counts[render_kind] = render_counts.get(render_kind, 0) + 1
        rec = waypoints.setdefault(
            wp_id,
            {
                "waypoint_id": wp_id,
                "waypoint_kind": str(row.get("waypoint_kind") or ""),
                "waypoint_name": str(row.get("waypoint_name") or ""),
                "x_cm": float(row.get("x_cm", 0.0)),
                "y_cm": float(row.get("y_cm", 0.0)),
                "render_kinds": set(),
            },
        )
        rec["render_kinds"].add(render_kind)
    return waypoints, render_counts


def _make_view(scenario_dir: Path, waypoints: Dict[str, Dict[str, Any]]) -> View:
    world = load_world(scenario_dir)
    xmin, xmax, ymin, ymax = world.bounds
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    for wp in waypoints.values():
        xs.append(float(wp["x_cm"]))
        ys.append(float(wp["y_cm"]))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(xmax - xmin, ymax - ymin) * 0.055
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


def _draw_waypoints(draw: ImageDraw.ImageDraw, view: View, waypoints: Dict[str, Dict[str, Any]]) -> None:
    small_font = _font(22)
    label_font = _font(26)
    for wp_id, wp in sorted(waypoints.items(), key=lambda item: item[0]):
        x, y = view.to_px(float(wp["x_cm"]), float(wp["y_cm"]))
        kind = str(wp.get("waypoint_kind") or "").lower()
        render_kinds = wp.get("render_kinds", set())
        color = WAYPOINT_COLORS.get(kind, "#424242")
        radius = 8 if kind == "dock" else 11
        if "traffic_light" in render_kinds:
            radius = max(radius, 13)
            draw.ellipse([x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4],
                         outline=TRAFFIC_COLOR, width=4)
        if "obstacle" in render_kinds:
            draw.rectangle([x - radius - 6, y - radius - 6, x + radius + 6, y + radius + 6],
                           outline=OBSTACLE_COLOR, width=4)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline="#FFFFFF", width=2)
        should_label = kind == "intersection" or "traffic_light" in render_kinds or "obstacle" in render_kinds
        if should_label:
            _draw_text_with_halo(
                draw,
                (x + radius + 6, y - radius - 2),
                wp_id,
                label_font if kind == "intersection" else small_font,
                "#202124",
                "#FFFFFF",
                anchor="la",
            )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    scenario_name: str,
    waypoints: Dict[str, Dict[str, Any]],
    render_counts: Dict[str, int],
    out_size: Tuple[int, int],
) -> None:
    font = _font_fixed(34)
    small = _font_fixed(28)
    x0, y0 = 36, 30
    box_w, box_h = 900, 270
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=12,
                           fill=(255, 255, 255, 235), outline="#DADCE0", width=2)
    n_int = sum(1 for w in waypoints.values() if str(w.get("waypoint_kind")).lower() == "intersection")
    n_dock = sum(1 for w in waypoints.values() if str(w.get("waypoint_kind")).lower() == "dock")
    lines = [
        f"{scenario_name} FPV prep preview",
        f"waypoints: {len(waypoints)} ({n_int} intersections, {n_dock} docks)",
        f"render rows: plain {render_counts.get('plain', 0)}, "
        f"traffic {render_counts.get('traffic_light', 0)}, obstacle {render_counts.get('obstacle', 0)}",
    ]
    draw.text((x0 + 22, y0 + 18), lines[0], font=font, fill="#202124")
    draw.text((x0 + 22, y0 + 72), lines[1], font=small, fill="#3C4043")
    draw.text((x0 + 22, y0 + 112), lines[2], font=small, fill="#3C4043")

    legend_y = y0 + 180
    entries = [
        ("intersection", WAYPOINT_COLORS["intersection"], "circle"),
        ("dock", WAYPOINT_COLORS["dock"], "circle"),
        ("traffic-light job", TRAFFIC_COLOR, "ring"),
        ("obstacle job", OBSTACLE_COLOR, "box"),
    ]
    x = x0 + 24
    for label, color, shape in entries:
        cx, cy = x + 12, legend_y + 14
        if shape == "box":
            draw.rectangle([cx - 10, cy - 10, cx + 10, cy + 10], outline=color, width=4)
        elif shape == "ring":
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=color, width=4)
        else:
            draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=color, outline="#FFFFFF", width=2)
        draw.text((x + 32, legend_y), label, font=small, fill="#3C4043")
        try:
            bbox = draw.textbbox((0, 0), label, font=small)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(label) * 16
        x += 48 + text_w + 36


def render(scenario_dir: Path, manifest_path: Path, out_path: Path) -> Path:
    scenario_dir = scenario_dir.resolve()
    manifest_path = manifest_path.resolve()
    waypoints, render_counts = _load_waypoints(manifest_path)
    if not waypoints:
        raise ValueError(f"no ok waypoint rows found in {manifest_path}")

    world = load_world(scenario_dir)
    view = _make_view(scenario_dir, waypoints)
    img = Image.new("RGB", (view.out_w, view.out_h), STYLE["bg"])
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    widths = _stroke_widths(view)

    _draw_city_area(draw, view, world.road_bounds)
    shadow = _draw_building_shadow_layer(view, world.buildings + world.poi_buildings, img.size)
    img.alpha_composite(shadow)
    draw = ImageDraw.Draw(img)
    _draw_buildings(draw, view, world.buildings, STYLE["building_fill"], STYLE["building_border"])
    _draw_poi_buildings(draw, view, world.poi_buildings)
    _draw_bus_routes(draw, view, world.bus_routes, widths)
    _draw_roads(draw, view, world.roads, widths)
    _draw_crosswalks(draw, view, world.roads, widths)
    _draw_point_pois(draw, view, world.point_pois, [])
    _draw_waypoints(draw, view, waypoints)
    _draw_scale_bar(draw, view, view.out_w, view.out_h)
    _draw_compass(draw, view.out_w, flip_y_axis=view.flip_y_axis)
    _draw_legend(draw, scenario_dir.name, waypoints, render_counts, img.size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.manifest.with_name("prep_map_2d.png")
    print(render(args.scenario_dir, args.manifest, out))


if __name__ == "__main__":
    main()
