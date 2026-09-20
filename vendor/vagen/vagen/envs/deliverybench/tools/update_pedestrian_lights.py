"""Generate pedestrian-light metadata for a scenario.

Example:
    python -m vagen.envs.deliverybench.tools.update_pedestrian_lights \
        vagen/envs/deliverybench/maps/small-city-11

    python -m vagen.envs.deliverybench.tools.update_pedestrian_lights \
        vagen/envs/deliverybench/maps/small-city-11 \
        --map-out /tmp/pedestrian_lights_2d_map.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Tuple

from PIL import ImageDraw

from vagen.envs.deliverybench.tools.render_gmaps import (
    MARGIN_PX,
    STYLE,
    _font,
    compose_frame,
    render_background,
)
from vagen.envs.deliverybench.utils.pedestrian_lights import (
    PHASE_A,
    PHASE_B,
    apply_pedestrian_light_phase,
    build_pedestrian_light_nodes,
    is_pedestrian_light_node,
    upsert_generated_pedestrian_lights,
)


_DIR_VECTORS = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}


def _map_name_from_scenario_dir(scenario_dir: Path) -> str:
    return scenario_dir.name.replace("-", "_")


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    fill: Tuple[int, int, int, int],
    width: int,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([start, end], fill=fill, width=width)
    dx, dy = ex - sx, ey - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    left = -uy, ux
    head_len = max(8.0, width * 3.0)
    head_w = max(5.0, width * 1.6)
    draw.polygon(
        [
            (ex, ey),
            (ex - ux * head_len + left[0] * head_w, ey - uy * head_len + left[1] * head_w),
            (ex - ux * head_len - left[0] * head_w, ey - uy * head_len - left[1] * head_w),
        ],
        fill=fill,
    )


def _light_xy(node: Mapping[str, Any]) -> Tuple[float, float]:
    loc = (node.get("properties", {}) or {}).get("location", {}) or {}
    return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))


def _render_pedestrian_light_map(
    scenario_dir: Path,
    map_out: Path,
    world: Mapping[str, Any],
) -> None:
    bg, _, view = render_background(scenario_dir)
    img = compose_frame(bg, view, agent_xy=None).convert("RGBA")
    draw = ImageDraw.Draw(img)

    lights = [node for node in world.get("nodes", []) if is_pedestrian_light_node(node)]
    pole_r = max(5, int(round(view.px_per_m * 0.9)))
    arrow_len_cm = 360.0
    arrow_w = max(3, int(round(view.px_per_m * 0.45)))

    for node in lights:
        props = node.get("properties", {}) or {}
        lx, ly = _light_xy(node)
        px, py = view.to_px(lx, ly)
        draw.ellipse(
            [px - pole_r - 2, py - pole_r - 2, px + pole_r + 2, py + pole_r + 2],
            fill=(255, 255, 255, 245),
        )
        draw.ellipse(
            [px - pole_r, py - pole_r, px + pole_r, py + pole_r],
            fill=(156, 39, 176, 245),
            outline=(74, 20, 140, 255),
            width=2,
        )

        for face in (props.get("faces") or {}).values():
            face_dir = str(face.get("facing_direction") or "").lower()
            vx, vy = _DIR_VECTORS.get(face_dir, (0.0, 0.0))
            if vx == 0.0 and vy == 0.0:
                continue
            state = str(face.get("state") or "").lower()
            color = (34, 139, 34, 235) if state == "green" else (211, 47, 47, 235)
            end = view.to_px(lx + vx * arrow_len_cm, ly + vy * arrow_len_cm)
            _arrow(draw, (px, py), end, fill=color, width=arrow_w)

    font = _font(22)
    legend_x = MARGIN_PX
    legend_y = img.height - MARGIN_PX - 72
    legend = [
        ((156, 39, 176, 245), "generated pedestrian-light pole"),
        ((34, 139, 34, 235), "green pedestrian face"),
        ((211, 47, 47, 235), "red pedestrian face"),
    ]
    for i, (color, text) in enumerate(legend):
        y = legend_y + i * 24
        draw.rounded_rectangle([legend_x, y, legend_x + 18, y + 12], radius=2, fill=color)
        draw.text((legend_x + 26, y - 5), text, font=font, fill=STYLE["label_dark"])

    map_out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(map_out, "PNG", optimize=True)
    print(f"wrote {map_out}  px_per_m={view.px_per_m:.4f}")


def update_scenario(
    scenario_dir: Path,
    *,
    phase: str,
    map_out: Path | None,
) -> int:
    roads_path = scenario_dir / "roads.json"
    world_path = scenario_dir / "progen_world_enriched.json"
    with roads_path.open("r", encoding="utf-8") as f:
        roads = json.load(f)["roads"]
    world = json.loads(world_path.read_text(encoding="utf-8"), strict=False)

    map_name = _map_name_from_scenario_dir(scenario_dir)
    lights = build_pedestrian_light_nodes(roads, map_name=map_name, initial_phase=phase)
    upsert_generated_pedestrian_lights(world, lights)
    apply_pedestrian_light_phase(world, phase=phase)

    with world_path.open("w", encoding="utf-8") as f:
        json.dump(world, f, indent=2)
        f.write("\n")

    if map_out is not None:
        _render_pedestrian_light_map(scenario_dir, map_out, world)
    return len(lights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument(
        "--phase",
        default=PHASE_A,
        choices=(PHASE_A, PHASE_B, "left_green_right_red", "left_red_right_green"),
    )
    parser.add_argument(
        "--map-out",
        type=Path,
        default=None,
        help="Optional output PNG path for the 2D light-facing check map.",
    )
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    map_out = args.map_out.resolve() if args.map_out is not None else None
    count = update_scenario(scenario_dir, phase=args.phase, map_out=map_out)
    print(f"wrote {count} pedestrian lights to {scenario_dir / 'progen_world_enriched.json'}")
    if map_out is not None:
        print(f"wrote 2D light-facing map to {map_out}")


if __name__ == "__main__":
    main()
