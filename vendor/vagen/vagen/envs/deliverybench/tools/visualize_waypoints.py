"""
Visualize the DeliveryBench waypoint graph for a given map.

Renders:
- The road skeleton (light grey lines) as the city's "ground truth"
  road network.
- Every waypoint as a coloured dot:
    * intersection (int_*)        — small blue circle
    * dock (dock_*) coloured by the POI kind it hosts:
        building          : grey
        restaurant        : red
        store             : orange
        charging_station  : green
        rest_area         : cyan
        hospital          : magenta
        car_rental        : olive
        bus_station       : brown
- Every waypoint-graph adjacency as a thin line.
- Intersections + non-building POI docks are labeled with their
  waypoint id; building docks are unlabeled to avoid clutter.

Outputs:  <output_dir>/waypoints_<map_name>.png  (and .svg if requested)

Run:
    python -m vagen.envs.deliverybench.tools.visualize_waypoints \
        --map_name medium-city-22

Or programmatically:
    from vagen.envs.deliverybench.tools.visualize_waypoints import render
    render(map_name="medium-city-22", output_dir="/tmp")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Same palette used for POI dots; tweak if you add new kinds.
_POI_COLOURS = {
    "building": "#bdbdbd",
    "restaurant": "#d62728",
    "store": "#ff7f0e",
    "charging_station": "#2ca02c",
    "rest_area": "#17becf",
    "hospital": "#e377c2",
    "car_rental": "#bcbd22",
    "bus_station": "#8c564b",
}
_INTERSECTION_COLOUR = "#1f77b4"
_INTERSECTION_LABEL_COLOUR = "#0b3d75"


def _load_map(base_dir: Path, map_name: str):
    """Load the simulator's Map without dragging the env wrapper along."""
    from ..vlm_delivery.map.map import Map

    cfg = json.loads(
        (base_dir / "vlm_delivery/input/game_mechanics_config.json").read_text()
    )
    m = Map(cfg.get("map", {}))
    m.import_roads(str(base_dir / f"maps/{map_name}/roads.json"))
    m.import_pois(str(base_dir / f"maps/{map_name}/progen_world_enriched.json"))
    return m


def render(
    map_name: str = "medium-city-22",
    output_dir: Optional[str] = None,
    base_dir: Optional[str] = None,
    fig_w: float = 16.0,
    fig_h: float = 16.0,
    label_intersections: bool = True,
    label_non_building_docks: bool = True,
    save_svg: bool = False,
) -> Path:
    """
    Render the waypoint graph for ``map_name`` to a PNG (and SVG).
    Returns the path to the PNG.
    """
    bd = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
    out = Path(output_dir) if output_dir else bd / "tools" / "out"
    out.mkdir(parents=True, exist_ok=True)

    m = _load_map(bd, map_name)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#fafafa")
    ax.set_aspect("equal")

    # 1) Road skeleton (light grey)
    for e in m.graph_skel.edges:
        meta = m.graph_skel.get_edge_meta(e.node1, e.node2) or {}
        if meta.get("kind") != "road":
            continue
        xs = [e.node1.position.x / 100.0, e.node2.position.x / 100.0]
        ys = [e.node1.position.y / 100.0, e.node2.position.y / 100.0]
        ax.plot(xs, ys, color="#cccccc", linewidth=1.5, zorder=1)

    # 2) Waypoint adjacency edges (thin coloured lines)
    for e in m.waypoint_graph.edges:
        u, v = e.node1, e.node2
        # Colour by edge type:
        # dock-dock or dock-intersection along a road: green
        # intersection-intersection (endcap / crosswalk): purple
        u_kind = getattr(u, "waypoint_kind", "")
        v_kind = getattr(v, "waypoint_kind", "")
        if u_kind == "intersection" and v_kind == "intersection":
            colour = "#9467bd"  # purple
        else:
            colour = "#2ca02c"  # green
        xs = [u.position.x / 100.0, v.position.x / 100.0]
        ys = [u.position.y / 100.0, v.position.y / 100.0]
        ax.plot(xs, ys, color=colour, linewidth=0.8, alpha=0.55, zorder=2)

    # 3) Dock waypoints
    by_kind = {}
    for wp_id, node in m.waypoints_by_id.items():
        if not wp_id.startswith("dock_"):
            continue
        kind = getattr(node, "poi_kind", "") or "building"
        by_kind.setdefault(kind, []).append((wp_id, node))

    for kind, items in by_kind.items():
        xs = [n.position.x / 100.0 for _, n in items]
        ys = [n.position.y / 100.0 for _, n in items]
        colour = _POI_COLOURS.get(kind, "#888888")
        size = 22 if kind == "building" else 60
        edgecolor = "#222222" if kind != "building" else "none"
        ax.scatter(
            xs, ys, s=size, c=colour, edgecolors=edgecolor,
            linewidths=0.5, zorder=3, label=f"dock: {kind}",
        )
        if label_non_building_docks and kind != "building":
            for wp_id, node in items:
                ax.annotate(
                    wp_id,
                    (node.position.x / 100.0, node.position.y / 100.0),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6,
                    color="#333333",
                    zorder=4,
                )

    # 4) Intersection waypoints — bigger, blue, always labelled
    int_xs, int_ys = [], []
    for wp_id, node in m.waypoints_by_id.items():
        if not wp_id.startswith("int_"):
            continue
        int_xs.append(node.position.x / 100.0)
        int_ys.append(node.position.y / 100.0)
        if label_intersections:
            ax.annotate(
                wp_id,
                (node.position.x / 100.0, node.position.y / 100.0),
                xytext=(5, -7),
                textcoords="offset points",
                fontsize=7,
                color=_INTERSECTION_LABEL_COLOUR,
                fontweight="bold",
                zorder=5,
            )
    ax.scatter(
        int_xs, int_ys, s=85, c=_INTERSECTION_COLOUR, edgecolors="#0b3d75",
        linewidths=0.9, zorder=4, label="intersection",
    )

    # 5) Title + legend
    n_int = sum(1 for k in m.waypoints_by_id if k.startswith("int_"))
    n_dock = sum(1 for k in m.waypoints_by_id if k.startswith("dock_"))
    ax.set_title(
        f"DeliveryBench waypoint graph — {map_name}\n"
        f"{n_int} intersections + {n_dock} docks = {len(m.waypoints_by_id)} waypoints, "
        f"{len(m.waypoint_graph.edges)} adjacency edges",
        fontsize=12,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)

    # Custom legend (one entry per kind, no duplicates from scatter loop).
    legend_handles = []
    legend_handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_INTERSECTION_COLOUR,
               markeredgecolor="#0b3d75", markersize=9, label="intersection")
    )
    for kind in ("building", "restaurant", "store", "charging_station",
                 "rest_area", "hospital", "car_rental", "bus_station"):
        if kind not in by_kind:
            continue
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=_POI_COLOURS[kind],
                   markeredgecolor="#222222" if kind != "building" else "none",
                   markersize=8 if kind != "building" else 6,
                   label=f"dock: {kind} (×{len(by_kind[kind])})")
        )
    legend_handles += [
        Line2D([0], [0], color="#cccccc", linewidth=2, label="road skeleton"),
        Line2D([0], [0], color="#2ca02c", linewidth=2, label="dock↔* edge"),
        Line2D([0], [0], color="#9467bd", linewidth=2, label="intersection↔intersection"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    png = out / f"waypoints_{map_name}.png"
    fig.savefig(png, dpi=170)
    if save_svg:
        svg = out / f"waypoints_{map_name}.svg"
        fig.savefig(svg)
    plt.close(fig)

    return png


def main(
    map_name: str = "medium-city-22",
    output_dir: Optional[str] = None,
    save_svg: bool = False,
    label_intersections: bool = True,
    label_non_building_docks: bool = True,
):
    out = render(
        map_name=map_name,
        output_dir=output_dir,
        save_svg=save_svg,
        label_intersections=label_intersections,
        label_non_building_docks=label_non_building_docks,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
