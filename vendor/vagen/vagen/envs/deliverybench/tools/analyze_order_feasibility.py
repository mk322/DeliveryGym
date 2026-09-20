"""
Estimate which DeliveryBench order candidates can be completed within a step budget.

This is an oracle-style structural estimate: it does not call an LLM and does
not execute a rollout. It enumerates the same raw restaurant/building candidates
used by OrderManager, binds each candidate pair through Order._bind_nodes_initial,
then counts shortest-path waypoint edges as MOVE actions.

Run:
    python -m vagen.envs.deliverybench.tools.analyze_order_feasibility \
        --map_name small-city-11 \
        --spawn-x-m -17.0 \
        --spawn-y-m 256.58 \
        --max-steps 25
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..vlm_delivery.base.types import Vector
from ..vlm_delivery.entities.order import Order, OrderManager
from ..vlm_delivery.map.map import Map
from ..vlm_delivery.utils.order_feasibility import (
    DEFAULT_NON_MOVE_ACTIONS,
    estimate_order_steps,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def _xy_from_props(node: Dict[str, Any]) -> Tuple[float, float]:
    props = node.get("properties", {}) or {}
    loc = props.get("location", {}) or {}
    return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))


def _node_address(node: Any) -> str:
    return getattr(node, "address", "") or "?"


def _load_map(base_dir: Path, map_name: str) -> Tuple[Map, List[Dict[str, Any]]]:
    game_cfg = _load_json(base_dir / "vlm_delivery/input/game_mechanics_config.json")
    world_path = base_dir / "maps" / map_name / "progen_world_enriched.json"
    roads_path = base_dir / "maps" / map_name / "roads.json"

    city_map = Map(game_cfg.get("map", {}))
    city_map.import_roads(str(roads_path))
    city_map.import_pois(str(world_path))
    world_nodes = _load_json(world_path).get("nodes", [])
    return city_map, world_nodes


def estimate(
    *,
    base_dir: Path,
    map_name: str,
    spawn_x_m: float,
    spawn_y_m: float,
    max_steps: int,
    non_move_actions: int,
) -> Dict[str, Any]:
    city_map, world_nodes = _load_map(base_dir, map_name)
    raw_pickups, raw_dropoffs = OrderManager._collect_candidates(world_nodes)

    rows: List[Dict[str, Any]] = []
    endpoint_pairs = set()

    for pickup_idx, pickup_raw in enumerate(raw_pickups):
        px, py = _xy_from_props(pickup_raw)
        for dropoff_idx, dropoff_raw in enumerate(raw_dropoffs):
            dx, dy = _xy_from_props(dropoff_raw)

            order = Order(
                city_map=city_map,
                pickup_address=Vector(px, py),
                delivery_address=Vector(dx, dy),
                items=[],
            )

            est = estimate_order_steps(
                city_map=city_map,
                start_x_cm=spawn_x_m * 100.0,
                start_y_cm=spawn_y_m * 100.0,
                order=order,
                non_move_actions=non_move_actions,
            )
            approach_moves = est["approach_moves"]
            delivery_moves = est["delivery_moves"]
            total_steps = est["total_steps"]

            endpoint_pairs.add((id(order.pickup_node), id(order.dropoff_node)))
            rows.append(
                {
                    "pickup_idx": pickup_idx,
                    "dropoff_idx": dropoff_idx,
                    "pickup": _node_address(order.pickup_node),
                    "dropoff": _node_address(order.dropoff_node),
                    "approach_moves": approach_moves,
                    "delivery_moves": delivery_moves,
                    "total_steps": total_steps,
                    "approach_m": est["approach_m"],
                    "delivery_m": est["delivery_m"],
                }
            )

    feasible = [r for r in rows if r["total_steps"] <= max_steps]
    thresholds = list(range(max(0, max_steps - 5), max_steps + 6))
    threshold_counts = {str(t): sum(1 for r in rows if r["total_steps"] <= t) for t in thresholds}

    by_pickup = []
    for pickup_idx in range(len(raw_pickups)):
        subset = [r for r in rows if r["pickup_idx"] == pickup_idx]
        if not subset:
            continue
        by_pickup.append(
            {
                "pickup_idx": pickup_idx,
                "pickup": subset[0]["pickup"],
                "feasible": sum(1 for r in subset if r["total_steps"] <= max_steps),
                "total": len(subset),
                "min_steps": min(r["total_steps"] for r in subset),
                "median_steps": statistics.median(r["total_steps"] for r in subset),
            }
        )

    step_values = [r["total_steps"] for r in rows]
    return {
        "map_name": map_name,
        "spawn_m": [spawn_x_m, spawn_y_m],
        "max_steps": max_steps,
        "non_move_actions": non_move_actions,
        "raw_pickups": len(raw_pickups),
        "raw_dropoffs": len(raw_dropoffs),
        "raw_pairs": len(rows),
        "unique_endpoint_pairs": len(endpoint_pairs),
        "feasible_pairs": len(feasible),
        "feasible_share": (len(feasible) / len(rows)) if rows else 0.0,
        "threshold_counts": threshold_counts,
        "step_summary": {
            "min": min(step_values) if step_values else None,
            "median": statistics.median(step_values) if step_values else None,
            "max": max(step_values) if step_values else None,
        },
        "by_pickup": by_pickup,
        "best": sorted(rows, key=lambda r: r["total_steps"])[:20],
    }


def _print_report(result: Dict[str, Any]) -> None:
    print(f"map: {result['map_name']}")
    print(f"spawn_m: {result['spawn_m']}")
    print(f"max_steps: {result['max_steps']}")
    print(f"non_move_actions: {result['non_move_actions']}")
    print(f"raw_pickups: {result['raw_pickups']}")
    print(f"raw_dropoffs: {result['raw_dropoffs']}")
    print(f"raw_pairs: {result['raw_pairs']}")
    print(f"unique_endpoint_pairs: {result['unique_endpoint_pairs']}")
    print(
        "feasible_pairs: "
        f"{result['feasible_pairs']} / {result['raw_pairs']} "
        f"({result['feasible_share']:.2%})"
    )
    ss = result["step_summary"]
    print(f"step_min_median_max: {ss['min']} {ss['median']} {ss['max']}")

    print("\nthreshold_counts:")
    for threshold, count in result["threshold_counts"].items():
        print(f"  <= {threshold}: {count}")

    print("\nby_pickup:")
    for row in result["by_pickup"]:
        print(
            f"  {row['pickup_idx']}: {row['pickup']}  "
            f"{row['feasible']} / {row['total']}  "
            f"min={row['min_steps']} median={row['median_steps']}"
        )

    print("\nbest_20:")
    for row in result["best"]:
        print(
            f"  {row['total_steps']:>2} steps "
            f"({row['approach_moves']}+{row['delivery_moves']} moves)  "
            f"{row['pickup']} -> {row['dropoff']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_name", default="small-city-11")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--spawn-x-m", type=float, default=-17.0)
    parser.add_argument("--spawn-y-m", type=float, default=256.58)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument(
        "--non-move-actions",
        type=int,
        default=DEFAULT_NON_MOVE_ACTIONS,
        help="VIEW_ORDERS + ACCEPT_ORDER + NAVIGATE pickup + PICKUP + NAVIGATE dropoff + DROP_OFF",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = estimate(
        base_dir=Path(args.base_dir).resolve(),
        map_name=args.map_name,
        spawn_x_m=args.spawn_x_m,
        spawn_y_m=args.spawn_y_m,
        max_steps=args.max_steps,
        non_move_actions=args.non_move_actions,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_report(result)


if __name__ == "__main__":
    main()
