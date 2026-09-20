#!/usr/bin/env python3
"""Revalidate a V2 pedestrian order pool against a fresh live UE process.

This is a fail-closed preflight, not a model rollout. It rebuilds the same
region-scoped NavMesh used by rollout, exports the rollout pawn's Recast
polygons again, compares stable topology, inventories the real crosswalk and
entrance meshes again, and re-audits every graph node and edge against the
physical surface below it. It also resolves the exact order that the two
models will later share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from embodiedbench.runtime.pixel_goal_paris_poc import ParisPocRegion
from tools.audit_trusted_pedestrian_graph_live import (
    _call_after_recast_stabilizes,
    _digest_json,
    _sample_segment,
    _stable_poly_id,
    _wait_ready,
)
from tools.build_trusted_pedestrian_order_pool import (
    _topology_chain_is_continuous,
    crosswalk_corridor_rejection,
)
from tools.pixel_goal_order_pool import (
    CROSSWALK_ASSET_PREFIX,
    POOL_SCHEMA,
    OrderConstraints,
    load_validated_delivery_pool,
    resolve_delivery_scenario,
    scenario_report,
)
from tools.run_pixel_goal_1b_poc import (
    AGENT_TAG,
    AttachedParisGameSession,
    build_paris_setup_request,
)
from tools.run_pixel_goal_front_rear_delivery import (
    DEFAULT_VALIDATED_ORDER_POOL,
)
from tools.run_pixel_goal_m1a import SpearPixelGoalEndpoint, _cleanup_session


SCENE = "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _load_certified_audit(pool) -> tuple[dict[str, Any], Path]:
    certification = pool.pedestrian_graph
    if certification is None:
        raise ValueError("live trusted validation requires a V2 pedestrian pool")
    path = Path(certification.audit_path)
    if not path.is_absolute():
        path = (pool.path.parent / path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != certification.audit_sha256:
        raise ValueError("certified audit SHA-256 changed before live validation")
    audit = json.loads(raw)
    if not isinstance(audit, dict):
        raise ValueError("certified pedestrian audit is not a JSON object")
    return audit, path


def _export_live_topology(
    endpoint: SpearPixelGoalEndpoint,
    pool,
    *,
    page_size: int,
) -> tuple[
    list[dict[str, Any]], dict[str, str], dict[str, set[str]], str,
]:
    center = pool.region.nav_bounds_center_cm
    extent = pool.region.nav_bounds_extent_cm
    bounds_min = [center[index] - extent[index] for index in range(3)]
    bounds_max = [center[index] + extent[index] for index in range(3)]
    polygons: list[dict[str, Any]] = []
    total: int | None = None
    offset = 0
    while total is None or offset < total:
        page = endpoint.call("PixelGoal_GetRecastPolygonsJson", {
            "agent_tag": AGENT_TAG,
            "bounds_min_cm": bounds_min,
            "bounds_max_cm": bounds_max,
            "offset": offset,
            "limit": page_size,
        })
        if page.get("success") is not True:
            raise RuntimeError(f"fresh Recast polygon page {offset} failed: {page}")
        page_total = page.get("total_polygons")
        rows = page.get("polygons")
        if isinstance(page_total, bool) or not isinstance(page_total, int) \
                or not isinstance(rows, list):
            raise RuntimeError(f"fresh Recast polygon page {offset} is malformed")
        if total is None:
            total = page_total
        elif total != page_total:
            raise RuntimeError(
                f"fresh Recast polygon count changed: {total} -> {page_total}")
        if not rows:
            if offset != total:
                raise RuntimeError("fresh Recast export ended before its declared total")
            break
        polygons.extend(rows)
        offset += len(rows)
        print(f"fresh Recast polygons {offset}/{total}", flush=True)
    if total is None or len(polygons) != total:
        raise RuntimeError("fresh Recast polygon export is incomplete")

    refs = [row.get("poly_ref") for row in polygons]
    if any(not isinstance(ref, str) or not ref for ref in refs) \
            or len(refs) != len(set(refs)):
        raise RuntimeError("fresh Recast export has duplicate or missing refs")
    stable_by_ref = {
        str(row["poly_ref"]): _stable_poly_id(row) for row in polygons
    }
    if len(stable_by_ref) != len(set(stable_by_ref.values())):
        raise RuntimeError("fresh Recast export has duplicate stable geometry")
    for row in polygons:
        row["stable_poly_id"] = stable_by_ref[str(row["poly_ref"])]
        row["neighbor_stable_ids"] = sorted(
            stable_by_ref[str(ref)]
            for ref in row["neighbor_refs"]
            if str(ref) in stable_by_ref
        )
    stable_neighbours = {
        str(row["stable_poly_id"]): set(row["neighbor_stable_ids"])
        for row in polygons
    }
    stable_payload = [
        {
            "stable_poly_id": row["stable_poly_id"],
            "center_cm": [round(float(value), 1) for value in row["center_cm"]],
            "vertices_cm": sorted(
                [round(float(value), 1) for value in vertex]
                for vertex in row["vertices_cm"]
            ),
            "area_id": row["area_id"],
            "neighbor_stable_ids": row["neighbor_stable_ids"],
        }
        for row in sorted(polygons, key=lambda item: item["stable_poly_id"])
    ]
    return (
        polygons,
        stable_by_ref,
        stable_neighbours,
        _digest_json(stable_payload),
    )


def _catalog_rows(
    catalog: Mapping[str, Any], key: str, identity: str,
) -> dict[str, Mapping[str, Any]]:
    rows = catalog.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"catalog has no {key}")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"catalog {key} contains a non-object")
        value = row.get(identity)
        if not isinstance(value, str) or not value or value in indexed:
            raise ValueError(f"catalog {key} has invalid {identity}")
        indexed[value] = row
    return indexed


def _compare_live_catalogs(
    pool,
    certified_audit: Mapping[str, Any],
    live_crosswalks: Mapping[str, Any],
    live_entrances: Mapping[str, Any],
) -> dict[str, Any]:
    old_crosswalks = _catalog_rows(
        certified_audit["crosswalk_catalog"],
        "crosswalk_components", "label",
    )
    new_crosswalks = _catalog_rows(
        live_crosswalks, "crosswalk_components", "label")
    old_entrances = _catalog_rows(
        certified_audit["building_entrance_catalog"],
        "entrance_components", "entrance_asset_id",
    )
    new_entrances = _catalog_rows(
        live_entrances, "entrance_components", "entrance_asset_id")
    used_crosswalks = sorted({
        edge.source.split(":", 2)[1]
        for edge in pool.edges if edge.kind == "marked_crossing"
    })
    errors: list[str] = []
    if set(old_crosswalks) != set(new_crosswalks):
        errors.append("live crosswalk identity set differs from certification")
    if set(old_entrances) != set(new_entrances):
        errors.append("live entrance identity set differs from certification")
    for label in used_crosswalks:
        if not label.startswith(CROSSWALK_ASSET_PREFIX):
            errors.append(f"route uses a non-PR_Crossswalk asset: {label}")
        elif old_crosswalks.get(label) != new_crosswalks.get(label):
            errors.append(f"live transform changed for route crosswalk {label}")
    for stop in pool.stops:
        asset_id = str(stop.entrance_asset_id)
        old = old_entrances.get(asset_id)
        new = new_entrances.get(asset_id)
        if old is None or new is None or old != new:
            errors.append(f"live entrance asset changed for stop {stop.id}")
        elif new.get("static_mesh_path") != stop.entrance_static_mesh_path:
            errors.append(f"live entrance mesh changed for stop {stop.id}")
    return {
        "certified_crosswalk_count": len(old_crosswalks),
        "live_crosswalk_count": len(new_crosswalks),
        "certified_entrance_count": len(old_entrances),
        "live_entrance_count": len(new_entrances),
        "used_crosswalks": used_crosswalks,
        "certified_crosswalk_catalog_sha256": _canonical_sha256(old_crosswalks),
        "live_crosswalk_catalog_sha256": _canonical_sha256(new_crosswalks),
        "certified_entrance_catalog_sha256": _canonical_sha256(old_entrances),
        "live_entrance_catalog_sha256": _canonical_sha256(new_entrances),
        "errors": errors,
        "passed": not errors,
    }


def _node_z_from_certified_audit(
    pool, audit: Mapping[str, Any],
) -> dict[str, float]:
    out: dict[str, float] = {}
    grid = audit.get("pedestrian_recast_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("certified audit has no pedestrian grid")
    grid_ids, grid_points = grid.get("grid_ids"), grid.get("points")
    if not isinstance(grid_ids, list) or not isinstance(grid_points, list) \
            or len(grid_ids) != len(grid_points):
        raise ValueError("certified pedestrian grid is malformed")
    for node_id, point in zip(grid_ids, grid_points):
        projected = point.get("projected_cm") if isinstance(point, Mapping) else None
        if node_id in pool.nodes_by_id and isinstance(projected, list) \
                and len(projected) == 3:
            out[str(node_id)] = float(projected[2])

    for group in audit.get("crosswalk_centerline_audits", []):
        if not isinstance(group, Mapping) or not isinstance(group.get("label"), str) \
                or not isinstance(group.get("points"), list):
            continue
        for index, point in enumerate(group["points"]):
            node_id = f"crosswalk:{group['label']}:{index:03d}"
            projected = point.get("projected_cm") if isinstance(point, Mapping) else None
            if node_id in pool.nodes_by_id and isinstance(projected, list) \
                    and len(projected) == 3:
                out[node_id] = float(projected[2])

    connectors = audit.get("grid_connector_surface_audits")
    for stop in pool.stops:
        matches = [
            row for row in connectors
            if isinstance(row, Mapping)
            and row.get("kind") == "entrance_to_recast_grid"
            and row.get("building_id") == stop.building_id
            and row.get("entrance_asset_id") == stop.entrance_asset_id
            and row.get("entrance_face") == stop.entrance_face
            and isinstance(row.get("anchor_cm"), list)
            and len(row["anchor_cm"]) == 3
            and math.dist(
                pool.nodes_by_id[stop.handover_node_id].position,
                (float(row["anchor_cm"][0]), float(row["anchor_cm"][1])),
            ) <= 0.1
        ] if isinstance(connectors, list) else []
        if not matches:
            raise ValueError(f"certified audit lost entrance anchor {stop.id}")
        out[stop.handover_node_id] = float(matches[0]["anchor_cm"][2])
    missing = sorted(set(pool.nodes_by_id) - set(out))
    if missing:
        raise ValueError(f"certified audit has no Z for graph nodes: {missing[:5]}")
    return out


def _audit_groups(
    endpoint: SpearPixelGoalEndpoint,
    groups: list[dict[str, Any]],
    *,
    stable_by_ref: Mapping[str, str],
    batch_size: int,
    projection_extent_cm: float,
) -> None:
    flat: list[list[float]] = []
    slices: list[tuple[dict[str, Any], int, int]] = []
    for group in groups:
        points = group.pop("points_cm")
        start = len(flat)
        flat.extend(points)
        slices.append((group, start, len(flat)))
    audited: list[dict[str, Any]] = []
    for offset in range(0, len(flat), batch_size):
        batch = flat[offset:offset + batch_size]
        response = endpoint.call("PixelGoal_AuditPedestrianPointsJson", {
            "agent_tag": AGENT_TAG,
            "points_cm": batch,
            "projection_extent_cm": projection_extent_cm,
            "compact": True,
        })
        rows = response.get("points")
        if response.get("success") is not True or not isinstance(rows, list) \
                or len(rows) != len(batch):
            raise RuntimeError(
                f"fresh pedestrian surface audit failed at {offset}: {response}")
        audited.extend(rows)
        print(f"fresh pedestrian points {len(audited)}/{len(flat)}", flush=True)
    for group, start, end in slices:
        group["points"] = audited[start:end]
        for point in group["points"]:
            point["stable_poly_id"] = stable_by_ref.get(str(point.get("poly_ref")))


def _surface_summary(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    surfaces = Counter(str(point.get("surface_class")) for point in points)
    errors = [
        float(point["projection_error_cm"])
        for point in points
        if isinstance(point.get("projection_error_cm"), (int, float))
        and not isinstance(point.get("projection_error_cm"), bool)
    ]
    return {
        "point_count": len(points),
        "surfaces": dict(sorted(surfaces.items())),
        "max_projection_error_cm": max(errors) if errors else None,
    }


def _audit_live_graph_surfaces(
    endpoint: SpearPixelGoalEndpoint,
    pool,
    certified_audit: Mapping[str, Any],
    *,
    stable_by_ref: Mapping[str, str],
    stable_neighbours: Mapping[str, set[str]],
    batch_size: int,
    projection_extent_cm: float,
) -> dict[str, Any]:
    certification = pool.pedestrian_graph
    assert certification is not None
    node_z = _node_z_from_certified_audit(pool, certified_audit)
    node_ids = sorted(pool.nodes_by_id)
    node_group = {
        "kind": "all_graph_nodes",
        "points_cm": [
            [
                pool.nodes_by_id[node_id].x_cm,
                pool.nodes_by_id[node_id].y_cm,
                node_z[node_id],
            ]
            for node_id in node_ids
        ],
    }
    sidewalk_groups: list[dict[str, Any]] = []
    for edge in pool.edges:
        if edge.kind != "sidewalk":
            continue
        first, second = pool.nodes_by_id[edge.a], pool.nodes_by_id[edge.b]
        sidewalk_groups.append({
            "a": edge.a,
            "b": edge.b,
            "kind": edge.kind,
            "source": edge.source,
            "points_cm": _sample_segment(
                [first.x_cm, first.y_cm, node_z[edge.a]],
                [second.x_cm, second.y_cm, node_z[edge.b]],
                certification.grid_edge_sample_spacing_cm,
            ),
        })

    used_crosswalks = {
        edge.source.split(":", 2)[1]
        for edge in pool.edges if edge.kind == "marked_crossing"
    }
    certified_crosswalks = {
        str(group.get("label")): group
        for group in certified_audit.get("crosswalk_centerline_audits", [])
        if isinstance(group, Mapping)
    }
    crosswalk_groups: list[dict[str, Any]] = []
    for label in sorted(used_crosswalks):
        certified = certified_crosswalks.get(label)
        points = certified.get("points") if isinstance(certified, Mapping) else None
        if not isinstance(points, list) or not points:
            raise ValueError(f"certified audit has no corridor for {label}")
        point_inputs = [
            list(point["input_cm"])
            for point in points
            if isinstance(point, Mapping)
            and isinstance(point.get("input_cm"), list)
            and len(point["input_cm"]) == 3
        ]
        if len(point_inputs) != len(points):
            raise ValueError(f"certified corridor {label} has malformed samples")
        crosswalk_groups.append({
            "kind": "fresh_crosswalk_corridor",
            "label": label,
            "points_cm": point_inputs,
        })

    groups = [node_group, *sidewalk_groups, *crosswalk_groups]
    _audit_groups(
        endpoint,
        groups,
        stable_by_ref=stable_by_ref,
        batch_size=batch_size,
        projection_extent_cm=projection_extent_cm,
    )

    errors: list[str] = []
    node_rows: list[dict[str, Any]] = []
    for node_id, point in zip(node_ids, node_group["points"]):
        node = pool.nodes_by_id[node_id]
        surface = point.get("surface_class")
        passed = (
            point.get("projected") is True
            and isinstance(point.get("stable_poly_id"), str)
            and (
                surface == "pavement"
                if node.role != "crossing"
                else surface in ("pavement", "marked_crossing", "road")
            )
        )
        if not passed:
            errors.append(
                f"node {node_id} ({node.role}) is not on an allowed live surface: "
                f"{surface!r}")
        node_rows.append({
            "id": node_id,
            "role": node.role,
            "surface": surface,
            "stable_poly_id": point.get("stable_poly_id"),
            "projection_error_cm": point.get("projection_error_cm"),
            "passed": passed,
        })

    sidewalk_rows: list[dict[str, Any]] = []
    for group in sidewalk_groups:
        points = group["points"]
        reason = None
        if not points or any(
            point.get("projected") is not True
            or point.get("surface_class") != "pavement"
            or not isinstance(point.get("stable_poly_id"), str)
            for point in points
        ):
            reason = "non_pavement_or_unprojected_sample"
        elif not _topology_chain_is_continuous(points, stable_neighbours):
            reason = "recast_topology_discontinuity"
        if reason is not None:
            errors.append(f"sidewalk edge {group['a']}--{group['b']}: {reason}")
        sidewalk_rows.append({
            "a": group["a"],
            "b": group["b"],
            "source": group["source"],
            **_surface_summary(points),
            "rejection": reason,
            "passed": reason is None,
        })

    crosswalk_rows: list[dict[str, Any]] = []
    for group in crosswalk_groups:
        rejection = crosswalk_corridor_rejection(
            group,
            stable_neighbours,
            max_road_transition_cm=(
                certification.max_crosswalk_road_transition_cm),
            expected_sample_spacing_cm=(
                certification.crosswalk_sample_spacing_cm),
        )
        if rejection is not None:
            errors.append(f"crosswalk {group['label']}: {rejection}")
        crosswalk_rows.append({
            "label": group["label"],
            **_surface_summary(group["points"]),
            "surface_sequence": [
                point.get("surface_class") for point in group["points"]
            ],
            "rejection": rejection,
            "passed": rejection is None,
        })

    full_evidence = {
        "nodes": node_group["points"],
        "sidewalk_edges": [group["points"] for group in sidewalk_groups],
        "crosswalks": [group["points"] for group in crosswalk_groups],
    }
    return {
        "node_count": len(node_rows),
        "sidewalk_edge_count": len(sidewalk_rows),
        "marked_crossing_edge_count": sum(
            edge.kind == "marked_crossing" for edge in pool.edges),
        "crosswalk_corridor_count": len(crosswalk_rows),
        "fresh_evidence_sha256": _canonical_sha256(full_evidence),
        "nodes": node_rows,
        "sidewalk_edges": sidewalk_rows,
        "crosswalk_corridors": crosswalk_rows,
        "errors": errors,
        "passed": not errors,
    }


def validate_pool_live(args: argparse.Namespace) -> dict[str, Any]:
    pool = load_validated_delivery_pool(args.order_pool)
    if pool.schema != POOL_SCHEMA or pool.pedestrian_graph is None:
        raise ValueError("only trusted pedestrian pool V2 can pass this validator")
    certified_audit, certified_audit_path = _load_certified_audit(pool)
    constraints = OrderConstraints(
        min_delivery_cm=args.min_delivery_m * 100.0,
        max_delivery_cm=args.max_delivery_m * 100.0,
        min_turns=args.min_route_turns,
        require_different_streets=args.require_different_streets,
        require_marked_crossing=args.require_marked_crossing,
    )
    scenario = resolve_delivery_scenario(
        pool,
        mode="fixed",
        seed=args.seed,
        constraints=constraints,
        spawn_id=args.spawn_id,
        pickup_id=args.pickup_id,
        dropoff_id=args.dropoff_id,
    )
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log = output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)

    spawn_node = pool.nodes_by_id[scenario.spawn.node_id]
    region = ParisPocRegion(
        name=pool.region.name,
        agent_spawn_cm=(
            spawn_node.x_cm, spawn_node.y_cm, scenario.spawn.z_cm),
        agent_yaw_deg=scenario.spawn.yaw_deg,
        nav_bounds_center_cm=pool.region.nav_bounds_center_cm,
        nav_bounds_extent_cm=pool.region.nav_bounds_extent_cm,
    )
    session = AttachedParisGameSession(args.spear_config)
    play_started = False
    try:
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        setup = endpoint.call(
            "PixelGoal_SetupParisPocJson", build_paris_setup_request(region))
        if setup.get("success") is not True:
            raise RuntimeError(f"Paris setup failed: {setup}")
        status = _wait_ready(endpoint, args.navmesh_timeout_s)
        live_crosswalks = _call_after_recast_stabilizes(
            endpoint,
            "PixelGoal_GetCrosswalkCatalogJson",
            {"agent_tag": AGENT_TAG},
            args.navmesh_timeout_s,
        )
        live_entrances = _call_after_recast_stabilizes(
            endpoint,
            "PixelGoal_GetBuildingEntranceCatalogJson",
            {"agent_tag": AGENT_TAG},
            args.navmesh_timeout_s,
        )
        if live_crosswalks.get("success") is not True:
            raise RuntimeError(f"fresh crosswalk catalog failed: {live_crosswalks}")
        if live_entrances.get("success") is not True:
            raise RuntimeError(f"fresh entrance catalog failed: {live_entrances}")
        polygons, stable_by_ref, stable_neighbours, topology_sha = \
            _export_live_topology(endpoint, pool, page_size=args.page_size)
        topology_errors: list[str] = []
        if len(polygons) != pool.pedestrian_graph.polygon_count:
            topology_errors.append(
                "fresh Recast polygon count differs from certification")
        if topology_sha != pool.pedestrian_graph.stable_recast_topology_sha256:
            topology_errors.append(
                "fresh stable Recast topology differs from certification")
        catalogs = _compare_live_catalogs(
            pool, certified_audit, live_crosswalks, live_entrances)
        surfaces = _audit_live_graph_surfaces(
            endpoint,
            pool,
            certified_audit,
            stable_by_ref=stable_by_ref,
            stable_neighbours=stable_neighbours,
            batch_size=args.point_batch_size,
            projection_extent_cm=args.point_projection_extent_cm,
        )
    finally:
        _cleanup_session(
            session,
            launch_mode="attach",
            shutdown_attached_editor=args.shutdown_attached_editor,
            play_started=play_started,
        )

    report = {
        "schema": "embodiedbench/live-trusted-pedestrian-pool-validation/v2",
        "scene": SCENE,
        "pool": {
            "path": str(pool.path),
            "sha256": pool.sha256,
            "profile": pool.profile,
            "certified_audit_path": str(certified_audit_path),
            "certified_audit_sha256": pool.pedestrian_graph.audit_sha256,
        },
        "setup": setup,
        "status": status,
        "topology": {
            "certified_polygon_count": pool.pedestrian_graph.polygon_count,
            "live_polygon_count": len(polygons),
            "certified_stable_recast_topology_sha256": (
                pool.pedestrian_graph.stable_recast_topology_sha256),
            "live_stable_recast_topology_sha256": topology_sha,
            "errors": topology_errors,
            "passed": not topology_errors,
        },
        "catalogs": catalogs,
        "surfaces": surfaces,
        "scenario": scenario_report(pool, scenario),
    }
    report["passed"] = (
        setup.get("success") is True
        and status.get("poc_ready") is True
        and status.get("agent_on_navmesh") is True
        and report["topology"]["passed"]
        and catalogs["passed"]
        and surfaces["passed"]
    )
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise RuntimeError(f"live trusted-pool validation failed; see {report_path}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-config", required=True)
    parser.add_argument("--order-pool", default=str(DEFAULT_VALIDATED_ORDER_POOL))
    parser.add_argument("--output", required=True)
    parser.add_argument("--spawn-id", default="spawn-near-citycore-building-0529")
    parser.add_argument("--pickup-id", default="stop-citycore-building-0529")
    parser.add_argument("--dropoff-id", default="stop-citycore-building-0535")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-delivery-m", type=float, default=30.0)
    parser.add_argument("--max-delivery-m", type=float, default=80.0)
    parser.add_argument("--min-route-turns", type=int, default=1)
    parser.add_argument(
        "--require-different-streets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-marked-crossing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--page-size", type=int, choices=range(1, 251), default=250)
    parser.add_argument(
        "--point-batch-size", type=int, choices=range(1, 513), default=512)
    parser.add_argument("--point-projection-extent-cm", type=float, default=25.0)
    parser.add_argument("--navmesh-timeout-s", type=float, default=300.0)
    parser.add_argument("--shutdown-attached-editor", action="store_true")
    return parser


def main() -> int:
    report = validate_pool_live(_parser().parse_args())
    print(json.dumps({
        "passed": report["passed"],
        "scenario_id": report["scenario"]["scenario_id"],
        "node_count": report["surfaces"]["node_count"],
        "sidewalk_edge_count": report["surfaces"]["sidewalk_edge_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
