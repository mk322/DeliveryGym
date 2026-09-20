#!/usr/bin/env python3
"""Export live Recast topology and physical surface evidence for authoring.

This tool attaches to one freshly launched Paris UE process, creates the same
region-scoped Recast NavMesh used by Pixel Goal rollout, and pages the native
diagnostic API.  It does not infer or bless pedestrian connectivity: the raw
report is the input to the fail-closed trusted-graph builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from embodiedbench.runtime.pixel_goal_paris_poc import ParisPocRegion
from tools.run_pixel_goal_1b_poc import (
    AGENT_TAG,
    AttachedParisGameSession,
    build_paris_setup_request,
)
from tools.run_pixel_goal_m1a import SpearPixelGoalEndpoint, _cleanup_session


DEFAULT_CENTER_CM = (-5200.0, -7100.0, 100.0)
DEFAULT_EXTENT_CM = (3000.0, 5000.0, 300.0)
DEFAULT_SPAWN_CM = (-5934.8349, -9700.0, 90.0)
DEFAULT_BUILDINGS = (
    Path(__file__).resolve().parents[1]
    / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris/buildings.json"
)


def _vec3(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected exactly three numbers")
    return (float(values[0]), float(values[1]), float(values[2]))


def _wait_ready(
    endpoint: SpearPixelGoalEndpoint, timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = endpoint.call("PixelGoal_GetParisPocStatusJson", {})
        if last.get("poc_ready") is True:
            return last
        time.sleep(0.25)
    raise TimeoutError(f"Paris Recast did not become ready: {last}")


def _call_after_recast_stabilizes(
    endpoint: SpearPixelGoalEndpoint,
    method: str,
    request: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Retry only the known one-frame dynamic-NavMesh transition."""

    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = endpoint.call(method, request)
        if last.get("success") is True:
            return last
        if last.get("error") != "paris_recast_not_ready":
            return last
        time.sleep(0.25)
    return last


def _digest_json(value: object) -> str:
    payload = json.dumps(
        value, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_poly_id(poly: dict[str, Any]) -> str:
    geometry = {
        "area_id": poly["area_id"],
        "center_cm": [round(float(value), 1) for value in poly["center_cm"]],
        "vertices_cm": sorted(
            [round(float(value), 1) for value in vertex]
            for vertex in poly["vertices_cm"]
        ),
    }
    return "recast-poly-" + _digest_json(geometry)[:20]


def _sample_segment(
    start: list[float], end: list[float], spacing_cm: float,
) -> list[list[float]]:
    distance = math.dist(start[:2], end[:2])
    steps = max(1, math.ceil(distance / spacing_cm))
    return [
        [
            start[axis] + (end[axis] - start[axis]) * step / steps
            for axis in range(3)
        ]
        for step in range(steps + 1)
    ]


def _crosswalk_centerline_groups(
    catalog: dict[str, Any],
    bounds_min: list[float],
    bounds_max: list[float],
    spacing_cm: float,
    approach_cm: float,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for component in catalog.get("crosswalk_components", []):
        location = component.get("location_cm")
        rotation = component.get("rotation_deg")
        scale = component.get("scale")
        local_min = component.get("local_min_cm")
        local_max = component.get("local_max_cm")
        if not all(isinstance(value, list) and len(value) == 3 for value in (
                location, rotation, scale, local_min, local_max)):
            raise RuntimeError(f"invalid crosswalk transform row: {component}")
        yaw = math.radians(float(rotation[1]))
        # Extend beyond both painted ends.  A crossing is usable only if this
        # exact line proves pavement -> PR_Crossswalk_* -> pavement; a zebra
        # mesh stranded in a vehicle island must not connect two graph sides.
        local_start = [0.0, float(local_min[1]) - approach_cm, 20.0]
        local_end = [0.0, float(local_max[1]) + approach_cm, 20.0]

        def world(local: list[float]) -> list[float]:
            x = local[0] * float(scale[0])
            y = local[1] * float(scale[1])
            return [
                float(location[0]) + math.cos(yaw) * x - math.sin(yaw) * y,
                float(location[1]) + math.sin(yaw) * x + math.cos(yaw) * y,
                float(location[2]) + local[2],
            ]

        start, end = world(local_start), world(local_end)
        points = _sample_segment(start, end, spacing_cm)
        if not any(
            all(bounds_min[axis] <= point[axis] <= bounds_max[axis]
                for axis in range(3))
            for point in points
        ):
            continue
        points = [
            point for point in points
            if all(bounds_min[axis] <= point[axis] <= bounds_max[axis]
                   for axis in range(3))
        ]
        if points:
            groups.append({
                "kind": "crosswalk_centerline",
                "label": component["label"],
                "component_name": component["component_name"],
                "points_cm": points,
            })
    return groups


def _entrance_candidate_groups(
    entrance_catalog: dict[str, Any],
    buildings_path: Path,
    bounds_min: list[float],
    bounds_max: list[float],
    min_gap_cm: float,
    max_gap_cm: float,
    spacing_cm: float,
) -> list[dict[str, Any]]:
    """Generate pavement candidates from actual UE *_Entrance* mesh faces."""

    raw_bytes = buildings_path.read_bytes()
    root = json.loads(raw_bytes)
    records = root.get("buildings") if isinstance(root, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("CityCore building source has no buildings list")
    catalog_rows = entrance_catalog.get("entrance_components")
    if not isinstance(catalog_rows, list):
        raise RuntimeError("UE building entrance catalog has no component list")
    buildings_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) \
                or row.get("deliverybench_navigable") is not True:
            continue
        actor_name = row.get("source_actor")
        actor_label = row.get("source_label")
        if not isinstance(actor_name, str) or not isinstance(actor_label, str):
            continue
        identity = (actor_name, actor_label)
        if identity in buildings_by_identity:
            raise RuntimeError(f"duplicate CityCore actor identity: {identity}")
        buildings_by_identity[identity] = row

    groups: list[dict[str, Any]] = []
    gaps: list[float] = []
    gap = min_gap_cm
    while gap <= max_gap_cm + 1e-6:
        gaps.append(round(gap, 6))
        gap += spacing_cm
    seen_asset_ids: set[str] = set()
    for component in catalog_rows:
        if not isinstance(component, dict):
            continue
        asset_id = component.get("entrance_asset_id")
        actor_name = component.get("actor_name")
        actor_label = component.get("actor_label")
        mesh_path = component.get("static_mesh_path")
        location = component.get("location_cm")
        rotation = component.get("rotation_deg")
        scale = component.get("scale")
        local_min = component.get("local_min_cm")
        local_max = component.get("local_max_cm")
        if not isinstance(asset_id, str) or not asset_id \
                or asset_id in seen_asset_ids \
                or not isinstance(actor_name, str) \
                or not isinstance(actor_label, str) \
                or not isinstance(mesh_path, str) \
                or "entrance" not in mesh_path.casefold() \
                or not all(isinstance(value, list) and len(value) == 3 for value in (
                    location, rotation, scale, local_min, local_max)):
            continue
        seen_asset_ids.add(asset_id)
        building = buildings_by_identity.get((actor_name, actor_label))
        if building is None:
            continue
        yaw = math.radians(float(rotation[1]))
        local_x = (float(local_min[0]) + float(local_max[0])) / 2.0
        local_z = float(local_min[2])
        for face, local_y, sign in (
                ("local_y_min", float(local_min[1]), -1.0),
                ("local_y_max", float(local_max[1]), 1.0)):
            scaled_x = local_x * float(scale[0])
            scaled_y = local_y * float(scale[1])
            door = [
                float(location[0]) + math.cos(yaw) * scaled_x
                    - math.sin(yaw) * scaled_y,
                float(location[1]) + math.sin(yaw) * scaled_x
                    + math.cos(yaw) * scaled_y,
            ]
            # The ground-facing XY of a facade module is unaffected by its
            # local Z for these source assets (pitch/roll are zero). Retain it
            # in the source record for auditability, but sample at pedestrian
            # ground height just like every other graph-authoring point.
            direction = (-math.sin(yaw) * sign, math.cos(yaw) * sign)
            points = [
                [
                    door[0] + direction[0] * distance,
                    door[1] + direction[1] * distance,
                    20.0,
                ]
                for distance in gaps
            ]
            kept = [
                (distance, point) for distance, point in zip(gaps, points)
                if all(bounds_min[axis] <= point[axis] <= bounds_max[axis]
                       for axis in range(3))
            ]
            if not kept:
                continue
            groups.append({
                "kind": "real_building_entrance_candidates",
                "building_id": building.get("id"),
                "source_actor": actor_name,
                "source_label": actor_label,
                "poi_type": building.get("poi_type") or "building",
                "entrance_asset_id": asset_id,
                "entrance_face": face,
                "component_name": component.get("component_name"),
                "instance_index": component.get("instance_index"),
                "static_mesh_path": mesh_path,
                "entrance_transform_location_cm": location,
                "entrance_transform_rotation_deg": rotation,
                "entrance_transform_scale": scale,
                "entrance_local_ground_z_cm": local_z,
                "door_cm": door,
                "candidate_gaps_cm": [item[0] for item in kept],
                "points_cm": [item[1] for item in kept],
            })
    return [{
        "kind": "building_source",
        "path": str(buildings_path.resolve()),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "entrance_geometry_source": "ue_static_mesh_instance_containing_Entrance",
        "matched_ue_actor_count": len({
            (row["source_actor"], row["source_label"])
            for row in groups
        }),
        "groups": groups,
    }]


def _pedestrian_grid_group(
    bounds_min: list[float],
    bounds_max: list[float],
    spacing_cm: float,
) -> tuple[dict[str, Any], dict[tuple[int, int], str]]:
    """A deterministic XY lattice to preserve pavement inside mixed polygons."""

    if spacing_cm <= 0.0:
        raise ValueError("pedestrian grid spacing must be positive")
    min_ix = math.ceil(bounds_min[0] / spacing_cm)
    max_ix = math.floor(bounds_max[0] / spacing_cm)
    min_iy = math.ceil(bounds_min[1] / spacing_cm)
    max_iy = math.floor(bounds_max[1] / spacing_cm)
    grid_ids: list[str] = []
    grid_indices: list[list[int]] = []
    points: list[list[float]] = []
    id_by_index: dict[tuple[int, int], str] = {}
    for ix in range(min_ix, max_ix + 1):
        for iy in range(min_iy, max_iy + 1):
            grid_id = f"recast-grid-{ix}-{iy}"
            id_by_index[(ix, iy)] = grid_id
            grid_ids.append(grid_id)
            grid_indices.append([ix, iy])
            points.append([ix * spacing_cm, iy * spacing_cm, 20.0])
    return ({
        "kind": "pedestrian_recast_grid",
        "spacing_cm": spacing_cm,
        "grid_ids": grid_ids,
        "grid_indices": grid_indices,
        "points_cm": points,
    }, id_by_index)


def _nearest_grid_candidates(
    point: list[float],
    grid_points: dict[str, dict[str, Any]],
    *,
    radius_cm: float,
    limit: int,
) -> list[tuple[float, str, dict[str, Any]]]:
    candidates = [
        (math.dist(point[:2], row["projected_cm"][:2]), grid_id, row)
        for grid_id, row in grid_points.items()
        if math.dist(point[:2], row["projected_cm"][:2]) <= radius_cm
    ]
    return sorted(candidates, key=lambda row: (row[0], row[1]))[:limit]


def _stable_ids_are_nearby(
    first: str,
    second: str,
    neighbours: dict[str, set[str]],
    *,
    max_hops: int = 3,
) -> bool:
    if first == second:
        return True
    seen = {first}
    frontier = {first}
    for _hop in range(max_hops):
        following = {
            candidate
            for stable_id in frontier
            for candidate in neighbours.get(stable_id, ())
            if candidate not in seen
        }
        if second in following:
            return True
        if not following:
            return False
        seen.update(following)
        frontier = following
    return False


def _audited_pavement_segment(
    points: list[dict[str, Any]],
    neighbours: dict[str, set[str]],
) -> bool:
    """Whether a sampled chord is wholly pavement on one Recast chain."""

    if not points:
        return False
    for point in points:
        if point.get("projected") is not True \
                or point.get("surface_class") != "pavement" \
                or not isinstance(point.get("stable_poly_id"), str):
            return False
    return all(
        _stable_ids_are_nearby(
            str(first["stable_poly_id"]),
            str(second["stable_poly_id"]),
            neighbours,
        )
        for first, second in zip(points, points[1:])
    )


def _grid_components(
    grid_ids: list[str],
    edge_groups: list[dict[str, Any]],
    neighbours: dict[str, set[str]],
) -> dict[str, int]:
    adjacency = {grid_id: set() for grid_id in grid_ids}
    for group in edge_groups:
        if not _audited_pavement_segment(group["points"], neighbours):
            continue
        first = str(group["a_grid_id"])
        second = str(group["b_grid_id"])
        adjacency[first].add(second)
        adjacency[second].add(first)
    component_by_id: dict[str, int] = {}
    component = 0
    for start in sorted(adjacency):
        if start in component_by_id:
            continue
        component_by_id[start] = component
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current]):
                if neighbour not in component_by_id:
                    component_by_id[neighbour] = component
                    stack.append(neighbour)
        component += 1
    return component_by_id


def _visibility_connector_groups(
    safe_grid_points: dict[str, dict[str, Any]],
    component_by_id: dict[str, int],
    *,
    radius_cm: float,
    candidates_per_component_pair: int,
    sample_spacing_cm: float,
) -> list[dict[str, Any]]:
    """Propose short chords only where the fixed lattice split pavement.

    Proposals are not trusted here. UE audits every sample below, and the
    compiler retains only all-pavement, Recast-continuous results.
    """

    candidates: dict[tuple[int, int], list[tuple[float, str, str]]] = {}
    rows = sorted(safe_grid_points.items())
    for index, (first_id, first) in enumerate(rows):
        first_component = component_by_id[first_id]
        first_xy = first["projected_cm"][:2]
        for second_id, second in rows[index + 1:]:
            second_component = component_by_id[second_id]
            if first_component == second_component:
                continue
            distance = math.dist(first_xy, second["projected_cm"][:2])
            if distance > radius_cm:
                continue
            pair = tuple(sorted((first_component, second_component)))
            candidates.setdefault(pair, []).append(
                (distance, first_id, second_id))

    groups: list[dict[str, Any]] = []
    for pair in sorted(candidates):
        for rank, (distance, first_id, second_id) in enumerate(
                sorted(candidates[pair])[:candidates_per_component_pair], 1):
            groups.append({
                "kind": "recast_grid_visibility_connector",
                "a_grid_id": first_id,
                "b_grid_id": second_id,
                "base_component_pair": list(pair),
                "candidate_rank": rank,
                "direct_cm": distance,
                "sample_spacing_cm": sample_spacing_cm,
                "points_cm": _sample_segment(
                    safe_grid_points[first_id]["projected_cm"],
                    safe_grid_points[second_id]["projected_cm"],
                    sample_spacing_cm,
                ),
            })
    return groups


def _audit_point_groups(
    endpoint: SpearPixelGoalEndpoint,
    groups: list[dict[str, Any]],
    *,
    batch_size: int,
    projection_extent_cm: float,
    stable_id_by_ref: dict[str, str],
    compact: bool = False,
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
        request: dict[str, Any] = {
            "agent_tag": AGENT_TAG,
            "points_cm": batch,
            "projection_extent_cm": projection_extent_cm,
        }
        if compact:
            request["compact"] = True
        response = endpoint.call(
            "PixelGoal_AuditPedestrianPointsJson", request,
        )
        if response.get("success") is not True \
                or not isinstance(response.get("points"), list) \
                or len(response["points"]) != len(batch):
            raise RuntimeError(f"pedestrian point audit failed at {offset}: {response}")
        audited.extend(response["points"])
        print(f"audited pedestrian points {len(audited)}/{len(flat)}", flush=True)
    for group, start, end in slices:
        group["points"] = audited[start:end]
        for point in group["points"]:
            ref = point.get("poly_ref")
            point["stable_poly_id"] = stable_id_by_ref.get(ref)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log = output_dir / "events.jsonl"
    event_log.unlink(missing_ok=True)

    center = _vec3(args.center_cm)
    extent = _vec3(args.extent_cm)
    spawn = _vec3(args.spawn_cm)
    bounds_min = [center[index] - extent[index] for index in range(3)]
    bounds_max = [center[index] + extent[index] for index in range(3)]
    region = ParisPocRegion(
        name="paris_trusted_pedestrian_graph_authoring",
        agent_spawn_cm=spawn,
        agent_yaw_deg=args.spawn_yaw_deg,
        nav_bounds_center_cm=center,
        nav_bounds_extent_cm=extent,
    )

    session = AttachedParisGameSession(args.spear_config)
    play_started = False
    try:
        session.begin_play()
        play_started = True
        endpoint = SpearPixelGoalEndpoint(session, event_log)
        setup = endpoint.call(
            "PixelGoal_SetupParisPocJson", build_paris_setup_request(region),
        )
        if setup.get("success") is not True:
            raise RuntimeError(f"Paris setup failed: {setup}")
        status = _wait_ready(endpoint, args.navmesh_timeout_s)
        catalog = _call_after_recast_stabilizes(
            endpoint,
            "PixelGoal_GetCrosswalkCatalogJson",
            {"agent_tag": AGENT_TAG},
            args.navmesh_timeout_s,
        )
        if catalog.get("success") is not True:
            raise RuntimeError(f"crosswalk catalog failed: {catalog}")
        entrance_catalog = _call_after_recast_stabilizes(
            endpoint,
            "PixelGoal_GetBuildingEntranceCatalogJson",
            {"agent_tag": AGENT_TAG},
            args.navmesh_timeout_s,
        )
        if entrance_catalog.get("success") is not True:
            raise RuntimeError(
                f"building entrance catalog failed: {entrance_catalog}")

        polygons: list[dict[str, Any]] = []
        total: int | None = None
        offset = 0
        while total is None or offset < total:
            page = endpoint.call(
                "PixelGoal_GetRecastPolygonsJson",
                {
                    "agent_tag": AGENT_TAG,
                    "bounds_min_cm": bounds_min,
                    "bounds_max_cm": bounds_max,
                    "offset": offset,
                    "limit": args.page_size,
                },
            )
            if page.get("success") is not True:
                raise RuntimeError(f"Recast page {offset} failed: {page}")
            page_total = page.get("total_polygons")
            if isinstance(page_total, bool) or not isinstance(page_total, int):
                raise RuntimeError(f"Recast page {offset} has invalid total: {page}")
            if total is None:
                total = page_total
            elif total != page_total:
                raise RuntimeError(
                    f"Recast polygon count changed during export: {total} -> {page_total}",
                )
            rows = page.get("polygons")
            if not isinstance(rows, list) or not rows:
                if offset != total:
                    raise RuntimeError(
                        f"Recast page {offset} returned no polygons before EOF",
                    )
                break
            polygons.extend(rows)
            offset += len(rows)
            print(f"exported Recast polygons {offset}/{total}", flush=True)

        poly_refs = [row.get("poly_ref") for row in polygons]
        if total is None or len(polygons) != total:
            raise RuntimeError(
                f"incomplete Recast export: expected {total}, got {len(polygons)}",
            )
        if any(not isinstance(ref, str) or not ref for ref in poly_refs) \
                or len(poly_refs) != len(set(poly_refs)):
            raise RuntimeError("Recast export contains missing or duplicate poly refs")

        polygons_by_ref = {row["poly_ref"]: row for row in polygons}
        stable_id_by_ref = {
            ref: _stable_poly_id(row) for ref, row in polygons_by_ref.items()
        }
        if len(stable_id_by_ref) != len(set(stable_id_by_ref.values())):
            raise RuntimeError("Recast export contains duplicate stable polygon geometry")
        for ref, poly in polygons_by_ref.items():
            poly["stable_poly_id"] = stable_id_by_ref[ref]
            poly["neighbor_stable_ids"] = sorted(
                stable_id_by_ref[neighbor]
                for neighbor in poly["neighbor_refs"]
                if neighbor in stable_id_by_ref
            )
        stable_neighbours = {
            poly["stable_poly_id"]: set(poly["neighbor_stable_ids"])
            for poly in polygons_by_ref.values()
        }
        edge_groups: list[dict[str, Any]] = []
        for poly_ref in sorted(polygons_by_ref, key=int):
            poly = polygons_by_ref[poly_ref]
            for neighbor_ref in sorted(poly["neighbor_refs"], key=int):
                if neighbor_ref not in polygons_by_ref \
                        or int(neighbor_ref) <= int(poly_ref):
                    continue
                edge_groups.append({
                    "kind": "recast_adjacency",
                    "a_poly_ref": poly_ref,
                    "b_poly_ref": neighbor_ref,
                    "a_stable_poly_id": stable_id_by_ref[poly_ref],
                    "b_stable_poly_id": stable_id_by_ref[neighbor_ref],
                    "sample_spacing_cm": args.edge_sample_spacing_cm,
                    "points_cm": _sample_segment(
                        poly["center_cm"],
                        polygons_by_ref[neighbor_ref]["center_cm"],
                        args.edge_sample_spacing_cm,
                    ),
                })
        crosswalk_groups = _crosswalk_centerline_groups(
            catalog, bounds_min, bounds_max,
            args.crosswalk_sample_spacing_cm, args.crosswalk_approach_cm,
        )
        building_source_rows = _entrance_candidate_groups(
            entrance_catalog, Path(args.buildings), bounds_min, bounds_max,
            args.min_entrance_gap_cm, args.max_entrance_gap_cm,
            args.entrance_sample_spacing_cm,
        )
        entrance_groups = building_source_rows[0].pop("groups")
        all_groups = [*edge_groups, *crosswalk_groups, *entrance_groups]
        _audit_point_groups(
            endpoint,
            all_groups,
            batch_size=args.point_batch_size,
            projection_extent_cm=args.point_projection_extent_cm,
            stable_id_by_ref=stable_id_by_ref,
        )

        grid_group, grid_id_by_index = _pedestrian_grid_group(
            bounds_min, bounds_max, args.pedestrian_grid_spacing_cm,
        )
        _audit_point_groups(
            endpoint,
            [grid_group],
            batch_size=args.point_batch_size,
            projection_extent_cm=args.point_projection_extent_cm,
            stable_id_by_ref=stable_id_by_ref,
            compact=True,
        )
        safe_grid_points: dict[str, dict[str, Any]] = {}
        grid_index_by_id: dict[str, tuple[int, int]] = {}
        for grid_id, indices, point in zip(
                grid_group["grid_ids"],
                grid_group["grid_indices"],
                grid_group["points"]):
            projected = point.get("projected_cm")
            if point.get("projected") is True \
                    and point.get("surface_class") == "pavement" \
                    and point.get("stable_poly_id") \
                    and isinstance(projected, list) \
                    and len(projected) == 3 \
                    and all(
                        bounds_min[axis] <= projected[axis] <= bounds_max[axis]
                        for axis in range(3)):
                safe_grid_points[grid_id] = point
                grid_index_by_id[grid_id] = (int(indices[0]), int(indices[1]))

        grid_edge_groups: list[dict[str, Any]] = []
        for grid_id in sorted(safe_grid_points):
            ix, iy = grid_index_by_id[grid_id]
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                neighbour_id = grid_id_by_index.get((ix + dx, iy + dy))
                if neighbour_id not in safe_grid_points:
                    continue
                grid_edge_groups.append({
                    "kind": "recast_grid_adjacency",
                    "a_grid_id": grid_id,
                    "b_grid_id": neighbour_id,
                    "sample_spacing_cm": args.grid_edge_sample_spacing_cm,
                    "points_cm": _sample_segment(
                        safe_grid_points[grid_id]["projected_cm"],
                        safe_grid_points[neighbour_id]["projected_cm"],
                        args.grid_edge_sample_spacing_cm,
                    ),
                })
        _audit_point_groups(
            endpoint,
            grid_edge_groups,
            batch_size=args.point_batch_size,
            projection_extent_cm=args.point_projection_extent_cm,
            stable_id_by_ref=stable_id_by_ref,
            compact=True,
        )

        component_by_grid_id = _grid_components(
            sorted(safe_grid_points), grid_edge_groups, stable_neighbours,
        )
        visibility_groups = _visibility_connector_groups(
            safe_grid_points,
            component_by_grid_id,
            radius_cm=args.grid_visibility_radius_cm,
            candidates_per_component_pair=(
                args.grid_visibility_candidates_per_component_pair),
            sample_spacing_cm=args.grid_edge_sample_spacing_cm,
        )
        _audit_point_groups(
            endpoint,
            visibility_groups,
            batch_size=args.point_batch_size,
            projection_extent_cm=args.point_projection_extent_cm,
            stable_id_by_ref=stable_id_by_ref,
            compact=True,
        )

        grid_connector_groups: list[dict[str, Any]] = []
        for group in crosswalk_groups:
            for side, point in (
                    ("start", group["points"][0]),
                    ("end", group["points"][-1])):
                projected = point.get("projected_cm")
                if point.get("surface_class") != "pavement" \
                        or not isinstance(projected, list) \
                        or not all(
                            bounds_min[axis] <= projected[axis] <= bounds_max[axis]
                            for axis in range(3)):
                    continue
                for rank, (distance, grid_id, grid_point) in enumerate(
                        _nearest_grid_candidates(
                            projected, safe_grid_points,
                            radius_cm=args.grid_connector_radius_cm,
                            limit=args.grid_connector_candidates,
                        ), 1):
                    grid_connector_groups.append({
                        "kind": "crosswalk_to_recast_grid",
                        "crosswalk_label": group["label"],
                        "side": side,
                        "candidate_rank": rank,
                        "direct_cm": distance,
                        "grid_id": grid_id,
                        "sample_spacing_cm": args.grid_edge_sample_spacing_cm,
                        "points_cm": _sample_segment(
                            projected, grid_point["projected_cm"],
                            args.grid_edge_sample_spacing_cm,
                        ),
                    })
        for group in entrance_groups:
            for candidate_gap, point in zip(
                    group["candidate_gaps_cm"], group["points"]):
                projected = point.get("projected_cm")
                if point.get("surface_class") != "pavement" \
                        or not isinstance(projected, list) \
                        or not all(
                            bounds_min[axis] <= projected[axis] <= bounds_max[axis]
                            for axis in range(3)):
                    continue
                for rank, (distance, grid_id, grid_point) in enumerate(
                        _nearest_grid_candidates(
                            projected, safe_grid_points,
                            radius_cm=args.grid_connector_radius_cm,
                            limit=args.grid_connector_candidates,
                        ), 1):
                    grid_connector_groups.append({
                        "kind": "entrance_to_recast_grid",
                        "building_id": group["building_id"],
                        "entrance_asset_id": group["entrance_asset_id"],
                        "entrance_face": group["entrance_face"],
                        "static_mesh_path": group["static_mesh_path"],
                        "candidate_gap_cm": candidate_gap,
                        "actual_door_anchor_gap_cm": math.dist(
                            group["door_cm"], projected[:2]),
                        "anchor_cm": projected,
                        "candidate_rank": rank,
                        "direct_cm": distance,
                        "grid_id": grid_id,
                        "sample_spacing_cm": args.grid_edge_sample_spacing_cm,
                        "points_cm": _sample_segment(
                            projected, grid_point["projected_cm"],
                            args.grid_edge_sample_spacing_cm,
                        ),
                    })
        _audit_point_groups(
            endpoint,
            grid_connector_groups,
            batch_size=args.point_batch_size,
            projection_extent_cm=args.point_projection_extent_cm,
            stable_id_by_ref=stable_id_by_ref,
            compact=True,
        )

        # Polygon-centre connectors belonged to the superseded sparse graph.
        # The v2 graph uses only the denser, independently audited grid
        # connectors above. Keeping the old diagnostics would duplicate tens
        # of thousands of samples and can reference centres outside a bounded
        # polygon export, while contributing no route edge.
        connector_groups: list[dict[str, Any]] = []

        topology_payload = [
            {
                "poly_ref": row["poly_ref"],
                "center_cm": row["center_cm"],
                "vertices_cm": row["vertices_cm"],
                "neighbor_refs": row["neighbor_refs"],
                "area_id": row["area_id"],
            }
            for row in polygons
        ]
        stable_topology_payload = [
            {
                "stable_poly_id": row["stable_poly_id"],
                "center_cm": [round(float(value), 1)
                              for value in row["center_cm"]],
                "vertices_cm": sorted(
                    [round(float(value), 1) for value in vertex]
                    for vertex in row["vertices_cm"]
                ),
                "area_id": row["area_id"],
                "neighbor_stable_ids": row["neighbor_stable_ids"],
            }
            for row in sorted(polygons, key=lambda row: row["stable_poly_id"])
        ]
        report = {
            "schema": "embodiedbench/live-recast-surface-audit/v1",
            "scene": "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints",
            "agent_tag": AGENT_TAG,
            "region": {
                "name": region.name,
                "nav_bounds_center_cm": list(center),
                "nav_bounds_extent_cm": list(extent),
            },
            "setup": setup,
            "ready_status": status,
            "crosswalk_catalog": catalog,
            "building_entrance_catalog": entrance_catalog,
            "recast_topology_sha256": _digest_json(topology_payload),
            "stable_recast_topology_sha256": _digest_json(
                stable_topology_payload),
            "polygon_count": len(polygons),
            "polygons": polygons,
            "edge_surface_audits": edge_groups,
            "crosswalk_centerline_audits": crosswalk_groups,
            "building_source": building_source_rows[0],
            "entrance_candidate_audits": entrance_groups,
            "pedestrian_recast_grid": grid_group,
            "grid_edge_surface_audits": grid_edge_groups,
            "grid_visibility_surface_audits": visibility_groups,
            "grid_connector_surface_audits": grid_connector_groups,
            "connector_surface_audits": connector_groups,
        }
        output_path = output_dir / "live_recast_surface_audit.json"
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output_path}", flush=True)
        return report
    finally:
        _cleanup_session(
            session,
            launch_mode="attach",
            shutdown_attached_editor=args.shutdown_attached_editor,
            play_started=play_started,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--center-cm", type=float, nargs=3, default=DEFAULT_CENTER_CM,
    )
    parser.add_argument(
        "--extent-cm", type=float, nargs=3, default=DEFAULT_EXTENT_CM,
    )
    parser.add_argument(
        "--spawn-cm", type=float, nargs=3, default=DEFAULT_SPAWN_CM,
    )
    parser.add_argument("--spawn-yaw-deg", type=float, default=90.0)
    parser.add_argument("--page-size", type=int, choices=range(1, 251), default=250)
    parser.add_argument("--point-batch-size", type=int, choices=range(1, 513), default=512)
    parser.add_argument("--point-projection-extent-cm", type=float, default=25.0)
    parser.add_argument("--edge-sample-spacing-cm", type=float, default=25.0)
    parser.add_argument("--crosswalk-sample-spacing-cm", type=float, default=5.0)
    parser.add_argument("--crosswalk-approach-cm", type=float, default=150.0)
    parser.add_argument("--crosswalk-connector-radius-cm", type=float, default=600.0)
    parser.add_argument("--crosswalk-connector-candidates", type=int, default=12)
    parser.add_argument("--pedestrian-grid-spacing-cm", type=float, default=100.0)
    parser.add_argument("--grid-edge-sample-spacing-cm", type=float, default=25.0)
    parser.add_argument("--grid-visibility-radius-cm", type=float, default=450.0)
    parser.add_argument(
        "--grid-visibility-candidates-per-component-pair",
        type=int,
        default=8,
    )
    parser.add_argument("--grid-connector-radius-cm", type=float, default=250.0)
    parser.add_argument("--grid-connector-candidates", type=int, default=8)
    parser.add_argument("--buildings", default=str(DEFAULT_BUILDINGS))
    parser.add_argument("--min-entrance-gap-cm", type=float, default=50.0)
    parser.add_argument("--max-entrance-gap-cm", type=float, default=250.0)
    parser.add_argument("--entrance-sample-spacing-cm", type=float, default=25.0)
    parser.add_argument("--navmesh-timeout-s", type=float, default=300.0)
    parser.add_argument("--shutdown-attached-editor", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
