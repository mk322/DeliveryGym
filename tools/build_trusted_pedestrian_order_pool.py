#!/usr/bin/env python3
"""Compile a fail-closed delivery pool from one live UE Recast audit.

Connectivity comes only from the rollout pawn's UE Recast NavMesh plus the
physical surface classifications retained by ``audit_trusted_pedestrian_graph_live``.
CityCore is consulted only after that graph exists, to attach stable street
names, address numbers and source-authored building entrances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from embodiedbench.compiler.road_network import (
    RoadNetwork,
    build_road_network,
    project_to_polyline,
)
from tools.pixel_goal_order_pool import (
    AGENT_NAV_DATA_SOURCE,
    AUDIT_SCHEMA,
    CITYCORE_SEMANTIC_USE,
    CROSSWALK_ASSET_PREFIX,
    PEDESTRIAN_CONNECTIVITY_SOURCE,
    POOL_SCHEMA,
)


DEFAULT_MAP_DIR = (
    REPO_ROOT / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
)
PAVEMENT_SURFACES = frozenset(("pavement",))
CROSSWALK_SURFACES = frozenset(("pavement", "marked_crossing"))
SCENE = "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _xy(point: Mapping[str, Any]) -> tuple[float, float]:
    projected = point.get("projected_cm")
    if not isinstance(projected, Sequence) or len(projected) != 3:
        raise ValueError("audited point has no projected 3D position")
    return (float(projected[0]), float(projected[1]))


def _input_xyz(point: Mapping[str, Any]) -> tuple[float, float, float]:
    value = point.get("input_cm")
    if not isinstance(value, Sequence) or len(value) != 3:
        value = point.get("projected_cm")
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("audited point has no input/projected position")
    return (float(value[0]), float(value[1]), float(value[2]))


def _point_is_projected(point: Mapping[str, Any]) -> bool:
    return (
        point.get("projected") is True
        and isinstance(point.get("stable_poly_id"), str)
        and bool(point["stable_poly_id"])
    )


def _all_surfaces(
    points: Sequence[Mapping[str, Any]], allowed: frozenset[str],
) -> bool:
    return bool(points) and all(
        _point_is_projected(point) and point.get("surface_class") in allowed
        for point in points
    )


def _within_topology_hops(
    start: str,
    end: str,
    neighbours: Mapping[str, set[str]],
    max_hops: int,
) -> bool:
    if start == end:
        return True
    seen = {start}
    frontier = {start}
    for _hop in range(max_hops):
        following = {
            candidate
            for node_id in frontier
            for candidate in neighbours.get(node_id, ())
            if candidate not in seen
        }
        if end in following:
            return True
        if not following:
            return False
        seen.update(following)
        frontier = following
    return False


def _topology_chain_is_continuous(
    points: Sequence[Mapping[str, Any]],
    neighbours: Mapping[str, set[str]],
    *,
    max_skipped_polygons: int = 3,
) -> bool:
    if not points or not all(_point_is_projected(point) for point in points):
        return False
    stable_ids = [str(point["stable_poly_id"]) for point in points]
    return all(
        _within_topology_hops(
            first, second, neighbours, max_skipped_polygons)
        for first, second in zip(stable_ids, stable_ids[1:])
    )


def _road_run_cm(
    points: Sequence[Mapping[str, Any]], first: int, last: int,
) -> float:
    distance = 0.0
    if first > 0:
        distance += 0.5 * math.dist(
            _input_xyz(points[first - 1])[:2], _input_xyz(points[first])[:2])
    for index in range(first, last):
        distance += math.dist(
            _input_xyz(points[index])[:2], _input_xyz(points[index + 1])[:2])
    if last + 1 < len(points):
        distance += 0.5 * math.dist(
            _input_xyz(points[last])[:2], _input_xyz(points[last + 1])[:2])
    return distance


def crosswalk_corridor_rejection(
    group: Mapping[str, Any],
    neighbours: Mapping[str, set[str]],
    *,
    max_road_transition_cm: float,
    expected_sample_spacing_cm: float,
) -> str | None:
    """Return why an asset cannot connect pavements, or ``None`` if certified."""

    label = group.get("label")
    points = group.get("points")
    if not isinstance(label, str) or not label.startswith(CROSSWALK_ASSET_PREFIX):
        return "not_an_exact_crosswalk_asset"
    if not isinstance(points, list) or len(points) < 3:
        return "missing_crosswalk_samples"
    if not all(isinstance(point, Mapping) for point in points):
        return "malformed_crosswalk_samples"
    if not _topology_chain_is_continuous(points, neighbours):
        return "crosswalk_not_continuous_on_recast"
    surfaces = [point.get("surface_class") for point in points]
    if surfaces[0] != "pavement" or surfaces[-1] != "pavement":
        return "crosswalk_does_not_end_on_two_pavements"
    if "marked_crossing" not in surfaces:
        return "crosswalk_has_no_marked_surface"
    if any(surface not in (*CROSSWALK_SURFACES, "road") for surface in surfaces):
        return "crosswalk_surface_unverified"
    for point in points:
        if point.get("surface_class") == "marked_crossing" \
                and label not in (point.get("crosswalk_labels") or []):
            return "marked_surface_belongs_to_another_asset"
    for first_point, second_point in zip(points, points[1:]):
        if math.dist(
                _input_xyz(first_point)[:2], _input_xyz(second_point)[:2]) \
                > expected_sample_spacing_cm + 0.2:
            return "crosswalk_sampling_gap"

    index = 0
    while index < len(points):
        if surfaces[index] != "road":
            index += 1
            continue
        first = index
        while index + 1 < len(points) and surfaces[index + 1] == "road":
            index += 1
        last = index
        touches_marking = (
            (first > 0 and surfaces[first - 1] == "marked_crossing")
            or (last + 1 < len(points)
                and surfaces[last + 1] == "marked_crossing")
        )
        if not touches_marking:
            return "road_run_is_not_a_crosswalk_kerb_seam"
        if _road_run_cm(points, first, last) > max_road_transition_cm + 1e-6:
            return "crosswalk_kerb_seam_too_wide"
        index += 1
    return None


def _base_components(
    node_ids: Iterable[str], edges: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        a, b = str(edge["a"]), str(edge["b"])
        adjacency[a].add(b)
        adjacency[b].add(a)
    component_by_node: dict[str, int] = {}
    component = 0
    for start in sorted(adjacency):
        if start in component_by_node:
            continue
        stack = [start]
        component_by_node[start] = component
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current]):
                if neighbour not in component_by_node:
                    component_by_node[neighbour] = component
                    stack.append(neighbour)
        component += 1
    return component_by_node


def _decimated_indices(
    points: Sequence[Mapping[str, Any]], max_edge_cm: float,
) -> list[int]:
    if len(points) < 2:
        return list(range(len(points)))
    indices = [0]
    distance = 0.0
    for index in range(1, len(points)):
        segment = math.dist(
            _xy(points[index - 1]), _xy(points[index]))
        if distance + segment > max_edge_cm and index - 1 > indices[-1]:
            indices.append(index - 1)
            distance = segment
        else:
            distance += segment
    if indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)
    return indices


def _nearest_street_index(
    network: RoadNetwork, position: tuple[float, float],
) -> int:
    if not network.streets:
        raise ValueError("CityCore has no streets for address/name semantics")
    return min(
        network.streets,
        key=lambda street: (
            project_to_polyline(position, street.polyline)[0], street.index),
    ).index


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _add_edge(
    edges: dict[tuple[str, str], dict[str, Any]],
    a: str,
    b: str,
    *,
    kind: str,
    source: str,
) -> None:
    if a == b:
        raise ValueError(f"refusing self-loop from {source}")
    key = _edge_key(a, b)
    row = {"a": key[0], "b": key[1], "kind": kind,
           "verified": True, "source": source}
    previous = edges.get(key)
    if previous is not None and previous != row:
        raise ValueError(f"conflicting evidence for edge {key}: {previous} vs {row}")
    edges[key] = row


def _graph_components(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[set[str]]:
    adjacency = {node_id: set() for node_id in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components: list[set[str]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        component = {start}
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _relative_path(path: Path, output_path: Path) -> str:
    return os.path.relpath(path.resolve(), output_path.resolve().parent)


def build_pool_document(
    audit: Mapping[str, Any],
    *,
    audit_path: Path,
    audit_sha256: str,
    map_dir: Path,
    output_path: Path,
    max_road_transition_cm: float = 50.0,
    crosswalk_sample_spacing_cm: float = 5.0,
    max_crosswalk_graph_edge_cm: float = 250.0,
    max_entrance_connector_cm: float = 400.0,
    min_door_anchor_cm: float = 50.0,
    max_door_anchor_cm: float = 250.0,
    require_crossing_component: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"unsupported live audit schema: {audit.get('schema')!r}")
    if audit.get("scene") != SCENE:
        raise ValueError("live audit is not from the configured Paris scene")
    if audit.get("agent_tag") != "PixelGoalParisPocAgent":
        raise ValueError("live audit is not bound to the rollout pawn")
    polygons = audit.get("polygons")
    edge_audits = audit.get("edge_surface_audits")
    crosswalk_audits = audit.get("crosswalk_centerline_audits")
    pedestrian_grid = audit.get("pedestrian_recast_grid")
    grid_edge_audits = audit.get("grid_edge_surface_audits")
    visibility_audits = audit.get("grid_visibility_surface_audits")
    connector_audits = audit.get("grid_connector_surface_audits")
    entrance_audits = audit.get("entrance_candidate_audits")
    if not all(isinstance(value, list) for value in (
            polygons, edge_audits, crosswalk_audits,
            grid_edge_audits, visibility_audits,
            connector_audits, entrance_audits)) \
            or not isinstance(pedestrian_grid, Mapping):
        raise ValueError("live audit is missing graph authoring arrays")
    if len(polygons) != audit.get("polygon_count"):
        raise ValueError("live audit polygon export is incomplete")

    polygon_by_id = {
        str(row["stable_poly_id"]): row for row in polygons
        if isinstance(row, Mapping) and isinstance(row.get("stable_poly_id"), str)
    }
    if len(polygon_by_id) != len(polygons):
        raise ValueError("live audit has duplicate/missing stable polygon ids")
    neighbours = {
        stable_id: set(str(item) for item in row.get("neighbor_stable_ids", []))
        for stable_id, row in polygon_by_id.items()
    }
    if any(
        neighbour not in polygon_by_id
        for rows in neighbours.values() for neighbour in rows
    ):
        raise ValueError("live audit topology references an unavailable polygon")

    network = build_road_network(map_dir, map_name="citycore-paris")
    compiled_addresses = {address.building_id: address for address in network.addresses}
    if len(compiled_addresses) != len(network.addresses):
        raise ValueError("compiled CityCore contains duplicate building addresses")

    nodes: dict[str, dict[str, Any]] = {}
    node_z_cm: dict[str, float] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    grid_ids = pedestrian_grid.get("grid_ids")
    grid_points = pedestrian_grid.get("points")
    if not isinstance(grid_ids, list) or not isinstance(grid_points, list) \
            or len(grid_ids) != len(grid_points) or not grid_ids:
        raise ValueError("live audit Recast grid is incomplete")
    for grid_id, point in zip(grid_ids, grid_points):
        if not isinstance(grid_id, str) or not isinstance(point, Mapping) \
                or not _point_is_projected(point) \
                or point.get("surface_class") != "pavement":
            continue
        projected = point.get("projected_cm")
        if not isinstance(projected, Sequence) or len(projected) != 3:
            continue
        position = (float(projected[0]), float(projected[1]))
        nodes[grid_id] = {
            "id": grid_id,
            "x_cm": position[0],
            "y_cm": position[1],
            "street_index": _nearest_street_index(network, position),
            "role": "recast_grid",
        }
        node_z_cm[grid_id] = float(projected[2])

    rejected_adjacencies: dict[str, int] = {}
    for row in grid_edge_audits:
        if not isinstance(row, Mapping):
            continue
        a = str(row.get("a_grid_id") or "")
        b = str(row.get("b_grid_id") or "")
        points = row.get("points")
        reason = None
        if a not in nodes or b not in nodes:
            reason = "unsafe_endpoint"
        elif not isinstance(points, list) \
                or not all(isinstance(point, Mapping) for point in points) \
                or not _all_surfaces(points, PAVEMENT_SURFACES):
            reason = "non_pedestrian_surface"
        elif not _topology_chain_is_continuous(points, neighbours):
            reason = "topology_discontinuity"
        if reason is not None:
            rejected_adjacencies[reason] = rejected_adjacencies.get(reason, 0) + 1
            continue
        _add_edge(
            edges, a, b, kind="sidewalk",
            source=f"ue_recast_grid:{min(a, b)}:{max(a, b)}",
        )

    accepted_visibility_edges = 0
    rejected_visibility_edges: dict[str, int] = {}
    for row in visibility_audits:
        if not isinstance(row, Mapping):
            continue
        a = str(row.get("a_grid_id") or "")
        b = str(row.get("b_grid_id") or "")
        points = row.get("points")
        reason = None
        if row.get("kind") != "recast_grid_visibility_connector":
            reason = "unexpected_kind"
        elif a not in nodes or b not in nodes:
            reason = "unsafe_endpoint"
        elif not isinstance(points, list) \
                or not all(isinstance(point, Mapping) for point in points) \
                or not _all_surfaces(points, PAVEMENT_SURFACES):
            reason = "non_pavement_surface"
        elif not _topology_chain_is_continuous(points, neighbours):
            reason = "topology_discontinuity"
        if reason is not None:
            rejected_visibility_edges[reason] = (
                rejected_visibility_edges.get(reason, 0) + 1)
            continue
        _add_edge(
            edges, a, b, kind="sidewalk",
            source=f"ue_recast_visibility:{min(a, b)}:{max(a, b)}",
        )
        accepted_visibility_edges += 1

    base_component = _base_components(nodes, edges.values())
    crosswalk_connectors: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in connector_audits:
        if not isinstance(row, Mapping) \
                or row.get("kind") != "crosswalk_to_recast_grid":
            continue
        label, side = row.get("crosswalk_label"), row.get("side")
        target = row.get("grid_id")
        points = row.get("points")
        if not isinstance(label, str) or side not in ("start", "end") \
                or target not in nodes or not isinstance(points, list) \
                or not all(isinstance(point, Mapping) for point in points) \
                or not _all_surfaces(points, PAVEMENT_SURFACES) \
                or not _topology_chain_is_continuous(points, neighbours):
            continue
        crosswalk_connectors.setdefault((label, str(side)), []).append(row)

    accepted_crosswalks: list[str] = []
    rejected_crosswalks: dict[str, str] = {}
    for group in sorted(crosswalk_audits, key=lambda row: str(row.get("label"))):
        if not isinstance(group, Mapping):
            continue
        label = str(group.get("label") or "")
        rejection = crosswalk_corridor_rejection(
            group, neighbours,
            max_road_transition_cm=max_road_transition_cm,
            expected_sample_spacing_cm=crosswalk_sample_spacing_cm,
        )
        if rejection is not None:
            rejected_crosswalks[label] = rejection
            continue
        start_rows = crosswalk_connectors.get((label, "start"), [])
        end_rows = crosswalk_connectors.get((label, "end"), [])
        connector_pairs = [
            (start, end)
            for start in start_rows for end in end_rows
            if start.get("grid_id") != end.get("grid_id")
            and base_component.get(str(start.get("grid_id")))
                != base_component.get(str(end.get("grid_id")))
        ]
        if not connector_pairs:
            rejected_crosswalks[label] = "does_not_bridge_two_sidewalk_components"
            continue
        points = group["points"]
        indices = _decimated_indices(points, max_crosswalk_graph_edge_cm)
        line_node_ids: list[str] = []
        for point_index in indices:
            point = points[point_index]
            node_id = f"crosswalk:{label}:{point_index:03d}"
            position = _xy(point)
            nodes[node_id] = {
                "id": node_id,
                "x_cm": position[0],
                "y_cm": position[1],
                "street_index": _nearest_street_index(network, position),
                "role": "crossing",
            }
            node_z_cm[node_id] = float(point["projected_cm"][2])
            line_node_ids.append(node_id)
        for first_id, second_id, first_index, second_index in zip(
                line_node_ids, line_node_ids[1:], indices, indices[1:]):
            _add_edge(
                edges, first_id, second_id, kind="marked_crossing",
                source=f"ue_crosswalk:{label}:samples-{first_index}-{second_index}",
            )
        # Retain every independently surface-audited pavement connector.  A
        # Recast polygon centre graph is intentionally sparse; limiting an end
        # to only its closest centre can strand a real door on the same piece
        # of pavement.  These do not create guessed connectivity: each retained
        # chord was itself sampled and classified above.
        for side, connector_rows, line_node_id in (
                ("start", start_rows, line_node_ids[0]),
                ("end", end_rows, line_node_ids[-1])):
            seen_targets: set[str] = set()
            for connector in sorted(
                    connector_rows,
                    key=lambda row: (
                        float(row.get("direct_cm", math.inf)),
                        int(row.get("candidate_rank", 1_000_000)),
                        str(row.get("grid_id")),
                    )):
                target = str(connector["grid_id"])
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                rank = int(connector.get("candidate_rank", 0))
                _add_edge(
                    edges, line_node_id, target, kind="sidewalk",
                    source=(f"ue_crosswalk_connector:{label}:{side}:"
                            f"rank-{rank}:{target}"),
                )
        accepted_crosswalks.append(label)

    entrance_groups: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in entrance_audits:
        if not isinstance(row, Mapping) \
                or row.get("kind") != "real_building_entrance_candidates" \
                or not isinstance(row.get("building_id"), str) \
                or not isinstance(row.get("entrance_asset_id"), str) \
                or row.get("entrance_face") not in (
                    "local_y_min", "local_y_max") \
                or not isinstance(row.get("static_mesh_path"), str) \
                or "entrance" not in str(row["static_mesh_path"]).casefold():
            continue
        key = (
            str(row["building_id"]),
            str(row["entrance_asset_id"]),
            str(row["entrance_face"]),
        )
        if key in entrance_groups:
            raise ValueError(f"duplicate real entrance audit group: {key}")
        entrance_groups[key] = row
    entrance_point_by_gap: dict[
        tuple[str, str, str], dict[float, Mapping[str, Any]]
    ] = {}
    for key, group in entrance_groups.items():
        gaps, points = group.get("candidate_gaps_cm"), group.get("points")
        if isinstance(gaps, list) and isinstance(points, list) and len(gaps) == len(points):
            entrance_point_by_gap[key] = {
                round(float(gap), 6): point
                for gap, point in zip(gaps, points)
                if isinstance(point, Mapping)
            }

    entrance_candidates: dict[
        str, list[tuple[tuple[str, str, str], Mapping[str, Any]]]
    ] = {}
    for row in connector_audits:
        if not isinstance(row, Mapping) \
                or row.get("kind") != "entrance_to_recast_grid":
            continue
        building_id = row.get("building_id")
        asset_id = row.get("entrance_asset_id")
        entrance_face = row.get("entrance_face")
        target = row.get("grid_id")
        points = row.get("points")
        gap = float(row.get("actual_door_anchor_gap_cm", math.inf))
        direct_cm = float(row.get("direct_cm", math.inf))
        key = (str(building_id), str(asset_id), str(entrance_face))
        group = entrance_groups.get(key)
        if not isinstance(building_id, str) \
                or not isinstance(asset_id, str) \
                or entrance_face not in ("local_y_min", "local_y_max") \
                or group is None \
                or row.get("static_mesh_path") != group.get("static_mesh_path") \
                or target not in nodes \
                or not min_door_anchor_cm <= gap <= max_door_anchor_cm \
                or direct_cm > max_entrance_connector_cm \
                or not isinstance(points, list) \
                or not all(isinstance(point, Mapping) for point in points) \
                or not _all_surfaces(points, PAVEMENT_SURFACES) \
                or not _topology_chain_is_continuous(points, neighbours):
            continue
        entrance_candidates.setdefault(building_id, []).append((key, row))

    stops: list[dict[str, Any]] = []
    entrance_target_by_stop: dict[str, str] = {}
    for building_id in sorted(entrance_candidates):
        address = compiled_addresses.get(building_id)
        if address is None:
            continue
        entrance_key, connector = min(
            entrance_candidates[building_id],
            key=lambda item: (
                float(item[1]["actual_door_anchor_gap_cm"]),
                float(item[1].get("candidate_gap_cm", math.inf)),
                float(item[1].get("direct_cm", math.inf)),
                item[0],
                str(item[1]["grid_id"]),
            ),
        )
        group = entrance_groups[entrance_key]
        anchor = connector.get("anchor_cm")
        door = group.get("door_cm")
        if not isinstance(anchor, Sequence) or len(anchor) != 3 \
                or not isinstance(door, Sequence) or len(door) != 2:
            continue
        node_id = f"entrance:{building_id}"
        stop_id = "stop-" + building_id.casefold().replace("_", "-")
        nodes[node_id] = {
            "id": node_id,
            "x_cm": float(anchor[0]),
            "y_cm": float(anchor[1]),
            "street_index": address.street_index,
            "role": "entrance_connector",
        }
        node_z_cm[node_id] = float(anchor[2])
        target = str(connector["grid_id"])
        _add_edge(
            edges, node_id, target, kind="sidewalk",
            source=(f"ue_entrance_connector:{building_id}:"
                    f"{entrance_key[1]}:{entrance_key[2]}:"
                    f"gap-{float(connector['actual_door_anchor_gap_cm']):.3f}:"
                    f"{target}"),
        )
        candidate_gap = round(float(connector.get("candidate_gap_cm")), 6)
        source_point = entrance_point_by_gap.get(entrance_key, {}).get(candidate_gap)
        projection_error = (
            float(source_point.get("projection_error_cm", 25.0))
            if isinstance(source_point, Mapping) else 25.0
        )
        stops.append({
            "id": stop_id,
            "building_id": building_id,
            "entrance_asset_id": entrance_key[1],
            "entrance_face": entrance_key[2],
            "entrance_static_mesh_path": group["static_mesh_path"],
            "street_index": address.street_index,
            "street_name": address.street_name,
            "number": address.number,
            "door_cm": [float(door[0]), float(door[1])],
            "handover_node_id": node_id,
            "poi_type": address.poi_type,
            "roles": ["pickup", "dropoff"],
            "entrance_source": (
                "live UE *_Entrance* static-mesh instance face; "
                f"pavement anchor at {float(connector['actual_door_anchor_gap_cm']):.1f} cm"
            ),
            "entrance_verified": True,
            "surface": "pavement",
            "navmesh_verified": True,
            "max_navmesh_adjustment_cm": projection_error,
        })
        entrance_target_by_stop[stop_id] = target

    stop_by_anchor = {stop["handover_node_id"]: stop for stop in stops}
    components = _graph_components(nodes, edges)
    eligible: list[tuple[set[str], list[dict[str, Any]], int, int]] = []
    component_summary: list[dict[str, Any]] = []
    for component in components:
        component_stops = [
            stop for anchor, stop in stop_by_anchor.items() if anchor in component
        ]
        crossing_edges = sum(
            1 for key, edge in edges.items()
            if key[0] in component and edge["kind"] == "marked_crossing"
        )
        natural_crossing_pairs = sum(
            1
            for index, first in enumerate(component_stops)
            for second in component_stops[index + 1:]
            if base_component.get(entrance_target_by_stop[first["id"]])
                != base_component.get(entrance_target_by_stop[second["id"]])
        )
        component_summary.append({
            "nodes": len(component),
            "stops": [stop["id"] for stop in component_stops],
            "crosswalks": sorted({
                str(edge["source"]).split(":", 2)[1]
                for key, edge in edges.items()
                if key[0] in component and edge["kind"] == "marked_crossing"
            }),
            "marked_crossing_edges": crossing_edges,
            "natural_crossing_stop_pairs": natural_crossing_pairs,
        })
        if len(component_stops) >= 2 \
                and (natural_crossing_pairs > 0
                     or not require_crossing_component):
            eligible.append((
                component, component_stops,
                crossing_edges, natural_crossing_pairs,
            ))
    if not eligible:
        requirement = " with a certified crossing" if require_crossing_component else ""
        raise ValueError(f"no connected pedestrian component has two stops{requirement}")
    selected_component, selected_stops, crossing_edge_count, \
        natural_crossing_pair_count = max(
        eligible,
        key=lambda item: (item[3], len(item[1]), item[2], len(item[0])),
    )
    nodes = {
        node_id: row for node_id, row in nodes.items()
        if node_id in selected_component
    }
    node_z_cm = {
        node_id: value for node_id, value in node_z_cm.items()
        if node_id in selected_component
    }
    edges = {
        key: row for key, row in edges.items()
        if key[0] in selected_component and key[1] in selected_component
    }
    stops = sorted(selected_stops, key=lambda row: row["id"])

    spawns: list[dict[str, Any]] = []
    used_spawn_nodes: set[str] = set()
    for stop in stops:
        target = entrance_target_by_stop[stop["id"]]
        if target in used_spawn_nodes or target not in nodes:
            continue
        used_spawn_nodes.add(target)
        origin = (nodes[target]["x_cm"], nodes[target]["y_cm"])
        destination = (
            nodes[stop["handover_node_id"]]["x_cm"],
            nodes[stop["handover_node_id"]]["y_cm"],
        )
        yaw = math.degrees(math.atan2(
            destination[1] - origin[1], destination[0] - origin[0],
        )) % 360.0
        spawns.append({
            "id": "spawn-near-" + stop["id"].removeprefix("stop-"),
            "node_id": target,
            "z_cm": node_z_cm[target] + 70.0,
            "yaw_deg": yaw,
            "verified": True,
            "max_navmesh_adjustment_cm": 25.0,
        })
    if not spawns:
        raise ValueError("selected pedestrian component has no safe spawn")

    node_rows = sorted(nodes.values(), key=lambda row: row["id"])
    edge_rows = sorted(edges.values(), key=lambda row: (row["a"], row["b"]))
    spawns.sort(key=lambda row: row["id"])
    graph_payload = {
        "nodes": node_rows,
        "edges": edge_rows,
        "spawns": spawns,
        "stops": stops,
    }
    graph_sha256 = _canonical_sha256(graph_payload)
    buildings_path = map_dir / "buildings.json"
    buildings_bytes = buildings_path.read_bytes()
    audit_building = audit.get("building_source")
    if not isinstance(audit_building, Mapping) \
            or audit_building.get("sha256") != _sha256_bytes(buildings_bytes):
        raise ValueError("CityCore building source differs from the live audit")
    catalog = audit.get("crosswalk_catalog")
    catalog_rows = (
        catalog.get("crosswalk_components")
        if isinstance(catalog, Mapping) else None
    )
    if not isinstance(catalog_rows, list) or not catalog_rows:
        raise ValueError("live audit has no exact crosswalk catalog")
    edge_spacings = {
        float(row.get("sample_spacing_cm")) for row in edge_audits
        if isinstance(row, Mapping) and row.get("sample_spacing_cm") is not None
    }
    if len(edge_spacings) != 1:
        raise ValueError("live audit uses inconsistent Recast edge sample spacing")
    grid_edge_spacings = {
        float(row.get("sample_spacing_cm")) for row in grid_edge_audits
        if isinstance(row, Mapping) and row.get("sample_spacing_cm") is not None
    }
    if len(grid_edge_spacings) != 1:
        raise ValueError("live audit uses inconsistent Recast grid edge spacing")
    visibility_spacings = {
        float(row.get("sample_spacing_cm")) for row in visibility_audits
        if isinstance(row, Mapping) and row.get("sample_spacing_cm") is not None
    }
    if visibility_audits and visibility_spacings != grid_edge_spacings:
        raise ValueError("live audit uses inconsistent visibility edge spacing")
    grid_spacing_cm = float(pedestrian_grid.get("spacing_cm", math.nan))
    if not math.isfinite(grid_spacing_cm) or grid_spacing_cm <= 0.0:
        raise ValueError("live audit has invalid Recast grid spacing")
    region = audit.get("region")
    if not isinstance(region, Mapping):
        raise ValueError("live audit has no region")
    document = {
        "schema": POOL_SCHEMA,
        "pool_id": "paris-ue-recast-pedestrian-deliveries",
        "version": 2,
        "scene": SCENE,
        "region": {
            "name": "paris_ue_recast_pedestrian_v2",
            "nav_bounds_center_cm": region["nav_bounds_center_cm"],
            "nav_bounds_extent_cm": region["nav_bounds_extent_cm"],
        },
        "validation": {
            "source_path": _relative_path(buildings_path, output_path),
            "source_sha256": _sha256_bytes(buildings_bytes),
            "min_door_anchor_cm": min_door_anchor_cm,
            "max_door_anchor_cm": max_door_anchor_cm,
            "max_navmesh_adjustment_cm": 25.0,
            "max_entrance_connector_cm": max_entrance_connector_cm,
            "max_nav_path_deviation_cm": 150.0,
            "max_nav_path_endpoint_error_cm": 100.0,
            "max_spawn_displacement_cm": 50.0,
        },
        "pedestrian_graph": {
            "audit_path": _relative_path(audit_path, output_path),
            "audit_sha256": audit_sha256,
            "audit_schema": AUDIT_SCHEMA,
            "stable_recast_topology_sha256":
                audit["stable_recast_topology_sha256"],
            "polygon_count": len(polygons),
            "recast_adjacency_count": len(edge_audits),
            "visibility_edge_audit_count": len(visibility_audits),
            "crosswalk_asset_count": len(catalog_rows),
            "graph_method": "recast_projected_surface_grid_v2",
            "grid_spacing_cm": grid_spacing_cm,
            "grid_edge_sample_spacing_cm": next(iter(grid_edge_spacings)),
            "edge_sample_spacing_cm": next(iter(edge_spacings)),
            "crosswalk_sample_spacing_cm": crosswalk_sample_spacing_cm,
            "max_crosswalk_road_transition_cm": max_road_transition_cm,
            "crosswalk_asset_prefix": CROSSWALK_ASSET_PREFIX,
            "agent_nav_data_source": AGENT_NAV_DATA_SOURCE,
            "connectivity_source": PEDESTRIAN_CONNECTIVITY_SOURCE,
            "semantic_source": CITYCORE_SEMANTIC_USE,
            "graph_sha256": graph_sha256,
        },
        **graph_payload,
    }
    summary = {
        "audit_sha256": audit_sha256,
        "stable_recast_topology_sha256": audit["stable_recast_topology_sha256"],
        "raw_polygons": len(polygons),
        "raw_recast_adjacencies": len(edge_audits),
        "raw_grid_points": len(grid_points),
        "raw_grid_edges": len(grid_edge_audits),
        "raw_visibility_edges": len(visibility_audits),
        "accepted_visibility_edges": accepted_visibility_edges,
        "rejected_visibility_edges": rejected_visibility_edges,
        "trusted_nodes": len(node_rows),
        "trusted_edges": len(edge_rows),
        "trusted_stops": len(stops),
        "trusted_spawns": len(spawns),
        "marked_crossing_edges": crossing_edge_count,
        "natural_crossing_stop_pairs": natural_crossing_pair_count,
        "accepted_crosswalks": accepted_crosswalks,
        "rejected_crosswalks": rejected_crosswalks,
        "rejected_recast_adjacencies": rejected_adjacencies,
        "candidate_components": sorted(
            component_summary,
            key=lambda row: (-len(row["stops"]), -row["nodes"]),
        ),
        "stops": [
            {"id": stop["id"], "address": f"{stop['number']} {stop['street_name']}",
             "building_id": stop["building_id"]}
            for stop in stops
        ],
        "spawns": [spawn["id"] for spawn in spawns],
        "graph_sha256": graph_sha256,
    }
    return document, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json"),
    )
    parser.add_argument("--map-dir", default=str(DEFAULT_MAP_DIR))
    parser.add_argument("--max-crosswalk-road-transition-cm", type=float, default=50.0)
    parser.add_argument("--crosswalk-sample-spacing-cm", type=float, default=5.0)
    parser.add_argument("--max-crosswalk-graph-edge-cm", type=float, default=250.0)
    parser.add_argument("--max-entrance-connector-cm", type=float, default=400.0)
    parser.add_argument(
        "--allow-component-without-crossing", action="store_true",
        help="Allow authoring a local pool without a usable crosswalk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit).resolve()
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    output_path = Path(args.output).resolve()
    document, summary = build_pool_document(
        audit,
        audit_path=audit_path,
        audit_sha256=_sha256_bytes(audit_bytes),
        map_dir=Path(args.map_dir).resolve(),
        output_path=output_path,
        max_road_transition_cm=args.max_crosswalk_road_transition_cm,
        crosswalk_sample_spacing_cm=args.crosswalk_sample_spacing_cm,
        max_crosswalk_graph_edge_cm=args.max_crosswalk_graph_edge_cm,
        max_entrance_connector_cm=args.max_entrance_connector_cm,
        require_crossing_component=not args.allow_component_without_crossing,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
