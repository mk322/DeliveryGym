"""Compile a procgen DeliveryBench map into a ``WorldBundle``.

This is the first compiler producer (design plan §6). It reads the navigation graph
from the engine's own ``city_map`` rather than re-deriving one from the map
JSON, because the graph the runtime navigates is the only graph a bundle may
claim to describe. Re-deriving would produce a bundle that is plausible and
subtly disagrees with the environment — exactly the failure design plan §17 lists as
"Paris export topology is visually plausible but not traversable".

Certification is deliberately conservative and honest about what M1 can prove.
Every edge here is marked ``certified`` **only** because a procgen graph is
generated rather than authored, so no synthetic connectors exist to validate;
the NavMesh evidence the design plan M2 requires is not available for procgen maps and
none is claimed. Certificates issued from this producer are grade C
(visualization/research) until a runtime conformance check upgrades them.
"""

from __future__ import annotations

from typing import Any

from embodiedbench.schemas.environment import (
    Certificate,
    CertificationGrade,
    EnvironmentBundle,
    NavigationMode,
    SensorSpec,
)
from embodiedbench.schemas.geometry import CoordinateFrame, FrameName, Vec3
from embodiedbench.schemas.world import (
    EdgeProvenance,
    EdgeStatus,
    InteractionSite,
    LabelProvenance,
    LocomotionMode,
    NavEdge,
    NavGraph,
    NavNode,
    NodeKind,
    SemanticEntity,
    SourceProvenance,
    WorldBundle,
)

# The engine's node kinds, mapped onto the schema's vocabulary. design plan §3.2
# counts junctions, pass-through vertices, and dead ends separately, so the
# distinction is preserved rather than flattened to "node".
_KIND_BY_ENGINE_TYPE = {
    "intersection": NodeKind.JUNCTION,
    "dock": NodeKind.DOCK,
    "door": NodeKind.DOCK,
    "normal": NodeKind.PASS_THROUGH,
}


def _safe_id(raw: Any, fallback: str) -> str:
    """Coerce an engine identifier into the schema's stable-id character set."""
    text = str(raw) if raw is not None else fallback
    # ASCII only: the schema's StableId admits no more, and str.isalnum()
    # would let "Sèvres" through to fail validation as an unknown load error.
    cleaned = "".join(ch if ((ch.isascii() and ch.isalnum()) or ch in "_.:-") else "_"
                      for ch in text)
    return cleaned or fallback


def _node_kind(node: Any, degree: int) -> NodeKind:
    engine_type = getattr(node, "type", None) or getattr(node, "waypoint_kind", None)
    kind = _KIND_BY_ENGINE_TYPE.get(str(engine_type))
    if kind is not None:
        return kind
    return NodeKind.DEAD_END if degree <= 1 else NodeKind.PASS_THROUGH


def extract_nav_graph(city_map: Any) -> NavGraph:
    """Build a ``NavGraph`` from the engine's waypoint graph."""
    graph = city_map.waypoint_graph
    adjacency = getattr(graph, "adjacency_list", {}) or {}

    id_by_object: dict[int, str] = {}
    nodes: list[NavNode] = []
    used_ids: set[str] = set()

    for index, node in enumerate(adjacency):
        node_id = _safe_id(getattr(node, "waypoint_id", None), f"n_{index}")
        # The engine can reuse a waypoint id across distinct node objects; the
        # schema requires uniqueness, so disambiguate rather than drop a node.
        if node_id in used_ids:
            node_id = f"{node_id}__{index}"
        used_ids.add(node_id)
        id_by_object[id(node)] = node_id
        degree = len(adjacency.get(node, []) or [])
        nodes.append(
            NavNode(
                node_id=node_id,
                kind=_node_kind(node, degree),
                position=Vec3(
                    x_cm=float(node.position.x),
                    y_cm=float(node.position.y),
                    z_cm=float(getattr(node.position, "z", 0.0) or 0.0),
                ),
                road_name=str(getattr(node, "road_name", "") or ""),
                address=str(getattr(node, "address", "") or ""),
            )
        )

    node_by_id = {n.node_id: n for n in nodes}
    edges: list[NavEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    for node, neighbours in adjacency.items():
        from_id = id_by_object.get(id(node))
        if from_id is None:
            continue
        for neighbour in neighbours or []:
            to_id = id_by_object.get(id(neighbour))
            if to_id is None or to_id == from_id:
                continue
            # The engine stores both directions; emit one bidirectional edge.
            pair = tuple(sorted((from_id, to_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            start, end = node_by_id[from_id].position, node_by_id[to_id].position
            length = start.distance_cm(end)
            if length <= 0.0:
                # design plan §3.3.1 filters zero/near-zero segments before
                # normalization; a zero-length edge cannot carry a direction.
                continue
            edges.append(
                NavEdge(
                    edge_id=f"{from_id}__{to_id}",
                    from_node=from_id,
                    to_node=to_id,
                    polyline=[start, end],
                    length_cm=length,
                    provenance=EdgeProvenance.AUTHORED_CENTERLINE,
                    status=EdgeStatus.CERTIFIED,
                    bidirectional=True,
                    allowed_modes=[LocomotionMode.WALK, LocomotionMode.SCOOTER],
                )
            )

    return NavGraph(nodes=nodes, edges=edges, components=[sorted(node_by_id)])


def extract_entities(city_map: Any, order_manager: Any) -> tuple[list[SemanticEntity], list[InteractionSite]]:
    """Pull labelled POIs and their interaction sites from the engine.

    Labels are ``authored_metadata``: a procgen map's POI types come from the
    generator, not from a heuristic or a model.
    """
    entities: list[SemanticEntity] = []
    sites: list[InteractionSite] = []
    seen_entities: set[str] = set()
    seen_sites: set[str] = set()

    address_book = getattr(city_map, "address_book", None) or {}
    items = address_book.items() if hasattr(address_book, "items") else []

    for raw_key, record in items:
        if not isinstance(record, dict):
            continue
        entity_id = _safe_id(raw_key, f"poi_{len(entities)}")
        if entity_id in seen_entities:
            continue
        node = record.get("dock_node") or record.get("door_node")
        position = None
        if node is not None and hasattr(node, "position"):
            position = Vec3(
                x_cm=float(node.position.x),
                y_cm=float(node.position.y),
                z_cm=float(getattr(node.position, "z", 0.0) or 0.0),
            )
        if position is None:
            continue
        entity_type = str(record.get("poi_type") or record.get("type") or "building")
        seen_entities.add(entity_id)
        entities.append(
            SemanticEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                position=position,
                provenance=LabelProvenance.AUTHORED_METADATA,
                display_name=str(record.get("name", "") or ""),
                attributes={"road_name": str(record.get("road_name", "") or "")},
            )
        )
        site_id = f"site_{entity_id}"
        if site_id not in seen_sites:
            seen_sites.add(site_id)
            sites.append(
                InteractionSite(
                    site_id=site_id,
                    site_type=f"{entity_type}_dock",
                    position=position,
                    nearest_node=_safe_id(getattr(node, "waypoint_id", None), "unknown"),
                    entity_id=entity_id,
                    reachable_modes=[LocomotionMode.WALK],
                )
            )
    return entities, sites


def compile_procgen_world(
    city_map: Any,
    *,
    map_name: str,
    version: str = "0.1.0",
    order_manager: Any = None,
    compiler_version: str = "0.1.0",
) -> WorldBundle:
    """Produce a ``WorldBundle`` for a loaded procgen map."""
    nav_graph = extract_nav_graph(city_map)
    entities, sites = extract_entities(city_map, order_manager)

    known_nodes = {n.node_id for n in nav_graph.nodes}
    # Drop sites whose nearest node did not survive graph extraction rather than
    # emit a bundle that fails its own referential-integrity check.
    sites = [s for s in sites if s.nearest_node in known_nodes]

    return WorldBundle(
        world_id=_safe_id(map_name, "procgen"),
        version=version,
        frames=[
            CoordinateFrame(
                name=FrameName.BUNDLE_WORLD,
                units="cm",
                description="DeliveryBench procgen world frame; centimetres, z up",
            )
        ],
        nav_graph=nav_graph,
        entities=entities,
        interaction_sites=sites,
        source=SourceProvenance(
            ue_project="",
            engine_version="",
            level_package=f"procgen/{map_name}",
            compiler_version=compiler_version,
            actor_count=len(nav_graph.nodes),
        ),
        notes=[
            "Compiled from the DeliveryBench engine's own waypoint graph, so the bundle "
            "describes the graph the runtime actually navigates.",
            "No NavMesh evidence: procgen maps have no UE NavMesh to validate against "
            "(the design plan M2 applies to UE-sourced maps).",
        ],
    )


def environment_for(
    world: WorldBundle,
    *,
    environment_id: str | None = None,
    embodiment_profile: str = "abstract_courier_v1",
    task_plugin: str = "delivery",
) -> EnvironmentBundle:
    """Wrap a base world as a text-only environment with an honest certificate."""
    environment_id = environment_id or f"{world.world_id}-delivery"
    certificate = Certificate(
        environment_id=environment_id,
        environment_version=world.version,
        task_plugin=task_plugin,
        task_plugin_version="0.1.0",
        embodiment_profile=embodiment_profile,
        navigation_mode=NavigationMode.NAV_WAYPOINT,
        runtime="text",
        grade=CertificationGrade.C,
        checks_passed=["schema", "graph_referential_integrity"],
        limitations=[
            "no NavMesh validation (procgen map, the design plan M2 not applicable)",
            "no cached observation album bound to this environment",
            "text runtime only",
        ],
        certified_road_coverage=world.nav_graph.certified_road_coverage(),
    )
    return EnvironmentBundle(
        environment_id=environment_id,
        version=world.version,
        base_world_id=world.world_id,
        base_world_version=world.version,
        base_world_sha256=world.content_hash(),
        supported_embodiment_profiles=[embodiment_profile],
        supported_navigation_modes=[NavigationMode.NAV_WAYPOINT],
        supported_runtimes=["text"],
        sensors=SensorSpec(channels=["text"]),
        certificates=[certificate],
    )
