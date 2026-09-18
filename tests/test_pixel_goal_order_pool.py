"""Fail-closed contracts for certified random and fixed delivery orders."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import (
    Address,
    RoadNetwork,
    Street,
    StreetNode,
    bearing_deg,
)
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
    CAMERA_VIEW_FRONT_REAR,
    EmbodiedCourierEnv,
)
from embodiedbench.runtime.live.protocol import EpisodeResponse
from tools.pixel_goal_courier_backend import (
    ValidatedPoolDeliveryEnv,
    trusted_phone_route_instruction,
)
from tools.pixel_goal_order_pool import (
    OrderConstraints,
    build_validated_pool_network,
    load_validated_delivery_pool,
    resolve_delivery_scenario,
    scenario_report,
    selected_route_report,
    shortest_pool_path,
)


def _pool_document(source: Path) -> dict[str, object]:
    return {
        "schema": "embodiedbench/validated-delivery-region/v1",
        "pool_id": "test-entrances",
        "version": 1,
        "scene": "/Game/Test",
        "region": {
            "name": "test_region",
            "nav_bounds_center_cm": [0.0, 0.0, 100.0],
            "nav_bounds_extent_cm": [1_000.0, 1_000.0, 300.0],
        },
        "validation": {
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "min_door_anchor_cm": 50.0,
            "max_door_anchor_cm": 200.0,
            "max_navmesh_adjustment_cm": 25.0,
            "max_entrance_connector_cm": 400.0,
            "max_nav_path_deviation_cm": 150.0,
            "max_nav_path_endpoint_error_cm": 100.0,
            "max_spawn_displacement_cm": 50.0,
        },
        "spawns": [{
            "id": "south",
            "node_id": "p0",
            "z_cm": 90.0,
            "yaw_deg": 0.0,
            "verified": True,
            "max_navmesh_adjustment_cm": 25.0,
        }],
        "nodes": [
            {"id": "p0", "x_cm": 0.0, "y_cm": 0.0,
             "street_index": 0, "role": "spawn"},
            {"id": "p1", "x_cm": 300.0, "y_cm": 0.0,
             "street_index": 0, "role": "route"},
            {"id": "a", "x_cm": 300.0, "y_cm": 200.0,
             "street_index": 0, "role": "entrance_connector"},
            {"id": "p2", "x_cm": 600.0, "y_cm": 0.0,
             "street_index": 1, "role": "crossing"},
            {"id": "b", "x_cm": 600.0, "y_cm": 300.0,
             "street_index": 1, "role": "entrance_connector"},
            {"id": "c", "x_cm": 900.0, "y_cm": 300.0,
             "street_index": 2, "role": "entrance_connector"},
        ],
        "edges": [
            {"a": "p0", "b": "p1", "kind": "sidewalk",
             "verified": True, "source": "test pavement"},
            {"a": "p1", "b": "a", "kind": "sidewalk",
             "verified": True, "source": "test entrance"},
            {"a": "p1", "b": "p2", "kind": "marked_crossing",
             "verified": True, "source": "test zebra"},
            {"a": "p2", "b": "b", "kind": "sidewalk",
             "verified": True, "source": "test entrance"},
            {"a": "b", "b": "c", "kind": "sidewalk",
             "verified": True, "source": "test pavement"},
        ],
        "stops": [
            {
                "id": "alpha", "building_id": "building-a",
                "street_index": 0, "street_name": "Alpha Street",
                "number": 1, "door_cm": [300.0, 300.0],
                "handover_node_id": "a", "poi_type": "building",
                "roles": ["pickup", "dropoff"],
                "entrance_source": "source entrance a",
                "entrance_verified": True, "surface": "pavement",
                "navmesh_verified": True, "max_navmesh_adjustment_cm": 25.0,
            },
            {
                "id": "bravo", "building_id": "building-b",
                "street_index": 1, "street_name": "Bravo Street",
                "number": 2, "door_cm": [600.0, 400.0],
                "handover_node_id": "b", "poi_type": "shop",
                "roles": ["pickup", "dropoff"],
                "entrance_source": "source entrance b",
                "entrance_verified": True, "surface": "pavement",
                "navmesh_verified": True, "max_navmesh_adjustment_cm": 25.0,
            },
            {
                "id": "charlie", "building_id": "building-c",
                "street_index": 2, "street_name": "Charlie Street",
                "number": 3, "door_cm": [900.0, 400.0],
                "handover_node_id": "c", "poi_type": "building",
                "roles": ["pickup", "dropoff"],
                "entrance_source": "source entrance c",
                "entrance_verified": True, "surface": "pavement",
                "navmesh_verified": True, "max_navmesh_adjustment_cm": 25.0,
            },
        ],
    }


@pytest.fixture
def pool_path(tmp_path: Path) -> Path:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "units": "centimeters",
        "buildings": [
            {
                "id": "building-a", "center_cm": {"x": 300.0, "y": 200.0},
                "bbox_cm": {"x": 200.0, "y": 200.0},
                "entrance_yaw_deg": 90.0, "poi_type": "building",
                "deliverybench_navigable": True,
            },
            {
                "id": "building-b", "center_cm": {"x": 600.0, "y": 300.0},
                "bbox_cm": {"x": 200.0, "y": 200.0},
                "entrance_yaw_deg": 90.0, "poi_type": "shop",
                "deliverybench_navigable": True,
            },
            {
                "id": "building-c", "center_cm": {"x": 900.0, "y": 300.0},
                "bbox_cm": {"x": 200.0, "y": 200.0},
                "entrance_yaw_deg": 90.0, "poi_type": "building",
                "deliverybench_navigable": True,
            },
        ],
    }, indent=2) + "\n", encoding="utf-8")
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(_pool_document(source), indent=2) + "\n", encoding="utf-8")
    return path


def _rewrite(path: Path, mutate) -> Path:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def _compiled_city() -> RoadNetwork:
    stop_rows = (
        ("building-a", 0, "Alpha Street", 1, (300.0, 300.0), "building"),
        ("building-b", 1, "Bravo Street", 2, (600.0, 400.0), "shop"),
        ("building-c", 2, "Charlie Street", 3, (900.0, 400.0), "building"),
    )
    return RoadNetwork(
        map_name="test",
        streets=[
            Street(index=index, name=name, source=f"source-{index}",
                   width_cm=600.0, polyline=[(0.0, 0.0), (1.0, 0.0)])
            for index, name in enumerate(
                ("Alpha Street", "Bravo Street", "Charlie Street"))
        ],
        addresses=[
            Address(
                building_id=building_id,
                street_index=street_index,
                street_name=street_name,
                number=number,
                arc_cm=0.0,
                offset_cm=0.0,
                side="odd",
                door=door,
                nearest_node=None,
                poi_type=poi_type,
            )
            for building_id, street_index, street_name, number, door, poi_type
            in stop_rows
        ],
    )


def test_seeded_random_orders_are_reproducible_and_only_use_certified_pairs(
    pool_path: Path,
) -> None:
    pool = load_validated_delivery_pool(pool_path)
    constraints = OrderConstraints(
        min_delivery_cm=300.0,
        max_delivery_cm=2_000.0,
        require_different_streets=True,
    )

    first = resolve_delivery_scenario(
        pool, mode="random", seed=17, constraints=constraints)
    replay = resolve_delivery_scenario(
        pool, mode="random", seed=17, constraints=constraints)
    sampled = {
        (resolve_delivery_scenario(
            pool, mode="random", seed=seed, constraints=constraints).pickup.id,
         resolve_delivery_scenario(
            pool, mode="random", seed=seed, constraints=constraints).dropoff.id)
        for seed in range(20)
    }

    assert first == replay
    assert len(sampled) > 1
    assert first.pickup.id != first.dropoff.id
    assert first.pickup.street_index != first.dropoff.street_index
    assert first.delivery.node_ids[0] == first.pickup.handover_node_id
    assert first.delivery.node_ids[-1] == first.dropoff.handover_node_id
    assert first.candidate_count == 6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_delivery_cm": True}, "minimum delivery distance"),
        ({"max_delivery_cm": "far"}, "maximum delivery distance"),
        ({"min_turns": 1.5}, "minimum turns"),
        ({"require_different_streets": 1}, "require_different_streets"),
        ({"require_marked_crossing": 0}, "require_marked_crossing"),
    ],
)
def test_order_constraints_reject_coercible_but_wrong_types(
    kwargs: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OrderConstraints(**kwargs)


def test_fixed_order_uses_the_same_constraints_and_records_the_exact_request(
    pool_path: Path,
) -> None:
    pool = load_validated_delivery_pool(pool_path)
    constraints = OrderConstraints(
        min_delivery_cm=300.0,
        max_delivery_cm=2_000.0,
        require_different_streets=True,
        require_marked_crossing=True,
    )

    scenario = resolve_delivery_scenario(
        pool,
        mode="fixed",
        seed=999,
        constraints=constraints,
        spawn_id="south",
        pickup_id="alpha",
        dropoff_id="bravo",
    )
    report = scenario_report(pool, scenario)

    assert scenario.pickup.id == "alpha"
    assert scenario.dropoff.id == "bravo"
    assert scenario.delivery.uses_marked_crossing is True
    assert report["request"] == {
        "spawn_id": "south", "pickup_id": "alpha", "dropoff_id": "bravo"}
    with pytest.raises(ValueError, match="does not satisfy"):
        resolve_delivery_scenario(
            pool,
            mode="fixed",
            seed=0,
            constraints=OrderConstraints(min_delivery_cm=2_001.0),
            pickup_id="alpha",
            dropoff_id="bravo",
        )
    with pytest.raises(ValueError, match="does not accept fixed"):
        resolve_delivery_scenario(
            pool,
            mode="random",
            seed=0,
            constraints=constraints,
            pickup_id="alpha",
        )


def test_network_contains_only_pool_addresses_and_reports_selected_route(
    pool_path: Path,
) -> None:
    pool = load_validated_delivery_pool(pool_path)
    city = _compiled_city()
    network = build_validated_pool_network(city, pool)
    scenario = resolve_delivery_scenario(
        pool,
        mode="fixed",
        seed=0,
        constraints=OrderConstraints(
            min_delivery_cm=300.0,
            max_delivery_cm=2_000.0,
            require_different_streets=True,
        ),
        pickup_id="alpha",
        dropoff_id="charlie",
    )
    route = selected_route_report(network, pool, scenario)

    assert {address.building_id for address in network.addresses} == {
        "building-a", "building-b", "building-c"}
    assert all(address.kerb_node.startswith(("a", "b", "c"))
               for address in network.addresses)
    assert route["pickup_waypoint_id"] == "a"
    assert route["dropoff_waypoint_id"] == "c"
    assert route["planned_delivery_cm"] == pytest.approx(
        scenario.delivery.length_cm)


def test_network_rejects_a_pool_stop_that_drifted_from_the_compiled_address(
    pool_path: Path,
) -> None:
    pool = load_validated_delivery_pool(pool_path)
    city = _compiled_city()
    city.addresses[0] = replace(city.addresses[0], number=99)

    with pytest.raises(ValueError, match="compiled city address"):
        build_validated_pool_network(city, pool)


def test_validated_env_starts_and_issues_the_resolved_order_without_overwrite(
    pool_path: Path, tmp_path: Path,
) -> None:
    class RecordingClient:
        events: tuple[object, ...] = ()

        def __init__(self) -> None:
            self.episodes = []

        def episode(self, request):
            self.episodes.append(request)
            return EpisodeResponse(request.episode_id, request.spawn, 1.0 / 30.0)

    pool = load_validated_delivery_pool(pool_path)
    city = _compiled_city()
    network = build_validated_pool_network(city, pool)
    scenario = resolve_delivery_scenario(
        pool,
        mode="fixed",
        seed=4,
        constraints=OrderConstraints(
            min_delivery_cm=300.0,
            require_different_streets=True,
        ),
        spawn_id="south",
        pickup_id="alpha",
        dropoff_id="bravo",
    )
    client = RecordingClient()
    env = ValidatedPoolDeliveryEnv(
        network,
        client,
        delivery_pool=pool,
        scenario=scenario,
        episode_id="certified-order-test",
        cache_root=tmp_path / "cache",
        action_space=ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
        camera_view=CAMERA_VIEW_FRONT_REAR,
        embodiment="human_on_foot",
        difficulty="solo",
        seed=4,
        spawn_z_cm=scenario.spawn.z_cm,
    )

    env.reset()

    assert env.node_id == scenario.spawn.node_id
    assert len(env.orders) == 1
    assert env.orders[0].pickup.building_id == scenario.pickup.building_id
    assert env.orders[0].dropoff.building_id == scenario.dropoff.building_id
    assert env._task_action_tolerance_cm(collected=False) == 300.0
    assert env._task_action_tolerance_cm(collected=True) == 300.0
    assert env.summary()["viewpoint_expected"] == "pavement"
    assert env.summary()["viewpoint_served"] == "pavement"
    assert env.summary()["viewpoint_matches_embodiment"] is True
    assert len(client.episodes) == 1
    assert client.episodes[0].spawn.x_cm == pool.nodes_by_id[
        scenario.spawn.node_id].x_cm
    assert client.episodes[0].spawn.y_cm == pool.nodes_by_id[
        scenario.spawn.node_id].y_cm
    assert client.episodes[0].spawn.z_cm == scenario.spawn.z_cm


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["stops"][0].update(surface="road"),
         "not certified as pavement"),
        (lambda raw: raw["stops"][0].update(entrance_verified=False),
         "entrance is not certified"),
        (lambda raw: raw["stops"][0].update(max_navmesh_adjustment_cm=25.1),
         "exceeds the NavMesh adjustment"),
        (lambda raw: raw["edges"][0].update(kind="road"),
         "unsafe kind"),
        (lambda raw: raw["edges"][0].update(verified=False),
         "is not certified"),
        (lambda raw: raw["stops"][0].update(door_cm=[900.0, 900.0]),
         "does not match its source entrance"),
        (lambda raw: raw["nodes"][2].update(y_cm=0.0),
         "door/anchor gap"),
        (lambda raw: raw["nodes"][2].update(y_cm=401.0),
         "entrance connector"),
    ],
)
def test_pool_loader_rejects_uncertified_or_implausible_geometry(
    pool_path: Path, mutation, message: str,
) -> None:
    _rewrite(pool_path, mutation)

    with pytest.raises(ValueError, match=message):
        load_validated_delivery_pool(pool_path)


def test_pool_loader_rejects_certification_source_drift(pool_path: Path) -> None:
    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    source = Path(raw["validation"]["source_path"])
    source.write_text('{"immutable":false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source SHA-256"):
        load_validated_delivery_pool(pool_path)


def test_report_identity_changes_when_the_versioned_pool_changes(
    pool_path: Path,
) -> None:
    first = load_validated_delivery_pool(pool_path)
    constraints = OrderConstraints(min_delivery_cm=0.0)
    first_scenario = resolve_delivery_scenario(
        first, mode="fixed", seed=0, constraints=constraints,
        pickup_id="alpha", dropoff_id="bravo")

    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    raw["version"] = 2
    pool_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    second = load_validated_delivery_pool(pool_path)
    second_scenario = resolve_delivery_scenario(
        second, mode="fixed", seed=0, constraints=constraints,
        pickup_id="alpha", dropoff_id="bravo")

    assert first.sha256 != second.sha256
    assert first_scenario.scenario_id != second_scenario.scenario_id


TRUSTED_V2_POOL = (
    Path(__file__).resolve().parents[1]
    / "configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json"
)


@pytest.fixture(scope="module")
def trusted_v2_pool():
    """Load the produced UE-backed pool as an integration fixture."""

    return load_validated_delivery_pool(TRUSTED_V2_POOL)


def test_trusted_v2_pool_resolves_random_and_fixed_crossing_orders(
    trusted_v2_pool,
) -> None:
    pool = trusted_v2_pool
    constraints = OrderConstraints(
        min_delivery_cm=3_000.0,
        max_delivery_cm=8_000.0,
        min_turns=1,
        require_different_streets=False,
        require_marked_crossing=True,
    )

    fixed = resolve_delivery_scenario(
        pool,
        mode="fixed",
        seed=0,
        constraints=constraints,
        spawn_id="spawn-near-citycore-building-0529",
        pickup_id="stop-citycore-building-0529",
        dropoff_id="stop-citycore-building-0535",
    )
    random_first = resolve_delivery_scenario(
        pool, mode="random", seed=19, constraints=constraints)
    random_replay = resolve_delivery_scenario(
        pool, mode="random", seed=19, constraints=constraints)

    assert pool.schema == "embodiedbench/trusted-pedestrian-delivery-region/v2"
    assert pool.pedestrian_graph is not None
    assert pool.pedestrian_graph.connectivity_source \
        == "ue_recast_navmesh_surface_audit"
    assert pool.pedestrian_graph.semantic_source \
        == "address_and_street_names_only"
    assert fixed.delivery.length_cm == pytest.approx(5_769.1017, abs=0.1)
    assert fixed.delivery.turns >= 1
    assert fixed.delivery.uses_marked_crossing is True
    assert random_first == random_replay
    assert random_first.delivery.uses_marked_crossing is True
    assert all(stop.entrance_asset_id for stop in pool.stops)
    assert all(
        "entrance" in stop.entrance_static_mesh_path.casefold()
        for stop in pool.stops
        if stop.entrance_static_mesh_path is not None
    )

    with pytest.raises(ValueError, match="no certified order pair"):
        resolve_delivery_scenario(
            pool,
            mode="random",
            seed=19,
            constraints=replace(
                constraints, require_different_streets=True),
        )
    with pytest.raises(ValueError, match="does not satisfy"):
        resolve_delivery_scenario(
            pool,
            mode="fixed",
            seed=0,
            constraints=replace(
                constraints, min_delivery_cm=8_001.0,
                max_delivery_cm=20_000.0),
            pickup_id="stop-citycore-building-0529",
            dropoff_id="stop-citycore-building-0535",
        )


def test_trusted_phone_previews_the_real_crosswalk_before_at_and_after_overshoot(
    trusted_v2_pool,
) -> None:
    """Regression for GPT rollout turns 8--16 on the exact certified pool.

    The pawn was 2.4 m before PR_Crossswalk_94, then 1.8 m beyond its entry.
    The old banner exposed only a one-metre lattice edge and alternated between
    those positions.  The new banner must retain the recovery leg while also
    announcing the same route's crosswalk turn early enough to stop for it.
    """

    pool = trusted_v2_pool
    dropoff = "entrance:CITYCORE_Building_0535"
    cases = (
        (
            "recast-grid--193-8",
            (-19302.796262352418, 779.2608002034558),
            "north-west",
            "south-west",
            241.8,
        ),
        (
            "recast-grid--192-7",
            (-19137.959102041288, 660.4570504896296),
            "south-west",
            "west",
            909.9,
        ),
        (
            "recast-grid--190-4",
            (-19042.686506404505, 442.4960413356186),
            "south-east",
            "south-west",
            184.6,
        ),
    )
    for start, pawn, heading, maneuver, maneuver_cm in cases:
        path = shortest_pool_path(pool, start, dropoff)
        assert path is not None
        route = [pawn, *(
            pool.nodes_by_id[node_id].position
            for node_id in path.node_ids[1:]
        )]

        instruction = trusted_phone_route_instruction(route)

        assert instruction["next_heading"] == heading
        assert instruction["next_maneuver_heading"] == maneuver
        assert instruction["next_maneuver_distance_cm"] == pytest.approx(
            maneuver_cm, abs=0.2)


def test_trusted_phone_primary_heading_looks_past_one_metre_recast_jitter() -> None:
    route = [
        (0.0, 0.0),
        (0.0, 50.0),
        (100.0, 50.0),
        (200.0, 50.0),
        (300.0, 50.0),
        (400.0, 50.0),
    ]

    instruction = trusted_phone_route_instruction(route)

    # The first lattice edge points east in this coordinate convention, but
    # a three-metre point on the same exact route is predominantly north.  A
    # multi-metre pixel action needs the latter, while the exact zig-zag stays
    # available in the blue route and UE movement guard.
    assert instruction["next_heading"] == "north"


def test_trusted_v2_network_discards_citycore_carriageway_connectivity(
    trusted_v2_pool,
) -> None:
    pool = trusted_v2_pool
    max_street_index = max(node.street_index for node in pool.nodes)
    streets = [
        Street(
            index=index,
            name=("Rue Oberkampf" if index == 9 else f"Semantic Street {index}"),
            source=f"citycore-{index}",
            width_cm=600.0,
            polyline=[(0.0, 0.0), (100.0, 0.0)],
        )
        for index in range(max_street_index + 1)
    ]
    city = RoadNetwork(
        map_name="citycore-semantics-only",
        streets=streets,
        nodes={
            "citycore-road-a": StreetNode(
                id="citycore-road-a", x_cm=0.0, y_cm=0.0,
                street_index=9, arc_cm=0.0,
                neighbours={"citycore-road-b"}),
            "citycore-road-b": StreetNode(
                id="citycore-road-b", x_cm=100.0, y_cm=0.0,
                street_index=9, arc_cm=100.0,
                neighbours={"citycore-road-a"}),
        },
        addresses=[
            Address(
                building_id=stop.building_id,
                street_index=stop.street_index,
                street_name=stop.street_name,
                number=stop.number,
                arc_cm=0.0,
                offset_cm=0.0,
                side="semantic",
                # Deliberately use the obsolete CityCore door estimate. V2
                # may consume its address text, but never its entrance geometry.
                door=(0.0, 0.0),
                nearest_node="citycore-road-a",
                poi_type=stop.poi_type,
            )
            for stop in pool.stops
        ],
    )

    network = build_validated_pool_network(city, pool)

    assert "citycore-road-a" not in network.nodes
    assert "citycore-road-b" not in network.nodes
    assert set(network.nodes) == {node.id for node in pool.nodes}
    assert network.edges() == {
        tuple(sorted((edge.a, edge.b))) for edge in pool.edges
    }


def test_trusted_v2_live_crosswalk_landing_advances_the_phone_route(
    trusted_v2_pool,
) -> None:
    """The exact failed GPT landing must map onto PR_Crossswalk_94.

    This uses the certified pool's real 540-node topology and the UE-reported
    start/end poses from the interrupted live rollout.  It protects the link
    between physical movement, map matching and the path subsequently drawn by
    the phone; a synthetic geometry test alone cannot prove those assets still
    agree after regenerating the pool.
    """
    pool = trusted_v2_pool
    streets = [
        Street(
            index=index,
            name=f"Semantic Street {index}",
            source=f"citycore-{index}",
            width_cm=600.0,
            polyline=[(0.0, 0.0), (100.0, 0.0)],
        )
        for index in range(max(node.street_index for node in pool.nodes) + 1)
    ]
    nodes = {
        row.id: StreetNode(
            id=row.id,
            x_cm=row.x_cm,
            y_cm=row.y_cm,
            street_index=row.street_index,
            arc_cm=0.0,
        )
        for row in pool.nodes
    }
    for edge in pool.edges:
        nodes[edge.a].neighbours.add(edge.b)
        nodes[edge.b].neighbours.add(edge.a)
    env = object.__new__(EmbodiedCourierEnv)
    env.network = RoadNetwork(
        map_name="trusted-v2-live-crosswalk-regression",
        streets=streets,
        nodes=nodes,
    )
    env.node_id = "recast-grid--193-9"
    env.action_space = ACTION_SPACE_PIXEL_GOAL_FRONT_REAR
    env._last_pose_match = None
    start = (-19294.4, 838.3)
    landing = (-19391.1, 370.3)
    env._walked_bearing = bearing_deg(start, landing)

    selected, gap = env._match_pose_node(
        landing, movement_cm=math.dist(start, landing))
    phone_path = env.route_nodes(
        selected, "entrance:CITYCORE_Building_0535")

    assert selected == "crosswalk:PR_Crossswalk_94:050"
    assert gap == pytest.approx(105.22, abs=0.1)
    assert phone_path is not None
    assert phone_path[:4] == [
        "crosswalk:PR_Crossswalk_94:050",
        "crosswalk:PR_Crossswalk_94:100",
        "crosswalk:PR_Crossswalk_94:150",
        "crosswalk:PR_Crossswalk_94:175",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["nodes"][0].update(x_cm=123456.0),
         "graph SHA-256"),
        (lambda raw: raw["pedestrian_graph"].update(
            audit_sha256="0" * 64),
         "audit source SHA-256"),
    ],
)
def test_trusted_v2_pool_rejects_mutated_certification(
    tmp_path: Path, mutation, message: str,
) -> None:
    raw = json.loads(TRUSTED_V2_POOL.read_text(encoding="utf-8"))
    raw["pedestrian_graph"]["audit_path"] = str(
        (TRUSTED_V2_POOL.parent / raw["pedestrian_graph"]["audit_path"]).resolve())
    raw["validation"]["source_path"] = str(
        (TRUSTED_V2_POOL.parent / raw["validation"]["source_path"]).resolve())
    mutation(raw)
    copied = tmp_path / "mutated-v2-pool.json"
    copied.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_validated_delivery_pool(copied)


def test_a_spawn_certified_staring_into_its_own_door_starts_facing_the_street():
    """The seed-2 spawn of the shipped pool looks into a shopfront 0.8 m away;
    it is turned round at resolution, the other three are not."""
    from tools.pixel_goal_order_pool import (
        OrderConstraints, face_away_from_own_door, load_validated_delivery_pool,
        resolve_delivery_scenario)
    pool = load_validated_delivery_pool(
        Path(__file__).resolve().parents[1]
        / "configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json")
    turned = {s.id: face_away_from_own_door(pool, s).yaw_deg != s.yaw_deg for s in pool.spawns}
    assert turned == {
        "spawn-near-citycore-building-0044": True,
        "spawn-near-citycore-building-0283": False,
        "spawn-near-citycore-building-0529": False,
        "spawn-near-citycore-building-0535": False,
    }
    constraints = OrderConstraints(
        min_delivery_cm=3000.0, max_delivery_cm=8000.0, min_turns=1,
        require_different_streets=False, require_marked_crossing=True)
    scenario = resolve_delivery_scenario(pool, mode="random", seed=2, constraints=constraints)
    assert scenario.spawn.id == "spawn-near-citycore-building-0044"
    certified = pool.spawns_by_id[scenario.spawn.id].yaw_deg
    assert abs(scenario.spawn.yaw_deg - (certified + 180.0) % 360.0) < 1e-9
