"""The four-view any-point harness: two engine pairs a quarter turn apart,
a pixel resolved without walking, and the walk itself along certified
pedestrian ways -- against fakes, at the same boundaries the live runner
crosses."""
from __future__ import annotations

import base64
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiedbench.compiler.road_network import Address, RoadNetwork, Street, StreetNode
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
    CAMERA_VIEW_FRONT_REAR,
    NUDGE_MAX_CM,
)
from embodiedbench.runtime.live.protocol import (
    PIXEL_VIEWS,
    PIXEL_VIEWS_QUAD,
    CameraSpec,
    EpisodeResponse,
    ObservedView,
    ObserveViewsRequest,
    ObserveViewsResponse,
    PixelSpec,
    Pose,
    ResolvedPixel,
    WalkPixelRequest,
    WalkPixelResponse,
    WalkResponse,
)
from embodiedbench.runtime.pixel_goal import (
    PixelGoalFrame,
    PixelGoalRejected,
    PixelGoalViewPair,
    project_world_point_to_pixel,
)
from embodiedbench.schemas.runtime import ControllerOutcomeCode
from tools.pixel_goal_courier_backend import (
    SpearTrackBClient,
    ValidatedPoolDeliveryEnv,
    WorldLegRefused,
)
from tools.pixel_goal_order_pool import (
    ENGINE_CERTIFICATION_METHOD,
    EnginePathCertification,
    OrderConstraints,
    build_validated_pool_network,
    candidate_legs,
    certified_leg_chains,
    load_validated_delivery_pool,
    nearest_pool_node,
    plan_pool_legs,
    resolve_delivery_scenario,
)

TRUSTED_V2_POOL = (
    Path(__file__).resolve().parents[1]
    / "configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json"
)


def with_every_leg_certified(pool):
    """The pool with every leg its geometry offers in its leg table, as if
    the engine had accepted them all: the harness is tested on its own
    rules here, not on the engine's verdicts (those are the shipped v3)."""
    from dataclasses import replace
    legs = frozenset(candidate_legs(pool))
    block = EnginePathCertification(
        method=ENGINE_CERTIFICATION_METHOD, certified_at="2026-09-14T00:00:00Z",
        source_pool_sha256=pool.sha256, verdicts_sha256="0" * 64,
        legs=len(legs), accepted=len(legs), refused=0, unjudged=0,
        removed_edges=(), removed_nodes=(), pass_through_nodes=(), placement_tolerance_cm=15.0,
        min_leg_cm=300.0, navmesh_adjustment_cm=100.0, leg_epsilon_cm=40.0, max_leg_cm=1000.0,
        max_path_excess_cm=150.0)
    return replace(pool, certified_legs=legs, engine_certification=block)
# a 1x1 PNG, so the environment can write and hash a real image file
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
ONE_PIXEL_B64 = base64.b64encode(ONE_PIXEL_PNG).decode()
CAMERA = CameraSpec(640, 360, 90.0)


# --------------------------------------------------------------- the client --

def frame(view: str, group: str, snapshot: str, yaw: float) -> PixelGoalFrame:
    return PixelGoalFrame(
        rgb_data_url="data:image/png;base64," + ONE_PIXEL_B64,
        camera_snapshot_id=snapshot,
        camera_intrinsics_id="perspective-640x360-hfov90",
        width_px=640, height_px=360, agent_tag="agent", view_id=view,
        capture_group_id=group, camera_yaw_deg=yaw,
        capture_timing={"capture_read_ms": 1.0, "encode_ms": 2.0, "wall_ms": 3.0},
        camera_location_cm=(20.0 * math.cos(math.radians(yaw)),
                            20.0 * math.sin(math.radians(yaw)), 163.0),
    )


class TurningRuntime:
    """An engine that reports the yaw the turner last set, resolves any pixel
    to a point 5 m ahead on the pavement, and walks there."""

    def __init__(self) -> None:
        self.yaw = 45.0
        self.position = [0.0, 0.0]
        self.captures = 0
        self.executed: list[tuple[str, str, float, float, float]] = []
        self.endpoint = None  # no status RPC: feet come from the pose
        self.config = None

    def capture_view_pair(self, **kwargs):
        self.captures += 1
        group = f"view-pair-{self.captures}"
        return PixelGoalViewPair(
            capture_group_id=group,
            pose=(self.position[0], self.position[1], 100.0, self.yaw),
            front=frame("front", group, f"{group}-front", self.yaw),
            rear=frame("rear", group, f"{group}-rear", (self.yaw + 180.0) % 360.0),
            capture_timing={"capture_read_ms": 2.0, "encode_ms": 4.0, "wall_ms": 8.0},
        )

    def _started(self, frame_: PixelGoalFrame, action, distance_cm: float):
        yaw = math.radians(frame_.camera_yaw_deg)
        hit = (self.position[0] + distance_cm * math.cos(yaw),
               self.position[1] + distance_cm * math.sin(yaw))
        vec = SimpleNamespace(x_cm=hit[0], y_cm=hit[1], z_cm=15.0)
        return SimpleNamespace(
            request=SimpleNamespace(
                request_id="move-1", raw_world_hit=vec, projected_target=vec,
                navmesh_adjustment_cm=0.0, controller_path_length_cm=distance_cm),
            raw_hit_actor="BP_SplineSidewalk13", controller_request_result="ok")

    def start(self, action, frame_):
        self.executed.append((frame_.view_id, frame_.capture_group_id,
                              action.target.u_norm, action.target.v_norm, 0.0))
        return self._started(frame_, action, 500.0)

    def cancel(self, request_id, *, reason):
        return SimpleNamespace(outcome=ControllerOutcomeCode.ACCEPTED, final_pose=None,
                               elapsed_sim_s=0.0, distance_travelled_cm=0.0)

    def execute(self, action, frame_):
        # the walk: 5 m ahead of the camera the pixel was taken from
        started = self._started(frame_, action, 500.0)
        hit = started.request.projected_target
        self.position = [hit.x_cm, hit.y_cm]
        self.yaw = frame_.camera_yaw_deg
        self.executed.append((frame_.view_id, frame_.capture_group_id,
                              action.target.u_norm, action.target.v_norm, 500.0))
        result = SimpleNamespace(
            outcome=ControllerOutcomeCode.ACCEPTED,
            final_pose=SimpleNamespace(
                position=SimpleNamespace(x_cm=hit.x_cm, y_cm=hit.y_cm, z_cm=100.0),
                yaw_deg=self.yaw),
            elapsed_sim_s=2.5, distance_travelled_cm=500.0)
        return started, result


def quad_client(runtime: TurningRuntime) -> SpearTrackBClient:
    turns: list[float] = []

    def turner(yaw: float) -> None:
        turns.append(yaw)
        runtime.yaw = yaw

    client = SpearTrackBClient(
        runtime, agent_tag="agent",
        spawn_pose=Pose(x_cm=0.0, y_cm=0.0, z_cm=100.0, yaw_deg=45.0),
        turner=turner, views=PIXEL_VIEWS_QUAD, turn_settle_s=0.0)
    client.episode(SimpleNamespace(episode_id="ep"))
    client.turns = turns  # type: ignore[attr-defined]
    return client


def test_four_views_need_a_turner():
    with pytest.raises(ValueError, match="turner"):
        SpearTrackBClient(TurningRuntime(), agent_tag="agent",
                          spawn_pose=Pose(0.0, 0.0, 100.0, 45.0),
                          views=PIXEL_VIEWS_QUAD)


def test_quad_observation_is_two_pairs_a_quarter_turn_apart_and_faces_back():
    runtime = TurningRuntime()
    client = quad_client(runtime)

    response = client.observe_views(
        ObserveViewsRequest(episode_id="ep", camera=CAMERA, views=PIXEL_VIEWS_QUAD))

    assert tuple(view.view for view in response.views) == PIXEL_VIEWS_QUAD
    assert [view.yaw_offset_deg for view in response.views] == [0.0, 270.0, 90.0, 180.0]
    # front/rear from the first pair, right/left from the second
    groups = [view.capture_group_id for view in response.views]
    assert groups == ["view-pair-1", "view-pair-2", "view-pair-2", "view-pair-1"]
    assert response.capture_group_id == "view-pair-1"
    assert len({view.camera_snapshot_id for view in response.views}) == 4
    # turned right for the second pair, then back to the observation's facing
    assert client.turns == [135.0, 45.0]
    assert runtime.yaw == 45.0
    response.to_dict()  # the wire accepts four views


def test_the_pair_observation_is_unchanged_by_the_quad_code():
    runtime = TurningRuntime()
    client = SpearTrackBClient(
        runtime, agent_tag="agent", spawn_pose=Pose(0.0, 0.0, 100.0, 45.0))
    client.episode(SimpleNamespace(episode_id="ep"))
    response = client.observe_views(ObserveViewsRequest(episode_id="ep", camera=CAMERA))
    assert tuple(view.view for view in response.views) == PIXEL_VIEWS
    assert all(view.capture_group_id is None for view in response.views)
    assert runtime.captures == 1


def test_a_left_view_pixel_resolves_against_the_second_pairs_rear_camera():
    runtime = TurningRuntime()
    client = quad_client(runtime)
    response = client.observe_views(
        ObserveViewsRequest(episode_id="ep", camera=CAMERA, views=PIXEL_VIEWS_QUAD))
    left = next(view for view in response.views if view.view == "left")

    walk = client.walk_pixel(WalkPixelRequest(
        episode_id="ep", pixel=PixelSpec(0.5, 0.8), camera=CAMERA, view="left",
        capture_group_id=left.capture_group_id,
        camera_snapshot_id=left.camera_snapshot_id, resolve_only=True))

    view_id, group, u, v, walked = runtime.executed[-1]
    assert (view_id, group) == ("rear", "view-pair-2")
    assert walk.arrived is False and walk.walked_cm == 0.0
    assert walk.resolved.raw_hit_actor == "BP_SplineSidewalk13"
    assert walk.resolved.raw_world_hit_z_cm == 15.0
    assert walk.resolved.direct_path_legal is True
    # the second pair's rear camera looks at 45 + 90 + 180 = 315 degrees
    hit = walk.resolved.raw_world_hit_cm
    assert hit == pytest.approx((500 * math.cos(math.radians(315.0)),
                                 500 * math.sin(math.radians(315.0))), abs=1e-6)
    assert (walk.view, walk.capture_group_id, walk.camera_snapshot_id) == (
        "left", left.capture_group_id, left.camera_snapshot_id)
    # a resolve turns the pawn back to the observation's facing
    assert runtime.yaw == 45.0


def test_a_mismatched_binding_is_refused_not_guessed():
    runtime = TurningRuntime()
    client = quad_client(runtime)
    response = client.observe_views(
        ObserveViewsRequest(episode_id="ep", camera=CAMERA, views=PIXEL_VIEWS_QUAD))
    front = response.views[0]
    with pytest.raises(Exception, match="group_mismatch"):
        client.walk_pixel(WalkPixelRequest(
            episode_id="ep", pixel=PixelSpec(0.5, 0.8), camera=CAMERA, view="front",
            capture_group_id="view-pair-2",
            camera_snapshot_id=front.camera_snapshot_id))


def test_walk_world_faces_the_point_names_its_pixel_and_walks():
    runtime = TurningRuntime()
    client = quad_client(runtime)
    client.observe_views(
        ObserveViewsRequest(episode_id="ep", camera=CAMERA, views=PIXEL_VIEWS_QUAD))

    walk = client.walk_world(0.0, 500.0, camera=CAMERA)

    assert client.turns[-1] == 90.0           # +Y is yaw 90 in UE
    view_id, group, u, v, walked = runtime.executed[-1]
    assert view_id == "front"
    assert u == pytest.approx(0.5, abs=1e-6)
    # the camera is 20 cm ahead and 163 cm up; the point 480 cm ahead of it
    expected = project_world_point_to_pixel(
        (20.0 * math.cos(math.radians(90.0)), 20.0 * math.sin(math.radians(90.0)), 163.0),
        90.0, (0.0, 500.0, 100.0 - 88.0))
    assert v == pytest.approx(expected[1], abs=1e-6)
    assert walk.arrived and walk.walked_cm == 500.0 and walk.sim_seconds == 2.5
    assert walk.start_pose is not None


def test_a_move_inside_a_leg_aims_at_the_legs_end_and_stops_at_the_node():
    # a node 1.2 m ahead is nearer than the picture reaches: the pawn aims
    # at the certified leg's end 3 m ahead and the acceptance radius stops
    # it at the node
    runtime = TurningRuntime()
    client = quad_client(runtime)
    client.observe_views(
        ObserveViewsRequest(episode_id="ep", camera=CAMERA, views=PIXEL_VIEWS_QUAD))

    client.walk_world(300.0, 0.0, camera=CAMERA, stop_cm=(120.0, 0.0))

    view_id, group, u, v, walked = runtime.executed[-1]
    aimed = project_world_point_to_pixel((20.0, 0.0, 163.0), 0.0, (300.0, 0.0, 12.0))
    assert v == pytest.approx(aimed[1], abs=1e-6)
    event = client.events[-1]
    assert event["kind"] == "world_leg" and event["target_cm"] == [300.0, 0.0]
    assert event["stop_cm"] == [120.0, 0.0]
    assert event["acceptance_radius_cm"] == pytest.approx(180.0 + 15.0)
    # a move to the leg's own end has no stop and the engine's own radius
    client.walk_world(600.0, 0.0, camera=CAMERA)
    assert client.events[-1]["stop_cm"] is None
    assert client.events[-1]["acceptance_radius_cm"] is None


def test_a_refused_leg_raises_with_the_engines_verdict():
    runtime = TurningRuntime()

    def refuse(action, frame_):
        raise PixelGoalRejected("controller_path_enters_unmarked_road",
                                audit={"raw_hit_actor": "BP_SplineRoad2"})
    runtime.execute = refuse
    client = quad_client(runtime)
    client.observe_views(
        ObserveViewsRequest(episode_id="ep", camera=CAMERA, views=PIXEL_VIEWS_QUAD))
    with pytest.raises(WorldLegRefused, match="unmarked_road"):
        client.walk_world(0.0, 500.0, camera=CAMERA)


# ------------------------------------------------------------ the pool graph --

@pytest.fixture(scope="module")
def pool():
    return with_every_leg_certified(load_validated_delivery_pool(TRUSTED_V2_POOL))


def test_nearest_pool_node_snaps_within_the_radius_only(pool):
    node = pool.nodes_by_id["recast-grid--193-9"]
    beside = (node.x_cm + 40.0, node.y_cm - 30.0)
    found = nearest_pool_node(pool, beside, radius_cm=150.0)
    assert found is not None and found[0].id == node.id
    assert found[1] == pytest.approx(50.0)
    assert nearest_pool_node(pool, (node.x_cm + 4000.0, node.y_cm), radius_cm=150.0) is None


def test_planned_legs_follow_the_phone_route_and_end_at_the_destination(pool):
    scenario = resolve_delivery_scenario(
        pool, mode="random", seed=9,
        constraints=OrderConstraints(min_delivery_cm=3000.0, max_delivery_cm=8000.0,
                                     min_turns=1, require_marked_crossing=True))
    planned = plan_pool_legs(pool, scenario.spawn.node_id, scenario.pickup.handover_node_id)
    assert planned is not None
    path, legs = planned
    assert path.node_ids[0] == scenario.spawn.node_id
    assert path.node_ids[-1] == scenario.pickup.handover_node_id
    assert legs[-1].stop == scenario.pickup.handover_node_id
    assert legs[-1].stop_cm == pool.nodes_by_id[scenario.pickup.handover_node_id].position
    assert 3 <= len(legs) <= len(path.node_ids)
    # every move rides a certified leg: it aims at the leg's end and stops
    # at a node on the leg's chain; the chains join up into the path
    chains = certified_leg_chains(pool)
    start = scenario.spawn.node_id
    walked = [start]
    for move in legs:
        chain = chains[(start, move.aim)]
        assert move.stop in chain[1:]
        walked.extend(chain[1:chain.index(move.stop) + 1])
        start = move.stop
    assert tuple(walked) == path.node_ids
    assert path.uses_marked_crossing is True
    assert path.length_cm == pytest.approx(sum(
        math.dist(pool.nodes_by_id[a].position, pool.nodes_by_id[b].position)
        for a, b in zip((scenario.spawn.node_id, *[m.stop for m in legs]), [m.stop for m in legs])))


def test_a_pool_without_a_leg_table_cannot_be_walked(pool):
    from dataclasses import replace
    bare = replace(pool, certified_legs=None, engine_certification=None)
    with pytest.raises(ValueError, match="no certified legs"):
        plan_pool_legs(bare, pool.spawns[0].node_id, pool.stops[0].handover_node_id)


def test_a_route_never_uses_a_leg_it_is_told_to_avoid(pool):
    start, end = pool.spawns[0].node_id, pool.stops[0].handover_node_id
    _path, legs = plan_pool_legs(pool, start, end)
    first = (start, legs[0].aim)
    _path, again = plan_pool_legs(pool, start, end, avoid=[first])
    assert (start, again[0].aim) != first
    assert again[-1].stop == end


# ------------------------------------------------------- the environment --

class RoutingClient:
    """A four-view engine client that resolves every pixel to one chosen
    world point and walks certified legs by teleporting the pose."""

    can_walk_world = True
    can_nudge = True

    def __init__(self, spawn: Pose, resolves_to: tuple[float, float], actor: str,
                 hit_z: float = 15.0) -> None:
        self.pose = spawn
        self.nudges: list[tuple[float, float]] = []
        self.resolves_to = resolves_to
        self.actor = actor
        self.hit_z = hit_z
        self.legs: list[tuple[float, float]] = []
        self.aims: list[tuple[float, float]] = []
        # land this far short of the stop, along the leg, on the walks listed
        # (1-based walk numbers); the engine's controller stops 5-30 cm
        # short as a rule and once in a while much more
        self.land_short_cm: dict[int, float] = {}
        self.refuse_legs_to: set[tuple[float, float]] = set()
        self.refuse_from: tuple[float, float] | None = None
        self.events = ()
        self.captures = 0

    def episode(self, request):
        return EpisodeResponse(episode_id=request.episode_id, pose=self.pose,
                               fixed_dt=1.0 / 30.0)

    def episode_end(self, request):
        return True

    def observe_views(self, request):
        self.captures += 1
        group = f"pair-{self.captures}"
        views = tuple(
            ObservedView(
                view=name, yaw_offset_deg=offset,
                camera_snapshot_id=f"{group}-{name}",
                camera_intrinsics_id="perspective-640x360-hfov90", status="ok",
                png_base64=ONE_PIXEL_B64, width=640, height=360,
                capture_group_id=(f"{group}-b" if name in ("left", "right") else group))
            for name, offset in (("front", 0.0), ("left", 270.0), ("right", 90.0), ("rear", 180.0)))
        return ObserveViewsResponse(capture_group_id=group, pose=self.pose, views=views)

    def walk_pixel(self, request):
        assert request.resolve_only, "the routing harness only resolves"
        return WalkPixelResponse(
            arrived=False, stuck=False, timeout=False, ticks=0, sim_seconds=0.0,
            pose=self.pose, walked_cm=0.0,
            resolved=ResolvedPixel(
                raw_world_hit_cm=self.resolves_to, accepted_target_cm=self.resolves_to,
                raw_hit_actor=self.actor, raw_world_hit_z_cm=self.hit_z,
                direct_path_legal=False),
            view=request.view, capture_group_id=request.capture_group_id,
            camera_snapshot_id=request.camera_snapshot_id)

    def nudge_world(self, x_cm, y_cm):
        self.nudges.append((x_cm, y_cm))
        self.pose = Pose(x_cm=x_cm, y_cm=y_cm, z_cm=self.pose.z_cm, yaw_deg=self.pose.yaw_deg)
        return self.pose

    def walk_world(self, x_cm, y_cm, *, camera, stop_cm=None):
        """Teleport to the stop (or the aim): the fake engine walks every
        leg it is not told to refuse. ``legs`` records where each walk
        landed, ``aims`` what it aimed at."""
        aim = (round(x_cm, 1), round(y_cm, 1))
        start = self.pose
        if aim in self.refuse_legs_to and (
                self.refuse_from is None
                or math.dist(self.refuse_from, (start.x_cm, start.y_cm)) < 1.0):
            raise WorldLegRefused("controller_path_enters_unmarked_road")
        landing = (x_cm, y_cm) if stop_cm is None else (float(stop_cm[0]), float(stop_cm[1]))
        short = self.land_short_cm.get(len(self.legs) + 1, 0.0)
        if short:
            length = math.dist((start.x_cm, start.y_cm), landing) or 1.0
            landing = (landing[0] - (landing[0] - start.x_cm) / length * short,
                       landing[1] - (landing[1] - start.y_cm) / length * short)
        distance = math.dist((start.x_cm, start.y_cm), landing)
        self.legs.append((round(landing[0], 1), round(landing[1], 1)))
        self.aims.append(aim)
        self.pose = Pose(x_cm=landing[0], y_cm=landing[1], z_cm=start.z_cm,
                         yaw_deg=math.degrees(math.atan2(y_cm - start.y_cm, x_cm - start.x_cm)))
        return WalkResponse(arrived=True, stuck=False, timeout=False, ticks=1,
                            sim_seconds=distance / 70.0, pose=self.pose,
                            walked_cm=distance, start_pose=start)


def pool_city(pool) -> RoadNetwork:
    named = {stop.street_index: stop.street_name for stop in pool.stops}
    streets = [
        Street(index=index, name=named.get(index, f"Street {index}"),
               source=f"s{index}", width_cm=600.0,
               polyline=[(0.0, 0.0), (100.0, 0.0)])
        for index in range(max(node.street_index for node in pool.nodes) + 1)
    ]
    addresses = [
        Address(building_id=stop.building_id, street_index=stop.street_index,
                street_name=stop.street_name, number=stop.number, arc_cm=0.0,
                offset_cm=0.0, side="semantic", door=(0.0, 0.0),
                nearest_node="x", poi_type=stop.poi_type)
        for stop in pool.stops
    ]
    city = RoadNetwork(map_name="pool-city", streets=streets,
                       nodes={"x": StreetNode(id="x", x_cm=0.0, y_cm=0.0,
                                              street_index=0, arc_cm=0.0)},
                       addresses=addresses)
    return build_validated_pool_network(city, pool)


def make_pool_env(tmp_path, pool, client, scenario):
    env = ValidatedPoolDeliveryEnv(
        pool_city(pool), client, delivery_pool=pool, scenario=scenario,
        street_camera=CAMERA, episode_id="quad-routing",
        cache_root=tmp_path / "album",
        action_space=ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
        camera_view=CAMERA_VIEW_FRONT_REAR, pixel_views=PIXEL_VIEWS_QUAD,
        embodiment="human_on_foot", difficulty="solo", seed=9,
        spawn_z_cm=scenario.spawn.z_cm)
    env.reset()
    return env


@pytest.fixture
def seed9(pool):
    return resolve_delivery_scenario(
        pool, mode="random", seed=9,
        constraints=OrderConstraints(min_delivery_cm=3000.0, max_delivery_cm=8000.0,
                                     min_turns=1, require_marked_crossing=True))


def observe(env):
    rows = env.candidates() if hasattr(env, "candidates") else []
    return env.photo_rows(rows)


def test_a_pixel_beside_the_pavement_walks_the_certified_legs_to_that_node(tmp_path, pool, seed9):
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    # a point 1.2 m off a certified node twelve metres along the route
    path, legs = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
    destination = pool.nodes_by_id[path.node_ids[12]]
    client = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(destination.x_cm + 30.0, destination.y_cm + 20.0),
        actor="BP_SplineSidewalk13")
    env = make_pool_env(tmp_path, pool, client, seed9)
    assert env.pedestrian_routing is True
    rows = observe(env)
    assert [row["view"] for row in rows] == list(PIXEL_VIEWS_QUAD)
    assert rows[1]["street"] == "to your left"

    outcome = env.walk_to_pixel(view="left", u=0.4, v=0.8)

    assert outcome.ok and outcome.moved, outcome.message
    assert "along the pavement" in outcome.message
    assert env.node_id == destination.id
    assert client.legs, "no leg was walked"
    assert client.legs[-1] == (round(destination.x_cm, 1), round(destination.y_cm, 1))
    assert env.walked_cm > 0 and env.turns == 1 and env.rejected_actions == 0
    record = [row for row in env.embodied_log if row.get("kind") == "pixel_goal"][-1]
    assert record["destination_node"] == destination.id
    assert record["surface"] == "pedestrian" and record["complete"] is True
    assert env.pixel_view_counts["left"] == 1
    # the summary reads the routed walk like any other movement record
    summary = env.summary()
    assert summary["embodied"]["pixel_walks"] == 1
    assert summary["embodied"]["max_pose_error_cm"] == record["pose_error_cm"]
    assert summary["embodied"]["pixel_view_counts"]["left"] == 1


def test_a_point_on_the_far_pavement_is_walked_over_the_marked_crossing(tmp_path, pool, seed9):
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    pickup = pool.nodes_by_id[seed9.pickup.handover_node_id]
    client = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(pickup.x_cm + 30.0, pickup.y_cm), actor="BP_SplineSidewalk9")
    env = make_pool_env(tmp_path, pool, client, seed9)
    observe(env)

    outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)

    assert outcome.ok and "marked crossing" in outcome.message
    assert env.node_id == pickup.id
    assert env.house_numbers_near(env.node_id) == "11"
    assert env.collect().ok


def test_the_carriageway_is_refused_unless_at_the_gutter(tmp_path, pool, seed9):
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    client = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(spawn_node.x_cm + 400.0, spawn_node.y_cm + 400.0),
        actor="BP_SplineRoad4")
    env = make_pool_env(tmp_path, pool, client, seed9)
    observe(env)

    outcome = env.walk_to_pixel(view="right", u=0.5, v=0.7)

    assert not outcome.ok and outcome.code == "unwalkable_pixel"
    assert "carriageway" in outcome.message
    assert client.legs == [] and env.rejected_actions == 1


def test_a_wall_above_the_ground_is_refused_but_its_foot_is_the_paving(tmp_path, pool, seed9):
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    path, _legs = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
    near = pool.nodes_by_id[path.node_ids[6]]
    high = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(near.x_cm + 20.0, near.y_cm), actor="BP_ProceduralBuilding3",
        hit_z=180.0)
    env = make_pool_env(tmp_path, pool, high, seed9)
    observe(env)
    outcome = env.walk_to_pixel(view="front", u=0.5, v=0.4)
    assert not outcome.ok and "wall" in outcome.message

    low = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(near.x_cm + 20.0, near.y_cm), actor="BP_ProceduralBuilding3",
        hit_z=20.0)
    env = make_pool_env(tmp_path, pool, low, seed9)
    observe(env)
    outcome = env.walk_to_pixel(view="front", u=0.5, v=0.9)
    assert outcome.ok and env.node_id == near.id


def test_a_refused_leg_is_gone_round_by_planning_again_without_it(tmp_path, pool, seed9):
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    path, legs = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
    destination = pool.nodes_by_id[path.node_ids[12]]
    client = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(destination.x_cm, destination.y_cm), actor="BP_SplineSidewalk13")
    _path, plan = plan_pool_legs(pool, seed9.spawn.node_id, destination.id)
    first_aim = (round(plan[0].aim_cm[0], 1), round(plan[0].aim_cm[1], 1))
    client.refuse_legs_to.add(first_aim)
    client.refuse_from = (spawn_node.x_cm, spawn_node.y_cm)
    env = make_pool_env(tmp_path, pool, client, seed9)
    observe(env)

    outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)

    assert outcome.ok, outcome.message
    assert client.aims[0] != first_aim, "the refused leg was not gone round"
    assert env.node_id == destination.id
    refused = [row for row in env.embodied_log if row.get("kind") == "pedestrian_leg_refused"]
    assert refused and refused[0]["reason"] == "controller_path_enters_unmarked_road"
    assert refused[0]["aim"] == plan[0].aim
    record = [row for row in env.embodied_log if row.get("kind") == "pixel_goal"][-1]
    assert record["replans"] == 1 and record["refused_legs"] == [[seed9.spawn.node_id, plan[0].aim]]
    assert env.summary()["embodied"]["pedestrian_replans"] == 1


def test_a_walk_the_engine_refuses_everywhere_ends_where_the_pawn_stands(tmp_path, pool, seed9):
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    path, _legs = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
    destination = pool.nodes_by_id[path.node_ids[12]]
    client = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(destination.x_cm, destination.y_cm), actor="BP_SplineSidewalk13")
    # every leg out of the spawn node is refused
    for (start, end) in pool.certified_legs:
        if start == seed9.spawn.node_id:
            node = pool.nodes_by_id[end]
            client.refuse_legs_to.add((round(node.x_cm, 1), round(node.y_cm, 1)))
    client.refuse_from = (spawn_node.x_cm, spawn_node.y_cm)
    env = make_pool_env(tmp_path, pool, client, seed9)
    observe(env)

    outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)

    assert not outcome.ok and outcome.code == "stuck"
    assert env.node_id == seed9.spawn.node_id and client.legs == []
    record = [row for row in env.embodied_log if row.get("kind") == "pixel_goal"][-1]
    assert record["complete"] is False and record["landed_node"] == seed9.spawn.node_id
    assert 1 <= record["replans"] <= env.max_leg_replans + 1


def test_the_quad_prompt_names_four_photographs_and_the_pedestrian_walk(tmp_path, pool, seed9):
    from embodiedbench.agent.courier.session import CourierSession
    spawn_node = pool.nodes_by_id[seed9.spawn.node_id]
    client = RoutingClient(
        Pose(spawn_node.x_cm, spawn_node.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
        resolves_to=(0.0, 0.0), actor="BP_SplineSidewalk13")
    env = make_pool_env(tmp_path, pool, client, seed9)
    session = CourierSession(env, city="Paris")
    prompt = session.system_prompt()
    assert '"front", "left", "right" or "rear"' in prompt
    assert "along pedestrian ways" in prompt
    assert "front/rear" not in prompt.split("TOOLS")[0]
    observation = session.observe()
    text = getattr(observation, "text", None) or getattr(observation, "obs_str", None) or str(observation)
    assert "[left," in text and "[right," in text


# ------------------------------------------------------------- the nudge --

def test_nudge_world_sets_the_pawn_on_the_point_and_records_it():
    runtime = TurningRuntime()
    moves: list[tuple[float, float, float, float]] = []
    client = SpearTrackBClient(
        runtime, agent_tag="agent", spawn_pose=Pose(x_cm=0.0, y_cm=0.0, z_cm=100.0, yaw_deg=45.0),
        turner=lambda yaw: None, mover=lambda x, y, z, yaw: moves.append((x, y, z, yaw)),
        views=PIXEL_VIEWS_QUAD, turn_settle_s=0.0)
    client.episode(SimpleNamespace(episode_id="ep"))
    assert client.can_nudge
    pose = client.nudge_world(20.0, -10.0)
    # the pawn is placed at the point at its own height, facing as it was
    assert moves == [(20.0, -10.0, 100.0, 45.0)]
    assert (pose.x_cm, pose.y_cm, pose.z_cm, pose.yaw_deg) == (20.0, -10.0, 100.0, 45.0)
    event = client.events[-1]
    assert event["kind"] == "world_nudge" and event["to_cm"] == [20.0, -10.0]
    assert event["distance_cm"] == round(math.hypot(20.0, 10.0), 2) and event["residual_cm"] == 0.0


def test_a_client_without_a_mover_cannot_nudge():
    client = quad_client(TurningRuntime())
    assert not client.can_nudge
    with pytest.raises(RuntimeError, match="mover"):
        client.nudge_world(1.0, 1.0)


class TestPawnOnTheNode:
    """Before a certified leg the pawn is set back onto the node it stands
    off by more than a step and less than ``NUDGE_MAX_CM``; on it, nothing
    is done; further off, the harness has lost the pawn and the walk is
    refused rather than walked from an uncertified spot."""

    def _env_with_offset(self, tmp_path, pool, seed9, dx_cm):
        spawn = pool.nodes_by_id[seed9.spawn.node_id]
        # a pixel on the pavement a few metres along the approach
        ahead = pool.nodes_by_id[seed9.approach.node_ids[min(4, len(seed9.approach.node_ids) - 1)]]
        client = RoutingClient(
            Pose(x_cm=spawn.x_cm + dx_cm, y_cm=spawn.y_cm, z_cm=seed9.spawn.z_cm, yaw_deg=seed9.spawn.yaw_deg),
            resolves_to=(ahead.x_cm, ahead.y_cm), actor="BP_SplineSidewalk13")
        env = make_pool_env(tmp_path, pool, client, seed9)
        return env, client, (spawn.x_cm, spawn.y_cm)

    def test_a_step_off_the_node_is_corrected_before_the_first_leg(self, tmp_path, pool, seed9):
        env, client, node_xy = self._env_with_offset(tmp_path, pool, seed9, 20.0)
        observe(env)
        outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)
        assert outcome.ok, outcome.message
        assert client.nudges and client.nudges[0] == node_xy
        assert env.pedestrian_nudges >= 1
        nudge = next(e for e in env.embodied_log if e.get("kind") == "pedestrian_nudge")
        assert nudge["distance_cm"] == 20.0
        assert env.summary()["embodied"]["pedestrian_nudges"] == env.pedestrian_nudges

    def test_a_walk_ends_with_the_pawn_set_back_onto_its_node(self, tmp_path, pool, seed9):
        # the controller leaves the pawn 30 cm short of the last stop: the
        # next photograph, and the pixel resolved from it, must come from
        # the node, so the walk ends by setting the pawn onto it
        spawn = pool.nodes_by_id[seed9.spawn.node_id]
        path, _legs = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
        destination = pool.nodes_by_id[path.node_ids[12]]
        client = RoutingClient(
            Pose(spawn.x_cm, spawn.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
            resolves_to=(destination.x_cm, destination.y_cm), actor="BP_SplineSidewalk13")
        _p, plan = plan_pool_legs(pool, seed9.spawn.node_id, destination.id)
        client.land_short_cm = {len(plan): 30.0}
        env = make_pool_env(tmp_path, pool, client, seed9)
        observe(env)
        outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)
        assert outcome.ok, outcome.message
        assert env.node_id == destination.id
        assert client.nudges[-1] == (destination.x_cm, destination.y_cm)
        assert (client.pose.x_cm, client.pose.y_cm) == (destination.x_cm, destination.y_cm)
        record = [row for row in env.embodied_log if row.get("kind") == "pixel_goal"][-1]
        assert record["snap_cm"] == 0.0

    def test_a_move_that_stops_far_short_is_planned_again_from_where_the_pawn_stands(
            self, tmp_path, pool, seed9):
        spawn = pool.nodes_by_id[seed9.spawn.node_id]
        path, _legs = plan_pool_legs(pool, seed9.spawn.node_id, seed9.pickup.handover_node_id)
        destination = pool.nodes_by_id[path.node_ids[12]]
        client = RoutingClient(
            Pose(spawn.x_cm, spawn.y_cm, seed9.spawn.z_cm, seed9.spawn.yaw_deg),
            resolves_to=(destination.x_cm, destination.y_cm), actor="BP_SplineSidewalk13")
        client.land_short_cm = {1: 120.0}        # the first move, 1.2 m short
        env = make_pool_env(tmp_path, pool, client, seed9)
        observe(env)
        outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)
        short = [row for row in env.embodied_log if row.get("kind") == "pedestrian_stopped_short"]
        assert short and short[0]["distance_cm"] == pytest.approx(120.0, abs=1.0)
        record = [row for row in env.embodied_log if row.get("kind") == "pixel_goal"][-1]
        assert record["replans"] >= 1
        # planned again from a certified node within a nudge of the pawn,
        # and the walk still got there
        assert outcome.ok, outcome.message
        assert env.node_id == destination.id

    def test_on_the_node_nothing_is_done(self, tmp_path, pool, seed9):
        env, client, _node = self._env_with_offset(tmp_path, pool, seed9, 2.0)
        observe(env)
        assert env.walk_to_pixel(view="front", u=0.5, v=0.7).ok
        assert client.nudges == [] and env.pedestrian_nudges == 0

    def test_a_pawn_lost_off_its_node_is_not_walked(self, tmp_path, pool, seed9):
        env, client, _node = self._env_with_offset(tmp_path, pool, seed9, NUDGE_MAX_CM + 25.0)
        observe(env)
        outcome = env.walk_to_pixel(view="front", u=0.5, v=0.7)
        assert not outcome.ok and outcome.code == "stuck"
        assert client.nudges == [] and client.legs == []
        lost = [e for e in env.embodied_log if e.get("kind") == "pedestrian_off_node"]
        assert lost and lost[0]["node"] == seed9.spawn.node_id
        assert lost[0]["distance_cm"] == pytest.approx(NUDGE_MAX_CM + 25.0)
