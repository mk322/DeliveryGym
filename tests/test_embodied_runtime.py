"""Track B: the courier whose legs are a UE pawn.

The claim under test is UE ownership of locomotion and locomotion time
(spec 3b), asserted against a kinematic reference service whose tick counts
are exact by construction:

1. **the bytes** -- the Track B messages reproduce the ``track_b_*`` golden
   fixtures byte for byte, and ``RenderResult`` gained its ``pose`` without
   moving a single Track A byte;
2. **the client and the lease** -- the four stateful endpoints over real
   HTTP, and the pool lease that takes an instance out of /render dispatch
   for exactly the episode's duration;
3. **the env** -- on the real Paris graph: the node trajectory is the stock
   GRAPH's, the clock is the integrator's ``ticks * fixed_dt`` plus the
   declared non-movement costs and nothing else, a wall maps to the stock
   refusal semantics with an engine-measured charge, and the ``embodied``
   summary block carries the I/O evidence the spec demands;
4. **the adapter** -- the training contract (obs shape, info keys, success)
   unchanged over the embodied backend, hazards refused, the lease returned
   on close.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import math
from pathlib import Path
import urllib.error
import urllib.request

import contextlib
import pytest

from embodiedbench.compiler.road_network import (
    RoadNetwork,
    Street,
    StreetNode,
    bearing_deg,
    build_road_network,
)
from embodiedbench.runtime.city.courier_env import (
    REJECTED_ACTION_SECONDS,
    CourierEnv,
    relative_of,
)
from embodiedbench.runtime.live.client import (
    BadRequestError,
    ServiceBusy,
    UERenderClient,
)
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL,
    ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
    CAMERA_VIEW_FORWARD,
    CAMERA_VIEW_FRONT_REAR,
    EmbodiedCourierEnv,
)
from embodiedbench.runtime.live.gym_adapter import EmbodiedCourierGymEnv
from embodiedbench.runtime.live.pool import NoHealthyInstance, RenderPool
from embodiedbench.runtime.live.protocol import (
    CameraSpec,
    EpisodeEndRequest,
    EpisodeRequest,
    EpisodeResponse,
    ObserveRequest,
    ObserveViewsRequest,
    ObserveViewsResponse,
    ObservedView,
    PixelSpec,
    PIXEL_VIEW_FRONT,
    PIXEL_VIEW_REAR,
    Pose,
    ProtocolViolation,
    ResolvedPixel,
    RenderResult,
    AgentSpec,
    WalkPixelRequest,
    WalkPixelResponse,
    WalkRequest,
    WalkResponse,
    dumps,
)
from embodiedbench.runtime.pixel_goal import PixelGoalPairIntegrityError

from live_stub import FakeTrackBService, write_endpoints

GOLDEN = Path(__file__).resolve().parent / "golden" / "nav_render_v0"
MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps")
PARIS = MAPS / "citycore-paris"
needs_maps = pytest.mark.skipif(not PARIS.exists(), reason="vendored maps not present")

EPISODE = "courier-citycore-paris-s5-embodied"
CAMERA = CameraSpec(640, 480, 90.0)
FIXED_DT = FakeTrackBService.FIXED_DT


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


@pytest.fixture()
def service(tmp_path):
    stub = FakeTrackBService(tmp_path / "svc").start()
    yield stub
    stub.stop()


def embodied_env(paris, client, tmp_path, *, seed=5, **kwargs):
    env = EmbodiedCourierEnv(paris, client,
                             episode_id=EPISODE,
                             cache_root=tmp_path / "cache",
                             seed=seed, **kwargs)
    env.reset()
    return env


def stand_at(env, node_id):
    """Move the graph position directly, the way the reference tests do. The
    pawn deliberately stays where it is: the divergence is the scenario."""
    env.node_id = node_id
    env.arrived_from = None


def expected_ticks(start_xy, target_xy, arrive_cm, speed_cm_s=140.0):
    """The integrator's own arithmetic: ticks to bring the remaining distance
    within ``arrive_cm`` at ``speed * dt`` per tick, never overshooting."""
    distance = math.dist(start_xy, target_xy)
    if distance <= arrive_cm:
        return 0
    return math.ceil((distance - arrive_cm) / (speed_cm_s * FIXED_DT))


def pose_match_env(
    nodes: dict[str, StreetNode], *, current: str, walked_bearing: float,
) -> EmbodiedCourierEnv:
    """The pure map-matching slice, without opening a renderer episode."""
    env = object.__new__(EmbodiedCourierEnv)
    env.network = RoadNetwork(
        map_name="pose-match-test",
        streets=[Street(0, "Test Street", "test", 600.0,
                        [(0.0, 0.0), (1000.0, 0.0)])],
        nodes=nodes,
    )
    env.node_id = current
    env.action_space = ACTION_SPACE_PIXEL_GOAL
    env._walked_bearing = walked_bearing
    env._last_pose_match = None
    return env


def post_track_b_raw(base_url, path, body):
    request = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# ─────────────────────────────────────────────────────────────────────────────


class TestTheTrackBGoldenBytes:
    """The same compatibility contract as Track A: the fixture files are the
    protocol, shared byte for byte with the SimWorld2 branch."""

    @pytest.mark.parametrize("name, cls", [
        ("track_b_episode_request.json", EpisodeRequest),
        ("track_b_episode_response.json", EpisodeResponse),
        ("track_b_walk_request.json", WalkRequest),
        ("track_b_walk_response.json", WalkResponse),
        ("track_b_walk_pixel_request.json", WalkPixelRequest),
        ("track_b_walk_pixel_response.json", WalkPixelResponse),
        ("track_b_observe_request.json", ObserveRequest),
        ("track_b_observe_result.json", RenderResult),
        ("track_b_episode_end_request.json", EpisodeEndRequest),
    ])
    def test_each_track_b_message_round_trips_byte_exactly(self, name, cls):
        golden = (GOLDEN / name).read_text()
        message = cls.from_dict(json.loads(golden))
        assert dumps(message.to_dict()) == golden

    def test_the_dual_view_messages_round_trip_byte_exactly(self):
        for name, cls in (
            ("track_b_observe_views_request.json", ObserveViewsRequest),
            ("track_b_observe_views_response.json", ObserveViewsResponse),
            ("track_b_walk_pixel_bound_request.json", WalkPixelRequest),
        ):
            golden = (GOLDEN / name).read_text()
            assert dumps(cls.from_dict(json.loads(golden)).to_dict()) == golden

    def test_the_old_walk_pixel_fixture_stays_byte_exact(self):
        golden = (GOLDEN / "track_b_walk_pixel_request.json").read_text()
        parsed = WalkPixelRequest.from_dict(json.loads(golden))
        assert parsed.view is None
        assert parsed.capture_group_id is None
        assert parsed.camera_snapshot_id is None
        assert dumps(parsed.to_dict()) == golden

    def test_the_walk_response_fixture_is_kinematically_coherent(self):
        """The exemplar's numbers must be the reference integrator's own, or
        the fixture teaches the wrong arithmetic to whoever implements the
        service side from it."""
        request = WalkRequest.from_dict(
            json.loads((GOLDEN / "track_b_walk_request.json").read_text()))
        response = WalkResponse.from_dict(
            json.loads((GOLDEN / "track_b_walk_response.json").read_text()))
        spawn = EpisodeRequest.from_dict(
            json.loads((GOLDEN / "track_b_episode_request.json").read_text())).spawn
        ticks = expected_ticks((spawn.x_cm, spawn.y_cm),
                               (request.target_x_cm, request.target_y_cm),
                               request.arrive_cm)
        assert response.ticks == ticks
        assert response.sim_seconds == pytest.approx(ticks * FIXED_DT, abs=1e-4)

    def test_a_result_without_a_pose_serialises_without_the_key(self):
        """``pose`` postdates the Track A fixtures; omitted-when-absent is
        what lets both repos gain it with zero golden bytes moving."""
        result = RenderResult(key="n0/toward_n1", status="ok",
                              path="/tmp/frame.png", sha256="00", width=64,
                              height=48)
        assert "pose" not in result.to_dict()

    def test_the_track_a_response_fixture_still_round_trips_unchanged(self):
        """The direct proof that the pose field cannot move the old bytes:
        the Track A response fixture parses to pose-less results and
        re-serialises to its own bytes."""
        golden = (GOLDEN / "render_response.json").read_text()
        parsed = json.loads(golden)
        results = [RenderResult.from_dict(r) for r in parsed["results"]]
        assert all(r.pose is None for r in results)
        assert dumps({"results": [r.to_dict() for r in results]}) == golden

    def test_a_pose_round_trips_through_a_result(self):
        result = RenderResult(
            key="observe/000001", status="ok", path="/tmp/f.png", sha256="00",
            width=640, height=480,
            pose=Pose(x_cm=1.5, y_cm=-2.5, z_cm=100.0, yaw_deg=90.0))
        assert RenderResult.from_dict(result.to_dict()) == result

    def test_a_wrong_protocol_string_is_refused(self):
        data = json.loads((GOLDEN / "track_b_episode_request.json").read_text())
        data["protocol"] = "nav-render/v1"
        with pytest.raises(ProtocolViolation):
            EpisodeRequest.from_dict(data)


def episode_request(episode_id=EPISODE, spawn=Pose(0.0, 0.0, 100.0, 0.0)):
    return EpisodeRequest(
        episode_id=episode_id, map_name="citycore-paris",
        agent=AgentSpec(speed_cm_s=140.0, eye_z_cm=160.0, camera=CAMERA),
        spawn=spawn)


class TestTheClientSpeaksTrackB:
    def test_episode_spawns_the_agent_and_reports_the_fixed_dt(self, service):
        client = UERenderClient(service.base_url)
        response = client.episode(episode_request(
            spawn=Pose(10.0, 20.0, 100.0, 45.0)))
        assert response.episode_id == EPISODE
        assert response.fixed_dt == pytest.approx(FIXED_DT)
        assert (response.pose.x_cm, response.pose.y_cm) == (10.0, 20.0)
        assert service.episodes[0]["agent"]["speed_cm_s"] == 140.0

    def test_a_walk_arrives_with_exact_tick_counts(self, service):
        """The integrator is deterministic on purpose: 950 cm of travel at
        4.662 cm per tick is 204 ticks, and ``sim_seconds`` is that times the
        fixed dt -- the arithmetic every clock assertion downstream uses."""
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        walk = client.walk(WalkRequest(
            episode_id=EPISODE, target_x_cm=1000.0, target_y_cm=0.0,
            arrive_cm=50.0))
        ticks = expected_ticks((0.0, 0.0), (1000.0, 0.0), 50.0)
        assert ticks == 204  # the worked example, pinned
        assert walk.arrived and not walk.stuck and not walk.timeout
        assert walk.ticks == ticks
        assert walk.sim_seconds == pytest.approx(ticks * FIXED_DT)
        assert walk.walked_cm == pytest.approx(ticks * 140.0 * FIXED_DT)
        assert math.dist((walk.pose.x_cm, walk.pose.y_cm),
                         (1000.0, 0.0)) <= 50.0

    def test_observe_returns_a_frame_with_the_pose_echo(self, service):
        from PIL import Image

        client = UERenderClient(service.base_url)
        client.episode(episode_request(spawn=Pose(5.0, 6.0, 100.0, 0.0)))
        result = client.observe(ObserveRequest(
            episode_id=EPISODE, camera=CAMERA, yaw_deg=90.0))
        assert result.ok and result.key.startswith("observe/")
        assert result.pose is not None
        assert (result.pose.x_cm, result.pose.y_cm) == (5.0, 6.0)
        assert result.pose.yaw_deg == 90.0
        with Image.open(result.path) as image:
            assert image.size == (640, 480)

    def test_episode_end_despawns_and_answers_ok(self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        assert client.episode_end(EpisodeEndRequest(episode_id=EPISODE))
        assert service.agent is None
        assert service.episode_ends == [EPISODE]

    def test_a_busy_instance_raises_service_busy_not_a_strike_signal(self, service):
        """A stateful endpoint's busy means "another episode holds this
        instance" -- the same transient taxonomy as a saturated render, and
        just as much not a health verdict."""
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        service.busy_track_b = True
        with pytest.raises(ServiceBusy):
            client.walk(WalkRequest(episode_id=EPISODE,
                                    target_x_cm=100.0, target_y_cm=0.0))

    def test_a_walk_for_an_unknown_episode_is_the_callers_bug(self, service):
        client = UERenderClient(service.base_url)
        with pytest.raises(BadRequestError):
            client.walk(WalkRequest(episode_id="never-spawned",
                                    target_x_cm=0.0, target_y_cm=0.0))

    def test_a_walk_pixel_resolves_and_arrives_with_exact_tick_counts(
            self, service):
        """Same integrator as ``/walk``, driven by a target the service
        resolved from a pixel instead of one the caller named directly --
        FakeTrackBService's ``_resolve_pixel`` puts the on-axis pixel 800 cm
        straight ahead of whatever the agent's spawn yaw is, so the expected
        tick count is exactly the same closed form ``/walk`` uses."""
        client = UERenderClient(service.base_url)
        client.episode(episode_request(spawn=Pose(0.0, 0.0, 100.0, 0.0)))
        walk = client.walk_pixel(WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(u=0.5, v=0.8), camera=CAMERA))
        assert walk.resolved.rejection_reason is None
        assert walk.resolved.raw_world_hit_cm is not None
        assert walk.resolved.accepted_target_cm == walk.resolved.raw_world_hit_cm
        expected_target = (800.0, 0.0)
        assert walk.resolved.accepted_target_cm == pytest.approx(
            expected_target, abs=0.5)
        ticks = expected_ticks((0.0, 0.0), expected_target, 50.0)
        assert walk.arrived and not walk.stuck and not walk.timeout
        assert walk.ticks == ticks
        assert walk.sim_seconds == pytest.approx(ticks * FIXED_DT)

    def test_a_walk_pixel_above_the_horizon_never_walks_and_says_why(
            self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        walk = client.walk_pixel(WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(u=0.5, v=0.3), camera=CAMERA))
        assert walk.resolved.rejection_reason == "no_ground_hit"
        assert walk.resolved.raw_world_hit_cm is None
        assert not walk.arrived and walk.ticks == 0
        assert walk.sim_seconds == 0.0

    def test_a_busy_instance_refuses_walk_pixel_too(self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        service.busy_track_b = True
        with pytest.raises(ServiceBusy):
            client.walk_pixel(WalkPixelRequest(
                episode_id=EPISODE, pixel=PixelSpec(u=0.5, v=0.8),
                camera=CAMERA))

    def test_a_view_pair_keeps_the_pawn_still_and_binds_the_rear_pixel(
            self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())

        pair = client.observe_views(ObserveViewsRequest(
            episode_id=EPISODE, camera=CAMERA, return_mode="path"))

        assert tuple(view.view for view in pair.views) == ("front", "rear")
        assert pair.pose.yaw_deg == 0.0
        assert service.agent["yaw"] == 0.0

        rear = pair.views[1]
        walk = client.walk_pixel(WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(0.5, 0.8), camera=CAMERA,
            view="rear", capture_group_id=pair.capture_group_id,
            camera_snapshot_id=rear.camera_snapshot_id,
        ))

        assert walk.resolved.accepted_target_cm[0] < 0.0
        assert (walk.view, walk.capture_group_id, walk.camera_snapshot_id) == (
            "rear", pair.capture_group_id, rear.camera_snapshot_id)

    def test_a_mismatched_view_snapshot_is_refused_before_walking(self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        pair = client.observe_views(ObserveViewsRequest(
            episode_id=EPISODE, camera=CAMERA, return_mode="path"))

        before = len(service.walk_pixels)
        with pytest.raises(BadRequestError):
            client.walk_pixel(WalkPixelRequest(
                episode_id=EPISODE, pixel=PixelSpec(0.5, 0.8), camera=CAMERA,
                view="rear", capture_group_id=pair.capture_group_id,
                camera_snapshot_id=pair.views[0].camera_snapshot_id,
            ))
        assert len(service.walk_pixels) == before

    def test_a_bound_pixel_resolution_rejection_echoes_its_capture_binding(
            self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        pair = client.observe_views(ObserveViewsRequest(
            episode_id=EPISODE, camera=CAMERA, return_mode="path"))
        rear = pair.views[1]

        walk = client.walk_pixel(WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(0.5, 0.3), camera=CAMERA,
            view="rear", capture_group_id=pair.capture_group_id,
            camera_snapshot_id=rear.camera_snapshot_id,
        ))

        assert walk.resolved.rejection_reason == "no_ground_hit"
        assert (walk.view, walk.capture_group_id, walk.camera_snapshot_id) == (
            "rear", pair.capture_group_id, rear.camera_snapshot_id)

    def test_an_unbound_pixel_keeps_the_existing_three_argument_resolver(
            self, service, monkeypatch):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        calls = []

        def resolve(agent, u, v):
            calls.append((agent["yaw"], u, v))
            return (800.0, 0.0), None

        monkeypatch.setattr(service, "_resolve_pixel", resolve)
        client.walk_pixel(WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(0.5, 0.8), camera=CAMERA))

        assert calls == [(0.0, 0.5, 0.8)]

    def test_a_partial_pixel_binding_is_refused_before_walking(self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        body = WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(0.5, 0.8), camera=CAMERA,
        ).to_dict()
        body["view"] = "rear"

        status, response = post_track_b_raw(service.base_url, "/walk_pixel", body)

        assert status == 400
        assert response["error"]["code"] == "bad_request"
        assert response["view"] == "rear"
        assert service.walk_pixels == []

    def test_a_bound_snapshot_mismatch_echoes_the_capture_identity(self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        pair = client.observe_views(ObserveViewsRequest(
            episode_id=EPISODE, camera=CAMERA, return_mode="path"))
        body = WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(0.5, 0.8), camera=CAMERA,
            view="rear", capture_group_id=pair.capture_group_id,
            camera_snapshot_id=pair.views[0].camera_snapshot_id,
        ).to_dict()

        status, response = post_track_b_raw(service.base_url, "/walk_pixel", body)

        assert status == 400
        assert response["error"]["code"] == "bad_request"
        assert (response["view"], response["capture_group_id"],
                response["camera_snapshot_id"]) == (
                    "rear", pair.capture_group_id,
                    pair.views[0].camera_snapshot_id)
        assert service.walk_pixels == []

    def test_an_inactive_bound_pixel_error_echoes_the_capture_identity(
            self, service):
        body = WalkPixelRequest(
            episode_id=EPISODE, pixel=PixelSpec(0.5, 0.8), camera=CAMERA,
            view="rear", capture_group_id="view-pair-17",
            camera_snapshot_id="view-pair-17-rear",
        ).to_dict()

        status, response = post_track_b_raw(service.base_url, "/walk_pixel", body)

        assert status == 400
        assert response["error"]["code"] == "bad_request"
        assert (response["view"], response["capture_group_id"],
                response["camera_snapshot_id"]) == (
                    "rear", "view-pair-17", "view-pair-17-rear")
        assert service.walk_pixels == []


class TestTheEmbodiedLease:
    def batch(self, key="n0/toward_n1"):
        from embodiedbench.runtime.live.protocol import RenderBatch, RenderItem

        return RenderBatch(
            episode_id="ep-render", return_mode="path",
            camera=CameraSpec(64, 48, 90.0),
            requests=(RenderItem(key=key, x_cm=0.0, y_cm=0.0, z_cm=160.0,
                                 yaw_deg=0.0, render_kind="street_view"),))

    def test_a_leased_instance_is_skipped_by_render_dispatch(self, tmp_path):
        first = FakeTrackBService(tmp_path / "a", instance_id="ue-a").start()
        second = FakeTrackBService(tmp_path / "b", instance_id="ue-b").start()
        try:
            pool = RenderPool(write_endpoints(tmp_path / "endpoints.json",
                                              [first, second]))
            with pool.lease_embodied("ep-emb") as client:
                leased = next(m for m in pool.members
                              if m.client is client)
                assert leased.leased_to == "ep-emb"
                for index in range(4):
                    pool.render(self.batch(f"n0/toward_{index}"))
                spare = first if leased.id == "ue-b" else second
                busy = first if spare is second else second
                assert len(spare.batches) == 4, (
                    "every batch must land on the unleased instance")
                assert len(busy.batches) == 0
            # Released: dispatch spreads over both again.
            assert all(m.leased_to is None for m in pool.members)
            for index in range(2):
                pool.render(self.batch(f"n1/toward_{index}"))
            assert len(first.batches) + len(second.batches) == 6
            assert min(len(first.batches), len(second.batches)) >= 1
        finally:
            first.stop()
            second.stop()

    def test_a_fully_leased_fleet_refuses_renders_rather_than_sharing(
            self, tmp_path, service):
        """Exclusivity is the point of the lease: an embodied episode's
        instance must not also photograph other episodes' streets, because
        the stateful scene (the pawn) would be in them."""
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        with pool.lease_embodied("ep-emb"):
            with pytest.raises(NoHealthyInstance):
                pool.render(self.batch())
        assert pool.render(self.batch())[0].ok

    def test_the_lease_is_released_on_error_too(self, tmp_path, service):
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        with pytest.raises(RuntimeError, match="episode died"):
            with pool.lease_embodied("ep-emb"):
                raise RuntimeError("episode died mid-walk")
        assert all(m.leased_to is None for m in pool.members)

    def test_leasing_probes_the_quarantined_before_giving_up(
            self, tmp_path, service):
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        service.healthz_ok = False
        pool.check_health()
        pool.check_health()
        assert pool.members[0].quarantined
        service.healthz_ok = True
        with pool.lease_embodied("ep-emb") as client:
            assert client is pool.members[0].client
        assert not pool.members[0].quarantined


# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestAnEmbodiedEpisodeWalksTheStockGraph:
    def scripted_nodes(self, env, turns=8):
        """Drive by geometry alone (street names and headings, never images)
        and record the node after each move -- the same decision rule for the
        embodied env and the stock one, so the sequences diverge only if the
        *transitions* diverge."""
        nodes = []
        for turn in range(turns):
            rows = env.candidates()
            row = rows[turn % len(rows)]
            env.walk_to(row["street"], row["heading"])
            nodes.append(env.node_id)
        return nodes

    def test_the_node_trajectory_is_the_stock_graphs(
            self, paris, service, tmp_path):
        """UE owns the seconds; the GRAPH still owns the topology. Same seed,
        same action script, same node sequence as a stock env -- the hop
        lands on the node ``_step_to`` names, however the pawn got there."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        bare = CourierEnv(paris, seed=5, difficulty="solo")
        bare.reset()
        assert env.node_id == bare.node_id, "same seed, same spawn"
        assert self.scripted_nodes(env) == self.scripted_nodes(bare)

    def test_the_clock_is_engine_ticks_plus_declared_costs_and_nothing_else(
            self, paris, service, tmp_path):
        """The Track B clock contract, end to end: after a scripted walk the
        env clock equals the integrator's ticks * fixed_dt summed over hops;
        a declared non-movement cost (wait, a refused action's floor) adds
        exactly its declared seconds on top."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        assert env.fixed_dt == pytest.approx(FIXED_DT)
        self.scripted_nodes(env, turns=6)

        total_ticks = sum(hop["ticks"] for hop in env.embodied_log)
        walk_seconds = sum(hop["sim_seconds"] for hop in env.embodied_log)
        assert walk_seconds == pytest.approx(total_ticks * FIXED_DT)
        assert env.sim_seconds == pytest.approx(walk_seconds), (
            "movement seconds must be the engine's, with no arithmetic beside")

        waited = env.wait()
        refused = env.walk_to("No Such Street Anywhere")
        assert not refused.ok
        assert refused.sim_seconds == REJECTED_ACTION_SECONDS
        assert env.sim_seconds == pytest.approx(
            walk_seconds + waited.sim_seconds + refused.sim_seconds)

    def test_every_hops_ticks_match_the_integrators_own_arithmetic(
            self, paris, service, tmp_path):
        """The tick math, hop by hop: each walk starts where the last one
        landed (within arrive_cm of the previous node, not on it), and its
        tick count is exactly ceil((distance - arrive) / (speed * dt))."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        self.scripted_nodes(env, turns=6)
        assert service.walks, "the script must have walked"
        for row in service.walks:
            ticks = expected_ticks(row["start_xy"], row["target_xy"],
                                   row["arrive_cm"])
            assert row["ticks"] == ticks
            assert row["sim_seconds"] == pytest.approx(ticks * FIXED_DT)

    def test_the_pawn_lands_within_arrive_cm_of_every_node(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        self.scripted_nodes(env, turns=6)
        errors = [hop["pose_error_cm"] for hop in env.embodied_log]
        assert errors and max(errors) <= env.arrive_cm
        assert env.summary()["embodied"]["max_pose_error_cm"] <= env.arrive_cm

    def test_the_episode_spawns_the_pawn_at_the_reset_node(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           spawn_z_cm=120.0)
        node = paris.nodes[env.node_id]
        spawn = service.episodes[0]["spawn"]
        assert (spawn["x_cm"], spawn["y_cm"]) == (node.x_cm, node.y_cm)
        assert spawn["z_cm"] == 120.0
        assert service.episodes[0]["agent"]["speed_cm_s"] == pytest.approx(
            env.embodiment.speed_cm_s)

    def test_frames_land_in_album_shape_from_the_pawns_camera(
            self, paris, service, tmp_path):
        from PIL import Image

        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        rows = env.candidates()
        assert rows
        for row in rows:
            assert row["image"], "every neighbour view must be observed"
            path = Path(row["image"])
            assert path.parent.parent == env.live_album.images
            with Image.open(path) as image:
                image.load()
            # v1 hazards off: nothing may show a lamp or an obstacle.
            assert row["signal_image"] is None
        before = len(service.observes)
        env.candidates()
        assert len(service.observes) == before, (
            "a second look is a cache hit, not a second observe")

    def test_the_summary_carries_the_embodied_evidence_block(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        self.scripted_nodes(env, turns=4)
        block = env.summary()["embodied"]
        # The evidence contract, pinned exactly. The last three were added
        # after an audit found that an episode could degrade to album frames
        # or queue for minutes behind another episode and report neither
        # anywhere a training run looks.
        # ``action_space`` joined them for the same reason: the coordinate
        # space is measured against this one, and an episode that does not
        # record which of the two it ran cannot be put on either side of that
        # comparison afterwards.
        # ``pedestrian_nudges`` and ``pedestrian_replans`` count what the
        # four-view harness did to keep a walk on certified legs (zero for
        # an episode that never walked one).
        assert set(block) == {"hops", "recoveries", "total_ticks",
                              "total_walk_seconds", "max_pose_error_cm",
                              "stuck_count", "walk_timeout_count",
                              "degraded", "busy_waits", "action_space",
                              "pedestrian_nudges", "pedestrian_replans"}
        assert block["pedestrian_nudges"] == 0 and block["pedestrian_replans"] == 0
        assert block["degraded"] is False
        assert block["walk_timeout_count"] == 0
        assert block["hops"] == len(env.embodied_log) > 0
        assert block["total_ticks"] == sum(h["ticks"] for h in env.embodied_log)
        assert block["total_walk_seconds"] == pytest.approx(
            block["total_ticks"] * FIXED_DT)
        assert block["stuck_count"] == 0
        # graph_seconds/chord_m are what the OFFLINE env would have charged
        # for the same hop. They are recorded on every hop so the gap between
        # engine-priced movement and the graph-priced budgets it is spent
        # against stays measurable instead of being an argument.
        log_keys = {"target_node", "target_xy", "ticks", "sim_seconds",
                    "walked_cm", "end_pose", "node_xy", "pose_error_cm",
                    "outcome", "graph_seconds", "chord_m"}
        assert all(set(hop) == log_keys for hop in env.embodied_log)

    def test_close_ends_the_episode_on_the_service(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        env.close()
        assert service.episode_ends == [EPISODE]
        env.close()  # idempotent: no second despawn, no exception
        assert service.episode_ends == [EPISODE]


@needs_maps
class TestConstructionRefusals:
    @pytest.mark.parametrize("key", [
        "album_root", "signal_album_root", "obstacle_album_root",
        "pavement_album_root", "pavement_obstacle_album_root",
        "obstacle_sidecar_root", "signal_sidecar_root",
    ])
    def test_hazard_albums_and_sidecars_are_refused_at_the_door(
            self, paris, tmp_path, key):
        """v1 embodied runs hazards OFF: locomotion realism is the thing
        under test, and a hazard root smuggled in would attach charges to a
        renderer that cannot show them."""
        with pytest.raises(ValueError, match="hazards OFF"):
            EmbodiedCourierEnv(paris, object(), episode_id=EPISODE,
                               cache_root=tmp_path,
                               **{key: str(tmp_path / "somewhere")})

    def test_difficulty_defaults_solo_but_other_tiers_are_not_refused(
            self, paris, service, tmp_path):
        solo = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        assert solo.difficulty == "solo"
        pair = EmbodiedCourierEnv(paris, UERenderClient(service.base_url),
                                  episode_id=EPISODE,
                                  cache_root=tmp_path / "pair",
                                  seed=5, difficulty="pair")
        assert pair.difficulty == "pair"


@needs_maps
class TestStuckAndTimeoutMapToTheStockRefusal:
    def hop_beyond(self, env, min_cm=2000.0):
        """A hop whose /walk distance from the pawn's actual pose clears
        ``min_cm``, found deterministically: the graph position is stood at a
        junction whose neighbour is far from the spawn pose (the pawn stays
        put -- the /walk starts from where the pawn really is, which is the
        distance that decides the tick count). Every map has one, so the
        engine-measured-charge scenarios never skip."""
        origin = (env.ue_pose.x_cm, env.ue_pose.y_cm)
        for node_id in sorted(env.network.nodes):
            for neighbour in sorted(env.network.nodes[node_id].neighbours):
                there = env.network.nodes[neighbour]
                if math.dist(origin, (there.x_cm, there.y_cm)) >= min_cm:
                    stand_at(env, node_id)
                    rows = {row["node"]: row for row in env.candidates()}
                    return rows[neighbour]
        pytest.fail("no edge far enough from spawn on this map")

    def test_a_wall_maps_to_the_stock_refusal_with_the_burned_seconds(
            self, paris, service, tmp_path):
        """The way_blocked semantics with an engine-measured price: code
        ``stuck``, not ok, the turn and the rejected action counted, the
        courier still at the junction -- and the charge is exactly the sim
        seconds the engine burned walking into the wall."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        row = self.hop_beyond(env)
        service.wall_after_cm = 1000.0  # 10 m of progress, then nothing
        before = env.sim_seconds
        node_before = env.node_id
        outcome = env._step_to(row["k"])
        assert not outcome.ok and outcome.code == "stuck"
        assert env.node_id == node_before, "a stuck walk does not hop"
        assert env.rejected_actions == 1
        assert env.blocked_attempts == 1
        assert (node_before, row["node"]) in env.witnessed_blocks

        walk = service.walks[-1]
        assert walk["stuck"] and not walk["arrived"]
        burned = walk["ticks"] * FIXED_DT
        assert burned > REJECTED_ACTION_SECONDS, (
            "this scenario must clear the floor or it proves nothing")
        assert outcome.sim_seconds == pytest.approx(burned)
        assert env.sim_seconds - before == pytest.approx(burned)
        assert env.summary()["embodied"]["stuck_count"] == 1
        # The failed hop is followed by its recovery re-spawn entry — the
        # hop is one before the end now, and the recovery names its cause.
        assert env.embodied_log[-2]["outcome"] == "stuck"
        assert env.embodied_log[-1]["recovery"] == "respawn"
        assert env.embodied_log[-1]["after"] == "stuck"

    def test_a_cheap_wall_still_pays_the_stock_refusal_floor(
            self, paris, service, tmp_path):
        """A pawn that stalls after 10 cm burned 0.07 s of engine time; the
        charge is the stock REJECTED_ACTION_SECONDS floor, because a refusal
        cheaper than any other refusal would be a probe."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        row = env.candidates()[0]
        service.wall_after_cm = 10.0
        before = env.sim_seconds
        outcome = env._step_to(row["k"])
        assert not outcome.ok and outcome.code == "stuck"
        assert env.sim_seconds - before == REJECTED_ACTION_SECONDS
        assert outcome.sim_seconds == REJECTED_ACTION_SECONDS

    def test_a_timeout_is_the_same_refusal_under_its_own_code(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           max_walk_seconds=6.0)
        row = self.hop_beyond(env)
        before = env.sim_seconds
        outcome = env._step_to(row["k"])
        assert not outcome.ok and outcome.code == "walk_timeout"
        walk = service.walks[-1]
        assert walk["timeout"] and not walk["arrived"]
        burned = walk["ticks"] * FIXED_DT
        assert burned == pytest.approx(int(6.0 / FIXED_DT) * FIXED_DT)
        assert env.sim_seconds - before == pytest.approx(
            max(burned, REJECTED_ACTION_SECONDS))
        # A timeout is not a witnessed barrier: nothing was walked into.
        assert env.blocked_attempts == 0
        assert env.summary()["embodied"]["stuck_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────


class _EmbodiedAdapterUnderTest(EmbodiedCourierGymEnv):
    """The embodied adapter with the phone map rasterised by PIL instead of
    cairosvg, exactly as the live adapter's tests do; the rasteriser is
    inherited stock code and not what these tests defend."""

    def _rasterise(self, svg, index):
        from PIL import Image

        return self._fit(Image.new("RGB", (720, 540), (240, 240, 240)))


def walk_reply(env) -> str:
    street, heading = env._env.street_at(1)
    return f'THOUGHT: go\n```\nwalk_to("{street}", "{heading}")\n```'


@needs_maps
class TestTheEmbodiedTrainingAdapter:
    @pytest.fixture()
    def config(self, service, tmp_path):
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        return {"backend": "embodied",
                "ue_endpoints": str(endpoints),
                "live_cache_root": str(tmp_path / "cache"),
                "spawn_z_cm": 120.0,
                "difficulty": "solo", "stride": "block",
                "max_turns": 3, "max_images": 2}

    def test_the_observation_contract_holds_over_the_embodied_backend(
            self, config, service):
        """The training contract, unchanged: placeholder count matches the
        image list, images are PIL objects, the info dict keeps its keys --
        success and env_return included -- and max_turns ends the episode."""
        from PIL import Image

        env = _EmbodiedAdapterUnderTest(config)
        obs, info = run(env.reset(0))
        assert info["backend"] == "embodied"
        # The id carries the config digest AND a per-episode suffix: GRPO
        # runs one seed n times at once, so ids must not collide.
        assert env._cfg8 in info["episode_id"]
        assert info["episode_id"].endswith("-1")
        images = obs["multi_modal_input"]["<image>"]
        assert obs["obs_str"].count("<image>") == len(images)
        assert images and all(isinstance(image, Image.Image) for image in images)

        done, steps = False, 0
        while not done and steps < 5:
            obs, reward, done, info = run(env.step(walk_reply(env)))
            assert isinstance(reward, float)
            for key in ("status", "sim_seconds", "success", "env_return",
                        "earnings", "turns"):
                assert key in info
            steps += 1
        assert done, "max_turns did not end the episode"
        assert service.walks, "the steps must have walked the pawn"
        run(env.close())
        assert service.episode_ends, "close must end the embodied episode"

    def test_close_returns_the_lease_to_the_pool(self, config, service):
        env = _EmbodiedAdapterUnderTest(config)
        run(env.reset(0))
        pool = env._render_pool
        assert any(m.leased_to for m in pool.members), (
            "reset must have taken the exclusive lease")
        run(env.close())
        assert all(m.leased_to is None for m in pool.members)

    def test_a_second_reset_releases_the_first_episodes_lease(
            self, config, service):
        env = _EmbodiedAdapterUnderTest(config)
        run(env.reset(0))
        observed = len(service.observes)
        assert observed
        run(env.reset(0))
        pool = env._render_pool
        assert sum(m.leased_to is not None for m in pool.members) == 1, (
            "a reset storm must not hold one lease per reset")
        assert len(service.episode_ends) == 1, (
            "the first episode must have been ended, once")
        # Each episode now owns its album, because each episode owns its id:
        # GRPO runs one seed several times at once, and two live episodes
        # sharing an id fight over the pawn. Re-rendering is the price, and
        # for embodied frames it is also the honest answer -- they come from
        # the pawn's own trajectory, which the next episode does not share.
        assert len(service.observes) > observed, (
            "a fresh episode renders its own frames")
        run(env.close())

    def test_hazards_true_is_refused_not_stripped(self):
        with pytest.raises(ValueError, match="hazards"):
            EmbodiedCourierGymEnv({"backend": "embodied", "hazards": True})

    def test_hazards_defaults_false_without_boilerplate(self, config):
        env = _EmbodiedAdapterUnderTest(config)
        assert env.hazards is False

    @pytest.mark.parametrize("key", ["album_root", "obstacle_sidecar_root",
                                     "signal_sidecar_root",
                                     "sidecar_source_root"])
    def test_album_and_sidecar_keys_are_refused(self, key):
        with pytest.raises(ValueError, match=key):
            EmbodiedCourierGymEnv({"backend": "embodied",
                                   key: "/data/somewhere"})

    def test_a_config_meant_for_another_adapter_is_refused(self):
        with pytest.raises(ValueError, match="backend"):
            EmbodiedCourierGymEnv({"backend": "live"})

    def test_spawn_z_reaches_the_wire(self, config, service):
        env = _EmbodiedAdapterUnderTest(config)
        run(env.reset(0))
        assert service.episodes[0]["spawn"]["z_cm"] == 120.0
        run(env.close())

    def test_the_registry_launch_line_names_a_real_class(self):
        import importlib

        module = importlib.import_module(
            "embodiedbench.runtime.live.gym_adapter")
        assert getattr(module, "EmbodiedCourierGymEnv") is EmbodiedCourierGymEnv


class TestAnOversubscribedFleetWaitsItsTurn:
    """A trainer drives more concurrent episodes than the fleet has
    instances; the second episode parks until the first ends, rather than
    failing the reset (NoHealthyInstance used to be immediate)."""

    def test_a_lease_waits_for_the_previous_episode_to_end(self, service, tmp_path):
        import threading
        import time as _time
        from embodiedbench.runtime.live.pool import RenderPool

        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        pool = RenderPool(endpoints, lease_timeout_s=10.0, lease_poll_s=0.05)
        release = threading.Event()
        held = threading.Event()

        def first():
            with pool.lease_embodied("ep-one"):
                held.set()
                release.wait(timeout=5.0)

        thread = threading.Thread(target=first)
        thread.start()
        assert held.wait(timeout=5.0)
        t0 = _time.monotonic()
        threading.Timer(0.3, release.set).start()
        with pool.lease_embodied("ep-two") as client:
            waited = _time.monotonic() - t0
            assert client is not None
        thread.join(timeout=5.0)
        assert waited >= 0.25, "second lease should have parked until release"

    def test_a_dead_fleet_still_fails_fast(self, tmp_path):
        import json as _json
        from embodiedbench.runtime.live.pool import NoHealthyInstance, RenderPool

        endpoints = tmp_path / "endpoints.json"
        endpoints.write_text(_json.dumps({"version": 0, "instances": [
            {"id": "gone", "base_url": "http://127.0.0.1:9", "map_name": "x"}]}))
        pool = RenderPool(endpoints, lease_timeout_s=30.0, lease_poll_s=0.05)
        for member in pool.members:
            member.quarantined = True
        import pytest as _pytest
        import time as _time
        t0 = _time.monotonic()
        with _pytest.raises(NoHealthyInstance):
            with pool.lease_embodied("ep-dead"):
                pass
        assert _time.monotonic() - t0 < 5.0, "all-dead must not wait out the lease timeout"


# ── seats: many couriers per instance ────────────────────────────────────────
#
# The service carries N couriers per instance (SimWorld2 --max-episodes); the
# pool is the half that decides whether those seats are ever used. Exclusive
# leases here would leave them empty however high the service is configured.


class TestSeatsPerInstance:

    def test_a_pre_seats_endpoints_file_still_means_one_courier(self, tmp_path):
        """Files written before seats existed carry no key. Defaulting them to
        anything but 1 would oversubscribe every old fleet in place."""
        pool = RenderPool(write_endpoints(tmp_path / "e.json",
                                          [("ue-a", "http://127.0.0.1:1")]))
        assert [m.seats for m in pool.members] == [1]

    def test_seats_are_read_from_the_endpoints_file(self, tmp_path):
        pool = RenderPool(write_endpoints(tmp_path / "e.json",
                                          [("ue-a", "http://127.0.0.1:1")], seats=4))
        assert [m.seats for m in pool.members] == [4]

    def test_one_instance_leases_up_to_its_seats_then_parks(self, tmp_path):
        pool = RenderPool(write_endpoints(tmp_path / "e.json",
                                          [("ue-a", "http://127.0.0.1:1")], seats=3),
                          lease_timeout_s=0.3, lease_poll_s=0.05)
        with contextlib.ExitStack() as stack:
            for name in ("ep-0", "ep-1", "ep-2"):
                stack.enter_context(pool.lease_embodied(name))
            member = pool.members[0]
            assert member.leases == {"ep-0", "ep-1", "ep-2"}
            # The fourth has nowhere to sit and parks rather than colliding.
            with pytest.raises(NoHealthyInstance) as caught:
                with pool.lease_embodied("ep-3"):
                    pass
            assert "seats" in str(caught.value)
        assert pool.members[0].leases == set()

    def test_seats_fill_breadth_first_across_instances(self, tmp_path):
        """Two instances of two seats take one courier each before either
        takes a second: seats are cheap to stack but share one GPU's render
        throughput, so spreading first is strictly better."""
        pool = RenderPool(
            write_endpoints(tmp_path / "e.json",
                            [("ue-a", "http://127.0.0.1:1"),
                             ("ue-b", "http://127.0.0.1:2")], seats=2),
            lease_timeout_s=1.0, lease_poll_s=0.05)
        with contextlib.ExitStack() as stack:
            for name in ("ep-0", "ep-1"):
                stack.enter_context(pool.lease_embodied(name))
            assert sorted(len(m.leases) for m in pool.members) == [1, 1], (
                "the second courier stacked instead of spreading")
            for name in ("ep-2", "ep-3"):
                stack.enter_context(pool.lease_embodied(name))
            assert sorted(len(m.leases) for m in pool.members) == [2, 2]

    def test_render_still_avoids_any_instance_with_a_courier(self, tmp_path):
        """A seat left over does NOT make the instance available to /render:
        the service answers /render busy while any courier is alive, so one
        lease withdraws the whole instance from dispatch."""
        pool = RenderPool(
            write_endpoints(tmp_path / "e.json",
                            [("ue-a", "http://127.0.0.1:1"),
                             ("ue-b", "http://127.0.0.1:2")], seats=4))
        with pool.lease_embodied("ep-0"):
            leased = next(m for m in pool.members if m.leases)
            assert len(leased.leases) < leased.seats, "precondition: a seat is free"
            picked = pool._pick(set())
            assert picked is not None and picked.id != leased.id

    def test_releasing_one_courier_leaves_the_others_seated(self, tmp_path):
        pool = RenderPool(write_endpoints(tmp_path / "e.json",
                                          [("ue-a", "http://127.0.0.1:1")], seats=3),
                          lease_timeout_s=1.0, lease_poll_s=0.05)
        with pool.lease_embodied("ep-keep"):
            with pool.lease_embodied("ep-go"):
                assert pool.members[0].leases == {"ep-keep", "ep-go"}
            assert pool.members[0].leases == {"ep-keep"}
        assert pool.members[0].leases == set()


class TestAnOnlineRunCanRefuseCachedFrames:
    """An episode whose pictures came from an album measured the album.

    The fallback exists so a flaky renderer cannot kill a long run, and that
    is right for training. It is wrong for an experiment measuring live UE:
    walking stays live, the frames quietly stop being, and every metric still
    reads green. Measured on the development workstations: 11 of 11 episodes finished
    `degraded` on a frame path that could never have worked across machines,
    and the only symptom was a boolean nobody reads.
    """

    def test_the_default_still_survives_a_broken_renderer(self):
        from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv
        import inspect
        sig = inspect.signature(EmbodiedCourierEnv.__init__)
        assert sig.parameters["allow_album_fallback"].default is True, (
            "training runs should keep surviving a flaky renderer")

    def test_frames_default_to_base64_not_a_path_on_someone_elses_disk(self):
        """`path` returns a filename on the RENDERER's machine. It is only
        readable when the two share a filesystem, and the fleet is addressable
        over the network precisely so they need not."""
        from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv
        from embodiedbench.runtime.live.protocol import RETURN_MODE_BASE64
        import inspect
        sig = inspect.signature(EmbodiedCourierEnv.__init__)
        assert sig.parameters["return_mode"].default == RETURN_MODE_BASE64


@needs_maps
class TestPoseTrackedMapMatching:
    def test_a_closer_disconnected_node_cannot_steal_the_pawn(self):
        nodes = {
            "start": StreetNode("start", 0.0, 0.0, 0, 0.0, {"forward"}),
            "forward": StreetNode(
                "forward", 0.0, 1000.0, 0, 1000.0, {"start"}),
            # A different graph component happens to occupy the landing.
            "unrelated": StreetNode(
                "unrelated", 0.0, 950.0, 0, 0.0, set()),
        }
        env = pose_match_env(nodes, current="start", walked_bearing=90.0)

        selected, gap = env._match_pose_node((0.0, 950.0))

        assert env._nearest_node((0.0, 950.0))[0] == "unrelated"
        assert selected == "forward" and gap == pytest.approx(50.0)
        assert env._last_pose_match["prevented_global_jump"] is True
        assert env._last_pose_match["global_nearest_node"] == "unrelated"

    def test_continuing_straight_does_not_snap_onto_a_diagonal_turn(self):
        nodes = {
            "corner": StreetNode(
                "corner", 0.0, 0.0, 0, 0.0, {"diagonal"}),
            # Geometry from the failed rollout: this planned arm is closer
            # than the corner after a 4.4 m straight overshoot, but points 25
            # degrees away from the pawn's measured movement.
            "diagonal": StreetNode(
                "diagonal", 260.0, 590.0, 0, 645.0, {"corner"}),
        }
        env = pose_match_env(nodes, current="corner", walked_bearing=91.3)

        selected, _gap = env._match_pose_node((-10.0, 441.0))

        assert env._nearest_node((-10.0, 441.0))[0] == "diagonal"
        assert selected == "corner"
        assert env._last_pose_match["prevented_global_jump"] is True

    def test_a_straight_context_edge_wins_after_a_missed_turn(self):
        nodes = {
            "corner": StreetNode(
                "corner", 0.0, 0.0, 0, 0.0, {"diagonal", "straight"}),
            "diagonal": StreetNode(
                "diagonal", 260.0, 590.0, 0, 645.0, {"corner"}),
            "straight": StreetNode(
                "straight", 0.0, 470.0, 0, 470.0, {"corner"}),
        }
        env = pose_match_env(nodes, current="corner", walked_bearing=91.3)

        selected, gap = env._match_pose_node((-10.0, 441.0))

        assert selected == "straight"
        assert gap == pytest.approx(math.hypot(10.0, 29.0))

    def test_one_walk_cannot_skip_sixteen_metres_around_a_folded_corner(self):
        """Regression from the v3 live rollout at its exact landing pose."""
        nodes = {
            "straight4": StreetNode(
                "straight4", -5934.8349, -4300.0, 0, 0.0, {"pre"}),
            "pre": StreetNode(
                "pre", -5950.0, -4000.0, 0, 300.0,
                {"straight4", "corner"}),
            "corner": StreetNode(
                "corner", -5950.0, -3200.0, 0, 1100.0,
                {"pre", "after"}),
            # Folded spatially close to the approach, but more than 16 m of
            # graph path away through the real corner.
            "after": StreetNode(
                "after", -5644.6327536912495, -3606.5472318525804,
                0, 1608.0, {"corner"}),
        }
        start = (-5972.826182160346, -4339.169017188201)
        landing = (-5853.167057123334, -3666.4636815694575)
        movement_cm = math.dist(start, landing)
        env = pose_match_env(
            nodes,
            current="straight4",
            walked_bearing=bearing_deg(start, landing),
        )

        selected, _gap = env._match_pose_node(
            landing, movement_cm=movement_cm)

        assert env._nearest_node(landing)[0] == "after"
        assert selected == "pre"
        assert env._last_pose_match["movement_cm"] == pytest.approx(
            movement_cm, abs=0.1)
        assert env._last_pose_match["selected_graph_progress_cm"] == (
            pytest.approx(math.hypot(15.1651, 300.0), abs=0.1))
        assert env._last_pose_match["selected_progress_excess_cm"] == 0.0

    def test_dense_recast_lattice_tracks_a_ten_metre_walk_past_three_hops(self):
        """A metric walk must not be truncated by the graph's node density."""
        nodes = {}
        for index in range(13):
            neighbours = set()
            if index:
                neighbours.add(f"n{index - 1}")
            if index < 12:
                neighbours.add(f"n{index + 1}")
            nodes[f"n{index}"] = StreetNode(
                f"n{index}", index * 100.0, 0.0, 0,
                index * 100.0, neighbours,
            )
        # A disconnected survey point at the landing must remain ineligible.
        nodes["unrelated"] = StreetNode(
            "unrelated", 1000.0, 0.0, 0, 0.0, set())
        env = pose_match_env(nodes, current="n0", walked_bearing=0.0)

        selected, gap = env._match_pose_node(
            (1000.0, 0.0), movement_cm=950.0)

        assert env._nearest_node((1000.0, 0.0))[0] in {"n10", "unrelated"}
        assert selected == "n10"
        assert gap == 0.0
        assert env._last_pose_match["selected_hops"] == 10
        assert env._last_pose_match["search_progress_limit_cm"] == 1250.0
        assert env._last_pose_match["prevented_global_jump"] == (
            env._last_pose_match["global_nearest_node"] == "unrelated")

    def test_recast_connector_side_step_cannot_hide_a_crosswalk_landing(self):
        """Regression from the trusted-pool GPT rollout at its exact pose.

        UE carried the pawn 4.8 m onto PR_Crossswalk_94.  The certified route
        reaches that sample through a short south-east connector, although the
        action's net displacement is almost due south.  Scoring the connector's
        first one-metre edge therefore held the graph state on the approach
        pavement and made the phone order a false reversal.  Map matching must
        compare the whole previous-to-candidate displacement instead.
        """
        nodes = {
            "previous": StreetNode(
                "previous", -19300.0, 900.0, 0, 0.0,
                {"pavement-1", "connector"}),
            "pavement-1": StreetNode(
                "pavement-1", -19300.0, 800.0, 0, 0.0,
                {"previous", "pavement-2"}),
            "pavement-2": StreetNode(
                "pavement-2", -19300.0, 700.0, 0, 0.0,
                {"pavement-1", "stale-pavement"}),
            "stale-pavement": StreetNode(
                "stale-pavement", -19300.0, 600.0, 0, 0.0,
                {"pavement-2"}),
            "connector": StreetNode(
                "connector", -19200.0, 800.0, 0, 0.0,
                {"previous", "crossing-entry"}),
            "crossing-entry": StreetNode(
                "crossing-entry", -19144.1, 596.8, 0, 0.0,
                {"connector", "crossing-050"}),
            "crossing-050": StreetNode(
                "crossing-050", -19358.3, 470.3, 0, 0.0,
                {"crossing-entry"}),
        }
        start = (-19294.4, 838.3)
        landing = (-19391.1, 370.3)
        env = pose_match_env(
            nodes,
            current="previous",
            walked_bearing=bearing_deg(start, landing),
        )

        selected, gap = env._match_pose_node(
            landing, movement_cm=math.dist(start, landing))

        assert env._nearest_node(landing)[0] == "crossing-050"
        assert selected == "crossing-050"
        assert gap == pytest.approx(
            math.dist(nodes["crossing-050"].position, landing))
        assert env._last_pose_match["heading_reference"] == (
            "previous_to_candidate_net_displacement")


class TestTheCoordinateActionSpace:
    """Naming a point instead of naming a street.

    The reason the space exists: the street space is close to solved by the
    map's route line plus the three-step procedure, and a GRPO group whose
    rollouts nearly all succeed has an advantage of ~0 and a gradient to
    match. So these tests are less about the walk -- the wire has always taken
    coordinates -- than about the two things that would make the comparison
    meaningless: an episode able to fall back to the easy action, and an
    observation that describes somewhere the courier is not.
    """

    def coordinate(self, paris, service, tmp_path, **kwargs):
        return embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                            action_space="coordinate", **kwargs)

    # ── the two spaces are two tasks ─────────────────────────────────────────

    def test_the_menus_never_overlap(self, paris, service, tmp_path):
        """A menu holding both lets a run take the easy action and be
        reported under the hard one's name."""
        street = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        assert "walk_to" in street.allowed_tool_names()
        assert "walk_to_xy" not in street.allowed_tool_names()

        coords = self.coordinate(paris, service, tmp_path / "b")
        names = coords.allowed_tool_names()
        assert "walk_to_xy" in names
        assert "walk_to" not in names and "follow_street" not in names

    def test_an_unknown_action_space_is_refused_at_construction(
            self, paris, service, tmp_path):
        with pytest.raises(ValueError, match="action_space"):
            embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                         action_space="freeform")

    def test_the_prompt_states_the_cap_the_env_enforces(
            self, paris, service, tmp_path):
        """A manual quoting a limit the runtime does not keep is worse than
        one that quotes none: the placeholder reaches the model verbatim, or
        the number does and is a lie."""
        from embodiedbench.agent.courier.session import CourierSession

        env = self.coordinate(paris, service, tmp_path, max_step_m=10.0)
        prompt = CourierSession(env, city="Paris").system_prompt()
        assert "10 m" in prompt
        assert "{max_step_m}" not in prompt and "{" not in prompt

    def test_the_prompt_never_teaches_a_call_it_cannot_run(
            self, paris, service, tmp_path):
        """The whole system prompt, not only its menu: the worked example, the
        reply format and the paragraph under the list of streets all named
        walk_to by hand."""
        from embodiedbench.agent.courier.session import CourierSession

        env = self.coordinate(paris, service, tmp_path)
        session = CourierSession(env, city="Paris")   # asserts this itself
        prompt = session.system_prompt()
        assert "walk_to(" not in prompt.replace("walk_to_xy(", "")
        assert "walk_to_xy(" in prompt
        assert "walk_to(" not in session.observe().text

    # ── one call, and what bounds it ─────────────────────────────────────────

    def test_a_point_past_the_cap_is_refused_and_nothing_moves(
            self, paris, service, tmp_path):
        """Refused, not carried part of the way. A courier that asked for
        forty metres and was quietly walked one and a half cannot tell that
        from arriving, and neither can a reader of the log."""
        env = self.coordinate(paris, service, tmp_path, max_step_m=10.0)
        start = env._here_cm()
        walks_before = len(service.walks)

        out = env.walk_to_xy(start[0] / 100.0 + 40.0, start[1] / 100.0)

        assert not out.ok and out.code == "too_far"
        assert "at most 10 m" in out.message
        assert len(service.walks) == walks_before, "no walk was attempted"
        assert env._here_cm() == pytest.approx(start)
        refused = [h for h in env.embodied_log if h.get("code") == "too_far"]
        assert refused and refused[0]["gap_m"] == pytest.approx(40.0, abs=0.2)

    def test_each_call_of_a_chunk_is_judged_where_it_runs(
            self, paris, service, tmp_path):
        """Three waypoints are checked one at a time as they are reached,
        never all three up front: the second and third are named relative to a
        position the courier has not walked to yet, so judging them against
        where it stood when it wrote them measures a step it never asked for.

        Three steps of 1 m in a line: each is inside the cap from where the
        previous one lands, and the third is 3 m from where the first was
        written -- twice the cap.

        8 m and not 10: a walk stops as soon as it is within ``arrive_cm`` of
        its target, so a CHAINED plan advances ``max_step_m - arrive_cm`` per
        call, not ``max_step_m``. At a 10 m cap and a 1 m radius that is 9 m,
        and the shortfall accumulates down the chain."""
        env = self.coordinate(paris, service, tmp_path,
                              max_step_m=10.0, arrive_cm=100.0, tick_chunk=2)
        start = env._here_cm()
        for i in range(1, 4):
            out = env.walk_to_xy(start[0] / 100.0 + 8.0 * i, start[1] / 100.0)
            assert out.ok, f"step {i} was refused: {out.message}"
        assert math.dist(start, env._here_cm()) > 1600.0, "it really moved 16+ m"

    def test_the_point_you_are_standing_on_is_refused_not_walked(
            self, paris, service, tmp_path):
        env = self.coordinate(paris, service, tmp_path)
        here = env._here_cm()
        walks_before = len(service.walks)

        out = env.walk_to_xy(here[0] / 100.0, here[1] / 100.0)

        assert not out.ok and out.code == "already_here"
        assert out.sim_seconds == REJECTED_ACTION_SECONDS
        assert len(service.walks) == walks_before, "no walk was attempted"

    def test_a_coordinate_that_is_not_a_number_is_a_refusal_not_a_crash(
            self, paris, service, tmp_path):
        env = self.coordinate(paris, service, tmp_path)
        out = env.walk_to_xy("north", 12.0)
        assert not out.ok and out.code == "bad_coordinate"

    # ── where the courier is ─────────────────────────────────────────────────

    def test_the_position_it_is_told_is_the_one_its_next_call_counts_from(
            self, paris, service, tmp_path):
        """The fairness line and the arithmetic line at once. The courier is
        told its own position (never the delivery's), and that position is the
        pawn's -- so a coordinate it derives by adding an offset to what it
        was told lands where it meant."""
        env = self.coordinate(paris, service, tmp_path)
        env.walk_to_xy(*[v / 100.0 + 5.0 for v in env._here_cm()])

        pawn = env._here_cm()
        stated = env.pose_text()
        assert f"({pawn[0] / 100.0:.1f}, {pawn[1] / 100.0:.1f})" in stated
        assert env.position() == pytest.approx(pawn)
        assert stated in env.location_text()

    def test_the_street_space_is_not_told_its_coordinates(
            self, paris, service, tmp_path):
        """Not tidiness: the coordinate run is measured against the street
        run, and adding a fact to the baseline moves what it is a baseline
        of."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        assert "standing at (" not in env.location_text()
        # ...and it stays one flag away, for the controlled version.
        env2 = embodied_env(paris, UERenderClient(service.base_url),
                            tmp_path / "b", show_pose=True)
        assert "standing at (" in env2.location_text()

    def test_the_junction_being_described_is_re_derived_from_the_pawn(
            self, paris, service, tmp_path):
        """Everything the environment can SAY is node-shaped -- which street
        this is, what leaves it -- so a graph node is kept. It is selected
        from the previous node's connected neighbourhood with the measured
        walk direction as evidence; the unconstrained nearest node is logged
        beside it rather than silently trusted."""
        env = self.coordinate(paris, service, tmp_path)
        env.walk_to_xy(*[v / 100.0 + 5.0 for v in env._here_cm()])

        hop = env.embodied_log[-1]
        match = hop["map_match"]
        assert env.node_id == match["selected_node"] == hop["landed_node"]
        assert hop["snap_cm"] == pytest.approx(
            match["selected_gap_cm"], abs=0.1)
        nearest, gap = env._nearest_node(env._here_cm())
        assert match["global_nearest_node"] == nearest
        assert match["global_nearest_gap_cm"] == pytest.approx(gap, abs=0.1)

    def test_phone_route_begins_at_the_same_pawn_pose_as_the_camera(
            self, paris, service, tmp_path):
        env = self.coordinate(paris, service, tmp_path)
        start = env._here_cm()
        outcome = env.walk_to_xy(start[0] / 100.0 + 5.0,
                                 start[1] / 100.0 + 2.0)
        assert outcome.ok
        target = next(
            address for address in env.network.addresses
            if address.kerb_node
            and address.kerb_node != env.node_id
            and env.route_nodes(env.node_id, address.kerb_node)
        )

        env.map_drawing(target)

        assert env.screen_route
        assert env.screen_route[0] == pytest.approx(env._here_cm())
        assert math.dist(env.screen_route[0], env._here_cm()) < 1e-6

    def test_two_walks_ending_in_different_places_do_not_share_a_photograph(
            self, paris, service, tmp_path):
        """Keyed by node alone, the first visit's picture would be served for
        every later one: the observation would stop being of where the courier
        is, and nothing would say so."""
        env = self.coordinate(paris, service, tmp_path)
        toward = env.candidates()[0]["node"]

        first = env.frame_for(env.node_id, toward)
        env.walk_to_xy(*[v / 100.0 + 5.0 for v in env._here_cm()])
        second = env.frame_for(env.node_id, toward)

        assert first and second and first != second
        assert Path(first).parent != Path(second).parent, (
            "the vantage, not just the frame, has to differ")
        # ...and idempotency survives it: asked again from the same place,
        # the frame comes off disk. FrameAliases and observation_media_hash
        # both rest on the same picture having the same bytes forever.
        rendered = len(service.observes)
        assert env.frame_for(env.node_id, toward) == second
        assert len(service.observes) == rendered

    # ── a walk that does not get through ─────────────────────────────────────

    def test_a_blocked_walk_keeps_the_ground_it_covered(
            self, paris, service, tmp_path):
        """No re-spawn here. In this space the pawn's position IS the
        courier's, so standing it back on a node it may be twenty metres from
        would be the desynchronisation the re-spawn exists to prevent, applied
        backwards."""
        service.wall_after_cm = 200.0
        env = self.coordinate(paris, service, tmp_path)
        start = env._here_cm()

        out = env.walk_to_xy(*[v / 100.0 + 6.0 for v in start])

        assert not out.ok and out.code == "stuck"
        assert "no way to walk there" in out.message
        moved = math.dist(start, env._here_cm())
        assert moved > 100.0, "the pawn kept where its walk took it"
        assert env.walked_cm > 0, "and the metres are counted"
        assert not [h for h in env.embodied_log if h.get("recovery")]

    # ── what the run writes down ─────────────────────────────────────────────

    def test_the_summary_says_which_question_was_asked(
            self, paris, service, tmp_path):
        """Two runs whose action spaces have to be recalled from a launch
        command are not a comparison."""
        env = self.coordinate(paris, service, tmp_path, max_step_m=10.0)
        env.walk_to_xy(*[v / 100.0 + 5.0 for v in env._here_cm()])
        env.walk_to_xy(*[v / 100.0 + 400.0 for v in env._here_cm()])

        block = env.summary()["embodied"]
        assert block["action_space"] == "coordinate"
        assert block["max_step_m"] == 10.0
        assert block["coordinate_walks"] == 1 and block["coordinate_too_far"] == 1
        assert block["median_snap_cm"] is not None
        assert block["map_matching"]["method"] == (
            "metric_local_continuity_net_heading_progress_v4")
        assert block["map_matching"]["matches"] == 1
        assert block["map_matching"]["global_recoveries"] == 0

        street = embodied_env(paris, UERenderClient(service.base_url),
                              tmp_path / "b")
        assert street.summary()["embodied"]["action_space"] == "street"
        assert "map_matching" not in street.summary()["embodied"]

    def test_facing_is_the_way_it_walked_not_the_way_it_last_photographed(
            self, paris, service, tmp_path):
        """The pawn's own yaw is the obvious answer and the wrong one:
        /observe aims the camera by turning the agent, and an observation
        photographs every street leaving the junction -- so read off the pose,
        "facing" is whichever neighbour rendered last, which then decides
        every "on your left" in the same turn's list."""
        env = self.coordinate(paris, service, tmp_path)
        start = env._here_cm()
        env.walk_to_xy(start[0] / 100.0 + 6.0, start[1] / 100.0)
        walked = env.facing()
        assert walked == pytest.approx(bearing_deg(start, env._here_cm()),
                                       abs=1.0)

        env.candidates()          # renders a frame down every street
        assert env.ue_pose.yaw_deg != pytest.approx(walked, abs=1.0), (
            "the fixture must actually turn the pawn, or this proves nothing")
        assert env.facing() == pytest.approx(walked, abs=1.0)
    def test_the_coordinate_call_refuses_to_run_in_the_street_space(
            self, paris, service, tmp_path):
        """Unreachable through the menu, so this is a caller wiring the two
        spaces together -- and every half that makes the call honest (the pawn
        being the position, the vantage in the frame key, facing measured off
        the walk) is switched off under `street`."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        with pytest.raises(RuntimeError, match="coordinate"):
            env.walk_to_xy(0.0, 0.0)

    def test_the_album_fallback_is_reachable_from_a_config(self, tmp_path, service):
        """The env has had the knob since the cross-machine work and the
        quickstart's settings table says an experiment must turn it off -- but
        no config key reached it, so the only value any run could have was the
        training-friendly default. Measured the hard way: an instance died
        mid-validation and the episodes it was serving carried on against an
        album with every walking metric still reading green."""
        endpoints = write_endpoints(tmp_path / "e.json", [service])
        base = {"backend": "embodied", "ue_endpoints": str(endpoints),
                "live_cache_root": str(tmp_path / "cache")}
        assert EmbodiedCourierGymEnv(base).allow_album_fallback is True
        assert EmbodiedCourierGymEnv(
            {**base, "allow_album_fallback": False}).allow_album_fallback is False

    def test_a_sum_is_refused_by_naming_the_sum(self, paris, service, tmp_path):
        """Measured on Qwen3-VL-4B: 62 of 81 coordinate format errors were an
        unevaluated sum -- the model saying "eighteen metres west of here" the
        most direct way it knows. It is still refused, because working the
        position out IS the task, but "you used quotes" is advice it cannot
        act on and it repeated the reply until the three-strike rule ended the
        episode."""
        from embodiedbench.agent.courier.loop import FormatError, build_call

        with pytest.raises(FormatError, match="sum"):
            build_call("walk_to_xy", "-53.2, 297.7 - 18", {"walk_to_xy"})
        with pytest.raises(FormatError, match="quotes"):
            build_call("walk_to_xy", '"-53.2", "297.7"', {"walk_to_xy"})
        # ...and a plain negative number is not mistaken for one.
        name, args, _ = build_call("walk_to_xy", "-53.2, -297.7", {"walk_to_xy"})
        assert args == [-53.2, -297.7]

    def test_naming_your_own_position_says_so(self, paris, service, tmp_path):
        env = self.coordinate(paris, service, tmp_path)
        here = env._here_cm()
        out = env.walk_to_xy(round(here[0] / 100.0, 1), round(here[1] / 100.0, 1))
        assert not out.ok and out.code == "already_here"
        assert "the point you are standing on" in out.message
        assert "ADD" in out.message
    def test_a_refused_coordinate_is_written_down(
            self, paris, service, tmp_path):
        """Leaving it out cost an evening's reading. 83 of 205 turns in the
        first live run were refused before the hop log, so the coordinate that
        caused them was recorded exactly nowhere -- the only way to see what
        the policy had asked for was to parse it back out of the reply text.
        Same shape as every other measurement failure here: the zero I read
        was invisible, not absent."""
        env = self.coordinate(paris, service, tmp_path)
        here = env._here_cm()
        env.walk_to_xy(round(here[0] / 100.0, 1), round(here[1] / 100.0, 1))

        refused = [h for h in env.embodied_log
                   if h.get("kind") == "coordinate_refused"]
        assert len(refused) == 1
        assert refused[0]["code"] == "already_here"
        assert refused[0]["gap_m"] < 1.0
        # ...and it must not look like a walk, or every aggregation that
        # counts hops starts counting refusals too.
        assert "ticks" not in refused[0]
        assert env.summary()["embodied"]["hops"] == 0

    def test_the_distance_walked_comes_from_the_poses(
            self, paris, service, tmp_path):
        """Measured on the development workstation: 13 accepted walks in one episode each moved
        0.5 to 4.7 m by their own start and end pose, and each reported
        walked_cm = 0.00 -- so the episode's route quality read "walked 0 m"
        for a courier that had covered fifteen. Both numbers come back in the
        same response, so this is the service's count disagreeing with the
        poses beside it, and the poses are what the arrival test and the next
        walk both use."""
        env = self.coordinate(paris, service, tmp_path,
                              max_step_m=10.0, arrive_cm=100.0, tick_chunk=2)
        start = env._here_cm()
        out = env.walk_to_xy(start[0] / 100.0 + 6.0, start[1] / 100.0)

        assert out.ok
        moved_m = math.dist(start, env._here_cm()) / 100.0
        assert out.walked_m == pytest.approx(moved_m, abs=0.02)
        assert env.walked_cm == pytest.approx(moved_m * 100.0, abs=2.0)
        # ...and the service's own figure stays on the record beside it.
        hop = env.embodied_log[-1]
        assert "service_walked_cm" in hop
    def test_the_step_budget_is_stated_where_the_distances_are(
            self, paris, service, tmp_path):
        """Told the cap once in the tool manual and nowhere else, a courier
        named points a median 1.14 m off for thirteen calls running and was
        never refused -- naming a short step breaks no rule it had been given.
        The manual states a maximum and nothing states that a metre is a
        wasted turn, so the rule we were scoring it against was not one it
        could read. It belongs beside "36 m on, 2 junctions"."""
        from embodiedbench.agent.courier.session import CourierSession

        env = self.coordinate(paris, service, tmp_path, max_step_m=10.0,
                              arrive_cm=100.0, tick_chunk=2)
        text = CourierSession(env, city="Paris").observe().text
        assert "UP TO 10 m" in text
        assert "{max_step_m}" not in text
        # ...and the street arm's line is untouched: it has no step to state.
        street = embodied_env(paris, UERenderClient(service.base_url),
                              tmp_path / "b")
        assert "UP TO" not in CourierSession(street, city="Paris").observe().text
    def test_the_trace_says_which_way_the_camera_pointed(
            self, paris, service, tmp_path):
        """Three orientations and three different facts: the way the courier
        travelled, where its body points now, and where each photograph
        looked. They are not the same number -- /observe aims by turning the
        pawn, so after a look the body's yaw is the last frame's bearing --
        and a recording that keeps one of them cannot tell them apart."""
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.runtime.live.trace import EpisodeTrace

        env = self.coordinate(paris, service, tmp_path, max_step_m=10.0,
                              arrive_cm=100.0, tick_chunk=2)
        session = CourierSession(env, city="Paris")
        trace = EpisodeTrace(tmp_path / "trace", "ep", {})
        here = env._here_cm()
        session.step(f'```\nwalk_to_xy({here[0]/100 + 5.0:.1f}, {here[1]/100:.1f})\n```')
        trace.record(env, session.run.turns[-1])
        rec = json.loads(trace.close(env).read_text())
        e = rec["events"][0]

        assert len(e["frame_yaws_deg"]) == len(e["frames"])
        assert all(y is not None for y in e["frame_yaws_deg"]), e["frame_yaws_deg"]
        # Each frame looks at a different street, so no two share a bearing.
        assert len(set(e["frame_yaws_deg"])) == len(e["frame_yaws_deg"])
        assert e["yaw_after_deg"] is not None
        assert e["calls"][0]["walk"]["end_yaw_deg"] is not None
    def test_forward_view_photographs_one_thing_the_way_it_faces(
            self, paris, service, tmp_path):
        """Under `streets` a turn photographs every street leaving the
        junction, including the way it came -- which the coordinate courier
        cannot act on by name and which spends half a two-image budget. A
        walking person does not get a photograph of behind them each step."""
        from embodiedbench.agent.courier.session import CourierSession

        env = self.coordinate(paris, service, tmp_path, max_step_m=10.0,
                              arrive_cm=100.0, tick_chunk=2,
                              camera_view="forward")
        session = CourierSession(env, city="Paris")
        obs = session.observe()

        # The phone's map rides along as always; the CAMERA frames are one.
        shots = [f for f in obs.frames if f.kind != "map"]
        assert len(shots) == 1, [f.label for f in obs.frames]
        assert "ahead" in shots[0].label
        assert "the view straight ahead" in obs.text
        # ...and it looks the way the courier is facing, not down a street.
        here = env._here_cm()
        session.step(f'```\nwalk_to_xy({here[0]/100 + 6.0:.1f}, {here[1]/100:.1f})\n```')
        after = session.observe()
        assert len([f for f in after.frames if f.kind != "map"]) == 1
        yaw = env.frame_yaws[[k for k in env.frame_yaws if "ahead" in k][-1]]
        assert yaw == pytest.approx(env.facing(), abs=1.0)

    def test_forward_view_is_refused_for_the_street_space(
            self, paris, service, tmp_path):
        """It picks a street off the list, and the list's pictures are how it
        tells them apart."""
        with pytest.raises(ValueError, match="street action space"):
            embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                         camera_view="forward")

    def test_streets_remains_the_default(self, paris, service, tmp_path):
        """The arms are compared against each other; changing what one of them
        SEES makes the difference between them two things instead of one."""
        env = self.coordinate(paris, service, tmp_path)
        assert env.camera_view == "streets"
        assert len(env.photo_rows(env.candidates())) == len(env.candidates())

@needs_maps
class TestThePixelGoalActionSpace:
    """Naming a point IN the photograph instead of on the map.

    A third task, not a variant of the coordinate one: the courier is never
    given a metric position at all, only a normalized point in the picture
    it was just shown, and the whole geometric judgement of whether that
    point is walkable happens engine-side (``FakeTrackBService._resolve_pixel``
    stands in for the raycast + NavMesh projection ``SpPixelGoalSubsystem``
    does for real). What is under
    test here is the same pair of things the coordinate suite tests: that
    the two action spaces cannot be run together, and that everything
    downstream of a resolved walk (the pawn IS the position, facing measured
    off the walk, frames keyed by vantage) is switched on.
    """

    def pixel_goal(self, paris, service, tmp_path, **kwargs):
        return embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                            action_space=ACTION_SPACE_PIXEL_GOAL,
                            camera_view=CAMERA_VIEW_FORWARD, **kwargs)

    def dual_pixel_goal(self, paris, service, tmp_path, **kwargs):
        return embodied_env(
            paris, UERenderClient(service.base_url), tmp_path,
            action_space=ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
            camera_view=CAMERA_VIEW_FRONT_REAR,
            **kwargs,
        )

    # A pixel resolved on-axis, dead ahead: no lateral offset, comfortably
    # inside FakeTrackBService.LATERAL_LIMIT_CM, well below HORIZON_V.
    ON_AXIS = (0.50, 0.80)
    # Same forward distance, but far enough off-centre that the fake's
    # stand-in NavMesh projection gives up -- see FakeTrackBService's own
    # docstring for the geometry.
    OFF_NAVMESH = (0.95, 0.80)
    # Above the fake horizon: nothing for the ray to hit.
    ABOVE_HORIZON = (0.50, 0.30)

    def test_the_menus_never_overlap(self, paris, service, tmp_path):
        pixel = self.pixel_goal(paris, service, tmp_path)
        names = pixel.allowed_tool_names()
        assert "walk_to_pixel" in names
        assert "walk_to" not in names and "walk_to_xy" not in names

        dual = self.dual_pixel_goal(paris, service, tmp_path)
        dual_names = dual.allowed_tool_names()
        assert "walk_to_pixel" in dual_names
        assert "walk_to" not in dual_names and "walk_to_xy" not in dual_names
        assert dual.movement_env_actions() == ("MOVE_TO_PIXEL",)

    def test_only_the_dual_environment_swaps_the_walk_tool_declaration(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.tools import (
            TOOLS_BY_NAME,
            WALK_TO_PIXEL,
            WALK_TO_PIXEL_FRONT_REAR,
        )

        single = self.pixel_goal(paris, service, tmp_path)
        dual = self.dual_pixel_goal(paris, service, tmp_path)
        single_walk = next(
            tool for tool in single.tools_for_prompt()
            if tool.name == "walk_to_pixel")
        dual_walk = next(
            tool for tool in dual.tools_for_prompt()
            if tool.name == "walk_to_pixel")

        assert [param.name for param in single_walk.params] == ["u", "v"]
        assert [param.name for param in dual_walk.params] == ["view", "u", "v"]
        assert dual_walk is WALK_TO_PIXEL_FRONT_REAR
        assert TOOLS_BY_NAME["walk_to_pixel"] is WALK_TO_PIXEL
        assert WALK_TO_PIXEL_FRONT_REAR not in TOOLS_BY_NAME.values()

    def test_sessions_render_only_the_active_pixel_variant(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        single = CourierSession(
            self.pixel_goal(paris, service, tmp_path), city="Paris")
        dual = CourierSession(
            self.dual_pixel_goal(paris, service, tmp_path), city="Paris")

        single_prompt = single.system_prompt()
        dual_prompt = dual.system_prompt()
        assert "walk_to_pixel(u: number, v: number)" in single_prompt
        assert "view:" not in single_prompt
        assert "walk_to_pixel(view: str, u: number, v: number)" in dual_prompt
        assert 'view="front"' in dual_prompt
        assert '"front" or "rear"' in dual_prompt
        assert dual_prompt.count("walk_to_pixel(view=") == 1
        for unavailable in ("select_camera(", "turn_left(", "turn_right("):
            assert unavailable not in dual_prompt

    @pytest.mark.parametrize(("action_space", "camera_view", "message"), [
        (ACTION_SPACE_PIXEL_GOAL_FRONT_REAR, CAMERA_VIEW_FORWARD,
         "pixel_goal_front_rear requires camera_view='front_rear'"),
        (ACTION_SPACE_PIXEL_GOAL_FRONT_REAR, "streets",
         "pixel_goal_front_rear requires camera_view='front_rear'"),
        (ACTION_SPACE_PIXEL_GOAL, CAMERA_VIEW_FRONT_REAR,
         "pixel_goal"),
        ("coordinate", CAMERA_VIEW_FRONT_REAR,
         "camera_view='front_rear' requires action_space='pixel_goal_front_rear'"),
        ("street", CAMERA_VIEW_FRONT_REAR,
         "camera_view='front_rear' requires action_space='pixel_goal_front_rear'"),
    ])
    def test_the_dual_action_and_camera_are_an_exact_cross_product(
            self, paris, service, tmp_path, action_space, camera_view, message):
        with pytest.raises(ValueError, match=message):
            embodied_env(
                paris, UERenderClient(service.base_url), tmp_path,
                action_space=action_space, camera_view=camera_view,
            )

    def test_a_dual_observation_is_one_fresh_front_then_rear_pair(
            self, paris, service, tmp_path):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        service.agent["yaw"] = 90.0
        pose_before = tuple(service.agent[key] for key in ("x", "y", "z", "yaw"))

        photo_rows = env.photo_rows(env.candidates())

        assert len(service.view_pairs) == 1
        assert [(row["view"], row["heading"]) for row in photo_rows] == [
            ("front", "east"),
            ("rear", "west"),
        ]
        assert len({row["capture_group_id"] for row in photo_rows}) == 1
        assert service.observes == [], "discarded street rows never render"
        assert tuple(
            service.agent[key] for key in ("x", "y", "z", "yaw")
        ) == pytest.approx(pose_before)
        assert env._active_pixel_view_pair is not None
        assert env._active_pixel_view_pair.capture_group_id == photo_rows[0][
            "capture_group_id"]
        assert [row["camera_snapshot_id"] for row in photo_rows] == [
            env._active_pixel_view_pair.views[0].camera_snapshot_id,
            env._active_pixel_view_pair.views[1].camera_snapshot_id,
        ]
        for row, expected_yaw in zip(photo_rows, (90.0, 270.0), strict=True):
            assert row["camera_yaw_deg"] == pytest.approx(expected_yaw)
            assert row["bearing"] == pytest.approx(expected_yaw)
            assert row["camera_intrinsics_id"]
            assert row["width"] == CAMERA.width
            assert row["height"] == CAMERA.height
            assert len(row["sha256"]) == 64
            assert row["path"] == row["image"]
            assert Path(row["image"]).is_file()
            assert row["common_pose"] == photo_rows[0]["common_pose"]
            assert row["capture_pose"] == row["common_pose"]
            assert row["capture_timing"] == row["timing"]
            assert "pair_timing" in row

        second = env.photo_rows(env.candidates())
        assert len(service.view_pairs) == 2, "a new turn never reuses the album"
        assert second[0]["capture_group_id"] != photo_rows[0]["capture_group_id"]
        assert second[0]["image"] != photo_rows[0]["image"]
        assert tuple(
            service.agent[key] for key in ("x", "y", "z", "yaw")
        ) == pytest.approx(pose_before)

    def test_dual_navigation_facing_is_the_front_camera_not_the_walk_chord(
            self, paris, service, tmp_path):
        """A curved UE path can have a 25-degree chord/final-yaw gap.

        The phone's relative directions and the front photograph must share
        the preserved common-pose yaw, or both inputs describe different
        notions of "ahead" in the same model turn.
        """
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env._walked_bearing = 306.911079
        body_yaw = 281.656
        env.ue_pose = Pose(
            env.ue_pose.x_cm, env.ue_pose.y_cm, env.ue_pose.z_cm, body_yaw)
        service.agent["yaw"] = body_yaw

        candidates = env.candidates()
        front, rear = env.photo_rows(candidates)

        assert env.facing() == pytest.approx(body_yaw)
        assert front["camera_yaw_deg"] == pytest.approx(env.facing())
        assert rear["camera_yaw_deg"] == pytest.approx(
            (env.facing() + 180.0) % 360.0)
        assert all(
            row["relative"] == relative_of(row["bearing"], body_yaw)
            for row in candidates
        )

    def test_a_session_attaches_front_rear_then_map_with_private_audit_data(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        env = self.dual_pixel_goal(paris, service, tmp_path)
        assert env.navigate().ok
        session = CourierSession(env, city="Paris")

        observation = session.observe()

        assert [frame.kind for frame in observation.frames] == [
            "photograph", "photograph", "map"]
        assert observation.frames[0].label.startswith("[front, ")
        assert observation.frames[1].label.startswith("[rear, ")
        assert observation.frames[2].label == "[map] the map on your phone"
        front, rear = observation.frames[:2]
        assert front.view_id == "front" and rear.view_id == "rear"
        assert front.required_group and rear.required_group
        assert front.capture_group_id == rear.capture_group_id
        assert front.capture_pose == rear.capture_pose
        assert front.camera_snapshot_id != rear.camera_snapshot_id
        for frame in (front, rear):
            serialised = frame.to_dict()
            for key in (
                "view_id", "capture_group_id", "camera_snapshot_id",
                "camera_intrinsics_id", "camera_yaw_deg", "width", "height",
                "sha256", "capture_pose", "capture_timing", "pair_timing",
                "required_group",
            ):
                assert serialised[key] == getattr(frame, key)
            visible = observation.text + "\n" + frame.label
            for private in (
                frame.capture_group_id,
                frame.camera_snapshot_id,
                frame.camera_intrinsics_id,
            ):
                assert private not in visible

    def test_one_pending_observation_is_consumed_by_exactly_one_turn(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        session = CourierSession(
            self.dual_pixel_goal(paris, service, tmp_path), city="Paris")

        first = session.observe()
        second = session.observe()
        assert second is first
        assert len(service.view_pairs) == 1

        turn = session.step("THOUGHT: stay here\n```\nwait()\n```")
        assert turn.image_paths == first.image_paths
        assert turn.prompt == first.text
        assert turn.frame_metadata == [
            {key: value for key, value in frame.to_dict().items()
             if key not in {"path", "svg"}}
            for frame in first.frames
        ]
        assert len(service.view_pairs) == 1

        third = session.observe()
        assert third is not first
        assert len(service.view_pairs) == 2

    def test_format_and_bad_action_errors_each_consume_the_shown_pair(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        session = CourierSession(
            self.dual_pixel_goal(paris, service, tmp_path), city="Paris")

        first = session.observe()
        malformed = session.step("I forgot the action.")
        assert malformed.status == "format_error"
        assert malformed.prompt == first.text
        assert malformed.image_paths == first.image_paths

        second = session.observe()
        assert second is not first
        assert len(service.view_pairs) == 2
        bad_action = session.step(
            'THOUGHT: incomplete\n```\nwalk_to_pixel(view="front", u=0.5)\n```')
        assert bad_action.status == "rejected"
        assert bad_action.error == "bad_pixel"
        assert bad_action.prompt == second.text
        assert bad_action.image_paths == second.image_paths

        third = session.observe()
        assert third is not second
        assert len(service.view_pairs) == 3

    def test_dual_session_type_checks_against_its_view_aware_tool(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        session = CourierSession(
            self.dual_pixel_goal(paris, service, tmp_path), city="Paris")
        session.observe()

        accepted = session.step(
            'THOUGHT: select rear\n```\n'
            'walk_to_pixel(view="rear", u=0.50, v=0.80)\n```')
        assert accepted.status == "accepted"
        assert accepted.tool_kind == "act"

        session.observe()
        wrong_view_type = session.step(
            "THOUGHT: malformed view\n```\n"
            "walk_to_pixel(view=1, u=0.50, v=0.80)\n```")
        assert wrong_view_type.status == "format_error"
        assert "walk_to_pixel(view=…) takes text" in wrong_view_type.error

    def test_dual_chunk_consumes_exact_frames_then_walks_again_on_a_fresh_pair(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession

        env = self.dual_pixel_goal(paris, service, tmp_path)
        session = ChunkedCourierSession(
            env, city="Paris", action_chunk=3)

        shown = session.observe()
        first = session.step(
            'THOUGHT: use rear\n```\n'
            'walk_to_pixel(view="rear", u=0.50, v=0.80)\n```')

        assert first.status == "accepted"
        assert first.prompt == shown.text
        assert first.image_paths == shown.image_paths
        assert first.frame_metadata == [
            {key: value for key, value in frame.to_dict().items()
             if key not in {"path", "svg"}}
            for frame in shown.frames
        ]
        assert env._active_pixel_view_pair is None
        assert len(service.view_pairs) == 1

        fresh = session.observe()
        assert fresh is not shown
        assert fresh.image_paths != shown.image_paths
        assert len(service.view_pairs) == 2
        second = session.step(
            'THOUGHT: use front\n```\n'
            'walk_to_pixel(view="front", u=0.50, v=0.80)\n```')
        assert second.status == "accepted", second.error
        assert len(service.walk_pixels) == 2

    def test_dual_action_chunk_is_reachable_through_the_embodied_adapter(
            self, service, tmp_path):
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession

        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        adapter = _EmbodiedAdapterUnderTest({
            "backend": "embodied",
            "ue_endpoints": str(endpoints),
            "live_cache_root": str(tmp_path / "cache"),
            "difficulty": "solo",
            "stride": "waypoint",
            "max_turns": 3,
            "max_images": 3,
            "action_space": ACTION_SPACE_PIXEL_GOAL_FRONT_REAR,
            "camera_view": CAMERA_VIEW_FRONT_REAR,
            "action_chunk": 3,
        })
        run(adapter.reset(0))
        session = adapter._session
        assert isinstance(session, ChunkedCourierSession)
        assert [param.name for param in
                session._tools_by_name["walk_to_pixel"].params] == [
                    "view", "u", "v"]
        shown = session.observe()

        _, _, _, info = run(adapter.step(
            'THOUGHT: use rear\n```\n'
            'walk_to_pixel(view="rear", u=0.50, v=0.80)\n```'))

        turn = session.run.turns[-1]
        assert info["status"] == "accepted"
        assert turn.prompt == shown.text
        assert turn.image_paths == shown.image_paths
        assert turn.frame_metadata == [
            frame.metadata_dict() for frame in shown.frames]
        assert len(service.view_pairs) == 2
        fresh = session.observe()
        assert fresh is not shown
        assert len(service.view_pairs) == 2
        assert (fresh.frames[0].capture_group_id
                != shown.frames[0].capture_group_id)
        run(adapter.close())

    @pytest.mark.parametrize("exit_case", [
        "format", "budget", "overlong", "bad_action",
    ])
    def test_every_early_dual_chunk_return_consumes_the_shown_pair(
            self, paris, service, tmp_path, exit_case):
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession
        from embodiedbench.agent.courier.loop import Budgets

        env = self.dual_pixel_goal(paris, service, tmp_path)
        budgets = Budgets(steps=0) if exit_case == "budget" else None
        session = ChunkedCourierSession(
            env, city="Paris", action_chunk=3, budgets=budgets)
        shown = session.observe()
        replies = {
            "format": "I omitted the action.",
            "budget": "THOUGHT: wait\n```\nwait()\n```",
            "overlong": (
                "THOUGHT: too much\n```\nwait()\nwait()\nwait()\nwait()\n```"),
            "bad_action": (
                'THOUGHT: incomplete\n```\n'
                'walk_to_pixel(view="front", u=0.50)\n```'),
        }

        turn = session.step(replies[exit_case])

        assert turn.prompt == shown.text
        assert turn.image_paths == shown.image_paths
        assert turn.frame_metadata == [
            {key: value for key, value in frame.to_dict().items()
             if key not in {"path", "svg"}}
            for frame in shown.frames
        ]
        assert len(service.view_pairs) == 1
        fresh = session.observe()
        assert fresh is not shown
        assert len(service.view_pairs) == 2
        assert (fresh.frames[0].capture_group_id
                != shown.frames[0].capture_group_id)

    def test_dual_chunk_type_checks_the_session_view_signature(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession

        session = ChunkedCourierSession(
            self.dual_pixel_goal(paris, service, tmp_path),
            city="Paris", action_chunk=3)
        session.observe()

        turn = session.step(
            "THOUGHT: malformed view\n```\n"
            "walk_to_pixel(view=1, u=0.50, v=0.80)\n```")

        assert turn.status == "format_error"
        assert "walk_to_pixel(view=…) takes text" in turn.error

    @pytest.mark.parametrize(("case", "causal"), [
        ("missing", "photograph label"),
        ("unknown", "selected view"),
        ("resolution", "visible ground"),
        ("controller", "blocked"),
    ])
    def test_dual_chunk_pixel_refusals_are_causal_not_steering_advice(
            self, paris, service, tmp_path, case, causal):
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession

        if case == "controller":
            service.wall_after_cm = 200.0
        session = ChunkedCourierSession(
            self.dual_pixel_goal(paris, service, tmp_path),
            city="Paris", action_chunk=3)
        session.observe()
        calls = {
            "missing": "walk_to_pixel(u=0.50, v=0.80)",
            "unknown": 'walk_to_pixel(view="side", u=0.50, v=0.80)',
            "resolution": 'walk_to_pixel(view="front", u=0.50, v=0.30)',
            "controller": 'walk_to_pixel(view="front", u=0.50, v=0.80)',
        }

        turn = session.step(f"THOUGHT: test\n```\n{calls[case]}\n```")

        assert turn.status == "rejected"
        feedback = session.feedback.lower()
        assert causal in feedback
        assert "which way does the route" not in feedback
        assert "which street" not in feedback
        for hint in (
            "choose front", "choose rear", "turn around", "left", "right",
            "try u", "try v", "different pixel", "open pavement",
        ):
            assert hint not in feedback

    def test_it_needs_the_forward_camera(self, paris, service, tmp_path):
        with pytest.raises(ValueError, match="pixel_goal"):
            embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                        action_space=ACTION_SPACE_PIXEL_GOAL)

    @pytest.mark.parametrize(("positional", "keywords"), [
        ((0.5, 0.8), {}),
        ((), {"u": 0.5, "v": 0.8}),
    ])
    def test_the_legacy_decoder_accepts_only_its_two_exact_shapes(
            self, paris, service, tmp_path, positional, keywords):
        env = self.pixel_goal(paris, service, tmp_path)
        out = env.walk_to_pixel(*positional, **keywords)
        assert out.ok
        request = service.walk_pixels[-1]
        assert not ({"view", "capture_group_id", "camera_snapshot_id"}
                    & set(request))

    @pytest.mark.parametrize(("positional", "keywords"), [
        ((0.5,), {}),
        ((0.5, 0.8, 0.9), {}),
        ((0.5,), {"v": 0.8}),
        ((), {"view": "front", "u": 0.5, "v": 0.8}),
        ((), {"u": 0.5, "v": 0.8, "extra": True}),
    ])
    def test_the_legacy_decoder_refuses_mixing_view_and_extra_arguments(
            self, paris, service, tmp_path, positional, keywords):
        env = self.pixel_goal(paris, service, tmp_path)
        calls_before = len(service.walk_pixels)
        out = env.walk_to_pixel(*positional, **keywords)
        assert not out.ok and out.code == "bad_pixel"
        assert len(service.walk_pixels) == calls_before

    @pytest.mark.parametrize(("positional", "keywords"), [
        (("rear", 0.5, 0.8), {}),
        ((), {"view": "rear", "u": 0.5, "v": 0.8}),
    ])
    def test_the_dual_decoder_accepts_only_its_three_exact_shapes(
            self, paris, service, tmp_path, positional, keywords):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())
        start = env._here_cm()

        out = env.walk_to_pixel(*positional, **keywords)

        assert out.ok
        assert env._here_cm()[0] < start[0]
        assert env.pixel_view_counts == {"front": 0, "rear": 1}
        request = service.walk_pixels[-1]
        assert request["view"] == "rear"
        assert request["capture_group_id"]
        assert request["camera_snapshot_id"].endswith("-rear")

    @pytest.mark.parametrize(("positional", "keywords"), [
        (("rear", 0.5), {}),
        (("rear", 0.5, 0.8, 0.9), {}),
        (("rear",), {"u": 0.5, "v": 0.8}),
        ((), {"view": "rear", "u": 0.5, "v": 0.8, "extra": True}),
    ])
    def test_the_dual_decoder_refuses_mixing_and_extra_arguments(
            self, paris, service, tmp_path, positional, keywords):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())
        calls_before = len(service.walk_pixels)
        out = env.walk_to_pixel(*positional, **keywords)
        assert not out.ok and out.code == "bad_pixel"
        assert len(service.walk_pixels) == calls_before
        assert env._active_pixel_view_pair is None

    def test_missing_or_unknown_dual_views_are_neutral_and_never_walk(
            self, paris, service, tmp_path):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())
        calls_before = len(service.walk_pixels)

        missing = env.walk_to_pixel(u=0.5, v=0.8)
        assert not missing.ok and missing.code == "missing_pixel_view"
        assert missing.message == (
            "This action needs the photograph label from this turn.")
        assert len(service.walk_pixels) == calls_before
        assert env._active_pixel_view_pair is None

        env.photo_rows(env.candidates())
        unknown = env.walk_to_pixel(view="left", u=0.5, v=0.8)
        assert not unknown.ok and unknown.code == "unknown_pixel_view"
        assert unknown.message == (
            "The selected view is not part of this turn's photographs.")
        assert len(service.walk_pixels) == calls_before
        assert env._active_pixel_view_pair is None

        forbidden = (
            "choose front", "choose rear", "turn around", "left", "right",
            "try u", "try v", "different pixel", "open pavement",
        )
        for feedback in (missing.message, unknown.message):
            low = feedback.lower()
            assert not any(hint in low for hint in forbidden)

    @pytest.mark.parametrize(("point", "code"), [
        ((1.4, 0.8), "out_of_range"),
        ((0.5, 0.3), "unwalkable_pixel"),
    ])
    def test_dual_pixel_feedback_is_causal_but_never_steering_advice(
            self, paris, service, tmp_path, point, code):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())

        outcome = env.walk_to_pixel("front", *point)

        assert not outcome.ok and outcome.code == code
        low = outcome.message.lower()
        for hint in (
            "choose front", "choose rear", "turn around", "left", "right",
            "try u", "try v", "different pixel", "open pavement",
        ):
            assert hint not in low

    def test_a_dual_pair_is_single_use_after_success_refusal_and_new_capture(
            self, paris, service, tmp_path):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())
        assert env.walk_to_pixel("front", *self.ON_AXIS).ok
        calls_after_success = len(service.walk_pixels)

        reused = env.walk_to_pixel("front", *self.ON_AXIS)
        assert not reused.ok and reused.code == "missing_pixel_view"
        assert len(service.walk_pixels) == calls_after_success

        env.photo_rows(env.candidates())
        rejected = env.walk_to_pixel("front", *self.ABOVE_HORIZON)
        assert not rejected.ok and rejected.code == "unwalkable_pixel"
        calls_after_rejection = len(service.walk_pixels)
        reused = env.walk_to_pixel("front", *self.ON_AXIS)
        assert not reused.ok and reused.code == "missing_pixel_view"
        assert len(service.walk_pixels) == calls_after_rejection

        env.photo_rows(env.candidates())
        assert env.walk_to_pixel("rear", *self.ON_AXIS).ok

    @pytest.mark.parametrize("error", [
        PixelGoalPairIntegrityError("incomplete_view_pair"),
        ProtocolViolation("incomplete_view_pair"),
    ], ids=("pair-integrity", "wire-protocol"))
    def test_pair_capture_integrity_is_audited_once_and_reraised(
            self, paris, service, tmp_path, monkeypatch, error):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        monkeypatch.setattr(
            env._client, "observe_views",
            lambda request: (_ for _ in ()).throw(error),
        )
        clock_before = env.sim_seconds

        with pytest.raises(type(error), match="incomplete_view_pair"):
            env.photo_rows(env.candidates())

        audit = [row for row in env.embodied_log
                 if row.get("kind") == "pixel_view_integrity"]
        assert [row["code"] for row in audit] == ["incomplete_view_pair"]
        assert env._active_pixel_view_pair is None
        assert env.sim_seconds == clock_before

    def test_private_pair_metadata_integrity_fails_before_the_walk(
            self, paris, service, tmp_path):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())
        env._active_pixel_view_metadata["rear"]["camera_snapshot_id"] = "stale"
        calls_before = len(service.walk_pixels)

        with pytest.raises(PixelGoalPairIntegrityError,
                           match="camera_snapshot_stale"):
            env.walk_to_pixel("rear", *self.ON_AXIS)

        assert len(service.walk_pixels) == calls_before
        audit = [row for row in env.embodied_log
                 if row.get("kind") == "pixel_view_integrity"]
        assert [row["code"] for row in audit] == ["camera_snapshot_stale"]
        assert env._active_pixel_view_pair is None

    def test_a_bound_response_mismatch_is_not_feedback_or_a_retry(
            self, paris, service, tmp_path, monkeypatch):
        env = self.dual_pixel_goal(paris, service, tmp_path)
        env.photo_rows(env.candidates())
        real_walk = env._client.walk_pixel

        def mismatched(request):
            return replace(real_walk(request), view="front")

        monkeypatch.setattr(env._client, "walk_pixel", mismatched)
        calls_before = len(service.walk_pixels)
        clock_before = env.sim_seconds

        with pytest.raises(PixelGoalPairIntegrityError,
                           match="camera_snapshot_view_mismatch"):
            env.walk_to_pixel("rear", *self.ON_AXIS)

        assert len(service.walk_pixels) == calls_before + 1
        audit = [row for row in env.embodied_log
                 if row.get("kind") == "pixel_view_integrity"]
        assert [row["code"] for row in audit] == [
            "camera_snapshot_view_mismatch"]
        assert env._active_pixel_view_pair is None
        assert env.sim_seconds == clock_before

    def test_the_prompt_never_teaches_a_call_it_cannot_run(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        env = self.pixel_goal(paris, service, tmp_path)
        session = CourierSession(env, city="Paris")   # asserts this itself
        prompt = session.system_prompt()
        assert "walk_to_pixel(" in prompt
        assert "walk_to(" not in prompt and "walk_to_xy(" not in prompt

    def test_a_resolved_pixel_walks_and_lands_the_pawn(
            self, paris, service, tmp_path):
        env = self.pixel_goal(paris, service, tmp_path)
        start = env._here_cm()
        walks_before = len(service.walk_pixels)

        out = env.walk_to_pixel(*self.ON_AXIS)

        assert out.ok, out.message
        assert out.moved
        assert len(service.walk_pixels) == walks_before + 1
        assert math.dist(start, env._here_cm()) > 1.0, "the pawn really moved"
        assert env.pixel_walks == 1

    def test_a_pixel_above_the_horizon_is_refused_and_nothing_moves(
            self, paris, service, tmp_path):
        """No ground hit at all -- the fake's stand-in for a ray into the
        sky. Nothing walks, and the courier gets the causal class without
        engine internals or recovery advice."""
        env = self.pixel_goal(paris, service, tmp_path)
        start = env._here_cm()
        walks_before = len(service.walk_pixels)

        out = env.walk_to_pixel(*self.ABOVE_HORIZON)

        assert not out.ok and out.code == "unwalkable_pixel"
        assert out.message == "The selected pixel did not hit visible ground."
        assert "no_ground_hit" not in out.message, "the raw code must not leak"
        assert len(service.walk_pixels) == walks_before + 1, (
            "resolution was attempted and reported, even though it failed")
        assert env._here_cm() == pytest.approx(start)
        refused = [h for h in env.embodied_log if h.get("code") == "unwalkable_pixel"]
        assert len(refused) == 1 and refused[0]["kind"] == "pixel_refused"
        assert refused[0]["engine_rejection_reason"] == "no_ground_hit"

    def test_a_pixel_off_the_navmesh_reports_ground_outside_navigation(
            self, paris, service, tmp_path):
        """A different geometric failure -- a real ground hit that sits too
        far from any walkable surface to snap -- gets a distinct causal class
        without exposing the engine's code."""
        env = self.pixel_goal(paris, service, tmp_path)
        out = env.walk_to_pixel(*self.OFF_NAVMESH)
        assert not out.ok and out.code == "unwalkable_pixel"
        assert out.message == (
            "The selected pixel hit ground, but it was not on a reachable "
            "walkable navigation surface."
        )
        assert "off_navmesh" not in out.message
        assert env.embodied_log[-1]["engine_rejection_reason"] == "off_navmesh"

    @pytest.mark.parametrize(("engine_reason", "policy_message"), [
        ("no_geometry_hit", "The selected pixel did not hit visible ground."),
        ("hit_not_walkable_ground",
         "The selected pixel hit a non-walkable surface."),
        ("navmesh_projection_failed",
         "The selected pixel hit ground, but it was not on a reachable "
         "walkable navigation surface."),
        ("navmesh_adjustment_exceeded",
         "The selected pixel hit ground, but it was not on a reachable "
         "walkable navigation surface."),
        ("controller_path_detour_exceeded",
         "The selected pixel hit visible ground, but reaching it would "
         "require a long detour out of view."),
        ("controller_path_enters_unmarked_road",
         "The route to the selected point would leave the pedestrian way "
         "outside a marked crosswalk; the only crossing that counts here "
         "is the one the blue route on your phone uses."),
        ("controller_path_surface_unverified",
         "The route to the selected point could not be verified as "
         "pedestrian-only."),
        ("future_engine_detail=BP_ProceduralBuilding_490",
         "The selected pixel could not be resolved to a walkable destination."),
    ])
    def test_engine_pixel_rejections_are_coarsened_for_the_policy(
            self, paris, service, tmp_path, monkeypatch,
            engine_reason, policy_message):
        """A new/raw engine detail must stay diagnostic rather than becoming
        a prompt feature the policy can exploit."""
        monkeypatch.setattr(
            service, "_resolve_pixel",
            lambda agent, u, v: (None, engine_reason),
        )
        env = self.pixel_goal(paris, service, tmp_path)

        out = env.walk_to_pixel(*self.ON_AXIS)

        assert not out.ok and out.code == "unwalkable_pixel"
        assert out.message == policy_message
        assert engine_reason not in out.message
        assert env.embodied_log[-1]["engine_rejection_reason"] == engine_reason

    def test_pixel_rejection_feedback_has_no_recovery_hint(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        env = self.pixel_goal(paris, service, tmp_path)
        session = CourierSession(env, city="Paris")

        turn = session.step(
            "THOUGHT: test the visible point\n```\nwalk_to_pixel(0.50, 0.30)\n```"
        )

        assert turn.status == "rejected"
        assert "did not hit visible ground" in session.feedback
        feedback = session.feedback.lower()
        for hint in ("left", "right", "horizon", "different point",
                     "open pavement", "tiny nudge", "u/v"):
            assert hint not in feedback

    def test_blocked_pixel_feedback_reports_the_cause_without_recovery_hint(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        service.wall_after_cm = 200.0
        env = self.pixel_goal(paris, service, tmp_path)
        session = CourierSession(env, city="Paris")

        turn = session.step(
            "THOUGHT: test the visible point\n```\nwalk_to_pixel(0.50, 0.80)\n```"
        )

        assert turn.status == "rejected" and turn.error == "stuck"
        assert "blocked" in session.feedback.lower()
        feedback = session.feedback.lower()
        for hint in ("left", "right", "different point", "pick", "aim"):
            assert hint not in feedback

    def test_non_pixel_rejection_feedback_keeps_its_own_reason(
            self, paris, service, tmp_path):
        from embodiedbench.agent.courier.session import CourierSession

        env = self.pixel_goal(paris, service, tmp_path)
        session = CourierSession(env, city="Paris")

        turn = session.step("THOUGHT: try collection\n```\ncollect()\n```")

        assert turn.status == "rejected" and turn.error == "not_at_pickup"
        assert env.active_order().pickup.text in session.feedback
        feedback = session.feedback.lower()
        for unrelated in ("selected pixel", "different point", "compass point",
                          "which street", "left", "right"):
            assert unrelated not in feedback

    def test_a_pixel_outside_the_picture_is_refused_client_side(
            self, paris, service, tmp_path):
        env = self.pixel_goal(paris, service, tmp_path)
        walks_before = len(service.walk_pixels)

        out = env.walk_to_pixel(1.4, 0.8)

        assert not out.ok and out.code == "out_of_range"
        assert len(service.walk_pixels) == walks_before, (
            "an out-of-range pixel is never sent to the engine at all")

    def test_a_pixel_that_is_not_a_number_is_a_refusal_not_a_crash(
            self, paris, service, tmp_path):
        env = self.pixel_goal(paris, service, tmp_path)
        out = env.walk_to_pixel("centre", 0.8)
        assert not out.ok and out.code == "bad_pixel"

    def test_the_position_it_is_told_is_the_one_its_next_call_counts_from(
            self, paris, service, tmp_path):
        env = self.pixel_goal(paris, service, tmp_path)
        env.walk_to_pixel(*self.ON_AXIS)

        pawn = env._here_cm()
        assert env.position() == pytest.approx(pawn)

    def test_facing_is_the_way_it_walked(self, paris, service, tmp_path):
        env = self.pixel_goal(paris, service, tmp_path)
        start = env._here_cm()
        env.walk_to_pixel(*self.ON_AXIS)
        assert env.facing() == pytest.approx(
            bearing_deg(start, env._here_cm()), abs=1.0)

    def test_a_resolved_walk_carries_a_conformant_navigation_request(
            self, paris, service, tmp_path):
        """design plan §9.4's audit record, not just a plan: the log entry has to
        be the exact shape ``embodiedbench.schemas.embodiment`` already
        reserves for ``nav_pixel_goal``, round-trippable through the schema
        that validates it."""
        from embodiedbench.schemas.embodiment import NavigationRequest
        from embodiedbench.schemas.environment import NavigationMode

        env = self.pixel_goal(paris, service, tmp_path)
        env.walk_to_pixel(*self.ON_AXIS)

        hop = env.embodied_log[-1]
        assert hop["kind"] == "pixel_goal"
        payload = hop["navigation_request"]
        assert payload is not None
        request = NavigationRequest.model_validate(payload)
        assert request.mode is NavigationMode.NAV_PIXEL_GOAL
        assert request.source_image_point.u_norm == pytest.approx(self.ON_AXIS[0])
        assert request.source_image_point.v_norm == pytest.approx(self.ON_AXIS[1])
        assert request.raw_world_hit is not None
        assert request.navmesh_adjustment_cm is not None

    def test_a_refused_pixel_carries_no_navigation_request(
            self, paris, service, tmp_path):
        """The schema demands a raw world hit to record one; a pixel that
        never hit anything has nothing honest to put there."""
        env = self.pixel_goal(paris, service, tmp_path)
        env.walk_to_pixel(*self.ABOVE_HORIZON)
        assert "navigation_request" not in env.embodied_log[-1]

    def test_the_summary_says_which_question_was_asked(
            self, paris, service, tmp_path):
        env = self.pixel_goal(paris, service, tmp_path)
        env.walk_to_pixel(*self.ON_AXIS)
        env.walk_to_pixel(*self.OFF_NAVMESH)

        block = env.summary()["embodied"]
        assert block["action_space"] == ACTION_SPACE_PIXEL_GOAL
        # Both calls round-tripped the engine -- pixel_walks counts attempts,
        # not successes, the same as /walk_pixel calls made -- and exactly
        # one of them ever became a target worth walking to.
        assert block["pixel_walks"] == 2
        assert block["pixel_unresolved"] == 1
        # ...and the generic hop aggregation the coordinate space also
        # relies on keeps counting the one hop that actually walked.
        assert block["hops"] == 1

    def test_dual_pixel_telemetry_binds_the_selected_view_and_summarises_it(
            self, paris, service, tmp_path):
        env = self.dual_pixel_goal(paris, service, tmp_path)

        rejected_rows = env.photo_rows(env.candidates())
        rejected_group = rejected_rows[0]["capture_group_id"]
        rejected_snapshot = rejected_rows[1]["camera_snapshot_id"]
        rejected = env.walk_to_pixel("rear", *self.ABOVE_HORIZON)
        assert not rejected.ok and rejected.code == "unwalkable_pixel"
        rejection_log = env.embodied_log[-1]
        assert rejection_log["selected_view"] == "rear"
        assert rejection_log["capture_group_id"] == rejected_group
        assert rejection_log["camera_snapshot_id"] == rejected_snapshot

        success_rows = env.photo_rows(env.candidates())
        success_group = success_rows[0]["capture_group_id"]
        success_snapshot = success_rows[0]["camera_snapshot_id"]
        success = env.walk_to_pixel("front", *self.ON_AXIS)
        assert success.ok
        success_log = env.embodied_log[-1]
        assert success_log["selected_view"] == "front"
        assert success_log["capture_group_id"] == success_group
        assert success_log["camera_snapshot_id"] == success_snapshot
        assert success_log["controller_status"] == "arrived"

        env.embodied_log.extend([
            {"kind": "pixel_view_integrity", "code": "incomplete_view_pair"},
            {"kind": "pixel_view_integrity", "code": "camera_snapshot_stale"},
            {"kind": "pixel_view_integrity",
             "code": "camera_snapshot_view_mismatch"},
            {"kind": "pixel_view_integrity",
             "code": "camera_snapshot_group_mismatch"},
        ])
        block = env.summary()["embodied"]
        assert block["pixel_view_counts"] == {"front": 1, "rear": 1}
        assert block["pixel_view_unresolved"] == {"front": 0, "rear": 1}
        assert block["pixel_view_controller_success"] == {
            "front": 1, "rear": 0}
        assert block["pixel_view_integrity_failures"] == {
            "incomplete_view_pair": 1,
            "camera_snapshot_stale": 1,
            "camera_snapshot_view_mismatch": 1,
            "camera_snapshot_group_mismatch": 1,
        }

    def test_a_blocked_walk_keeps_the_ground_it_covered(
            self, paris, service, tmp_path):
        service.wall_after_cm = 200.0
        env = self.pixel_goal(paris, service, tmp_path)
        start = env._here_cm()

        out = env.walk_to_pixel(*self.ON_AXIS)

        assert not out.ok and out.code == "stuck"
        assert math.dist(start, env._here_cm()) > 100.0, (
            "the pawn kept the ground its walk covered")


@needs_maps
class TestTheTraceRecordsWhatTheCourierSaw:
    """One event per CALL, with the picture beside it.

    Episode telemetry is the right grain for a fleet and the wrong one for
    "what did it see, what did it decide, what did the world do" -- which is
    asked one call at a time, and which a chunked turn flattens into one
    action with one joined feedback paragraph.
    """

    def test_a_trace_carries_the_frames_after_the_cache_is_gone(
            self, paris, service, tmp_path, monkeypatch):
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.runtime.live.trace import EpisodeTrace

        monkeypatch.setenv("EB_LIVE_TRACE_DIR", str(tmp_path / "trace"))
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           action_space="coordinate", max_step_m=10.0,
                           arrive_cm=100.0, tick_chunk=2)
        session = CourierSession(env, city="Paris")
        trace = EpisodeTrace(tmp_path / "trace", "ep", {"action_space": "coordinate"})

        here = env._here_cm()
        session.step(f'```\nwalk_to_xy({here[0]/100 + 5.0:.1f}, {here[1]/100:.1f})\n```')
        trace.record(env, session.run.turns[-1])
        path = trace.close(env)

        assert path and path.exists()
        rec = json.loads(path.read_text())
        event = rec["events"][0]
        assert event["calls"] and event["calls"][0]["action"].startswith("walk_to_xy")
        assert event["calls"][0]["walk"]["kind"] == "coordinate"
        assert event["pose_before_m"] and len(event["pose_before_m"]) == 2
        # The frames were rendered before the call ran, so the pose they
        # belong to is not the one the turn ended at.
        assert event["pose_after_m"] != event["pose_before_m"]
        assert event["observation"] and event["reply"]
        # The frames were copied out of the cache, which the episode owns and
        # takes down with it -- the whole reason they are copied.
        assert event["frames"], "no frames kept"
        for rel in event["frames"]:
            assert (path.parent / rel).exists()

    def test_every_call_of_a_chunk_gets_its_own_row(
            self, paris, service, tmp_path):
        """The turn-level record says one action and one outcome; the reply
        named three, and the second and third were judged from positions the
        first two walks produced."""
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession
        from embodiedbench.runtime.live.trace import EpisodeTrace

        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           action_space="coordinate", max_step_m=10.0,
                           arrive_cm=100.0, tick_chunk=2)
        session = ChunkedCourierSession(env, city="Paris", action_chunk=3)
        trace = EpisodeTrace(tmp_path / "trace", "ep", {})

        x, y = (v / 100.0 for v in env._here_cm())
        session.step("```\n" + "\n".join(
            f"walk_to_xy({x + 8.0 * i:.1f}, {y:.1f})" for i in (1, 2, 3)) + "\n```")
        trace.record(env, session.run.turns[-1])
        rec = json.loads(trace.close(env).read_text())

        calls = rec["events"][0]["calls"]
        assert len(calls) == 3, "all three named waypoints are recorded"
        assert [c["index"] for c in calls] == [0, 1, 2]
        # One happened; the other two were the plan. A trace that kept only
        # what ran could not tell a plan that was never made from one that was
        # made and superseded, and the difference is the reason for asking for
        # three in the first place.
        assert calls[0]["status"] == "accepted"
        assert [c["status"] for c in calls[1:]] == ["planned", "planned"]
        assert calls[0]["from_xy_m"] is not None
        assert calls[0]["feedback"]
        assert calls[0]["walk"] is not None
        assert all(c["walk"] is None for c in calls[1:]), (
            "a planned call consumed no walk")


@needs_maps
class TestADeadInstanceDoesNotEndTheRun:
    """Twice in one day a training job ended on `ServiceUnreachable: ue-0
    /walk unreachable`. A lease hands the env one instance and the env had
    nowhere else to go, while the pool -- which strikes an unreachable
    instance instantly on the /render path -- never learned, because a leased
    episode talks to the client directly."""

    def test_the_pool_strikes_an_instance_that_dies_under_a_lease(self, tmp_path):
        from embodiedbench.runtime.live.client import ServiceUnreachable

        pool = RenderPool(write_endpoints(tmp_path / "e.json",
                                          [("ue-a", "http://127.0.0.1:1")], seats=2),
                          lease_timeout_s=1.0, lease_poll_s=0.05)
        assert pool.members[0].strikes == 0
        with contextlib.suppress(ServiceUnreachable):
            with pool.lease_embodied("ep"):
                raise ServiceUnreachable("ue-a: /walk unreachable")
        assert pool.members[0].strikes >= 1, "the corpse stayed the healthiest member"
        # ...and a busy does not count: it just answered.
        pool.members[0].strikes = 0
        with contextlib.suppress(ServiceBusy):
            with pool.lease_embodied("ep2"):
                raise ServiceBusy("another episode holds this instance")
        assert pool.members[0].strikes == 0

    def test_a_walk_onto_a_dead_instance_is_a_refusal_not_a_crash(
            self, paris, service, tmp_path):
        """The walk is not re-issued -- /walk is the one non-idempotent
        endpoint and an unreachable service may or may not have moved the pawn
        -- so the call is refused and the episode carries on from the node."""
        from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv

        endpoints = write_endpoints(tmp_path / "e.json", [service], seats=4)
        pool = RenderPool(endpoints, lease_timeout_s=2.0, lease_poll_s=0.05)
        env = EmbodiedCourierEnv(paris, pool, episode_id=EPISODE,
                                 cache_root=tmp_path / "cache", seed=5,
                                 action_space="coordinate", max_step_m=10.0,
                                 arrive_cm=100.0, tick_chunk=2)
        env.reset()
        here = env._here_cm()
        node_before = env.node_id

        # The instance goes away mid-walk.
        from embodiedbench.runtime.live.client import ServiceUnreachable
        real_walk = env._ue().walk
        calls = {"n": 0}

        def dies_once(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ServiceUnreachable("ue-0: /walk unreachable (timed out)")
            return real_walk(request)

        env._ue().walk = dies_once
        out = env.walk_to_xy(here[0] / 100.0 + 6.0, here[1] / 100.0)

        assert not out.ok and out.code == "instance_moved"
        assert env.node_id == node_before, "it stands where the graph believes"
        moved = [h for h in env.embodied_log if h.get("recovery") == "moved_instance"]
        assert moved, env.embodied_log
        # ...and the episode is usable afterwards.
        env._ue().walk = real_walk
        again = env.walk_to_xy(*[v / 100.0 + 5.0 for v in env._here_cm()])
        assert again.ok, again.message


class TestTheWalkReportsWhereItStarted:
    """`walked_cm` came back 0.00 on thirteen consecutive walks whose start
    and end poses differed by half a metre to five -- and could not be called
    wrong, because the only start pose available was the CALLER's, which
    /observe overwrites and other couriers' ticks move. Twice I called the
    count impossible against a baseline that was not the walk's own.

    With both ends in one response it is an arithmetic identity: a path is
    never shorter than the line it spans."""

    def test_the_field_is_optional_and_the_golden_bytes_do_not_move(self):
        golden = (GOLDEN / "track_b_walk_response.json").read_text()
        message = WalkResponse.from_dict(json.loads(golden))
        assert message.start_pose is None, "the fixture predates the field"
        assert dumps(message.to_dict()) == golden, "omitted when absent"

    def test_a_start_pose_round_trips(self):
        message = WalkResponse(
            arrived=True, stuck=False, timeout=False, ticks=7,
            sim_seconds=1.4, pose=Pose(10.0, 20.0, 100.0, 90.0),
            walked_cm=33.0, start_pose=Pose(0.0, 0.0, 100.0, 0.0))
        again = WalkResponse.from_dict(message.to_dict())
        assert again == message
        assert again.start_pose is not None
        assert "start_pose" in message.to_dict()

    @needs_maps
    def test_the_env_records_path_against_displacement(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           action_space="coordinate", max_step_m=10.0,
                           arrive_cm=100.0, tick_chunk=2)
        here = env._here_cm()
        env.walk_to_xy(here[0] / 100.0 + 6.0, here[1] / 100.0)
        hop = env.embodied_log[-1]
        assert "service_walked_cm" in hop and "service_displacement_cm" in hop
        # The identity, on a service that reports both ends.
        if hop["service_displacement_cm"] is not None:
            assert hop["service_walked_cm"] >= hop["service_displacement_cm"] - 0.5
    def test_a_fleet_of_one_re_seats_on_itself(self, paris, service, tmp_path):
        """The fleet we actually run has one instance, and "move to another"
        has nowhere to go there -- which is how a run kept dying with the move
        already written. `unreachable` is a timed-out or reset socket, not a
        death certificate: the engine behind it had been up for hours every
        time this was measured."""
        from embodiedbench.runtime.live.client import ServiceUnreachable
        from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv

        endpoints = write_endpoints(tmp_path / "e.json", [service], seats=4)
        pool = RenderPool(endpoints, lease_timeout_s=2.0, lease_poll_s=0.05)
        env = EmbodiedCourierEnv(paris, pool, episode_id=EPISODE,
                                 cache_root=tmp_path / "cache", seed=5,
                                 action_space="coordinate", max_step_m=10.0,
                                 arrive_cm=100.0, tick_chunk=2)
        env.reset()
        env.reseat_pause_s = 0.01
        assert len(pool.members) == 1, "the case under test is a fleet of one"

        real = env._ue().walk
        fired = {"n": 0}

        def dies_once(request):
            fired["n"] += 1
            if fired["n"] == 1:
                raise ServiceUnreachable("ue-0: /walk unreachable (timed out)")
            return real(request)

        env._ue().walk = dies_once
        here = env._here_cm()
        out = env.walk_to_xy(here[0] / 100.0 + 6.0, here[1] / 100.0)

        assert not out.ok and out.code == "instance_moved"
        env._ue().walk = real
        assert env.walk_to_xy(*[v / 100.0 + 5.0 for v in env._here_cm()]).ok

    def test_opening_an_episode_survives_a_wedged_instance(
            self, paris, service, tmp_path):
        """Opening the episode is the OTHER call that meets a wedged engine,
        and it was the one left unguarded: /walk learned to re-seat and
        reset() still ended the training job with `render_failed: spawn
        failed: AssertionError:` -- the engine answering /healthz green while
        every RPC into it raises."""
        from embodiedbench.runtime.live.client import RenderFailedError
        from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv

        endpoints = write_endpoints(tmp_path / "e.json", [service], seats=4)
        pool = RenderPool(endpoints, lease_timeout_s=2.0, lease_poll_s=0.05)
        env = EmbodiedCourierEnv(paris, pool, episode_id=EPISODE,
                                 cache_root=tmp_path / "cache", seed=5,
                                 action_space="coordinate", max_step_m=10.0,
                                 arrive_cm=100.0, tick_chunk=2)
        env.reseat_pause_s = 0.01

        real = pool.members[0].client.episode
        fired = {"n": 0}

        def wedged_once(request):
            fired["n"] += 1
            if fired["n"] == 1:
                raise RenderFailedError(
                    "render_failed: spawn failed: AssertionError:")
            return real(request)

        pool.members[0].client.episode = wedged_once
        env.reset()          # would have raised before

        assert fired["n"] == 2, "it retried exactly once"
        assert [h for h in env.embodied_log
                if h.get("recovery") == "reseat_on_open"]

    def test_it_gives_up_rather_than_spin_on_an_instance_that_cannot_spawn(
            self, paris, service, tmp_path):
        from embodiedbench.runtime.live.client import RenderFailedError
        from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv

        endpoints = write_endpoints(tmp_path / "e.json", [service], seats=4)
        pool = RenderPool(endpoints, lease_timeout_s=2.0, lease_poll_s=0.05)
        env = EmbodiedCourierEnv(paris, pool, episode_id=EPISODE,
                                 cache_root=tmp_path / "cache", seed=5,
                                 action_space="coordinate", max_step_m=10.0,
                                 arrive_cm=100.0, tick_chunk=2)
        env.reseat_pause_s = 0.01

        def always_wedged(request):
            raise RenderFailedError("render_failed: spawn failed: AssertionError:")

        pool.members[0].client.episode = always_wedged
        with pytest.raises(RenderFailedError):
            env.reset()
