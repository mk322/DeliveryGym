"""Regression coverage for the real-engine pixel-goal delivery seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.pixel_goal_courier_backend as courier_backend
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.live.embodied_env import (
    ACTION_SPACE_PIXEL_GOAL,
    CAMERA_VIEW_FORWARD,
)
from embodiedbench.runtime.live.protocol import (
    CameraSpec,
    EpisodeResponse,
    ObserveRequest,
    ObserveViewsRequest,
    PixelSpec,
    Pose,
    WalkPixelRequest,
)
from embodiedbench.runtime.pixel_goal import (
    PixelGoalFrame,
    PixelGoalPairIntegrityError,
    PixelGoalRejected,
    PixelGoalViewPair,
)
from embodiedbench.schemas.runtime import ControllerOutcomeCode
from tools.pixel_goal_courier_backend import CorridorDeliveryEnv, SpearTrackBClient


MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps" / "citycore-paris")
requires_maps = pytest.mark.skipif(
    not MAPS.exists(), reason="vendored maps not present")


class RecordingPairRuntime:
    def __init__(self, *, reject: bool = False) -> None:
        self.pairs = [runtime_pair()]
        self.pair = self.pairs[0]
        self.capture_calls: list[dict[str, object]] = []
        self.reject = reject
        self.executed_frames: list[PixelGoalFrame] = []

    def capture_view_pair(self, **kwargs):
        self.capture_calls.append(kwargs)
        return self.pairs[len(self.capture_calls) - 1]

    def capture_frame(self, **_kwargs):
        return self.pair.front

    def execute(self, action, frame):
        self.executed_frames.append(frame)
        if self.reject:
            raise PixelGoalRejected("no_geometry_hit")
        started = SimpleNamespace(request=SimpleNamespace(
            raw_world_hit=SimpleNamespace(x_cm=-800.0, y_cm=0.0),
            projected_target=SimpleNamespace(x_cm=-800.0, y_cm=0.0),
            navmesh_adjustment_cm=0.0,
        ))
        result = SimpleNamespace(
            outcome=ControllerOutcomeCode.ACCEPTED,
            final_pose=None,
            elapsed_sim_s=1.0,
            distance_travelled_cm=700.0,
        )
        return started, result


def runtime_pair(
    *,
    group: str = "view-pair-4",
    front_snapshot: str = "front-snapshot",
    rear_snapshot: str = "rear-snapshot",
    pose: tuple[float, float, float, float] = (1.0, 2.0, 90.0, 45.0),
) -> PixelGoalViewPair:
    timing = {"capture_read_ms": 1.0, "encode_ms": 2.0, "wall_ms": 3.0}
    return PixelGoalViewPair(
        capture_group_id=group,
        pose=pose,
        front=PixelGoalFrame(
            rgb_data_url="data:image/jpeg;base64,ZnJvbnQ=",
            camera_snapshot_id=front_snapshot,
            camera_intrinsics_id="perspective-640x360-hfov90",
            width_px=640,
            height_px=360,
            agent_tag="agent",
            view_id="front",
            capture_group_id=group,
            camera_yaw_deg=45.0,
            capture_timing=timing,
        ),
        rear=PixelGoalFrame(
            rgb_data_url="data:image/jpeg;base64,cmVhcg==",
            camera_snapshot_id=rear_snapshot,
            camera_intrinsics_id="perspective-640x360-hfov90",
            width_px=640,
            height_px=360,
            agent_tag="agent",
            view_id="rear",
            capture_group_id=group,
            camera_yaw_deg=225.0,
            capture_timing=timing,
        ),
        capture_timing={
            "capture_read_ms": 2.0,
            "encode_ms": 4.0,
            "wall_ms": 8.0,
        },
    )


def make_pair_client(runtime: RecordingPairRuntime) -> SpearTrackBClient:
    client = SpearTrackBClient(
        runtime,
        agent_tag="agent",
        spawn_pose=Pose(x_cm=1.0, y_cm=2.0, z_cm=90.0, yaw_deg=45.0),
    )
    client.episode(SimpleNamespace(episode_id="ep"))
    return client


def observe_pair(client: SpearTrackBClient, episode_id: str = "ep"):
    return client.observe_views(ObserveViewsRequest(
        episode_id=episode_id, camera=CameraSpec(640, 360, 90.0)))


def rear_walk(pair, **overrides):
    values = {
        "episode_id": "ep",
        "pixel": PixelSpec(0.5, 0.8),
        "camera": CameraSpec(640, 360, 90.0),
        "view": "rear",
        "capture_group_id": pair.capture_group_id,
        "camera_snapshot_id": pair.views[1].camera_snapshot_id,
    }
    values.update(overrides)
    return WalkPixelRequest(**values)


def test_spear_pair_observation_keeps_front_rear_order_data_and_timing():
    runtime = RecordingPairRuntime()

    pair = observe_pair(make_pair_client(runtime))

    assert tuple(view.view for view in pair.views) == ("front", "rear")
    assert tuple(view.png_base64 for view in pair.views) == ("ZnJvbnQ=", "cmVhcg==")
    assert pair.views[0].timing == {
        "capture_read_ms": 1.0,
        "encode_ms": 2.0,
        "wall_ms": 3.0,
    }
    assert pair.timing == {
        "capture_read_ms": 2.0,
        "encode_ms": 4.0,
        "wall_ms": 8.0,
    }
    assert runtime.capture_calls == [{
        "agent_tag": "agent",
        "width_px": 640,
        "height_px": 360,
        "fov_degrees": 90.0,
    }]


def test_spear_client_records_backend_only_pair_and_action_wall_timing(
        monkeypatch):
    """Removing either perf-counter bracket must lose an audit event.

    Timing is deliberately asserted only on the backend boundary.  The
    observation text belongs to ``CourierSession`` and is covered separately;
    no mock policy prose is involved here.
    """
    clock = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(
        courier_backend,
        "time",
        SimpleNamespace(perf_counter=lambda: next(clock)),
        raising=False,
    )
    runtime = RecordingPairRuntime()
    client = make_pair_client(runtime)

    pair = observe_pair(client)
    client.walk_pixel(rear_walk(pair))

    assert client.events == (
        {
            "kind": "view_pair_capture",
            "capture_group_id": "view-pair-4",
            "wall_latency_s": 0.25,
            "ue_timing": {
                "capture_read_ms": 2.0,
                "encode_ms": 4.0,
                "wall_ms": 8.0,
            },
            "views": {
                "front": {
                    "capture_read_ms": 1.0,
                    "encode_ms": 2.0,
                    "wall_ms": 3.0,
                },
                "rear": {
                    "capture_read_ms": 1.0,
                    "encode_ms": 2.0,
                    "wall_ms": 3.0,
                },
            },
        },
        {
            "kind": "pixel_action_execute",
            "capture_group_id": "view-pair-4",
            "view_id": "rear",
            "camera_snapshot_id": "rear-snapshot",
            "pixel_uv": [0.5, 0.8],
            "wall_latency_s": 0.5,
            "outcome": "accepted",
        },
    )

    mutable_copy = client.events[0]
    mutable_copy["kind"] = "tampered"
    assert client.events[0]["kind"] == "view_pair_capture"


def test_deterministic_percentile_uses_linear_interpolation():
    """A shortcut to a nearest-rank percentile would change the p95."""
    values = [4.0, 1.0, 3.0, 2.0]

    assert courier_backend.deterministic_percentile(values, 50.0) == 2.5
    assert courier_backend.deterministic_percentile(values, 95.0) == pytest.approx(3.85)
    assert courier_backend.latency_statistics(values) == {
        "count": 4,
        "median_s": 2.5,
        "p95_s": pytest.approx(3.85),
    }


def test_spear_bound_walk_selects_the_exact_retained_rear_frame():
    runtime = RecordingPairRuntime()
    client = make_pair_client(runtime)
    pair = observe_pair(client)

    response = client.walk_pixel(rear_walk(pair))

    assert runtime.executed_frames[-1] is runtime.pair.rear
    assert runtime.executed_frames[-1].camera_snapshot_id == "rear-snapshot"
    assert (response.view, response.capture_group_id, response.camera_snapshot_id) == (
        "rear", "view-pair-4", "rear-snapshot")


def test_spear_front_snapshot_presented_as_rear_fails_before_execute():
    runtime = RecordingPairRuntime()
    client = make_pair_client(runtime)
    pair = observe_pair(client)

    with pytest.raises(PixelGoalPairIntegrityError):
        client.walk_pixel(rear_walk(
            pair, camera_snapshot_id=pair.views[0].camera_snapshot_id))

    assert runtime.executed_frames == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"episode_id": "different-episode"},
        {"capture_group_id": "view-pair-older"},
    ],
)
def test_spear_bound_walk_validates_episode_and_group_before_execute(overrides):
    runtime = RecordingPairRuntime()
    client = make_pair_client(runtime)
    pair = observe_pair(client)

    with pytest.raises(PixelGoalPairIntegrityError):
        client.walk_pixel(rear_walk(pair, **overrides))

    assert runtime.executed_frames == []


def test_spear_bound_policy_rejection_echoes_validated_capture_identity():
    runtime = RecordingPairRuntime(reject=True)
    client = make_pair_client(runtime)
    pair = observe_pair(client)

    response = client.walk_pixel(rear_walk(pair))

    assert response.resolved.rejection_reason == "no_geometry_hit"
    assert (response.view, response.capture_group_id, response.camera_snapshot_id) == (
        "rear", "view-pair-4", "rear-snapshot")


def test_spear_pair_capture_pose_replaces_nominal_spawn_before_rejection():
    runtime = RecordingPairRuntime(reject=True)
    settled = runtime_pair(pose=(1.0, 2.0, 100.25, 45.0))
    runtime.pairs[0] = settled
    runtime.pair = settled
    client = make_pair_client(runtime)

    pair = observe_pair(client)
    response = client.walk_pixel(rear_walk(pair))

    assert pair.pose.z_cm == 100.25
    assert response.pose.z_cm == 100.25


def test_spear_episode_end_invalidates_the_retained_pair_before_execute():
    runtime = RecordingPairRuntime()
    client = make_pair_client(runtime)
    pair = observe_pair(client)

    client.episode_end(SimpleNamespace(episode_id="ep"))

    with pytest.raises(PixelGoalPairIntegrityError):
        client.walk_pixel(rear_walk(pair))
    assert runtime.executed_frames == []


@pytest.mark.parametrize("transition", ["episode_end", "replacement_episode"])
def test_spear_episode_transition_invalidates_legacy_frame_before_unbound_walk(
        transition):
    runtime = RecordingPairRuntime()
    client = make_pair_client(runtime)
    client.observe(ObserveRequest(
        episode_id="ep", camera=CameraSpec(640, 360, 90.0)))
    if transition == "episode_end":
        client.episode_end(SimpleNamespace(episode_id="ep"))
    else:
        client.episode(SimpleNamespace(episode_id="replacement"))

    with pytest.raises(RuntimeError, match="before any observe"):
        client.walk_pixel(WalkPixelRequest(
            episode_id=("ep" if transition == "episode_end" else "replacement"),
            pixel=PixelSpec(0.5, 0.8),
            camera=CameraSpec(640, 360, 90.0),
        ))

    assert runtime.executed_frames == []


def test_spear_replacement_episode_invalidates_old_pair_then_accepts_new_pair():
    runtime = RecordingPairRuntime()
    runtime.pairs.append(runtime_pair(
        group="view-pair-5",
        front_snapshot="new-front-snapshot",
        rear_snapshot="new-rear-snapshot",
    ))
    client = make_pair_client(runtime)
    old_pair = observe_pair(client)

    client.episode(SimpleNamespace(episode_id="replacement"))

    with pytest.raises(PixelGoalPairIntegrityError):
        client.walk_pixel(rear_walk(old_pair))
    assert runtime.executed_frames == []

    new_pair = observe_pair(client, episode_id="replacement")
    response = client.walk_pixel(rear_walk(
        new_pair, episode_id="replacement"))
    assert runtime.executed_frames == [runtime.pairs[1].rear]
    assert response.camera_snapshot_id == "new-rear-snapshot"


def test_spear_repeated_observe_replaces_pair_without_accepting_old_binding():
    runtime = RecordingPairRuntime()
    runtime.pairs.append(runtime_pair(
        group="view-pair-5",
        front_snapshot="new-front-snapshot",
        rear_snapshot="new-rear-snapshot",
    ))
    client = make_pair_client(runtime)
    old_pair = observe_pair(client)
    new_pair = observe_pair(client)

    with pytest.raises(PixelGoalPairIntegrityError):
        client.walk_pixel(rear_walk(old_pair))
    assert runtime.executed_frames == []

    client.walk_pixel(rear_walk(new_pair))
    assert runtime.executed_frames == [runtime.pairs[1].rear]


class FixedCorridorClient:
    """Only the Track-B lifecycle boundary this reset-level test needs."""

    spawn = Pose(x_cm=-5_934.8349, y_cm=-9_700.0, z_cm=90.0, yaw_deg=90.0)
    events = tuple(
        {"kind": "view_pair_capture", "wall_latency_s": value}
        for value in (0.1, 0.2, 0.3, 0.4)
    ) + tuple(
        {"kind": "pixel_action_execute", "wall_latency_s": value}
        for value in (0.5, 0.7)
    )

    def episode(self, request):
        return EpisodeResponse(
            episode_id=request.episode_id,
            pose=self.spawn,
            fixed_dt=1.0 / 30.0,
        )


def place_pawn_at(env: CorridorDeliveryEnv, point: tuple[float, float]) -> None:
    """Put both halves of the embodied position at one hand-checked point."""
    env.ue_pose = Pose(x_cm=point[0], y_cm=point[1], z_cm=90.0, yaw_deg=90.0)
    env.node_id, _ = env._nearest_node(point)
    env.arrived_from = None


def make_corridor_env(tmp_path: Path, episode_id: str) -> CorridorDeliveryEnv:
    paris = build_road_network(MAPS, map_name="citycore-paris")
    env = CorridorDeliveryEnv(
        paris,
        FixedCorridorClient(),
        episode_id=episode_id,
        cache_root=tmp_path / "album",
        action_space=ACTION_SPACE_PIXEL_GOAL,
        camera_view=CAMERA_VIEW_FORWARD,
        embodiment="human_on_foot",
        difficulty="solo",
        seed=0,
    )
    env.reset()
    return env


@requires_maps
def test_corridor_summary_aggregates_backend_event_latency_deterministically(
        tmp_path):
    """Dropping backend events from the environment must empty the report."""
    env = make_corridor_env(tmp_path, "pixel-goal-backend-latency")

    assert env.summary()["latency"] == {
        "view_pair_capture": {
            "count": 4,
            "median_s": pytest.approx(0.25),
            "p95_s": pytest.approx(0.385),
        },
        "pixel_action_execute": {
            "count": 2,
            "median_s": pytest.approx(0.6),
            "p95_s": pytest.approx(0.69),
        },
    }


@requires_maps
def test_synthetic_target_number_uses_the_same_arrival_truth_as_task_actions(
        tmp_path):
    """Entering each synthetic kerb must both announce and accept the action.

    The production change this catches is dropping the corridor target back
    out of the current-location narration while ``collect``/``hand_over``
    continue judging the pawn against that target's physical kerb.
    """
    env = make_corridor_env(tmp_path, "pixel-goal-arrival-truth")
    order = env.orders[0]

    assert "1" not in env.house_numbers_near(env.node_id).split(", ")

    place_pawn_at(env, order.pickup.kerb)
    assert "1" in env.house_numbers_near(env.node_id).split(", ")
    assert env.collect().ok

    assert "2" not in env.house_numbers_near(env.node_id).split(", ")

    place_pawn_at(env, order.dropoff.kerb)
    assert "2" in env.house_numbers_near(env.node_id).split(", ")
    assert env.hand_over().ok
    assert order.delivered


@requires_maps
def test_corridor_collect_refreshes_phone_route_to_dropoff(tmp_path):
    """The always-visible phone map must follow the order's current target."""
    env = make_corridor_env(tmp_path, "pixel-goal-phone-target-transition")
    order = env.orders[0]

    assert env.screen_target is order.pickup
    pickup_map = env.map_drawing().svg
    assert "to 1 Rue Crémieux" in pickup_map

    place_pawn_at(env, order.pickup.kerb)
    assert env.collect().ok

    assert env.screen_target is order.dropoff
    dropoff_map = env.map_drawing().svg
    assert "to 2 Rue Crémieux" in dropoff_map
    assert "to 1 Rue Crémieux" not in dropoff_map


@requires_maps
def test_corridor_pickup_keeps_the_standard_eight_metre_tolerance(tmp_path):
    """Tightening hand-over must not silently tighten parcel collection."""
    env = make_corridor_env(tmp_path, "pixel-goal-pickup-tolerance")
    order = env.orders[0]
    inside_standard_pickup = (order.pickup.kerb[0] + 799.0,
                              order.pickup.kerb[1])

    place_pawn_at(env, inside_standard_pickup)

    assert env.house_numbers_near(env.node_id) == "1"
    assert env.collect().ok


@requires_maps
def test_corridor_dropoff_requires_arms_reach_plus_a_step(tmp_path):
    """The road-middle 6.78 m hand-over must no longer count as delivery.

    The tolerance is 300 cm (``CORRIDOR_DROPOFF_TOLERANCE_CM``): the anchor
    sits about half a metre before the door, the pawn cannot stand closer
    than roughly 1.5 m to a facade, and nearer than 2.85 m the door is under
    the bottom edge of the picture and cannot be pointed at.
    """
    env = make_corridor_env(tmp_path, "pixel-goal-dropoff-tolerance")
    order = env.orders[0]
    place_pawn_at(env, order.pickup.kerb)
    assert env.collect().ok

    just_outside = (order.dropoff.kerb[0] + 301.0, order.dropoff.kerb[1])
    place_pawn_at(env, just_outside)

    assert env.house_numbers_near(env.node_id) != "2"
    assert "You have arrived" not in env._navigate_impl().message
    refusal = env.hand_over()
    assert not refusal.ok
    assert refusal.code == "not_at_dropoff"
    assert not order.delivered

    on_boundary = (order.dropoff.kerb[0] + 300.0, order.dropoff.kerb[1])
    place_pawn_at(env, on_boundary)

    assert env.house_numbers_near(env.node_id) == "2"
    assert "You have arrived" in env._navigate_impl().message
    assert env.hand_over().ok
    assert order.delivered
