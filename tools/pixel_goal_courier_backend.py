"""Drive real ``CourierEnv`` task logic (orders, ``collect``, ``hand_over``)
over this repo's own working SPEAR pixel-goal RPC surface, instead of the
SimWorld2 ``nav-render/v0`` HTTP service ``EmbodiedCourierEnv`` normally
expects.

Why this file exists: ``EmbodiedCourierEnv.walk_to_pixel`` (the DeliveryBench
Track B integration added for pixel-goal navigation) talks to a
``UERenderClient``-shaped object over HTTP -- five methods, ``episode``,
``observe``, ``walk``, ``walk_pixel``, ``episode_end`` (see
``embodiedbench/runtime/live/client.py``). That HTTP service is owned by the
separate SimWorld2 repository, which does not implement ``/walk_pixel`` yet
(the protocol dataclasses are its contract, not a running server). The
ONLY place pixel-goal navigation actually runs today is this repo's own
SPEAR ``.simworld-ue`` engine, reached through
``embodiedbench.runtime.pixel_goal.LivePixelGoalRuntime`` -- the same RPC
surface the manual PoC scripts (the early closed-loop probes and
friends) already exercise.

``SpearTrackBClient`` below is a duck-typed stand-in for ``UERenderClient``
that implements those same five methods directly against
``LivePixelGoalRuntime``, so ``EmbodiedCourierEnv`` -- and everything above
it (``CourierSession``, the tool dispatch, the prompts) -- runs completely
unmodified. ``CorridorDeliveryEnv`` is the small companion override needed
because the compiled Paris graph's own kerb coordinates sit on the road
centreline, not the sidewalk (measured: ~7.8 m off at the corridor this uses,
more than the 8 m arrival tolerance on an unlucky street -- see the
investigation this module exists to work around), and because one SPEAR
engine session only has ONE navmesh-ready region configured, not "teleport
anywhere on the map" the way a real Track B backend would.

Scope, stated once rather than re-discovered as a bug report: this is a
single-episode, single-corridor proof of concept that one full pickup ->
dropoff delivery can run under real pixel-goal navigation against a real
engine. It is not a general Track B backend -- ``episode()`` always resets to
the one configured ``ParisPocRegion`` regardless of the pose it is asked for,
and ``walk()`` (plain coordinate walk, for the street/coordinate action
spaces) is not implemented, because pixel-goal is the only mode this ever
runs.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import replace
from collections.abc import Iterable, Sequence
from typing import Any

from embodiedbench.compiler.road_network import (
    Address,
    RoadNetwork,
    StreetNode,
    bearing_deg,
)
from embodiedbench.runtime.city.courier_env import (
    HANDLING_SECONDS,
    Order,
    compass_of,
)
from embodiedbench.runtime.city.embodiment import Viewpoint
from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv
from embodiedbench.runtime.live.protocol import (
    PIXEL_VIEW_YAW_OFFSETS,
    PIXEL_VIEWS,
    PIXEL_VIEWS_QUAD,
    EpisodeEndRequest,
    EpisodeRequest,
    EpisodeResponse,
    ObserveRequest,
    ObservedView,
    ObserveViewsRequest,
    ObserveViewsResponse,
    Pose,
    RenderResult,
    ResolvedPixel,
    WalkPixelRequest,
    WalkPixelResponse,
    WalkRequest,
    WalkResponse,
)
from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalConfig,
    PixelGoalFrame,
    PixelGoalPairIntegrityError,
    PixelGoalRejected,
    PixelGoalViewPair,
    project_world_point_to_pixel,
)
from embodiedbench.schemas.runtime import ControllerOutcomeCode, NavPixelGoalAction
from tools.pixel_goal_order_pool import (
    ResolvedDeliveryScenario,
    ValidatedDeliveryPool,
    nearest_pool_node,
    plan_pool_legs,
)
from tools.pixel_goal_capture_preflight import CaptureReadinessGate

logger = logging.getLogger(__name__)


# The pool's certified connectivity is intentionally a one-metre Recast
# lattice.  That is the right resolution for proving where the pawn may walk,
# but the wrong resolution for a navigation banner consumed by an action that
# commonly travels several metres.  These values affect only the phone's
# description of the exact route; they never add an edge, move a route point,
# or relax the crossing/arrival validators.
TRUSTED_PHONE_ROUTE_SIMPLIFICATION_CM = 75.0
TRUSTED_PHONE_GUIDANCE_LOOKAHEAD_CM = 300.0
TRUSTED_PHONE_MANEUVER_PREVIEW_CM = 1_000.0
TRUSTED_PHONE_MIN_MANEUVER_DEG = 30.0
TRUSTED_PHONE_INSTRUCTION_METHOD = (
    "exact_recast_route_actionable_lookahead_maneuver_preview_v2")


def _trusted_phone_route_points(
    route: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Normalize a route without changing its geometry or order."""

    points: list[tuple[float, float]] = []
    for raw in route:
        point = (float(raw[0]), float(raw[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("phone route contains a non-finite point")
        if not points or math.dist(points[-1], point) > 1e-6:
            points.append(point)
    return points


def _point_to_segment_distance_cm(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator))
    projection = (
        start[0] + fraction * dx,
        start[1] + fraction * dy,
    )
    return math.dist(point, projection)


def _simplify_trusted_phone_route(
    route: list[tuple[float, float]],
    *,
    tolerance_cm: float = TRUSTED_PHONE_ROUTE_SIMPLIFICATION_CM,
) -> list[tuple[float, float]]:
    """Find meaningful bends without replacing the certified route.

    Ramer--Douglas--Peucker is used only to decide what the banner previews.
    ``render_map`` still receives and draws every original Recast node, and
    movement remains guarded by UE's live NavMesh.
    """

    points = _trusted_phone_route_points(route)
    if len(points) <= 2:
        return points
    distance, split = max(
        (
            _point_to_segment_distance_cm(point, points[0], points[-1]),
            index,
        )
        for index, point in enumerate(points[1:-1], 1)
    )
    if distance <= tolerance_cm:
        return [points[0], points[-1]]
    first = _simplify_trusted_phone_route(
        points[:split + 1], tolerance_cm=tolerance_cm)
    second = _simplify_trusted_phone_route(
        points[split:], tolerance_cm=tolerance_cm)
    return [*first[:-1], *second]


def _trusted_phone_point_along_route(
    route: list[tuple[float, float]],
    distance_cm: float,
) -> tuple[float, float]:
    """Interpolate a point by arc length on the exact Recast polyline."""

    points = _trusted_phone_route_points(route)
    if not points:
        raise ValueError("phone route is empty")
    remaining_cm = max(0.0, float(distance_cm))
    for start, end in zip(points, points[1:]):
        edge_cm = math.dist(start, end)
        if edge_cm <= 1e-6:
            continue
        if remaining_cm <= edge_cm:
            fraction = remaining_cm / edge_cm
            return (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
        remaining_cm -= edge_cm
    return points[-1]


def trusted_phone_route_instruction(
    route: list[tuple[float, float]],
) -> dict[str, str | float | None]:
    """Describe the current leg and, when close, its next real maneuver.

    The former pooled banner read ``route[1]`` directly.  On a one-metre grid
    it changed after almost every action and, at the audited zebra entrance,
    announced the turn only after a multi-metre pixel action had passed it.
    This function reads the same exact route polyline, removes only sub-metre
    stair-step noise for instruction purposes, and previews a >=30 degree bend
    within one maximum 10 m action.
    """

    points = _trusted_phone_route_points(route)
    simplified = _simplify_trusted_phone_route(points)
    empty: dict[str, str | float | None] = {
        "next_heading": "",
        "next_maneuver_heading": "",
        "next_maneuver_distance_cm": None,
    }
    if len(points) < 2 or len(simplified) < 2:
        return empty
    # The pawn's visual action normally covers roughly three or more metres.
    # Aim the primary banner at a point that far along the exact route instead
    # of exposing the arbitrary first one-metre Recast lattice tangent.  This
    # is an instruction lookahead only: the blue line still contains every
    # node, and UE still resolves and guards the selected photograph pixel.
    guidance_target = _trusted_phone_point_along_route(
        points, TRUSTED_PHONE_GUIDANCE_LOOKAHEAD_CM)
    first_bearing = bearing_deg(points[0], guidance_target)
    instruction: dict[str, str | float | None] = {
        "next_heading": compass_of(first_bearing),
        "next_maneuver_heading": "",
        "next_maneuver_distance_cm": None,
    }
    first_leg_cm = math.dist(simplified[0], simplified[1])
    if len(simplified) < 3 \
            or first_leg_cm > TRUSTED_PHONE_MANEUVER_PREVIEW_CM:
        return instruction
    route_first_bearing = bearing_deg(simplified[0], simplified[1])
    second_bearing = bearing_deg(simplified[1], simplified[2])
    turn_deg = abs(
        (second_bearing - route_first_bearing + 180.0) % 360.0 - 180.0)
    if turn_deg < TRUSTED_PHONE_MIN_MANEUVER_DEG:
        return instruction
    instruction["next_maneuver_heading"] = compass_of(second_bearing)
    instruction["next_maneuver_distance_cm"] = first_leg_cm
    return instruction


def deterministic_percentile(values: Iterable[float], percentile: float) -> float | None:
    """A linearly interpolated percentile with no library/version ambiguity."""

    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise ValueError("percentile must be a number in [0, 100]")
    percentile_f = float(percentile)
    if not math.isfinite(percentile_f) or not 0.0 <= percentile_f <= 100.0:
        raise ValueError("percentile must be a number in [0, 100]")
    samples: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("latency samples must be finite non-negative numbers")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0.0:
            raise ValueError("latency samples must be finite non-negative numbers")
        samples.append(sample)
    if not samples:
        return None
    samples.sort()
    position = (len(samples) - 1) * percentile_f / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    fraction = position - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction


def latency_statistics(values: Iterable[float]) -> dict[str, int | float | None]:
    """Count, conventional median, and p95 from one materialized sample set."""

    samples = list(values)
    return {
        "count": len(samples),
        "median_s": deterministic_percentile(samples, 50.0),
        "p95_s": deterministic_percentile(samples, 95.0),
    }


#: Which engine pair (0 = taken at the pawn's facing, 1 = taken a quarter
#: turn to the right) and which of its two cameras each of the four views is.
QUAD_VIEW_SOURCE = {
    "front": (0, "front"), "rear": (0, "rear"),
    "right": (1, "front"), "left": (1, "rear"),
}
QUAD_PAIR_YAW_OFFSETS = (0.0, 90.0)
#: Seconds to let the engine settle after the pawn is turned in place before
#: a capture; the camera follows the actor within a frame or two.
TURN_SETTLE_S = 0.4
#: Where the pawn's feet are relative to its reported position, for the
#: rare caller without the status RPC (test doubles).
PAWN_FEET_BELOW_POSITION_CM = 88.0
#: A move whose acceptance radius is larger than the map completes where it
#: starts: the engine still resolves the pixel and judges the straight path,
#: which makes it the resolve-only call the subsystem does not otherwise have.
RESOLVE_ONLY_ACCEPTANCE_RADIUS_CM = 100_000.0
#: Where the camera sits relative to the pawn's position when the engine did
#: not say (test doubles): a step ahead and 60 cm up.
CAMERA_AHEAD_CM = 20.0
CAMERA_ABOVE_POSITION_CM = 60.0


#: How far the engine may project a leg's ray hit onto the NavMesh.
LEG_NAVMESH_ADJUSTMENT_CM = 100.0
#: A walk that goes round refused legs re-plans at most this many times.
MAX_LEG_REPLANS = 4


def make_pawn_turner(session: Any, agent_tag: str) -> Any:
    """A callable that turns the pawn in place through SPEAR.

    The pixel-goal subsystem has no turn RPC; ``K2_SetActorRotation`` on the
    tagged pawn does it, inside the frame the SPEAR services expect.
    """

    instance, game = session._instance, session._game

    def turn(yaw_deg: float) -> None:
        with instance.begin_frame():
            pawn = game.unreal_service.find_actor_by_tag(
                agent_tag, "AActor", as_unreal_object=True)
            pawn.K2_SetActorRotation(
                NewRotation={"Pitch": 0.0, "Yaw": float(yaw_deg), "Roll": 0.0},
                bTeleportPhysics=True)
        with instance.end_frame():
            pass

    return turn


def make_pawn_mover(session: Any, agent_tag: str) -> Any:
    """A callable that sets the pawn down at a point through SPEAR.

    The subsystem has no place-the-pawn RPC either; ``K2_TeleportTo`` on
    the tagged pawn does it, inside the frame the SPEAR services expect.
    The harness uses it for one thing: putting the pawn back onto the
    certified node it just walked to, when the engine's controller stopped
    it a few tens of centimetres short or aside (``nudge_world``).
    """

    instance, game = session._instance, session._game

    def move(x_cm: float, y_cm: float, z_cm: float, yaw_deg: float) -> None:
        with instance.begin_frame():
            pawn = game.unreal_service.find_actor_by_tag(
                agent_tag, "AActor", as_unreal_object=True)
            pawn.K2_TeleportTo(
                DestLocation={"X": float(x_cm), "Y": float(y_cm), "Z": float(z_cm)},
                DestRotation={"Pitch": 0.0, "Yaw": float(yaw_deg), "Roll": 0.0})
        with instance.end_frame():
            pass

    return move


class WorldLegRefused(RuntimeError):
    """The engine would not walk one certified leg; ``reason`` is its verdict."""

    def __init__(self, reason: str, *, audit: dict[str, Any] | None = None) -> None:
        self.reason = reason
        self.audit = dict(audit or {})
        super().__init__(reason)


class SpearTrackBClient:
    """``UERenderClient``'s five Track B methods, backed by SPEAR directly.

    Holds the pawn's pose itself (``LivePixelGoalRuntime`` has no session
    concept of "current pose" -- each capture/move call is self-contained),
    updating it from every ``walk_pixel`` result so ``EmbodiedCourierEnv``'s
    ``ue_pose``/``_here_cm`` stay accurate exactly as they would against a
    real nav-render/v0 service.
    """

    def __init__(
        self,
        runtime: LivePixelGoalRuntime,
        *,
        agent_tag: str,
        spawn_pose: Pose,
        fixed_dt: float = 1.0 / 30.0,
        capture_gate: CaptureReadinessGate | None = None,
        turner: Any = None,
        views: tuple[str, ...] = PIXEL_VIEWS,
        turn_settle_s: float = TURN_SETTLE_S,
        mover: Any | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.runtime = runtime
        self.agent_tag = agent_tag
        self._pose = spawn_pose
        self._fixed_dt = float(fixed_dt)
        self._capture_gate = capture_gate
        self._needs_capture_settle = capture_gate is not None
        # ``turner(yaw_deg)`` turns the pawn in place. The engine has no such
        # RPC; the runner supplies one through SPEAR. With it the observation
        # can be four views (two pairs, a quarter turn apart) and a certified
        # leg can be walked by facing its end and naming the pixel it projects
        # to.
        views = tuple(views)
        if views not in (PIXEL_VIEWS, PIXEL_VIEWS_QUAD):
            raise ValueError(f"views must be {PIXEL_VIEWS} or {PIXEL_VIEWS_QUAD}")
        if views == PIXEL_VIEWS_QUAD and turner is None:
            raise ValueError("four views need a turner that sets the pawn's yaw")
        self._turner = turner
        self._mover = mover
        self._views = views
        self._turn_settle_s = float(turn_settle_s)
        self._sleep = sleep_fn
        self._view_pairs: dict[int, PixelGoalViewPair] = {}
        self._pair_yaws: dict[int, float] = {}
        self._engine_latest_pair: int | None = None
        # The frame ``walk_pixel`` resolves a pixel against -- must be the
        # most recent ``observe()``, per ``WalkPixelRequest``'s own contract
        # that nothing ticks between the two. Same invariant, enforced here
        # instead of on the wire.
        self._last_frame: PixelGoalFrame | None = None
        self._last_view_pair: PixelGoalViewPair | None = None
        self._last_view_pair_episode_id: str | None = None
        self._active_episode_id: str | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Backend wall/UE timing evidence, copied so callers cannot mutate it."""

        return tuple(copy.deepcopy(self._events))

    @property
    def capture_readiness_report(self) -> dict[str, Any] | None:
        return (
            self._capture_gate.report()
            if self._capture_gate is not None else None
        )

    def episode(self, request: EpisodeRequest) -> EpisodeResponse:
        # The requested spawn pose is IGNORED on purpose -- see module
        # docstring. This engine session has exactly one navmesh-ready
        # region, configured once before this client exists
        # (PixelGoal_SetupParisPocJson + PixelGoal_ResetParisPocTrialJson,
        # done by the runner script), and ``spawn_pose`` passed to __init__
        # already records where that reset actually left the agent.
        self._last_frame = None
        self._last_view_pair = None
        self._last_view_pair_episode_id = None
        self._view_pairs = {}
        self._pair_yaws = {}
        self._engine_latest_pair = None
        self._active_episode_id = request.episode_id
        self._events = []
        self._needs_capture_settle = self._capture_gate is not None
        return EpisodeResponse(
            episode_id=request.episode_id, pose=self._pose, fixed_dt=self._fixed_dt)

    def observe(self, request: ObserveRequest) -> RenderResult:
        # request.yaw_deg is not honoured: there is no "turn to face, then
        # capture" RPC on this surface, only "capture whatever the agent is
        # currently facing" -- which is exactly the pawn's own heading after
        # its last walk_pixel, the same thing request.yaw_deg would compute
        # from (see EmbodiedCourierEnv.facing() / photo_rows()). Nothing
        # turns the agent between a walk and its next observe in this mode,
        # so the two never disagree in practice.
        frame = self.runtime.capture_frame(
            agent_tag=self.agent_tag,
            width_px=request.camera.width, height_px=request.camera.height,
        )
        self._last_frame = frame
        _, _, b64 = frame.rgb_data_url.partition(",")
        # The bytes may be JPEG (PixelGoalConfig.jpeg_quality), written to a
        # ``.png``-named album path by LiveAlbum.store -- cosmetic only:
        # every downstream reader (PIL, this repo's data-URL builders) sniffs
        # image format from content, never from the extension.
        return RenderResult(
            key="", status="ok", png_base64=b64,
            width=frame.width_px, height=frame.height_px, pose=self._pose)

    def _turn(self, yaw_deg: float) -> None:
        if self._turner is None:
            raise RuntimeError("this client cannot turn the pawn")
        # UE rotators are [-180, 180); a yaw handed over as 312 came back as
        # 132 from the engine, so normalise before it leaves this process.
        self._turner((float(yaw_deg) + 180.0) % 360.0 - 180.0)
        if self._turn_settle_s > 0:
            self._sleep(self._turn_settle_s)

    def _capture(self, request: ObserveViewsRequest, *, gated: bool) -> PixelGoalViewPair:
        if gated and self._capture_gate is not None and self._needs_capture_settle:
            pair = self._capture_gate.capture(
                self.runtime,
                agent_tag=self.agent_tag,
                width_px=request.camera.width,
                height_px=request.camera.height,
                fov_degrees=request.camera.fov_deg,
            )
            self._needs_capture_settle = False
            return pair
        return self.runtime.capture_view_pair(
            agent_tag=self.agent_tag,
            width_px=request.camera.width,
            height_px=request.camera.height,
            fov_degrees=request.camera.fov_deg,
        )

    def observe_views(self, request: ObserveViewsRequest) -> ObserveViewsResponse:
        if request.episode_id != self._active_episode_id:
            raise PixelGoalPairIntegrityError("view_pair_episode_mismatch")
        wall_started = time.perf_counter()
        gate_applied = bool(
            self._capture_gate is not None and self._needs_capture_settle)
        pair = self._capture(request, gated=True)
        pairs = {0: pair}
        yaw = float(pair.pose[3])
        quad = self._views == PIXEL_VIEWS_QUAD
        if quad:
            # The second pair, a quarter turn to the right: its front camera
            # is the courier's right, its rear camera the left. The pawn is
            # turned back so its facing -- what "front" means, what the phone
            # narrates against -- is the one the first pair was taken at.
            self._turn(yaw + QUAD_PAIR_YAW_OFFSETS[1])
            pairs[1] = self._capture(request, gated=False)
            self._turn(yaw)
        self._engine_latest_pair = 1 if quad else 0
        self._view_pairs = pairs
        self._pair_yaws = {index: yaw + QUAD_PAIR_YAW_OFFSETS[index] for index in pairs}
        wall_elapsed = time.perf_counter() - wall_started
        event: dict[str, Any] = {
            "kind": "view_pair_capture",
            "capture_group_id": pair.capture_group_id,
            "wall_latency_s": wall_elapsed,
            "ue_timing": copy.deepcopy(pair.capture_timing),
            "views": {
                "front": copy.deepcopy(pair.front.capture_timing),
                "rear": copy.deepcopy(pair.rear.capture_timing),
            },
        }
        if quad:
            event["capture_group_ids"] = [
                pairs[index].capture_group_id for index in sorted(pairs)]
            event["views"].update({
                "right": copy.deepcopy(pairs[1].front.capture_timing),
                "left": copy.deepcopy(pairs[1].rear.capture_timing),
            })
            event["gate_applied"] = gate_applied
        self._events.append(event)
        self._last_view_pair = pair
        self._last_view_pair_episode_id = request.episode_id
        pose = Pose(
            x_cm=pair.pose[0],
            y_cm=pair.pose[1],
            z_cm=pair.pose[2],
            yaw_deg=pair.pose[3],
        )
        # The setup request supplies a nominal spawn Z, while UE settles the
        # capsule at its actual grounded actor Z.  A rejected pixel action has
        # no controller result from which to refresh pose, so it returns this
        # cached value.  Keep the cache synchronized with every atomic capture
        # pair; otherwise the first refusal appears to teleport vertically back
        # to the nominal spawn even though UE never moved the pawn.
        self._pose = pose
        views = []
        for name in self._views:
            index, engine_view = QUAD_VIEW_SOURCE[name]
            frame = pairs[index].frame(engine_view)
            views.append(ObservedView(
                view=name,
                yaw_offset_deg=PIXEL_VIEW_YAW_OFFSETS[name],
                camera_snapshot_id=frame.camera_snapshot_id,
                camera_intrinsics_id=frame.camera_intrinsics_id,
                status="ok",
                png_base64=frame.rgb_data_url.partition(",")[2],
                width=frame.width_px,
                height=frame.height_px,
                timing=(
                    dict(frame.capture_timing)
                    if frame.capture_timing is not None
                    else None
                ),
                capture_group_id=(pairs[index].capture_group_id if quad else None),
            ))
        return ObserveViewsResponse(
            capture_group_id=pair.capture_group_id,
            pose=pose,
            views=tuple(views),
            timing=dict(pair.capture_timing),
        )

    def walk(self, request: WalkRequest) -> WalkResponse:
        raise NotImplementedError(
            "SpearTrackBClient only serves action_space='pixel_goal'; a "
            "plain coordinate /walk has no SPEAR RPC behind it here. A "
            "certified leg is walked with walk_world().")

    def _frame_for(
        self, request: WalkPixelRequest,
    ) -> tuple[PixelGoalFrame, dict[str, str]]:
        """The engine frame a bound pixel resolves against, re-captured if the
        engine's snapshots have moved on to a later pair since."""
        request_binding = (
            request.view, request.capture_group_id, request.camera_snapshot_id)
        if not all(isinstance(value, str) and value for value in request_binding):
            raise PixelGoalPairIntegrityError("bound_walk_missing_capture_identity")
        if request.episode_id != self._last_view_pair_episode_id:
            raise PixelGoalPairIntegrityError("view_pair_episode_mismatch")
        source = QUAD_VIEW_SOURCE.get(request.view)
        if source is None or request.view not in self._views:
            raise PixelGoalPairIntegrityError("camera_snapshot_view_mismatch")
        index, engine_view = source
        pair = self._view_pairs.get(index)
        if pair is None:
            raise PixelGoalPairIntegrityError("view_pair_not_observed")
        if request.capture_group_id != pair.capture_group_id:
            raise PixelGoalPairIntegrityError("view_pair_group_mismatch")
        frame = pair.frame(engine_view)
        if request.camera_snapshot_id != frame.camera_snapshot_id:
            raise PixelGoalPairIntegrityError("view_pair_snapshot_mismatch")
        binding = {
            "view": request.view,
            "capture_group_id": request.capture_group_id,
            "camera_snapshot_id": request.camera_snapshot_id,
        }
        # The engine keeps every snapshot it committed and resolves a pixel
        # against the one named, provided the pawn stands as it did when that
        # snapshot was taken (measured live: a first-pair pixel resolved after
        # the second pair had been taken and the pawn turned back; a
        # second-pair pixel was "stale" until the pawn was turned to that
        # pair's yaw again). So: turn to the pair's yaw, never re-capture.
        if self._turner is not None and index in self._pair_yaws:
            self._turn(self._pair_yaws[index])
        return frame, binding

    @staticmethod
    def _resolved_from_rejection(rejected: PixelGoalRejected) -> ResolvedPixel:
        raw = rejected.audit.get("raw_world_hit_cm")
        raw_z = (float(raw[2]) if isinstance(raw, (list, tuple)) and len(raw) == 3
                 else None)
        raw_hit = (
            (rejected.raw_world_hit.x_cm, rejected.raw_world_hit.y_cm)
            if rejected.raw_world_hit is not None else None)
        actor = rejected.audit.get("raw_hit_actor")
        target = rejected.audit.get("validated_navigation_target_cm")
        length = rejected.audit.get("controller_path_length_cm")
        return ResolvedPixel(
            raw_world_hit_cm=raw_hit,
            accepted_target_cm=(
                (float(target[0]), float(target[1]))
                if isinstance(target, (list, tuple)) and len(target) >= 2 else None),
            rejection_reason=rejected.reason,
            controller_path_length_cm=(
                float(length) if isinstance(length, (int, float)) else None),
            raw_hit_actor=actor if isinstance(actor, str) and actor else None,
            raw_world_hit_z_cm=raw_z,
            direct_path_legal=(
                False if rejected.reason == "controller_path_enters_unmarked_road"
                else None),
        )

    @staticmethod
    def _resolved_from_started(started: Any, *, legal: bool) -> ResolvedPixel:
        raw_hit_v = started.request.raw_world_hit
        accepted_v = started.request.projected_target
        # Older protocol-compatible test doubles and remote runtimes predate
        # the controller-path diagnostics.  Keep those reporting fields
        # optional; movement acceptance is still determined by the runtime
        # result, never by a fabricated path.
        controller_path_points = getattr(
            started.request, "controller_path_points", None)
        return ResolvedPixel(
            raw_world_hit_cm=((raw_hit_v.x_cm, raw_hit_v.y_cm)
                              if raw_hit_v is not None else None),
            accepted_target_cm=((accepted_v.x_cm, accepted_v.y_cm)
                                if accepted_v is not None else None),
            navmesh_adjustment_cm=started.request.navmesh_adjustment_cm,
            controller_path_points_cm=(
                tuple(
                    (point.x_cm, point.y_cm, point.z_cm)
                    for point in controller_path_points)
                if controller_path_points else None),
            controller_path_length_cm=getattr(
                started.request, "controller_path_length_cm", None),
            controller_path_direct_cm=getattr(
                started.request, "controller_path_direct_cm", None),
            controller_path_stretch_ratio=(
                getattr(started.request, "controller_path_stretch_ratio", None)),
            raw_hit_actor=getattr(started, "raw_hit_actor", None),
            raw_world_hit_z_cm=(
                getattr(raw_hit_v, "z_cm", None) if raw_hit_v is not None else None),
            direct_path_legal=legal,
        )

    def walk_pixel(self, request: WalkPixelRequest) -> WalkPixelResponse:
        pose_before = self._pose
        request_binding = (
            request.view,
            request.capture_group_id,
            request.camera_snapshot_id,
        )
        is_bound = any(value is not None for value in request_binding)
        response_binding: dict[str, str] = {}
        if is_bound and request.episode_id != self._active_episode_id:
            raise PixelGoalPairIntegrityError("view_pair_episode_mismatch")
        if self._view_pairs:
            frame, response_binding = self._frame_for(request)
        else:
            if is_bound:
                raise PixelGoalPairIntegrityError("view_pair_not_observed")
            if self._last_frame is None:
                raise RuntimeError(
                    "walk_pixel called before any observe(); SPEAR resolves a "
                    "pixel against the most recently captured frame's camera "
                    "snapshot, and there is not one yet.")
            frame = self._last_frame
        action = NavPixelGoalAction(
            target={"u_norm": request.pixel.u, "v_norm": request.pixel.v})
        execute_started = time.perf_counter()
        if request.resolve_only:
            return self._resolve_only(
                action, frame, request, response_binding, execute_started)
        try:
            started, result = self.runtime.execute(action, frame)
        except PixelGoalRejected as rejected:
            self._record_action_event(
                request, frame,
                wall_latency_s=time.perf_counter() - execute_started,
                outcome="resolution_rejected",
            )
            return WalkPixelResponse(
                arrived=False, stuck=False, timeout=False, ticks=0,
                sim_seconds=0.0, pose=self._pose, walked_cm=0.0,
                resolved=self._resolved_from_rejection(rejected),
                **response_binding,
            )
        except Exception:
            self._record_action_event(
                request, frame,
                wall_latency_s=time.perf_counter() - execute_started,
                outcome="error",
            )
            raise
        self._record_action_event(
            request, frame,
            wall_latency_s=time.perf_counter() - execute_started,
            outcome=result.outcome.value,
        )
        self._engine_latest_pair = None
        resolved = self._resolved_from_started(started, legal=True)
        accepted = result.outcome is ControllerOutcomeCode.ACCEPTED
        timed_out = result.outcome is ControllerOutcomeCode.EXECUTION_TIMEOUT
        self._absorb_final_pose(result.final_pose, pose_before)
        return WalkPixelResponse(
            arrived=accepted, stuck=not accepted and not timed_out,
            timeout=timed_out,
            # No lockstep tick count on this RPC surface; 1 for a walk that
            # actually ran the controller, 0 for one that was rejected before
            # ever moving. Only used for reporting totals, never for logic.
            ticks=0 if result.final_pose is None else 1,
            sim_seconds=result.elapsed_sim_s, pose=self._pose,
            walked_cm=result.distance_travelled_cm, resolved=resolved,
            **response_binding,
        )

    def _absorb_final_pose(self, final_pose: Any, pose_before: Pose) -> None:
        if final_pose is None:
            return
        next_pose = Pose(
            x_cm=final_pose.position.x_cm, y_cm=final_pose.position.y_cm,
            z_cm=final_pose.position.z_cm, yaw_deg=final_pose.yaw_deg)
        yaw_delta = abs(
            (next_pose.yaw_deg - pose_before.yaw_deg + 180.0) % 360.0
            - 180.0)
        pose_changed = bool(
            math.dist(
                (next_pose.x_cm, next_pose.y_cm, next_pose.z_cm),
                (pose_before.x_cm, pose_before.y_cm, pose_before.z_cm),
            ) > 0.1
            or yaw_delta > 0.01
        )
        self._pose = next_pose
        if pose_changed and self._capture_gate is not None:
            self._needs_capture_settle = True

    def _resolve_only(
        self,
        action: NavPixelGoalAction,
        frame: PixelGoalFrame,
        request: WalkPixelRequest,
        response_binding: dict[str, str],
        execute_started: float,
    ) -> WalkPixelResponse:
        """Resolve the pixel and stop: the engine's ray hit, NavMesh
        projection and straight-path verdict, without the walk.

        The engine has no resolve-only call. A move with an acceptance radius
        wider than the map is one: the engine resolves and judges the pixel
        exactly as for a walk, then finds the pawn already within the radius
        of its target and completes without moving (measured live: zero
        distance, zero seconds). A runtime that cannot be re-configured
        (a test double) falls back to starting the move and cancelling it at
        once. Either way the pawn is turned back to the observation's facing.
        """
        pose_before = self._pose
        resolver = self._resolver()
        try:
            if resolver is not None:
                started, cancelled = resolver.execute(action, frame)
            else:
                started = self.runtime.start(action, frame)
                cancelled = self.runtime.cancel(
                    started.request.request_id, reason="resolve_only")
        except PixelGoalRejected as rejected:
            self._record_action_event(
                request, frame,
                wall_latency_s=time.perf_counter() - execute_started,
                outcome="resolution_rejected",
            )
            return WalkPixelResponse(
                arrived=False, stuck=False, timeout=False, ticks=0,
                sim_seconds=0.0, pose=self._pose, walked_cm=0.0,
                resolved=self._resolved_from_rejection(rejected),
                **response_binding,
            )
        self._record_action_event(
            request, frame,
            wall_latency_s=time.perf_counter() - execute_started,
            outcome="resolved_only",
        )
        self._engine_latest_pair = None
        # The pawn did not walk, so the pose the observation was taken from
        # is still its pose; the engine's completed-on-the-spot result is not
        # absorbed (it reports a zero yaw for a move that never started).
        if self._turner is not None and self._pair_yaws:
            self._turn(self._pair_yaws[0])
        self._pose = pose_before
        return WalkPixelResponse(
            arrived=False, stuck=False, timeout=False, ticks=0,
            sim_seconds=float(cancelled.elapsed_sim_s or 0.0), pose=self._pose,
            walked_cm=float(cancelled.distance_travelled_cm or 0.0),
            resolved=self._resolved_from_started(started, legal=True),
            **response_binding,
        )

    @property
    def can_walk_world(self) -> bool:
        """Whether certified legs can be walked: the pawn can be turned."""
        return self._turner is not None

    def _resolver(self) -> LivePixelGoalRuntime | None:
        """The same engine, asked with an acceptance radius wider than the
        map; ``None`` for a runtime without a config to widen."""
        endpoint = getattr(self.runtime, "endpoint", None)
        config = getattr(self.runtime, "config", None)
        if endpoint is None or not isinstance(config, PixelGoalConfig):
            return None
        cached = getattr(self, "_resolver_runtime", None)
        if cached is None:
            cached = LivePixelGoalRuntime(
                endpoint, replace(
                    config, acceptance_radius_cm=RESOLVE_ONLY_ACCEPTANCE_RADIUS_CM))
            self._resolver_runtime = cached
        return cached

    def feet_cm(self) -> tuple[float, float, float]:
        """Where the pawn's feet are, from the engine when it will say."""
        endpoint = getattr(self.runtime, "endpoint", None)
        if endpoint is not None and hasattr(endpoint, "call"):
            try:
                status = endpoint.call("PixelGoal_GetParisPocStatusJson", {})
                feet = status.get("agent_feet_position_cm")
                if isinstance(feet, (list, tuple)) and len(feet) == 3:
                    return (float(feet[0]), float(feet[1]), float(feet[2]))
            except Exception:  # noqa: BLE001 - the pose below is the fallback
                pass
        return (self._pose.x_cm, self._pose.y_cm,
                self._pose.z_cm - PAWN_FEET_BELOW_POSITION_CM)

    @property
    def can_nudge(self) -> bool:
        """Whether the pawn can be set back onto a certified node."""
        return self._mover is not None

    def nudge_world(self, x_cm: float, y_cm: float) -> Pose:
        """Set the pawn down on a point a few tens of centimetres away --
        the certified node it just walked to -- keeping its facing and its
        height. The engine's controller stops 5-30 cm from the point it was
        given, and its road check reads the path from where the pawn really
        stands, so a leg that is legal from the node can be refused from
        beside it; the harness certifies its ways from the nodes, and this
        keeps the pawn on them. Recorded as a ``world_nudge`` event."""
        if self._mover is None:
            raise RuntimeError("setting the pawn on a point needs a mover")
        feet = self.feet_cm()
        yaw = float(self._pose.yaw_deg)
        self._mover(x_cm, y_cm, feet[2] + PAWN_FEET_BELOW_POSITION_CM, yaw)
        if self._turn_settle_s > 0:
            self._sleep(self._turn_settle_s)
        endpoint = getattr(self.runtime, "endpoint", None)
        if endpoint is not None and hasattr(endpoint, "call"):
            landed = self.feet_cm()
        else:
            landed = (x_cm, y_cm, feet[2])
        self._pose = Pose(x_cm=landed[0], y_cm=landed[1],
                          z_cm=landed[2] + PAWN_FEET_BELOW_POSITION_CM, yaw_deg=yaw)
        self._engine_latest_pair = None
        if self._capture_gate is not None:
            self._needs_capture_settle = True
        self._events.append({
            "kind": "world_nudge", "from_cm": [feet[0], feet[1]], "to_cm": [x_cm, y_cm],
            "landed_cm": [landed[0], landed[1]],
            "distance_cm": round(math.dist((feet[0], feet[1]), (x_cm, y_cm)), 2),
            "residual_cm": round(math.dist((landed[0], landed[1]), (x_cm, y_cm)), 2)})
        return self._pose

    def walk_world(
        self, x_cm: float, y_cm: float, *, camera: Any,
        stop_cm: tuple[float, float] | None = None,
    ) -> WalkResponse:
        """Walk a certified leg the harness chose: face its end, capture,
        name the pixel the end projects to, and let the engine walk that
        pixel -- all the way, or only as far as ``stop_cm``, a point on the
        leg (the acceptance radius stops the pawn there).

        The end has to be far enough to be inside the picture (the bottom
        edge is about 2.9 m away at this camera height); certified legs are.
        Raises ``WorldLegRefused`` with the engine's verdict when the leg is
        not walkable as a straight line.
        """
        if self._turner is None:
            raise RuntimeError("walking to a world point needs a turner")
        pose_before = self._pose
        feet = self.feet_cm()
        bearing = math.degrees(math.atan2(y_cm - feet[1], x_cm - feet[0]))
        acceptance_radius_cm: float | None = None
        aim = (x_cm, y_cm)
        base_radius = float(getattr(
            getattr(self.runtime, "config", None), "acceptance_radius_cm", 15.0))
        if stop_cm is not None:
            # Stop short of the leg's end: the controller stops when it is
            # within the acceptance radius of the point it walks to. That
            # point is the engine's projection of the aim, up to a metre
            # nearer than the aim itself, so the radius is taken from the
            # projection (asked for below, once the pixel is known) and not
            # from the aim: measured from the aim, a 1 m move whose aim
            # projected 90 cm short was "arrived" before the pawn moved.
            acceptance_radius_cm = math.dist(stop_cm, aim) + base_radius
        self._turn(bearing)
        pair = self.runtime.capture_view_pair(
            agent_tag=self.agent_tag, width_px=camera.width,
            height_px=camera.height, fov_degrees=camera.fov_deg)
        frame = pair.front
        cam_yaw = (frame.camera_yaw_deg if frame.camera_yaw_deg is not None
                   else pair.pose[3])
        cam_location = frame.camera_location_cm
        if cam_location is None:
            yaw_rad = math.radians(cam_yaw)
            cam_location = (
                pair.pose[0] + CAMERA_AHEAD_CM * math.cos(yaw_rad),
                pair.pose[1] + CAMERA_AHEAD_CM * math.sin(yaw_rad),
                pair.pose[2] + CAMERA_ABOVE_POSITION_CM)
        uv = project_world_point_to_pixel(
            cam_location, cam_yaw, (aim[0], aim[1], feet[2]),
            width_px=frame.width_px, height_px=frame.height_px,
            hfov_deg=camera.fov_deg)
        if uv is None or not (0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0):
            self._engine_latest_pair = None
            raise WorldLegRefused(
                "leg_outside_picture",
                audit={"uv": uv, "target_cm": [x_cm, y_cm], "feet_cm": list(feet)})
        runtime = self.runtime
        endpoint = getattr(self.runtime, "endpoint", None)
        config = getattr(self.runtime, "config", None)
        action = NavPixelGoalAction(target={"u_norm": uv[0], "v_norm": uv[1]})
        started_at = time.perf_counter()

        def refused(reason: str, audit: dict[str, Any], projected: list[float] | None = None) -> None:
            hit = audit.get("raw_world_hit_cm")
            self._events.append({
                "kind": "world_leg", "outcome": "refused",
                "reason": reason, "target_cm": [x_cm, y_cm],
                "stop_cm": None if stop_cm is None else [stop_cm[0], stop_cm[1]],
                "pixel_uv": [uv[0], uv[1]],
                "from_cm": [feet[0], feet[1]], "bearing_deg": bearing,
                "camera_yaw_deg": cam_yaw, "aim_cm": [aim[0], aim[1]],
                "projected_cm": projected,
                "raw_hit_actor": audit.get("raw_hit_actor"),
                "raw_world_hit_cm": list(hit) if isinstance(hit, (list, tuple)) else None,
                "wall_latency_s": time.perf_counter() - started_at})
            self._engine_latest_pair = None

        projected_cm: list[float] | None = None
        if endpoint is not None and isinstance(config, PixelGoalConfig):
            # A leg's end is a certified node; the ray may land on the kerb
            # chamfer beside it, so let the engine project further than it
            # would for a policy's pixel.
            adjustment = max(config.max_navmesh_adjustment_cm, LEG_NAVMESH_ADJUSTMENT_CM)
            if stop_cm is not None:
                # Ask where the engine will walk to before asking it to walk:
                # the same resolve with a radius wider than the map judges
                # the leg and projects the aim without moving.
                probe = LivePixelGoalRuntime(endpoint, replace(
                    config, acceptance_radius_cm=RESOLVE_ONLY_ACCEPTANCE_RADIUS_CM,
                    max_navmesh_adjustment_cm=adjustment))
                try:
                    probed, _ = probe.execute(action, frame)
                except PixelGoalRejected as rejected:
                    refused(rejected.reason, rejected.audit)
                    raise WorldLegRefused(rejected.reason, audit=rejected.audit) from rejected
                target = probed.request.projected_target
                projected_cm = [float(target.x_cm), float(target.y_cm)]
                acceptance_radius_cm = math.dist(stop_cm, projected_cm) + base_radius
                if math.dist(feet[:2], projected_cm) <= acceptance_radius_cm:
                    # the engine walks to a point at or before the stop: this
                    # leg cannot stop there, and "arrived" would mean nothing
                    audit = {"projected_cm": projected_cm, "stop_cm": list(stop_cm),
                             "acceptance_radius_cm": acceptance_radius_cm}
                    refused("leg_stop_beyond_projection", audit, projected_cm)
                    raise WorldLegRefused("leg_stop_beyond_projection", audit=audit)
            runtime = LivePixelGoalRuntime(
                endpoint, replace(
                    config,
                    acceptance_radius_cm=(
                        config.acceptance_radius_cm if acceptance_radius_cm is None
                        else acceptance_radius_cm),
                    max_navmesh_adjustment_cm=adjustment))
        try:
            started, result = runtime.execute(action, frame)
        except PixelGoalRejected as rejected:
            refused(rejected.reason, rejected.audit, projected_cm)
            raise WorldLegRefused(rejected.reason, audit=rejected.audit) from rejected
        self._engine_latest_pair = None
        if projected_cm is None:
            target = getattr(started.request, "projected_target", None)
            if target is not None:
                projected_cm = [float(target.x_cm), float(target.y_cm)]
        self._events.append({
            "kind": "world_leg", "outcome": result.outcome.value,
            "target_cm": [x_cm, y_cm],
            "stop_cm": None if stop_cm is None else [stop_cm[0], stop_cm[1]],
            "pixel_uv": [uv[0], uv[1]],
            "from_cm": [feet[0], feet[1]], "bearing_deg": bearing,
            "camera_yaw_deg": cam_yaw, "aim_cm": [aim[0], aim[1]],
            "projected_cm": projected_cm,
            "acceptance_radius_cm": acceptance_radius_cm,
            "raw_hit_actor": getattr(started, "raw_hit_actor", None),
            "path_length_cm": getattr(started.request, "controller_path_length_cm", None),
            "wall_latency_s": time.perf_counter() - started_at})
        self._absorb_final_pose(result.final_pose, pose_before)
        accepted = result.outcome is ControllerOutcomeCode.ACCEPTED
        timed_out = result.outcome is ControllerOutcomeCode.EXECUTION_TIMEOUT
        return WalkResponse(
            arrived=accepted, stuck=not accepted and not timed_out,
            timeout=timed_out, ticks=0 if result.final_pose is None else 1,
            sim_seconds=float(result.elapsed_sim_s or 0.0), pose=self._pose,
            walked_cm=float(result.distance_travelled_cm or 0.0),
            start_pose=pose_before)

    def _record_action_event(
        self,
        request: WalkPixelRequest,
        frame: PixelGoalFrame,
        *,
        wall_latency_s: float,
        outcome: str,
    ) -> None:
        self._events.append({
            "kind": "pixel_action_execute",
            "capture_group_id": frame.capture_group_id,
            "view_id": frame.view_id,
            "camera_snapshot_id": frame.camera_snapshot_id,
            "pixel_uv": [request.pixel.u, request.pixel.v],
            "wall_latency_s": wall_latency_s,
            "outcome": outcome,
        })

    def episode_end(self, request: EpisodeEndRequest) -> bool:
        self._last_frame = None
        self._last_view_pair = None
        self._last_view_pair_episode_id = None
        self._active_episode_id = None
        self._needs_capture_settle = self._capture_gate is not None
        return True


# Where a synthetic delivery address sits relative to the validated corridor
# spawn, along its own sidewalk line (metres north of spawn -- the corridor
# runs north-south at a fixed sidewalk X). Kept apart so a caller can size a
# longer or shorter demo leg without touching the class.
DEFAULT_PICKUP_OFFSET_CM = 1_000.0
DEFAULT_DROPOFF_OFFSET_CM = 3_500.0
# A hand-over is a close-range interaction, unlike the stock graph benchmark's
# broad eight-metre notion of reaching an address. Keep the tighter rule local
# to this physical corridor PoC; pickup and every other CourierEnv stay stock.
# Arm's reach plus a step. The certified anchor stands about half a metre
# in front of the door; the pawn's capsule and the NavMesh setback keep it
# roughly 1.5 m from the facade, so at 100 cm the action succeeded only on
# the one NavMesh edge point directly in front of the anchor -- a relay
# policy that reached the door three times stopped 100, 106 and 114 cm
# from it and was refused each time (2026-09-12). 175 cm leaves that
# physical approach a quarter to three quarters of a metre of slack while
# still refusing a hand-over from across the pavement.
# ... and 300 cm since the four-view harness: the picture's bottom edge is
# about 2.85 m from the camera, so a courier standing nearer the door than
# that cannot point at it any more, while the tolerance said it had not
# arrived. The oracle walk of seed 9 stopped exactly there, 2.7 m from the
# door node with nothing left to aim at. Three metres closes that gap: a
# door a courier can no longer see the foot of is a door it has reached.
CORRIDOR_DROPOFF_TOLERANCE_CM = 300.0
VALIDATED_POOL_PICKUP_TOLERANCE_CM = 300.0

# A second, deliberately longer route for evaluating whether a vision policy
# can follow the phone's drawn line through a real corner. V1/V2 fixed the
# trigger point, but V3 made a more fundamental mistake: it copied three
# endpoints from a controller run that had silently routed around BP_Fence2.
# The phone consequently drew a north-west chord through ordinary roadway even
# though the real zebra crossing is north-east of the corner.
#
# V4 keeps the stock CourierEnv shortest-path algorithm and phone UI, but feeds
# them pedestrian geometry from the authoritative CityCore export. The crossing
# entry/centre/exit below are the AABB centreline of PR_Crossswalk_17. The bend
# then stays on the building side of the PR_SidewalkIsland/EdgeSeparation actors
# (x > -4979 cm) until the already validated door-side pavement.
#
#   <CityCore_Paris>/Exports/DeliveryBench/citycore-paris/elements.json
#   sha256 d1820f3d06e6828fda886a394346da6aad512d24569ddf4a790ef27c4bf2afca
#
# These are scene-authored pedestrian landmarks, not positions inferred from a
# controller detour. Intermediate nodes also turn the 90-degree far-side bend
# gradually enough to remain observable with the accepted 90-degree front/rear
# camera limitation.
#
# The route remains policy-neutral: node ids and coordinates are bookkeeping
# only.  The model sees the same phone-map image and front/rear photographs as
# in the straight run, never a textual "turn left/right" hint.
TURNING_ROUTE_PROFILE = "paris-poc-long-l-turn-v4-pedestrian"
TURNING_ROUTE_WAYPOINTS = (
    # id, x_cm, y_cm, street_index, role
    ("pixelgoal-turn-spawn", -5934.8349, -9700.0, 21, "spawn"),
    ("pixelgoal-turn-pickup", -5934.8349, -8700.0, 21, "pickup"),
    ("pixelgoal-turn-straight-1", -5934.8349, -7900.0, 21, "route"),
    ("pixelgoal-turn-straight", -5934.8349, -7000.0, 21, "route"),
    ("pixelgoal-turn-straight-2", -5934.8349, -6100.0, 21, "route"),
    ("pixelgoal-turn-straight-3", -5934.8349, -5200.0, 21, "route"),
    ("pixelgoal-turn-straight-4", -5934.8349, -4300.0, 21, "route"),
    ("pixelgoal-turn-pre-corner", -5950.0, -4000.0, 21, "route"),
    ("pixelgoal-turn-corner", -5950.0, -3200.0, 21, "corner"),
    # PR_Crossswalk_17: min-x edge, centre, max-x edge at its y centre.
    ("pixelgoal-turn-crossing-entry", -5704.827975050328,
     -2935.666851566335, 2, "crossing_entry"),
    ("pixelgoal-turn-crossing-centre", -5406.915006050115,
     -2935.666851566335, 2, "crossing"),
    ("pixelgoal-turn-crossing-exit", -5109.002037049902,
     -2935.666851566335, 2, "crossing_exit"),
    # Building-side pavement, curving around PR_SidewalkIslandEnd_01/08.
    ("pixelgoal-turn-sidewalk-bend-1", -4870.0, -3050.0, 2,
     "sidewalk_turn"),
    ("pixelgoal-turn-sidewalk-bend-2", -4775.0, -3250.0, 2,
     "sidewalk_turn"),
    ("pixelgoal-turn-sidewalk-1", -4775.0, -3500.0, 2, "route"),
    ("pixelgoal-turn-sidewalk-2", -4775.0, -3900.0, 2, "route"),
    ("pixelgoal-turn-dropoff", -4692.1603348614335,
     -4114.651577036972, 2, "dropoff"),
)
TURNING_PICKUP_NODE_ID = "pixelgoal-turn-pickup"
TURNING_DROPOFF_NODE_ID = "pixelgoal-turn-dropoff"
# A graph-only continuation behind the spawn.  The live rear photograph shows
# that pavement, so treating the spawn as degree one made the observation call
# a normal through street a "dead end" and confounded the first route choice.
# It is not part of the planned delivery polyline and never appears in route
# audit distances; it only makes the local street topology match the images.
TURNING_REAR_CONTEXT_WAYPOINT = (
    "pixelgoal-turn-rear-context", -5934.8349, -10700.0, 21,
)
# The physical Rue Crémieux pavement continues straight past the planned turn.
# This graph-only point covers that live-captured continuation, so a pawn that
# misses the corner stays on the local component and the phone can draw a
# truthful route back to the corner.  Without it, the nearest-node
# update eventually snapped to an unrelated original graph component and the
# blue route disappeared exactly when recovery was needed.
TURNING_OVERSHOOT_CONTEXT_WAYPOINTS = (
    ("pixelgoal-turn-overshoot-1", -5950.0, -2700.0, 21),
)


def build_turning_delivery_network(network: RoadNetwork) -> RoadNetwork:
    """Copy ``network`` and add the audited physical L-shaped sidewalk path.

    The overlay is intentionally disconnected from the approximate compiled
    road-centre graph.  This one proof-of-concept episode has targets on the
    overlay itself, and connecting it to displaced graph nodes would give the
    phone a shorter but physically false route through buildings or roadway.
    """

    out = copy.deepcopy(network)
    waypoint_ids = [row[0] for row in TURNING_ROUTE_WAYPOINTS]
    context_id, context_x, context_y, context_street_index = (
        TURNING_REAR_CONTEXT_WAYPOINT)
    overshoot_ids = [row[0] for row in TURNING_OVERSHOOT_CONTEXT_WAYPOINTS]
    new_node_ids = [*waypoint_ids, context_id, *overshoot_ids]
    collisions = sorted(set(new_node_ids) & set(out.nodes))
    if collisions:
        raise ValueError(
            f"turning delivery node ids already exist: {', '.join(collisions)}")
    for waypoint_id, x_cm, y_cm, street_index, _role in TURNING_ROUTE_WAYPOINTS:
        if not 0 <= street_index < len(out.streets):
            raise ValueError(
                f"turning delivery street index {street_index} is unavailable")
        out.nodes[waypoint_id] = StreetNode(
            id=waypoint_id,
            x_cm=x_cm,
            y_cm=y_cm,
            street_index=street_index,
            arc_cm=0.0,
        )
    if not 0 <= context_street_index < len(out.streets):
        raise ValueError(
            f"turning delivery street index {context_street_index} is unavailable")
    out.nodes[context_id] = StreetNode(
        id=context_id,
        x_cm=context_x,
        y_cm=context_y,
        street_index=context_street_index,
        arc_cm=-math.dist(
            (context_x, context_y), out.nodes[waypoint_ids[0]].position),
    )
    for overshoot_id, x_cm, y_cm, street_index in (
            TURNING_OVERSHOOT_CONTEXT_WAYPOINTS):
        if not 0 <= street_index < len(out.streets):
            raise ValueError(
                f"turning delivery street index {street_index} is unavailable")
        out.nodes[overshoot_id] = StreetNode(
            id=overshoot_id,
            x_cm=x_cm,
            y_cm=y_cm,
            street_index=street_index,
            arc_cm=0.0,
        )
    cumulative_by_street: dict[int, float] = {}
    for first_id, second_id in zip(waypoint_ids, waypoint_ids[1:]):
        first = out.nodes[first_id]
        second = out.nodes[second_id]
        first.neighbours.add(second_id)
        second.neighbours.add(first_id)
        edge_cm = math.dist(first.position, second.position)
        cumulative_by_street.setdefault(second.street_index, 0.0)
        cumulative_by_street[second.street_index] += edge_cm
        second.arc_cm = cumulative_by_street[second.street_index]
    out.nodes[context_id].neighbours.add(waypoint_ids[0])
    out.nodes[waypoint_ids[0]].neighbours.add(context_id)
    corner_id = next(
        row[0] for row in TURNING_ROUTE_WAYPOINTS if row[4] == "corner")
    overshoot_chain = [corner_id, *overshoot_ids]
    overshoot_arc = out.nodes[corner_id].arc_cm
    for first_id, second_id in zip(overshoot_chain, overshoot_chain[1:]):
        first = out.nodes[first_id]
        second = out.nodes[second_id]
        first.neighbours.add(second_id)
        second.neighbours.add(first_id)
        overshoot_arc += math.dist(first.position, second.position)
        second.arc_cm = overshoot_arc
    out.notes.append(
        "pixel-goal long-turn PoC v4: added a disconnected pedestrian overlay "
        "through PR_Crossswalk_17 and its building-side pavement")
    return out


def turning_route_report(network: RoadNetwork) -> dict[str, Any]:
    """Serializable, post-hoc audit metadata for the physical turn route."""

    rows = []
    for waypoint_id, x_cm, y_cm, street_index, role in TURNING_ROUTE_WAYPOINTS:
        rows.append({
            "id": waypoint_id,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "street": network.streets[street_index].name,
            "role": role,
        })
    points = [(row[1], row[2]) for row in TURNING_ROUTE_WAYPOINTS]
    waypoint_ids = [row[0] for row in TURNING_ROUTE_WAYPOINTS]
    pickup_index = waypoint_ids.index(TURNING_PICKUP_NODE_ID)
    segment_lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    return {
        "profile": TURNING_ROUTE_PROFILE,
        "waypoints": rows,
        "pickup_waypoint_id": TURNING_PICKUP_NODE_ID,
        "dropoff_waypoint_id": TURNING_DROPOFF_NODE_ID,
        "planned_total_cm": sum(segment_lengths),
        "planned_delivery_cm": sum(segment_lengths[pickup_index:]),
    }


class CorridorDeliveryEnv(EmbodiedCourierEnv):
    """``EmbodiedCourierEnv`` pinned to the one corridor a SPEAR session can
    currently serve pixel-goal navigation on, carrying a hand-picked
    pickup/dropoff pair instead of a graph-drawn one.

    The stock ``reset()`` (a) spawns at a random graph node anywhere on the
    compiled Paris map, and (b) draws an order between two real addresses.
    Both are wrong for a single SPEAR engine session: (a) this session's
    engine only has the ONE navmesh-ready region the runner configured, so a
    random spawn would ask the agent to walk somewhere the pawn never stood;
    and (b) a real address's ``kerb`` is the nearest same-street graph node
    to its door, which sits on the road CENTRELINE (nodes are resampled off
    ``roads_detailed.json``, never checked against the sidewalk) -- median
    ~7.8 m off the actual footway at this corridor, more than the 8 m arrival
    tolerance ``collect``/``hand_over`` test against (measured on the first
    real-engine runs, August 2026).

    So this override, after the stock reset has run: pins ``node_id`` to the
    corridor's own nearest graph node (used only for narration -- street
    name, "on your left" -- CourierEnv text keeps working unmodified), and
    replaces the drawn order with one whose pickup/dropoff ``kerb`` sit
    exactly on the validated sidewalk line, a real walking distance apart.
    Everything else -- the ``collect()``/``hand_over()`` state machines, the
    clock, the prompts, and tool dispatch -- is stock ``CourierEnv``. Only the
    drop-off proximity hook is tightened to one metre for a physically
    plausible hand-over.
    """

    def __init__(
        self,
        network: RoadNetwork,
        pool_or_client: Any,
        *,
        pickup_offset_cm: float = DEFAULT_PICKUP_OFFSET_CM,
        dropoff_offset_cm: float = DEFAULT_DROPOFF_OFFSET_CM,
        **kwargs: Any,
    ) -> None:
        if dropoff_offset_cm <= pickup_offset_cm:
            raise ValueError(
                "dropoff_offset_cm must be further along the corridor than "
                "pickup_offset_cm, or there is no delivery leg to walk")
        self._pickup_offset_cm = float(pickup_offset_cm)
        self._dropoff_offset_cm = float(dropoff_offset_cm)
        super().__init__(network, pool_or_client, **kwargs)

    def reset(self) -> None:
        super().reset()
        spawn_x, spawn_y = self._here_cm()
        corridor_node, _ = self._nearest_node((spawn_x, spawn_y))
        self.node_id = corridor_node
        self.arrived_from = None
        self._draw_cursor = self.node_id
        pickup = self._corridor_address(
            "pixelgoal-demo-pickup", 1, spawn_x, spawn_y + self._pickup_offset_cm)
        dropoff = self._corridor_address(
            "pixelgoal-demo-dropoff", 2, spawn_x, spawn_y + self._dropoff_offset_cm)
        self.orders = [Order(index=1, pickup=pickup, dropoff=dropoff,
                             fee=6.0, deadline_s=1_800.0)]
        # _optimal_route() still runs, off the real (if approximate) graph
        # distance between the two addresses' kerb NODES -- useful for the
        # end-of-run "walked how much more than optimal" report -- but the
        # shift clock is set generously and independently of it: a demo
        # corridor's graph routing is not trustworthy enough to let it also
        # decide when the one order expires.
        self.optimal_seconds, self.optimal_walk_cm = self._optimal_route()
        self.shift_seconds = 3_600.0
        self._issue()

    def _corridor_address(
        self, building_id: str, number: int, x_cm: float, y_cm: float,
    ) -> Address:
        """A synthetic ``Address`` whose ``kerb`` is exactly ``(x_cm, y_cm)``
        on the validated sidewalk, borrowing a real street name/graph
        connectivity from the nearest actual node for narration and routing.
        """
        node_id, _ = self._nearest_node((x_cm, y_cm))
        node = self.network.nodes[node_id]
        street = self.network.streets[node.street_index]
        return Address(
            building_id=building_id, street_index=node.street_index,
            street_name=street.name, number=number,
            arc_cm=node.arc_cm, offset_cm=0.0, side="left",
            door=(x_cm, y_cm), nearest_node=node_id, poi_type="residence",
            kerb_node=node_id, kerb=(x_cm, y_cm),
        )

    def house_numbers_near(self, node_id: str, limit: int = 10) -> str:
        """Name the synthetic target exactly when its task action would work.

        The corridor order is deliberately not part of the compiled Paris
        address book, so the stock implementation can only narrate unrelated
        real doors near the graph node.  At the pawn's current node, make the
        synthetic order's physical kerb authoritative within the same radius
        ``collect`` and ``hand_over`` use.  Neighbour lookups still delegate
        unchanged to the real address book.
        """
        order = self.active_order()
        tolerance_cm = self._task_action_tolerance_cm(
            collected=bool(order and order.picked_up))
        if (order is not None
                and self.ue_pose is not None
                and node_id == self.node_id
                and math.dist(self._here_cm(), order.target.kerb)
                <= tolerance_cm):
            return str(order.target.number)
        return super().house_numbers_near(node_id, limit)

    def _task_action_tolerance_cm(self, *, collected: bool) -> float:
        if collected:
            return CORRIDOR_DROPOFF_TOLERANCE_CM
        return super()._task_action_tolerance_cm(collected=collected)

    def summary(self) -> dict[str, Any]:
        """Add wall latency measured at the SPEAR capture/execute boundary."""

        out = super().summary()
        events = getattr(self._client, "events", ()) if self._client is not None else ()
        out["latency"] = {
            kind: latency_statistics(
                event["wall_latency_s"]
                for event in events
                if event.get("kind") == kind
            )
            for kind in ("view_pair_capture", "pixel_action_execute")
        }
        return out


class TurningDeliveryEnv(CorridorDeliveryEnv):
    """The same delivery mechanics on the audited longer L-shaped route."""

    def reset(self) -> None:
        # Let the corridor class initialize every stock CourierEnv and live-UE
        # lifecycle field first.  Its temporary straight order is then replaced
        # before the first observation or model call can see it.
        super().reset()
        pickup_node = self.network.nodes[TURNING_PICKUP_NODE_ID]
        dropoff_node = self.network.nodes[TURNING_DROPOFF_NODE_ID]
        pickup = self._corridor_address(
            "pixelgoal-turn-pickup", 1,
            pickup_node.x_cm, pickup_node.y_cm,
        )
        dropoff = self._corridor_address(
            "pixelgoal-turn-dropoff", 2,
            dropoff_node.x_cm, dropoff_node.y_cm,
        )
        self.orders = [Order(
            index=1,
            pickup=pickup,
            dropoff=dropoff,
            fee=6.0,
            deadline_s=1_800.0,
        )]
        self.optimal_seconds, self.optimal_walk_cm = self._optimal_route()
        self.shift_seconds = 3_600.0
        # The parent briefly lit the phone for its temporary straight target.
        # Clear ownership before issuing this real order so its L-shaped route
        # is the only route that can reach the first policy observation.
        self.screen_route = []
        self.screen_target = None
        self._screen_route_owner = self.SCREEN_ROUTE_NONE
        self._issue()


class ValidatedPoolDeliveryEnv(EmbodiedCourierEnv):
    """One resolved order from a certified regional entrance pool.

    Selection happens before UE setup so the physical pawn, graph spawn, phone
    route and task order share one source of truth from the first observation.
    Unlike the legacy corridor subclasses this class never creates a temporary
    random order and never replaces an already issued order during ``reset``.
    """

    def __init__(
        self,
        network: RoadNetwork,
        pool_or_client: Any,
        *,
        delivery_pool: ValidatedDeliveryPool,
        scenario: ResolvedDeliveryScenario,
        **kwargs: Any,
    ) -> None:
        self.delivery_pool = delivery_pool
        self.delivery_scenario = scenario
        super().__init__(network, pool_or_client, **kwargs)
        # CourierEnv's viewpoint field describes which pre-baked street album
        # it selected.  This backend serves neither street album: every frame
        # comes live from the pawn spawned on a pool point that the fresh UE
        # surface audit classifies as pavement.  Leaving the inherited default
        # reported "carriageway" for a physically pavement episode and made a
        # valid trusted run look like a viewpoint mismatch.  The pool is
        # pedestrian-only, so refuse a non-pavement embodiment rather than
        # silently relabel its camera.
        if self.embodiment.viewpoint != Viewpoint.PAVEMENT:
            raise ValueError(
                "trusted pedestrian delivery requires a pavement embodiment")
        self.viewpoint_served = Viewpoint.PAVEMENT
        self.viewpoint_matches_embodiment = True
        # A pixel names a destination on the certified graph and the pawn is
        # walked there along certified ways -- when the client can turn the
        # pawn to walk a leg. Without that (the two-view client) the engine
        # walks the straight line to the pixel or refuses it, as before.
        self.pedestrian_routing = bool(
            getattr(pool_or_client, "can_walk_world", False))
        if self.pedestrian_routing and delivery_pool.certified_legs is None:
            raise ValueError(
                "pedestrian routing walks only the legs the engine certified; "
                f"pool {delivery_pool.profile} carries none")

    def _pedestrian_snap(
        self, point_cm: tuple[float, float], radius_cm: float,
    ) -> tuple[str, float] | None:
        # a destination is a node the pawn may stop on: never the middle of
        # a crossing the engine walks over but accepts no leg from
        found = nearest_pool_node(
            self.delivery_pool, point_cm, radius_cm=radius_cm, stops_only=True)
        if found is None:
            return None
        node, gap = found
        return node.id, gap

    def _pedestrian_legs(
        self, start_node_id: str, end_node_id: str,
        avoid: Sequence[tuple[str, str]] = (),
    ) -> tuple[bool, list[Any]] | None:
        planned = plan_pool_legs(
            self.delivery_pool, start_node_id, end_node_id, avoid=avoid)
        if planned is None:
            return None
        path, legs = planned
        return path.uses_marked_crossing, legs

    @property
    def max_leg_replans(self) -> int:
        return MAX_LEG_REPLANS

    def _choose_spawn_node(self, _rng: Any) -> str:
        node_id = self.delivery_scenario.spawn.node_id
        if node_id not in self.network.nodes:
            raise ValueError(
                f"certified spawn node is absent from the regional graph: {node_id}")
        return node_id

    def _path_length_cm(self, node_ids: tuple[str, ...]) -> float:
        """Planar length of a node path, in the network's own coordinates."""
        total = 0.0
        for a, b in zip(node_ids, node_ids[1:]):
            total += math.dist(self.position(a), self.position(b))
        return total

    def _make_orders(self, _rng: Any) -> list[Order]:
        by_building = {address.building_id: address for address in self.network.addresses}
        pickup = by_building.get(self.delivery_scenario.pickup.building_id)
        dropoff = by_building.get(self.delivery_scenario.dropoff.building_id)
        if pickup is None or dropoff is None:
            raise ValueError("resolved scenario addresses are absent from the pool network")
        if pickup.text != self.delivery_scenario.pickup.text \
                or dropoff.text != self.delivery_scenario.dropoff.text:
            raise ValueError("resolved scenario address text drifted during network build")

        phone_approach = self.route_nodes(self.node_id, pickup.kerb_node)
        phone_delivery = self.route_nodes(pickup.kerb_node, dropoff.kerb_node)
        # The phone must route the courier over the certified pedestrian graph
        # and nowhere else -- which every path in this network is, by
        # construction. What the certification pins is the *route*, not the
        # tie-break: on a lattice two equal-length paths differ in node ids
        # and are the same walk, and requiring identity refused 2 of the 16
        # protocol scenarios at reset. So: same endpoints, same length.
        for phone, certified, leg in (
            (phone_approach, self.delivery_scenario.approach, "approach"),
            (phone_delivery, self.delivery_scenario.delivery, "delivery"),
        ):
            phone_ids = tuple(phone or ())
            if phone_ids == certified.node_ids:
                continue
            if (not phone_ids or phone_ids[0] != certified.node_ids[0]
                    or phone_ids[-1] != certified.node_ids[-1]):
                raise ValueError(
                    f"phone navigation {leg} path diverged from the selected "
                    "pedestrian path")
            phone_cm = self._path_length_cm(phone_ids)
            if abs(phone_cm - float(certified.length_cm)) > 5.0:
                raise ValueError(
                    f"phone navigation {leg} path is {phone_cm:.1f} cm against a "
                    f"certified {float(certified.length_cm):.1f} cm")

        approach = self.route_cost(
            self.node_id, pickup.kerb_node, obstacles=True)
        delivery = self.route_cost(
            pickup.kerb_node, dropoff.kerb_node, obstacles=True)
        if approach is None or delivery is None:
            raise ValueError("resolved certified scenario became unreachable at reset")
        walking_seconds = approach[0] + delivery[0]
        deadline_s = walking_seconds * self.deadline_slack + 2.0 * HANDLING_SECONDS
        self._draw_cursor = dropoff.kerb_node
        return [Order(
            index=1,
            pickup=pickup,
            dropoff=dropoff,
            fee=self.fee_for(self.delivery_scenario.delivery.length_cm),
            deadline_s=deadline_s,
        )]

    def _task_action_tolerance_cm(self, *, collected: bool) -> float:
        # Both ends are certified door-side interaction anchors. Collection from
        # eight metres away is no more physically plausible than handing a parcel
        # over from eight metres away.
        return (
            CORRIDOR_DROPOFF_TOLERANCE_CM
            if collected else VALIDATED_POOL_PICKUP_TOLERANCE_CM
        )

    def house_numbers_near(self, node_id: str, limit: int = 10) -> str:
        """Name a door only where its task action would work.

        The stock reader lists every door within READABLE_NUMBER_CM (8 m) of
        the matched graph node, while collect()/hand_over() here accept only
        the certified anchor inside the task tolerance. So "where you
        are" named the pickup from a lattice node a few metres short of it,
        the prompt told the policy to act on a match, and the refusal that
        followed carried no distance -- four of those end a run as `stuck`.
        At the pawn's own node the active target's number appears exactly
        when its action would succeed and nothing else is named; every other
        node keeps the stock reading.
        """
        if node_id != self.node_id:
            return super().house_numbers_near(node_id, limit)
        order = self.active_order()
        if order is None or self.ue_pose is None:
            return ""
        tolerance_cm = self._task_action_tolerance_cm(
            collected=bool(order.picked_up))
        if math.dist(self._here_cm(), order.target.kerb) <= tolerance_cm:
            return str(order.target.number)
        return ""

    def _next_instruction(
        self, route: list[tuple[float, float]],
    ) -> dict[str, str | float | None]:
        """Give an actionable banner while retaining the exact Recast route.

        ``CourierEnv`` correctly uses the immediate next node for its coarser
        street graph.  This subclass has a certified one-metre surface lattice,
        where that policy makes the banner oscillate and reveals a turn too
        late for a multi-metre pixel action.  Both headings below are computed
        solely from the route passed to the map renderer.
        """

        instruction = trusted_phone_route_instruction(route)
        if not instruction["next_heading"]:
            return {"next_street": "", **instruction}
        # Street identity is semantic metadata only.  Use the first exact route
        # node to label the already-computed Recast instruction; it contributes
        # no connectivity or geometry.
        nearest = min(
            self.network.nodes,
            key=lambda node_id: math.dist(self.position(node_id), route[1]),
        )
        return {
            "next_street": self.street_of(nearest) or "",
            **instruction,
        }

    def summary(self) -> dict[str, Any]:
        """Add wall latency measured at the SPEAR capture/execute boundary."""

        out = super().summary()
        events = getattr(self._client, "events", ()) if self._client is not None else ()
        out["latency"] = {
            kind: latency_statistics(
                event["wall_latency_s"]
                for event in events
                if event.get("kind") == kind
            )
            for kind in ("view_pair_capture", "pixel_action_execute")
        }
        out["phone_navigation"] = {
            "route_source": "same_certified_ue_recast_pedestrian_graph",
            "route_geometry": "exact_dynamic_shortest_path_node_polyline",
            "instruction_method": TRUSTED_PHONE_INSTRUCTION_METHOD,
            "simplification_tolerance_cm": (
                TRUSTED_PHONE_ROUTE_SIMPLIFICATION_CM),
            "guidance_lookahead_cm": TRUSTED_PHONE_GUIDANCE_LOOKAHEAD_CM,
            "maneuver_preview_cm": TRUSTED_PHONE_MANEUVER_PREVIEW_CM,
            "minimum_maneuver_deg": TRUSTED_PHONE_MIN_MANEUVER_DEG,
        }
        return out
