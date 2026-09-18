"""``EmbodiedCourierEnv``: the courier world whose legs are a UE pawn.

The embodied backend. Where ``LiveCourierEnv`` keeps every
transition in Python and uses UE as a camera, this env hands UE ownership of
**locomotion and locomotion time**: a hop along an edge is a ``/walk`` to the
next node's coordinates, and the seconds the clock advances by are the
engine's own ``ticks * fixed_dt`` instead of the stock distance/speed
arithmetic. Everything else the clock counts -- collect, hand_over, look,
wait, the refusal floor -- remains declared bookkeeping added on top, through
the same ``_charge``/``_refuse`` path the stock env uses, so budgets, turns
and stamina keep meaning what they mean.

The transition seam, stated precisely because it is the whole design:
``_step_to`` is the single-hop atom every movement tool reduces to
(``walk_to`` at waypoint stride calls it once; ``_run_street`` -- the block
stride and the ``follow_street`` macro -- loops it). Overriding it and
nothing else means every stock movement tool acquires UE-owned motion without
being touched. Within the override:

* an unknown ``k`` delegates to the stock method, whose refusal wording
  (including the dead-end coaching) is behaviour tests pin;
* an ``arrived`` walk replays the stock hop bookkeeping -- eight lines copied
  from ``CourierEnv._step_to`` (turns, stamina, node update, clock,
  walked_cm, ``_issue``, the outcome) -- with the engine's ``sim_seconds``
  where the arithmetic was and the engine's ``walked_cm`` where the graph
  length was. Copied rather than called, because the stock method computes
  time from speed mid-body and offers no narrower seam; the copy is small,
  and this docstring is its registration;
* ``stuck`` maps to the stock refusal path with code ``"stuck"`` and a
  charge equal to the sim seconds the engine actually burned (floored at the
  stock ``REJECTED_ACTION_SECONDS``, exactly like every refusal) -- the
  ``way_blocked`` semantics with an engine-measured price instead of the
  declared ``BLOCKED_SECONDS``;
* a walk that runs out of ``max_sim_seconds`` is the same refusal with code
  ``"walk_timeout"``.

v1 runs hazards OFF, by construction: locomotion realism is the thing under
test, and obstacle dressing / signal charging in embodied mode needs the
stateful scene dressing the spec reserves. Passing any hazard album or
sidecar root refuses at construction; ``signal_frame_for`` and
``obstacle_frame_for`` answer ``None`` unconditionally. Difficulty defaults
to solo but other tiers are allowed -- they change the order book, not the
physics.

Legacy observations come from ``/observe`` at the agent's ACTUAL pose (yawed
toward the neighbour first). The ``pixel_goal_front_rear`` variant instead
uses one atomic, no-pawn-rotation ``/observe_views`` pair and keeps its private
snapshot bindings with the pending turn. Both forms materialise into the lazy
album cache, so CourierSession, FrameAliases and the training adapter retain
their normal interfaces. The cache's first-render-wins rule is kept
deliberately: after a stuck walk the pawn stands off the node, and a
re-observe under the same key would break ``observation_media_hash``'s
same-bytes assumption for a frame whose divergence the embodied log already
records.

Two honest limitations, logged rather than hidden: the pawn's max speed is
set once at /episode from the embodiment, so the tired-courier slowdown does
not reach UE motion in v1 (stamina still drains, off the engine's walked_cm);
and after a stuck walk the env stands at the node while the pawn stands where
it stopped -- ``embodied_log``'s ``end_pose`` and ``pose_error_cm`` are the
evidence trail for both.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import heapq
import logging
import math
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from embodiedbench.compiler.road_network import RoadNetwork, bearing_deg
from embodiedbench.runtime.city.courier_env import (
    WALK_SPEED_CM_S,
    REJECTED_ACTION_SECONDS,
    CourierEnv,
    StepOutcome,
    compass_of,
)
from embodiedbench.runtime.pixel_goal import PixelGoalPairIntegrityError

from .cache import LiveAlbum
from .client import (
    BadRequestError,
    RenderFailedError,
    RenderServiceError,
    ServiceBusy,
    ServiceUnreachable,
    UERenderClient,
)
from .env import STREET_CAMERA, STREET_EYE_CM
from .protocol import (
    PIXEL_VIEWS,
    PIXEL_VIEWS_QUAD,
    ResolvedPixel,
    DEFAULT_ARRIVE_CM,
    DEFAULT_MAX_WALK_SIM_SECONDS,
    DEFAULT_TICK_CHUNK,
    RETURN_MODE_PATH,
    RETURN_MODE_BASE64,
    AgentSpec,
    EpisodeEndRequest,
    EpisodeRequest,
    ObserveRequest,
    ObserveViewsRequest,
    ObserveViewsResponse,
    ObservedView,
    PixelSpec,
    Pose,
    ProtocolViolation,
    WalkPixelRequest,
    WalkPixelResponse,
    WalkRequest,
    WalkResponse,
)

logger = logging.getLogger(__name__)


class InstanceMoved(Exception):
    """The instance died and the episode has been re-opened on another.

    Raised out of the walk path so the caller charges the call as a refusal
    rather than crediting a walk nobody can say happened. Not a
    ``RenderServiceError``: nothing about the service is wrong any more.
    """

# Where the pawn spawns on the z axis, in cm. The capsule needs clearance
# above the road surface; 100 is the spec's own example and what the SimWorld2
# scaffold uses.
DEFAULT_SPAWN_Z_CM = 100.0

#: The three action spaces this env can present. See ``action_space``.
ACTION_SPACE_STREET = "street"
ACTION_SPACE_COORDINATE = "coordinate"
#: Name a visible ground point in the current forward photograph, rather
#: than a street or a metric position. See ``walk_to_pixel``.
ACTION_SPACE_PIXEL_GOAL = "pixel_goal"
ACTION_SPACE_PIXEL_GOAL_FRONT_REAR = "pixel_goal_front_rear"

#: Pedestrian routing (the four-view harness): a resolved point becomes the
#: nearest certified pedestrian node within this distance, and the pawn is
#: walked there along certified ways.
PEDESTRIAN_SNAP_CM = 150.0
#: A ray that hit the carriageway snaps only from the gutter.
ROAD_SNAP_CM = 60.0
#: A hit on an object or a wall counts as the ground beside it only when it
#: is near the ground (a kerb stone, a planter's foot), not up a facade.
OBJECT_HIT_MAX_ABOVE_GROUND_CM = 60.0
#: Before a certified leg the pawn is set back onto the node it stands off
#: by more than a step and less than this: the engine's controller stops
#: 5-30 cm from the point it walks to, that point is the engine's own
#: projection of the leg's end (up to a metre off it), and the legs are
#: certified from the nodes. Further off than this the harness has lost the
#: pawn and says so rather than walk an uncertified line.
NUDGE_MIN_CM = 5.0
NUDGE_MAX_CM = 75.0
#: The pawn's feet relative to its reported position.
PAWN_FEET_BELOW_POSITION_CM = 88.0
ROAD_ACTOR_PREFIXES = ("BP_SplineRoad", "BP_CustomIntersection", "PR_Road")
PEDESTRIAN_ACTOR_PREFIXES = (
    "BP_SplineSidewalk", "PR_SidewalkIsland", "PR_Crossswalk", "Template_Map_Floor")
_PIXEL_VIEW_STREET = {"front": "ahead", "rear": "behind",
                      "left": "to your left", "right": "to your right"}


def _surface_class(actor: str | None) -> str:
    """What the engine's actor name says the ray hit: road, pedestrian, object."""
    name = actor or ""
    if name.startswith(ROAD_ACTOR_PREFIXES):
        return "road"
    if name.startswith(PEDESTRIAN_ACTOR_PREFIXES):
        return "pedestrian"
    return "object"
ACTION_SPACES = (ACTION_SPACE_STREET, ACTION_SPACE_COORDINATE,
                 ACTION_SPACE_PIXEL_GOAL, ACTION_SPACE_PIXEL_GOAL_FRONT_REAR)

# Engine diagnostics are translated rather than echoed. These are causal
# classes a policy can use to reassess what it saw, without actor names,
# coordinates, thresholds, or instructions about where to aim next.
_PIXEL_REJECTION_NO_GROUND = frozenset({
    "no_ground_hit",
    "no_geometry_hit",
})
_PIXEL_REJECTION_NON_WALKABLE = frozenset({
    "hit_not_walkable_ground",
})
_PIXEL_REJECTION_OUTSIDE_NAVIGATION = frozenset({
    "off_navmesh",
    "navmesh_projection_failed",
    "navmesh_adjustment_exceeded",
    "navmesh_unavailable",
})
_PIXEL_REJECTION_NO_PATH = frozenset({
    "no_controller_path",
})
_PIXEL_REJECTION_INDIRECT_PATH = frozenset({
    "controller_path_detour_exceeded",
})
_PIXEL_REJECTION_UNMARKED_ROAD = frozenset({
    "controller_path_enters_unmarked_road",
})
_PIXEL_REJECTION_UNVERIFIED_PEDESTRIAN_SURFACE = frozenset({
    "controller_path_surface_unverified",
})


def _pixel_rejection_message(reason: str) -> str:
    if reason in _PIXEL_REJECTION_NO_GROUND:
        return "The selected pixel did not hit visible ground."
    if reason in _PIXEL_REJECTION_NON_WALKABLE:
        return "The selected pixel hit a non-walkable surface."
    if reason in _PIXEL_REJECTION_OUTSIDE_NAVIGATION:
        return (
            "The selected pixel hit ground, but it was not on a reachable "
            "walkable navigation surface."
        )
    if reason in _PIXEL_REJECTION_NO_PATH:
        return (
            "The selected pixel hit walkable ground, but there is no walkable "
            "way from here to it."
        )
    if reason in _PIXEL_REJECTION_INDIRECT_PATH:
        return (
            "The selected pixel hit visible ground, but reaching it would "
            "require a long detour out of view."
        )
    if reason in _PIXEL_REJECTION_UNMARKED_ROAD:
        return (
            "The route to the selected point would leave the pedestrian way "
            "outside a marked crosswalk; the only crossing that counts here "
            "is the one the blue route on your phone uses."
        )
    if reason in _PIXEL_REJECTION_UNVERIFIED_PEDESTRIAN_SURFACE:
        return (
            "The route to the selected point could not be verified as "
            "pedestrian-only."
        )
    return "The selected pixel could not be resolved to a walkable destination."

#: What the courier is shown each turn.
#:
#: ``streets`` photographs every street leaving the junction, one frame per
#: candidate, each captioned with the street a ``walk_to`` would name. That
#: pairing is the street action space's whole design.
#:
#: ``forward`` photographs one thing: what is in front of the courier. It
#: exists because the pairing stops being a design once the courier no longer
#: takes streets by name -- under ``coordinate`` the per-street frames include
#: the way it came, which it cannot act on and which spends half a two-image
#: budget. A walking person does not get a photograph of behind them each time
#: they take a step.
#:
#: ``streets`` is the default and stays it: the two action spaces are compared
#: against each other, and changing what one of them SEES makes the difference
#: between them two things instead of one.
CAMERA_VIEW_STREETS = "streets"
CAMERA_VIEW_FORWARD = "forward"
CAMERA_VIEW_FRONT_REAR = "front_rear"
CAMERA_VIEWS = (CAMERA_VIEW_STREETS, CAMERA_VIEW_FORWARD,
                CAMERA_VIEW_FRONT_REAR)

# How far one ``walk_to_xy`` may carry, in metres. A request past it is
# REFUSED, not clamped: a courier that asked for forty metres and was quietly
# carried one and a half cannot tell that from arriving, and neither can a
# reader of the log.
#
# 10 m, and the number has to clear four things at once.
#
# The JOB has to be reachable. A median delivery on this map is 247 m end to
# end; at 9 m of ground per call (10 less the metre the walk stops short by)
# that is 27 calls, nine turns at a chunk of three, against a val budget of
# sixteen. A 1.5 m step wanted sixty-nine turns and the context window holds
# about thirty, so `delivered` would have read zero for every seed by
# arithmetic rather than by navigation.
#
# It has to sit near the scale the POLICY reaches for, or the refusals stop
# being about navigation. Measured over the first live run: when it named a
# point that was not the one under its feet, the request was 2.6 m at the
# smallest, 4.8 m median, 16.2 m at the largest. A 10 m cap admits thirteen of
# those fifteen, so what a refusal now means is "there is no way there", not
# "you guessed the limit wrong".
#
# It must not make a coordinate call CHEAPER than a street one, which is what
# the original 60 m was sized for: block-stride legs are a median 38.8 m and a
# p75 of 59.5 m, and 10 m is comfortably under both.
#
# And the walk geometry has to close: see DEFAULT_STEP_ARRIVE_CM.
DEFAULT_MAX_STEP_M = 10.0

# The arrival radius and the tick chunk are not free once the step is short.
# One tick carries ``speed * fixed_dt`` -- 28 cm at the 16x rollout settings --
# and arrival is only tested at chunk boundaries, so the radius has to be at
# least a chunk's worth of travel or the pawn sails past and reports stuck.
# It must also be well inside the step cap, or the band between "you are
# already there" and "that is too far" closes and every request lands in one
# refusal or the other.
DEFAULT_STEP_ARRIVE_CM = 100.0
DEFAULT_STEP_TICK_CHUNK = 2

# Pose-tracked movement is matched back to graph-shaped narration after every
# UE walk.  The graph is a survey abstraction, not a second position sensor:
# an unconstrained nearest-node search can therefore jump to a physically
# close but disconnected street (or the wrong arm of a junction).  Three hops
# are always searched for compatibility with the coarse street graph.  A
# Recast-derived pedestrian lattice is much denser, however: a valid ten-metre
# pixel walk can traverse ten one-metre edges.  When UE supplies a measured
# movement, search the same connected component out to that distance plus a
# bounded map-matching margin.  Switching nodes pays a small hysteresis cost,
# a candidate whose net displacement from the previous match contradicts the
# measured UE movement direction pays up to four metres more, and graph
# progress beyond the measured walk pays one-for-one.  The net displacement is
# deliberate: a dense Recast connector may begin with a one-metre side-step
# before joining the crosswalk actually traversed by a multi-metre action.  A
# first-edge bearing would mistake that sampling detail for the pawn's travel
# direction.  Distances are centimetres so all terms share one auditable unit.
POSE_MATCH_MIN_HOPS = 3
POSE_MATCH_SEARCH_MARGIN_CM = 300.0
POSE_MATCH_SWITCH_COST_CM = 75.0
POSE_MATCH_HEADING_COST_CM = 400.0
POSE_MATCH_PROGRESS_COST_PER_CM = 1.0
POSE_MATCH_METHOD = "metric_local_continuity_net_heading_progress_v4"

# The album roots this env owns (all of them: v1 embodied has exactly one
# album, the live cache, and no hazard frames at all), plus the hazard levers
# that would smuggle charges into a mode whose renderer cannot show them.
_REFUSED_KWARGS = (
    "album_root", "signal_album_root", "obstacle_album_root",
    "pavement_album_root", "pavement_obstacle_album_root",
    "obstacle_sidecar_root", "signal_sidecar_root",
)


def _metres(value: float) -> str:
    """A distance as the manual should read it: 60, not 60.0."""
    return f"{value:g}"


def _stable_id(text: str) -> str:
    """A best-effort ``StableId`` (``embodiedbench.schemas.world``): letters,
    digits, ``_.:-`` only, 1-128 characters. Audit ids are built from
    caller-chosen strings like ``episode_id``, which the schema was never
    asked to constrain -- so this launders rather than trusts them, on the
    same "the audit record must not crash the walk" principle as the
    try/except around ``NavigationRequest`` construction itself."""
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "-", text) or "id"
    return cleaned[:128]


def _point(xy: tuple[float, float]) -> str:
    """A position as the courier is told it, in the units it types back.

    One decimal and north first, identical to ``CourierEnv.pose_text`` -- the
    coordinates in "you walk 38 m and stop at (…)" and the ones in "you are
    standing at (…)" are the same position and have to read as the same
    number, or the courier is left deciding which of two spellings of its own
    position to count from.
    """
    return f"({xy[0] / 100.0:.1f}, {xy[1] / 100.0:.1f})"


class EmbodiedCourierEnv(CourierEnv):
    """CourierEnv whose hops are walked by a UE pawn under lockstep ticks.

    ``pool_or_client`` is either a ``RenderPool`` -- in which case the env
    takes an exclusive embodied lease on first engine contact and returns it
    on ``close()`` -- or a bare ``UERenderClient`` for single-instance setups
    and tests.
    """

    def __init__(
        self,
        network: RoadNetwork,
        pool_or_client: Any,
        street_camera: Any = None,
        *,
        episode_id: str,
        cache_root: str | Path,
        spawn_z_cm: float = DEFAULT_SPAWN_Z_CM,
        arrive_cm: float = DEFAULT_ARRIVE_CM,
        max_walk_seconds: float = DEFAULT_MAX_WALK_SIM_SECONDS,
        tick_chunk: int = DEFAULT_TICK_CHUNK,
        # base64 by default. `path` hands back a filename on the RENDERER's
        # disk, which is only readable when trainer and fleet share a
        # filesystem -- and the whole point of the fleet being addressable
        # over the network is that they no longer do. Measured: every
        # episode degraded on FileNotFoundError while walking looked
        # perfect, because walking is pure RPC and never touches a file.
        return_mode: str = RETURN_MODE_BASE64,
        # Whether a failed observe may finish the episode on cached
        # frames. On by default so a flaky renderer does not kill a
        # long run -- but an ONLINE experiment should turn it off: an
        # episode whose pictures came from an album measured the album.
        allow_album_fallback: bool = True,
        # Which action space this episode presents: "street" (name one of
        # the streets leaving this junction) or "coordinate" (name a point
        # and walk toward it). Never both -- see ``tools.COORDINATE_TOOLS``.
        action_space: str = ACTION_SPACE_STREET,
        pixel_views: tuple[str, ...] = PIXEL_VIEWS,
        max_step_m: float = DEFAULT_MAX_STEP_M,
        # What the turn's photographs are of. See CAMERA_VIEWS.
        camera_view: str = CAMERA_VIEW_STREETS,
        **courier_kwargs: Any,
    ):
        clash = sorted(set(_REFUSED_KWARGS) & set(courier_kwargs))
        if clash:
            raise ValueError(
                f"EmbodiedCourierEnv v1 runs hazards OFF; got {clash}. "
                "Locomotion realism is the thing under test, and hazard "
                "frames in embodied mode need the stateful scene dressing "
                "the spec reserves -- there is no album root to point at.")
        # Solo by default -- one order keeps the locomotion evidence clean --
        # but other tiers only change the order book, so they are allowed.
        courier_kwargs.setdefault("difficulty", "solo")
        self.live_album = LiveAlbum(cache_root, episode_id)
        self.spawn_z_cm = float(spawn_z_cm)
        self.arrive_cm = float(arrive_cm)
        self.max_walk_seconds = float(max_walk_seconds)
        self.tick_chunk = int(tick_chunk)
        self.return_mode = return_mode
        self.allow_album_fallback = bool(allow_album_fallback)
        if action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space={action_space!r}; it is one of {ACTION_SPACES}. "
                "These are different tasks and a run is exactly one of them "
                "-- a menu holding more than one lets an episode fall back "
                "to the easy action and be reported under a harder one's "
                "name.")
        self.action_space = action_space
        if camera_view not in CAMERA_VIEWS:
            raise ValueError(
                f"camera_view={camera_view!r}; it is one of {CAMERA_VIEWS}.")
        self.camera_view = camera_view
        if camera_view == CAMERA_VIEW_FORWARD and action_space == ACTION_SPACE_STREET:
            raise ValueError(
                "camera_view='forward' with the street action space would "
                "photograph nothing the courier can name: it picks a street "
                "off the list, and the list's pictures are how it tells them "
                "apart.")
        if (action_space == ACTION_SPACE_PIXEL_GOAL
                and camera_view != CAMERA_VIEW_FORWARD):
            raise ValueError(
                "action_space='pixel_goal' picks a point IN a photograph, so "
                "it needs camera_view='forward' -- the one photograph it can "
                "point at. 'streets' would show it pictures of streets it "
                "does not name and none of the one it does.")
        if (action_space == ACTION_SPACE_PIXEL_GOAL_FRONT_REAR
                and camera_view != CAMERA_VIEW_FRONT_REAR):
            raise ValueError(
                "pixel_goal_front_rear requires camera_view='front_rear'")
        if (camera_view == CAMERA_VIEW_FRONT_REAR
                and action_space != ACTION_SPACE_PIXEL_GOAL_FRONT_REAR):
            raise ValueError(
                "camera_view='front_rear' requires "
                "action_space='pixel_goal_front_rear'")
        self.max_step_m = float(max_step_m)
        if self.max_step_m <= 0:
            raise ValueError(
                f"max_step_m={max_step_m!r}; one coordinate call has to be "
                "able to cover some ground.")
        # The three numbers that have to move together, checked once here
        # rather than rediscovered as a stuck rate. See DEFAULT_STEP_ARRIVE_CM.
        if self.coordinate_mode and self.arrive_cm >= self.max_step_m * 100.0:
            raise ValueError(
                f"arrive_cm={self.arrive_cm} against a {self.max_step_m} m "
                "step cap leaves no distance a request can name: anything "
                "nearer counts as already arrived and anything further is "
                "refused. Bring arrive_cm well inside the cap.")
        # Naming a point is only answerable if the courier is told the point
        # it is naming from. Defaulted here rather than required from the
        # caller, because a coordinate run without it is not a harder task,
        # it is an unanswerable one -- but it stays overridable, so the
        # controlled comparison (street space, pose shown) is a config away.
        if self.action_space == ACTION_SPACE_COORDINATE:
            courier_kwargs.setdefault("show_pose", True)
        # Per-call evidence for the coordinate space, in the same log as the
        # hops: what was asked for, what became of it, and where the graph
        # ended up relative to the pawn.
        self.coordinate_walks = 0
        # Same bookkeeping for the pixel-goal space, kept apart from
        # ``coordinate_walks`` because the two spaces are never run together
        # (see the action_space guard above) and a shared counter would blur
        # which task a number in a report belongs to.
        self.pixel_walks = 0
        pixel_views = tuple(pixel_views)
        if pixel_views not in (PIXEL_VIEWS, PIXEL_VIEWS_QUAD):
            raise ValueError(
                f"pixel_views={pixel_views!r}; it is {PIXEL_VIEWS} or {PIXEL_VIEWS_QUAD}")
        if pixel_views == PIXEL_VIEWS_QUAD and action_space != ACTION_SPACE_PIXEL_GOAL_FRONT_REAR:
            raise ValueError("four pixel views need action_space='pixel_goal_front_rear'")
        self.pixel_views = pixel_views
        self.pixel_view_counts = {view: 0 for view in self.pixel_views}
        # Pedestrian routing: on, a pixel names a destination and the pawn is
        # walked there along certified pedestrian ways (see
        # ``_walk_to_pixel_along_pedestrian_ways``); off, the engine walks the
        # straight line to the pixel or refuses it, as it always did.
        self.pedestrian_routing = False
        self.pedestrian_nudges = 0
        self.pedestrian_replans = 0
        self._ground_z_cm: float | None = None
        self._active_pixel_view_pair: ObserveViewsResponse | None = None
        self._active_pixel_view_metadata: dict[
            str, dict[str, Any]] | None = None
        # The chord from the start pose to the end pose of the last walk.
        # Legacy street/forward observations turn the pawn to aim cameras, so
        # those modes still need this instead of the mutable pose yaw.  The
        # atomic front/rear capture does not turn the pawn; in that mode the
        # response pose yaw is the front camera's authoritative direction.
        self._walked_bearing: float | None = None
        # The latest continuity-aware graph match beside the unconstrained
        # nearest-node counterfactual. It is copied into the walk audit row.
        self._last_pose_match: dict[str, Any] | None = None
        #: Album path -> the yaw the camera was aimed at when it was taken.
        #: The pawn is turned to aim, so it is also the pawn's own yaw at the
        #: shutter -- which is why ``facing`` cannot be read off the pose.
        self.frame_yaws: dict[str, float] = {}
        #: How long reset() waits for a busy instance before giving up.
        self.episode_busy_timeout_s = 1800.0
        #: How long to leave an instance alone after it stopped answering,
        #: before asking the pool for a seat again.
        #:
        #: Sixty seconds, not five. A wedged engine is not repaired by asking
        #: it again -- it is repaired by the keeper noticing and rebuilding
        #: it, which is a four-minute cold boot. Measured: three attempts five
        #: seconds apart gave up at 18:50:34 and the engine came back at
        #: 18:56:40, so the trainer died of impatience six minutes before the
        #: fix landed. Patience here is free; the alternative is losing every
        #: step since the last checkpoint.
        self.reseat_pause_s = 60.0
        #: How many times one episode may re-seat while trying to open.
        #: Eight at a minute apart is eight minutes, which covers the cold
        #: boot the keeper needs with room to spare. Still bounded: an engine
        #: that cannot spawn after that is not one this episode can wait out.
        self.max_reseats = 8
        # Lease plumbing: a pool is leased lazily (the fleet may still be
        # launching when the env object is built); a bare client is used as
        # given and never "released".
        if hasattr(pool_or_client, "lease_embodied"):
            self._pool = pool_or_client
            self._lease: Any = None
            self._client: UERenderClient | None = None
        else:
            self._pool = None
            self._lease = None
            self._client = pool_or_client
        # The engine's own clock quantum, learned from /episode.
        self.fixed_dt: float | None = None
        # Where the pawn really is, per the last Track B response.
        self.ue_pose: Pose | None = None
        # The per-hop I/O evidence contract (spec 3b): one dict per /walk.
        self.embodied_log: list[dict[str, Any]] = []
        # How long this episode spent queueing behind another one on its
        # instance. Not an error -- it is the fleet being smaller than the
        # rollout width -- but it is the difference between "UE is slow" and
        # "UE was busy", and only one of those is fixed by buying more dt.
        self.busy_waits = 0
        self.busy_wait_seconds = 0.0
        # The bake's camera unless the caller renders at serving size.
        self.street_camera = street_camera or STREET_CAMERA
        # Same containment posture as the live env's observation path: a dead
        # observe degrades frames to album mode, never physics -- but /walk
        # failures PROPAGATE, because in this mode UE owns the physics and
        # there is nothing honest to degrade to.
        self.live_degraded = False
        self._episode_open = False
        super().__init__(network, album_root=self.live_album.root,
                         **courier_kwargs)

    @property
    def episode_id(self) -> str:
        return self.live_album.episode_id

    # ── the engine connection ────────────────────────────────────────────────

    def _ue(self) -> UERenderClient:
        if self._client is None:
            self._lease = self._pool.lease_embodied(self.episode_id)
            self._client = self._lease.__enter__()
        return self._client

    def _release_lease(self) -> None:
        if self._lease is not None:
            lease, self._lease = self._lease, None
            self._client = None
            lease.__exit__(None, None, None)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self.embodied_log = []
        self.live_degraded = False
        self.busy_waits = 0
        self.busy_wait_seconds = 0.0
        self.coordinate_walks = 0
        self.pixel_walks = 0
        self.pixel_view_counts = {view: 0 for view in self.pixel_views}
        self.pedestrian_nudges = 0
        self.pedestrian_replans = 0
        self._ground_z_cm = None
        self._active_pixel_view_pair = None
        self._active_pixel_view_metadata = None
        self._walked_bearing = None
        self._last_pose_match = None
        self.frame_yaws = {}
        node = self.network.nodes[self.node_id]
        request = EpisodeRequest(
            episode_id=self.episode_id,
            map_name=self.network.map_name,
            agent=AgentSpec(
                # The embodiment's cruising speed, applied once at spawn
                # (SetMaxSpeed). The tired slowdown does not reach UE in v1.
                speed_cm_s=float(self.embodiment.speed_cm_s),
                eye_z_cm=STREET_EYE_CM,
                camera=self.street_camera,
            ),
            # Spawn at the reset node's coordinates: the graph position and
            # the pawn agree from the first frame.
            spawn=Pose(x_cm=node.x_cm, y_cm=node.y_cm,
                       z_cm=self.spawn_z_cm, yaw_deg=0.0),
        )
        response = self._episode_when_free(request)
        self.fixed_dt = response.fixed_dt
        self.ue_pose = response.pose
        self._episode_open = True
        self._check_walk_geometry()

    def _check_walk_geometry(self) -> None:
        """Can this arrival radius be hit at this tick chunk and speed?

        Arrival is tested at chunk boundaries only, so a chunk carries the
        pawn ``speed * fixed_dt * tick_chunk`` between looks. A radius smaller
        than that is invisible: the pawn steps over the target and the walk
        reports stuck, which reads as a navigation failure and is arithmetic.
        Warned rather than raised -- the run is still meaningful, and taking a
        training job down over a tuning mistake is worse than saying so.
        """
        if not self.fixed_dt:
            return
        per_chunk = (self.embodiment.speed_cm_s or 140.0) * self.fixed_dt * self.tick_chunk
        if per_chunk > self.arrive_cm:
            logger.warning(
                "walk geometry: one chunk of %d tick(s) carries %.0f cm at "
                "dt=%.3f, but arrive_cm is %.0f -- the pawn can step over the "
                "target between arrival checks and report stuck. Lower "
                "tick_chunk or raise arrive_cm.",
                self.tick_chunk, per_chunk, self.fixed_dt, self.arrive_cm)

    def _episode_when_free(self, request: EpisodeRequest) -> Any:
        """Open the episode, waiting out a busy instance.

        ``busy`` is transient by contract (spec §3): another episode still
        holds this instance. Leases serialize episodes inside one process,
        but a trainer with several worker processes holds several lease
        views -- the development workstation measured exactly that, four concurrent GRPO
        episodes against one Paris instance, and ServiceBusy ended the run
        at the first rollout. Wait our turn instead.
        """
        deadline = time.monotonic() + self.episode_busy_timeout_s
        delay = 1.0
        reseats = 0
        while True:
            try:
                return self._ue().episode(request)
            except ServiceBusy:
                if time.monotonic() >= deadline:
                    raise
                logger.info("instance busy for episode %s; retrying in %.0fs",
                            self.episode_id, delay)
                self.busy_waits += 1
                self.busy_wait_seconds += delay
                time.sleep(delay)
                delay = min(delay * 1.5, 15.0)
            except RenderServiceError as error:
                # Opening the episode is the OTHER call that can meet a wedged
                # instance, and it was the one left unguarded: `/walk` learned
                # to re-seat and `reset()` still took the training job down
                # with `render_failed: spawn failed: AssertionError:` -- the
                # engine answering /healthz green while every RPC into it
                # raises. Same recovery, same reason, and bounded, because an
                # instance that cannot spawn twice running is not one this
                # episode can wait out.
                if reseats >= self.max_reseats or self._pool is None:
                    raise
                if not (self._instance_is_gone(error)
                        or isinstance(error, RenderFailedError)):
                    raise
                reseats += 1
                logger.warning(
                    "episode %s could not open (%s); re-seating (%d/%d)",
                    self.episode_id, error, reseats, self.max_reseats)
                try:
                    self._release_lease()
                except Exception:  # noqa: BLE001 — it is already unusable
                    pass
                self._client = None
                self.embodied_log.append({
                    "recovery": "reseat_on_open", "attempt": reseats,
                    "error": f"{type(error).__name__}: {error}"})
                time.sleep(self.reseat_pause_s)

    def close(self) -> None:
        """End the embodied episode and give the instance back.

        Safe to call twice, and safe when the service has died under us: the
        despawn is best-effort (the fleet owns lifecycle and a new episode_id
        tears down a stale agent anyway), but the lease release is
        unconditional.
        """
        try:
            if self._episode_open and self._client is not None:
                self._client.episode_end(
                    EpisodeEndRequest(episode_id=self.episode_id))
        except (RenderServiceError, ProtocolViolation, OSError) as error:
            logger.warning(
                "episode_end failed for %s (%s: %s); the fleet's next "
                "/episode tears the agent down anyway",
                self.episode_id, type(error).__name__, error)
        finally:
            self._episode_open = False
            self._release_lease()

    def _respawn_at_current_node(self, *, reason: str) -> None:
        """Stand the pawn back on the node the graph believes in.

        Same-id /episode is the spec's idempotent re-spawn (despawn + spawn +
        apply embodiment), so no new wire surface is needed. Failure here is
        contained: the episode keeps its refusal semantics either way, and a
        dead backend already has its own degrade path.
        """
        node = self.network.nodes[self.node_id]
        try:
            self._ue().episode(EpisodeRequest(
                episode_id=self.episode_id,
                map_name=self.network.map_name,
                agent=AgentSpec(
                    speed_cm_s=float(self.embodiment.speed_cm_s),
                    eye_z_cm=STREET_EYE_CM,
                    camera=self.street_camera,
                ),
                spawn=Pose(x_cm=node.x_cm, y_cm=node.y_cm,
                           z_cm=self.spawn_z_cm, yaw_deg=0.0),
            ))
            self.embodied_log.append({
                "recovery": "respawn", "after": reason,
                "node": self.node_id,
                "node_xy": (node.x_cm, node.y_cm),
            })
        except (RenderServiceError, ProtocolViolation, OSError) as error:
            logger.warning(
                "recovery respawn after %s failed for %s (%s: %s); later "
                "walks start from the pawn's stranded pose",
                reason, self.episode_id, type(error).__name__, error)
            # Record it, because this is the one branch where the pose error
            # stops being bounded. Every other path re-anchors the pawn: an
            # arrived hop lands within arrive_cm of an absolute target, and a
            # successful respawn puts it back on the node. A FAILED respawn
            # leaves it stranded, and every subsequent walk starts from the
            # wrong place. Logging alone made that invisible -- the episode
            # gained no recovery row, so nothing downstream could count it.
            self.embodied_log.append({
                "recovery": "reopen_failed", "after": reason,
                "node": self.node_id,
                "error": f"{type(error).__name__}: {error}",
            })

    # ── the transition seam ──────────────────────────────────────────────────

    def _step_to(self, k: int) -> StepOutcome:
        """One waypoint along street ``k``, walked by the pawn.

        The single-hop atom, overridden whole; see the module docstring for
        what is delegated, what is copied, and why.
        """
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            # The stock refusal, wording and all (including the dead-end
            # coaching). No walk was attempted, so nothing embodied applies.
            return super()._step_to(k)
        row = rows[k]
        try:
            walk = self._walk_hop(row["node"])
        except InstanceMoved:
            return self._refuse(StepOutcome(
                ok=False, code="instance_moved",
                message=("Something went wrong out of your sight and you are "
                         "back at the last junction. Nothing you did caused "
                         "it. Read where you are and go on from there."),
            ))
        if not walk.arrived:
            outcome = "stuck" if walk.stuck else "timeout"
            if walk.stuck:
                # The way_blocked ledger: the pawn personally ran into
                # something, which is exactly what witnessed_blocks records.
                self.blocked_attempts += 1
                self.witnessed_blocks.add((self.node_id, row["node"]))
            code = "stuck" if walk.stuck else "walk_timeout"
            # The engine-measured price of the failed walk, floored at the
            # refusal floor every refused action pays.
            charge = max(walk.sim_seconds, REJECTED_ACTION_SECONDS)
            self._log_hop(row, walk, outcome)
            # Recovery re-spawn: a failed walk leaves the pawn wherever
            # physics stopped it while the graph stays at the junction, and
            # without repair every LATER walk starts from the wrong place —
            # measured on the development workstation as a 160 m pose error cascading through
            # the episode. /episode with the SAME id is the spec's idempotent
            # re-spawn, so the pawn is stood back on the node the graph
            # believes in; the refusal above still stands and still charges.
            self._respawn_at_current_node(reason=code)
            return self._refuse(StepOutcome(
                ok=False, code=code,
                message=(
                    f"You cannot get through along {row['street']}. You stop "
                    "where you are; you will have to go round."
                    if walk.stuck else
                    f"You give up partway along {row['street']}; the way is "
                    "taking far longer than it should."
                ),
            ), seconds=charge)
        # The arrived hop: stock ``_step_to`` bookkeeping (copied -- see the
        # module docstring), with the engine's numbers where the arithmetic
        # was. Obstacle and signal branches are structurally absent because
        # v1 refuses their configuration at the door.
        self.turns += 1
        seconds = walk.sim_seconds
        walked_m = walk.walked_cm / 100.0
        self._spend_stamina(walked_m)          # per metre, off the engine's own count
        self.arrived_from = self.node_id
        self.node_id = row["node"]
        self.sim_seconds += seconds
        self.walked_cm += walk.walked_cm
        self._issue()
        self._log_hop(row, walk, "arrived")
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=walked_m,
            message=(
                f"You walk {walked_m:.0f} m {row['heading']} along "
                f"{row['street']}."
            ),
        )

    # ── the coordinate action space ──────────────────────────────────────────
    #
    # The street space asks which of the handful of streets leaving this
    # junction to take. The coordinate space asks for a point, and the pawn
    # walks toward it under the navmesh -- so the courier has to derive a
    # position from the map and its own, which is the harder question and the
    # reason the space exists. The two are never offered together: a menu with
    # both lets an episode take the easy action and be reported under the hard
    # one's name.
    #
    # Where the courier IS, in this space, is wherever the last walk left the
    # pawn -- not a node. That one decision settles the rest of this section.
    # ``position()`` answers with the pawn, so the map's "you are here", the
    # distances beside each street, the door tolerance ``collect`` measures and
    # the coordinates the observation prints are all the same point. The graph
    # node is kept as well, because everything the environment can *say* is
    # node-shaped -- which street this is, its house numbers, what leaves it --
    # and it is re-derived from the pawn after every walk. The gap between the
    # two is measured and logged as ``snap_cm`` rather than assumed small.

    def movement_env_actions(self) -> tuple[str, ...]:
        if self.action_space == ACTION_SPACE_COORDINATE:
            return ("MOVE_TO_XY",)
        if self.action_space in (
                ACTION_SPACE_PIXEL_GOAL, ACTION_SPACE_PIXEL_GOAL_FRONT_REAR):
            return ("MOVE_TO_PIXEL",)
        return super().movement_env_actions()

    def tool_limits(self) -> dict[str, Any]:
        limits = dict(super().tool_limits())
        limits["max_step_m"] = _metres(self.max_step_m)
        return limits

    def tools_for_prompt(self) -> list[Any]:
        """Use the view-aware same-name declaration only in dual mode."""
        tools = super().tools_for_prompt()
        if not self.front_rear_pixel_goal_mode:
            return tools
        from embodiedbench.agent.courier.tools import (
            WALK_TO_PIXEL_FRONT_REAR, WALK_TO_PIXEL_QUAD)

        declared = (WALK_TO_PIXEL_QUAD if self.pixel_views == PIXEL_VIEWS_QUAD
                    else WALK_TO_PIXEL_FRONT_REAR)
        return [
            declared if tool.name == "walk_to_pixel" else tool
            for tool in tools
        ]

    @property
    def coordinate_mode(self) -> bool:
        return self.action_space == ACTION_SPACE_COORDINATE

    @property
    def pixel_goal_mode(self) -> bool:
        return self.action_space in (
            ACTION_SPACE_PIXEL_GOAL, ACTION_SPACE_PIXEL_GOAL_FRONT_REAR)

    @property
    def front_rear_pixel_goal_mode(self) -> bool:
        return self.action_space == ACTION_SPACE_PIXEL_GOAL_FRONT_REAR

    @property
    def pose_tracked_mode(self) -> bool:
        """True in every action space where the courier is a pawn standing
        wherever its last walk left it, rather than always exactly on a
        graph node. Both non-street spaces share this -- they differ only in
        how a target is NAMED, never in what "where am I" means once one is
        walked to."""
        return self.coordinate_mode or self.pixel_goal_mode

    def _here_cm(self) -> tuple[float, float]:
        """Where the courier actually stands: the pawn, once the engine says.

        Before the first ``/episode`` there is no pawn, and the node the graph
        starts on is the honest answer -- it is where the pawn is about to be
        spawned.
        """
        pose = getattr(self, "ue_pose", None)
        if pose is None:
            return CourierEnv.position(self)
        return (float(pose.x_cm), float(pose.y_cm))

    def position(self, node_id: str | None = None) -> tuple[float, float]:
        """Where a node is, or -- with no argument -- where the courier is.

        Only the no-argument case changes, and only in the coordinate space:
        there the courier is a pawn standing anywhere, and every caller asking
        "where is the courier" should get that rather than the nearest
        junction. Asked about a named node this is the stock lookup, so the
        graph's own geometry is untouched.
        """
        if node_id is not None or not self.pose_tracked_mode:
            return super().position(node_id)
        return self._here_cm()

    def facing(self) -> float | None:
        """Which way the courier is looking in the photographs it receives.

        An atomic front/rear capture preserves the pawn pose and defines the
        front image at exactly ``pose.yaw_deg``.  Navigation-relative labels
        must therefore use that same yaw.  A curved controller path can end
        facing along its final segment while its start-to-end chord differs by
        tens of degrees; using the chord would make the phone and FPV disagree.

        Legacy street/forward ``/observe`` calls do turn the pawn to aim each
        photograph.  In those modes pose yaw is capture residue, so they keep
        using the measured start-to-end chord as before.
        """
        if not self.pose_tracked_mode:
            return super().facing()
        if self.front_rear_pixel_goal_mode:
            pose = getattr(self, "ue_pose", None)
            if pose is not None:
                return float(pose.yaw_deg) % 360.0
        if self._walked_bearing is None:
            return super().facing()
        return self._walked_bearing % 360.0

    def _nearest_node(self, point_cm: tuple[float, float]) -> tuple[str, float]:
        """The globally closest graph node, for initialization and addresses.

        Online movement must use :meth:`_match_pose_node` instead.  A global
        nearest lookup has no continuity evidence and may select a different,
        disconnected road merely because its survey centreline is closer.
        """
        node = min(self.network.nodes,
                   key=lambda n: math.dist(self.position(n), point_cm))
        return node, math.dist(self.position(node), point_cm)

    @staticmethod
    def _bearing_gap_deg(first: float, second: float) -> float:
        """Smallest unsigned separation between two compass bearings."""
        return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

    def _local_pose_match_candidates(
        self,
        start: str,
        *,
        movement_cm: float | None = None,
    ) -> dict[str, tuple[int, str | None, float]]:
        """Connected candidates as ``node -> (hops, first, progress_cm)``.

        A fixed hop radius is not a metric radius.  It happened to work for
        the old street graph, whose edges are long, but held the map match at
        the pickup after a perfectly valid six-to-ten-metre UE walk on the
        one-metre trusted Recast lattice.  The phone then joined the pawn to a
        stale node with an unaudited chord and appeared to reverse its route.

        With a measured movement, Dijkstra expansion follows the connected
        pedestrian graph far enough to cover that movement plus a small
        endpoint/sampling margin.  With no measurement, retain the original
        three-hop neighbourhood.  Disconnected geometry is never considered
        in either case.
        """
        progress_limit_cm = (
            None
            if movement_cm is None
            else max(0.0, float(movement_cm)) + POSE_MATCH_SEARCH_MARGIN_CM
        )
        found: dict[str, tuple[int, str | None, float]] = {
            start: (0, None, 0.0),
        }
        frontier: list[tuple[float, int, str]] = [(0.0, 0, start)]
        while frontier:
            progress_to_node, hops, node_id = heapq.heappop(frontier)
            recorded = found.get(node_id)
            if recorded is None or progress_to_node > recorded[2] + 1e-9:
                continue
            first_to_node = recorded[1]
            for neighbour in sorted(self.network.nodes[node_id].neighbours):
                edge_cm = math.dist(
                    self.position(node_id), self.position(neighbour))
                next_hops = hops + 1
                next_progress = progress_to_node + edge_cm
                if (next_hops > POSE_MATCH_MIN_HOPS
                        and (progress_limit_cm is None
                             or next_progress > progress_limit_cm)):
                    continue
                next_first = (
                    neighbour if first_to_node is None else first_to_node)
                previous = found.get(neighbour)
                candidate_key = (next_progress, next_hops, next_first)
                previous_key = (
                    (previous[2], previous[0], previous[1] or "")
                    if previous is not None else None
                )
                if previous_key is not None and candidate_key >= previous_key:
                    continue
                found[neighbour] = (
                    next_hops, next_first, next_progress)
                heapq.heappush(
                    frontier, (next_progress, next_hops, neighbour))
        return found

    def _match_pose_node(
        self,
        point_cm: tuple[float, float],
        *,
        movement_cm: float | None = None,
    ) -> tuple[str, float]:
        """Map-match a pawn landing without losing topological continuity.

        Observation distance is still the primary evidence.  Unlike a global
        nearest-node snap, candidates must be reachable from the previous
        match, and moving to a candidate is checked against both the direction
        and distance UE says the pawn actually travelled.  Direction compares
        the previous-to-candidate displacement, not the candidate path's first
        edge: the latter is merely a one-metre sampling detail on a dense
        Recast lattice and can point sideways while a longer UE action has
        already entered a certified crossing.  A folded route can put a
        three-hop post-corner node physically close to the approach; without
        the distance term, one six-metre action can be labelled as sixteen
        metres of graph progress and make the phone announce a turn the pawn
        has not taken.  The progress penalty is soft because the pawn may start
        between graph nodes.  The unconstrained nearest result is retained in
        ``_last_pose_match`` as counterfactual telemetry.
        """
        global_node, global_gap = self._nearest_node(point_cm)
        previous = self.node_id
        if previous not in self.network.nodes:
            self._last_pose_match = {
                "method": "global_recovery",
                "previous_node": previous,
                "selected_node": global_node,
                "selected_gap_cm": round(global_gap, 1),
                "global_nearest_node": global_node,
                "global_nearest_gap_cm": round(global_gap, 1),
                "local_candidate_count": 0,
                "prevented_global_jump": False,
            }
            return global_node, global_gap

        candidates = self._local_pose_match_candidates(
            previous, movement_cm=movement_cm)
        scored: list[tuple[float, float, int, str, float, float]] = []
        for node_id, (hops, first_hop, graph_progress_cm) in candidates.items():
            gap = math.dist(self.position(node_id), point_cm)
            score = gap
            progress_excess_cm = 0.0
            if first_hop is not None:
                score += POSE_MATCH_SWITCH_COST_CM
                if movement_cm is not None:
                    progress_excess_cm = max(
                        0.0, graph_progress_cm - float(movement_cm))
                    score += (POSE_MATCH_PROGRESS_COST_PER_CM
                              * progress_excess_cm)
                if self._walked_bearing is not None:
                    previous_point = self.position(previous)
                    candidate_point = self.position(node_id)
                    # Usually the net graph displacement is the honest
                    # counterpart of UE's measured start-to-end chord.  Two
                    # distinct graph states may occasionally be co-located at
                    # a junction; only there fall back to the first edge so
                    # the heading evidence remains defined.
                    heading_point = (
                        candidate_point
                        if math.dist(previous_point, candidate_point) > 1e-6
                        else self.position(first_hop)
                    )
                    graph_bearing = bearing_deg(
                        previous_point, heading_point)
                    heading_gap = min(
                        self._bearing_gap_deg(self._walked_bearing, graph_bearing),
                        90.0,
                    )
                    score += (POSE_MATCH_HEADING_COST_CM
                              * heading_gap / 90.0)
            scored.append((
                score, gap, hops, node_id, graph_progress_cm,
                progress_excess_cm,
            ))
        (_score, selected_gap, selected_hops, selected,
         selected_graph_progress_cm, selected_progress_excess_cm) = min(scored)
        self._last_pose_match = {
            "method": POSE_MATCH_METHOD,
            "heading_reference": "previous_to_candidate_net_displacement",
            "previous_node": previous,
            "selected_node": selected,
            "selected_gap_cm": round(selected_gap, 1),
            "selected_hops": selected_hops,
            "movement_cm": (
                round(float(movement_cm), 1)
                if movement_cm is not None else None
            ),
            "selected_graph_progress_cm": round(
                selected_graph_progress_cm, 1),
            "selected_progress_excess_cm": round(
                selected_progress_excess_cm, 1),
            "global_nearest_node": global_node,
            "global_nearest_gap_cm": round(global_gap, 1),
            "local_candidate_count": len(candidates),
            "search_progress_limit_cm": (
                round(float(movement_cm) + POSE_MATCH_SEARCH_MARGIN_CM, 1)
                if movement_cm is not None else None
            ),
            "prevented_global_jump": selected != global_node,
        }
        return selected, selected_gap

    def walk_to_xy(self, x: float, y: float) -> StepOutcome:
        """Walk toward a named point, as far as one step is allowed to carry.

        The three cases, and the reason each is what it is:

        * **Too near.** A point inside the arrival radius is the one the
          courier is already standing on. Walking to it would burn a turn to
          arrive where it started, so it is refused with the wording the
          manual declares.
        * **Too far.** Refused, and refused at the moment this call runs
          rather than when it was written -- so the second and third calls of
          a chunk are judged against the position the first two walks left
          the courier at, which is the position they were named relative to.
        * **Unwalkable.** The navmesh gets as far as it can and reports
          stuck; the pawn keeps the ground it covered and the refusal says
          there is no way through. No re-spawn: in this space the pawn's
          position is the truth and standing it back on a node it may be
          twenty metres from would be the desynchronisation the re-spawn
          exists to prevent, applied backwards.
        """
        if not self.coordinate_mode:
            # Unreachable through the menu -- the tool is not offered -- so
            # this is a caller wiring the two spaces together, and the halves
            # that make this method honest (the pawn being the courier's
            # position, the frame keys carrying the vantage, facing measured
            # off the walk) are all switched off under `street`.
            raise RuntimeError(
                "walk_to_xy needs action_space='coordinate'; this episode is "
                f"running {self.action_space!r}.")
        try:
            asked = (float(x) * 100.0, float(y) * 100.0)
        except (TypeError, ValueError):
            return self._refuse(StepOutcome(
                ok=False, code="bad_coordinate",
                message=("A point is two numbers, metres north and metres "
                         "east: walk_to_xy(-267.1, 97.8)."),
            ))
        if not all(math.isfinite(v) for v in asked):
            self._log_refused_point(asked, self._here_cm(), "bad_coordinate")
            return self._refuse(StepOutcome(
                ok=False, code="bad_coordinate",
                message=("A point is two ordinary numbers, metres north and "
                         "metres east: walk_to_xy(-267.1, 97.8)."),
            ))
        here = self._here_cm()
        reach = math.dist(here, asked)
        if reach <= self.arrive_cm:
            # Names the mistake rather than the rule. Measured on Qwen3-VL-4B:
            # 83 of 205 coordinate turns were this, and every one of them was
            # the model typing back the two numbers the observation had just
            # given it for its own position -- reasoning correctly about where
            # it wanted to go ("the next junction is 18 m north-east") and then
            # writing down where it already was. "Name somewhere you are not"
            # is true and was not enough; a courier repeating a refusal word
            # for word has not understood which word was wrong.
            self._log_refused_point(asked, here, "already_here")
            return self._refuse(StepOutcome(
                ok=False, code="already_here",
                message=(
                    f"{_point(asked)} is the point you are standing on -- the "
                    "same two numbers this turn gave you for your own "
                    "position. Naming it again does not move you. Decide how "
                    "far north and how far east you want to go, ADD that to "
                    "each of your numbers, and name the result."
                ),
            ))
        cap_cm = self.max_step_m * 100.0
        if reach > cap_cm:
            # Refused, not clamped, and refused HERE -- at the moment this
            # call runs, against the position it runs from.
            #
            # A chunk of three waypoints is checked one at a time as it is
            # reached, never all three up front: the second and third are
            # named relative to a position the courier has not walked to yet,
            # so judging them against the position it stood at when it wrote
            # them measures a step it never asked for. The turn stops at the
            # first refusal either way, so a chunk that opens well and drifts
            # loses only its tail.
            #
            # Clamping was the earlier behaviour and it hid the mistake:
            # a courier that asked for 40 m and was silently carried 1.5 m
            # cannot tell the two apart from where it lands, and neither can
            # a reader of the log.
            self._log_refused_point(asked, here, "too_far")
            return self._refuse(StepOutcome(
                ok=False, code="too_far",
                message=(
                    f"{_point(asked)} is {reach / 100.0:.1f} m away and one "
                    f"step is at most {_metres(self.max_step_m)} m. Name a "
                    "point on the way there instead -- you can take several "
                    "steps, and each one starts from where the last left you."
                ),
            ))
        target = asked
        self.coordinate_walks += 1
        try:
            walk = self._walk_to_point(*target)
        except InstanceMoved as error:
            # The instance died under this call and the episode is now open on
            # another one, standing at the node the graph believes in. The
            # call itself did not happen.
            self._log_refused_point(asked, here, "instance_moved")
            return self._refuse(StepOutcome(
                ok=False, code="instance_moved",
                message=("Something went wrong out of your sight and you are "
                         "back at the last junction. Nothing you did caused "
                         "it. Read where you are and go on from there."),
            ))
        # The distance covered is measured between two poses this env holds,
        # not taken from the service's own count.
        #
        # Measured on the development workstation, 13 accepted walks in one episode: every one
        # moved between 0.5 and 4.7 m by its own start and end pose, and every
        # one reported walked_cm = 0.00. The episode's route quality therefore
        # read "walked 0 m" for a courier that had covered fifteen. Both
        # numbers come back from the same response, so this is not a baseline
        # I picked badly -- it is the count disagreeing with the poses beside
        # it, and the poses are the ones the arrival test and the next walk
        # both use.
        #
        # The service's figure is kept in the log rather than discarded: the
        # gap between the two is the evidence for whatever is wrong in the
        # walk loop's per-chunk accumulation, and dropping it would hide the
        # defect this line is working around.
        landed_at = self._here_cm()
        walked_m = math.dist(here, landed_at) / 100.0
        # Every outcome moved the pawn some distance, and in this space that
        # distance is kept: there is no re-spawn undoing it, the next call
        # starts from where it left off, and a body that walked forty metres
        # into a dead end has walked forty metres. The stock hop drops them
        # because it puts the pawn back where it started.
        self._spend_stamina(walked_m)
        self.walked_cm += walked_m * 100.0
        # Measured, not assumed to be the bearing that was asked for: a
        # navmesh route round a corner ends the courier facing along the last
        # leg of it, which is not the direction of the point it named.
        if math.dist(here, landed_at) > 1.0:
            self._walked_bearing = bearing_deg(here, landed_at)
        landed, snap_cm = self._match_pose_node(
            self._here_cm(), movement_cm=math.dist(here, landed_at))
        if landed != self.node_id:
            self.arrived_from = self.node_id
            self.node_id = landed
        self._log_point_walk(asked, target, walk, start=here,
                             landed=landed, snap_cm=snap_cm)
        if not walk.arrived:
            code = "stuck" if walk.stuck else "walk_timeout"
            if walk.stuck:
                self.blocked_attempts += 1
            # No ``_issue`` here, matching both stock refusal paths: the queue
            # is topped up and swept after the clock is charged, and ``_refuse``
            # charges it on the way out. Sweeping first would expire orders
            # against a clock this turn has not yet paid.
            return self._refuse(StepOutcome(
                ok=False, code=code, walked_m=walked_m,
                message=(
                    (f"There is no way to walk there. You get {walked_m:.0f} m "
                     f"and stop at {_point(self._here_cm())}; something is "
                     "across the way. Read the map again and aim at somewhere "
                     "a person could walk to."
                     if walk.stuck else
                     f"You give up {walked_m:.0f} m along, at "
                     f"{_point(self._here_cm())}; getting there is taking far "
                     "longer than it should.")
                ),
            ), seconds=max(walk.sim_seconds, REJECTED_ACTION_SECONDS))
        self.turns += 1
        self.sim_seconds += walk.sim_seconds
        self._issue()
        return StepOutcome(
            ok=True, moved=True, sim_seconds=walk.sim_seconds,
            walked_m=walked_m,
            message=(
                f"You walk {walked_m:.0f} m and stop at "
                f"{_point(self._here_cm())}."

            ),
        )

    def _log_refused_point(self, asked: tuple[float, float],
                           here: tuple[float, float], code: str) -> None:
        """A coordinate request that never became a walk.

        Deliberately NOT shaped like a hop -- no ``ticks``, so every existing
        aggregation goes on counting walks and only walks.

        It exists because leaving it out cost a whole evening's reading. 83 of
        205 turns in the first live run were refused here, and because a
        refusal returned before the hop log, the telemetry recorded the
        coordinate that caused them exactly nowhere: the only way to see what
        the policy had asked for was to parse it back out of the model's own
        reply text. That is the same shape as every other measurement failure
        on this system -- the zero I read was invisible, not absent.

        ``gap_m`` is the number the refusal turns on, so a reader can tell
        "it named its own position" from "it named somewhere a metre away"
        without re-deriving the arithmetic.
        """
        self.embodied_log.append({
            "kind": "coordinate_refused",
            "code": code,
            "asked_xy": (round(asked[0], 1), round(asked[1], 1)),
            "from_xy": (round(here[0], 1), round(here[1], 1)),
            "gap_m": round(math.dist(here, asked) / 100.0, 2),
        })

    def _log_point_walk(self, asked: tuple[float, float],
                        target: tuple[float, float], walk: WalkResponse, *,
                        start: tuple[float, float],
                        landed: str, snap_cm: float) -> None:
        """One coordinate walk, in the same log shape as a hop.

        Same keys where the meaning survives -- ``ticks``, ``sim_seconds``,
        ``walked_cm``, ``end_pose``, ``outcome`` -- so every aggregation that
        already reads ``embodied_log`` keeps working without being told this
        action space exists. ``pose_error_cm`` keeps its name and changes its
        referent: it is still "how far the pawn ended from what it was aimed
        at", which here is the point it named rather than a node.
        """
        graph_seconds = (math.dist(start, target)
                         / max(self.travel_speed_cm_s(), 1e-6))
        self.embodied_log.append({
            "kind": "coordinate",
            "asked_xy": (round(asked[0], 1), round(asked[1], 1)),
            "target_xy": (round(target[0], 1), round(target[1], 1)),
            "chord_m": round(math.dist(start, target) / 100.0, 3),
            "graph_seconds": round(graph_seconds, 4),
            "ticks": walk.ticks,
            "sim_seconds": walk.sim_seconds,
            "walked_cm": round(math.dist(start, (walk.pose.x_cm, walk.pose.y_cm)), 2),
            # What the service counted, beside what the poses say. They
            # disagree, and the disagreement is the evidence.
            "service_walked_cm": walk.walked_cm,
            # The identity that makes the disagreement checkable, once the
            # service reports both ends of its own walk: a path is never
            # shorter than the line it spans, so `service_walked_cm` below
            # `service_displacement_cm` is the count being wrong and not the
            # baseline being badly chosen. Absent from a service that predates
            # the field, which is why it is written as None rather than
            # computed from something else.
            "service_displacement_cm": (
                round(math.dist(
                    (walk.start_pose.x_cm, walk.start_pose.y_cm),
                    (walk.pose.x_cm, walk.pose.y_cm)), 2)
                if walk.start_pose is not None else None),
            "end_pose": walk.pose.to_dict(),
            "pose_error_cm": math.dist(
                (walk.pose.x_cm, walk.pose.y_cm), target),
            # Which junction the environment will describe from here, and how
            # far that junction is from where the courier is really standing.
            # The one number that says whether "the streets leaving this
            # junction" is a description of the courier's surroundings or of
            # somewhere down the road.
            "landed_node": landed,
            "snap_cm": round(snap_cm, 1),
            "map_match": (dict(self._last_pose_match)
                          if self._last_pose_match is not None else None),
            "outcome": ("arrived" if walk.arrived else
                        "stuck" if walk.stuck else "timeout"),
        })

    # ── the pixel-goal action space ───────────────────────────────────────────
    #
    # Where the coordinate space asks the courier to derive a metric position
    # from the map and its own, this one asks it to point at a spot in the
    # photograph it was just shown -- no map arithmetic, no notion of its own
    # position at all. The resolution (does this pixel's ray hit the ground,
    # does the hit land on the NavMesh) happens engine-side; the courier is
    # never told which of those failed, only that the point did not work,
    # for the same reason a coordinate walk that hits a wall is not told the
    # wall's shape -- a policy cannot learn to game a threshold it never sees.
    #
    # Everything downstream of a resolved target -- the pawn IS the position,
    # ``facing`` is measured off the walk, frames are keyed by vantage -- is
    # the same machinery ``pose_tracked_mode`` already turns on for the
    # coordinate space; this section adds only how a target gets NAMED.

    def walk_to_pixel(self, *args: Any, **kwargs: Any) -> StepOutcome:
        """Dispatch the exact call shape advertised by the active variant."""
        if not self.pixel_goal_mode:
            raise RuntimeError(
                "walk_to_pixel needs action_space='pixel_goal'; this episode "
                f"is running {self.action_space!r}.")
        try:
            decoded = self._decode_walk_to_pixel_call(args, kwargs)
            if isinstance(decoded, StepOutcome):
                return self._refuse(decoded)
            view, u, v = decoded
            return self._walk_to_pixel(view=view, u=u, v=v)
        except (PixelGoalPairIntegrityError, ProtocolViolation) as error:
            if self.front_rear_pixel_goal_mode:
                self._record_pixel_view_integrity(error)
            raise
        finally:
            if self.front_rear_pixel_goal_mode:
                self._clear_active_pixel_view_pair()

    def _decode_walk_to_pixel_call(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[str | None, Any, Any] | StepOutcome:
        bad = StepOutcome(
            ok=False, code="bad_pixel",
            message=(
                "Use exactly the pixel arguments shown by this episode's "
                "walk_to_pixel action."),
        )
        if not self.front_rear_pixel_goal_mode:
            if len(args) == 2 and not kwargs:
                return None, args[0], args[1]
            if not args and set(kwargs) == {"u", "v"}:
                return None, kwargs["u"], kwargs["v"]
            return bad

        if len(args) == 3 and not kwargs:
            view, u, v = args
        elif not args and set(kwargs) == {"view", "u", "v"}:
            view, u, v = kwargs["view"], kwargs["u"], kwargs["v"]
        elif ((len(args) == 2 and not kwargs
               and not isinstance(args[0], str))
              or (not args and set(kwargs) == {"u", "v"})):
            return StepOutcome(
                ok=False, code="missing_pixel_view",
                message=(
                    "This action needs the photograph label from this turn."),
            )
        else:
            return bad
        if view not in self.pixel_views:
            return StepOutcome(
                ok=False, code="unknown_pixel_view",
                message=(
                    "The selected view is not part of this turn's photographs."),
            )
        return view, u, v

    def _walk_to_pixel(
            self, *, view: str | None, u: Any, v: Any) -> StepOutcome:
        """Walk toward a point picked in the current forward photograph.

        There is no client-side "too far" or "already here" refusal here,
        unlike ``walk_to_xy``: the courier never learns a metric position to
        compare a candidate against, only a normalized pixel, so the whole
        geometric judgement -- does this ray even meet the ground, does the
        hit sit on the NavMesh -- happens on the engine side. The raw
        diagnostic comes back as ``resolved.rejection_reason``; this method
        translates it to a coarse causal class and keeps the raw value in
        telemetry.
        """
        binding: dict[str, Any] | None = None
        if self.front_rear_pixel_goal_mode:
            selected = self._selected_pixel_view(view)
            if isinstance(selected, StepOutcome):
                return self._refuse(selected)
            _pair, _observed, binding = selected
            self.pixel_view_counts[view] += 1
        try:
            u_f, v_f = float(u), float(v)
        except (TypeError, ValueError):
            return self._refuse(StepOutcome(
                ok=False, code="bad_pixel",
                message=(
                    "The selected pixel coordinates were not numbers."
                    if self.front_rear_pixel_goal_mode else
                    "A pixel is two numbers between 0 and 1, across "
                    "then down: walk_to_pixel(0.5, 0.8)."
                ),
            ))
        if not (math.isfinite(u_f) and math.isfinite(v_f)):
            return self._refuse(StepOutcome(
                ok=False, code="bad_pixel",
                message=(
                    "The selected pixel coordinates were not finite numbers."
                    if self.front_rear_pixel_goal_mode else
                    "A pixel is two ordinary numbers between 0 and 1: "
                    "walk_to_pixel(0.5, 0.8)."
                ),
            ))
        if not (0.0 <= u_f <= 1.0 and 0.0 <= v_f <= 1.0):
            self._log_refused_pixel(
                (u_f, v_f), "out_of_range", selected=binding)
            return self._refuse(StepOutcome(
                ok=False, code="out_of_range",
                message=(
                    "The selected point is outside the selected photograph."
                    if self.front_rear_pixel_goal_mode else
                    f"({u_f:.2f}, {v_f:.2f}) is outside the photograph -- "
                    "both numbers have to be between 0 and 1. Pick a point "
                    "inside the picture you were just shown."
                ),
            ))
        here = self._here_cm()
        self.pixel_walks += 1
        if self.pedestrian_routing:
            return self._walk_to_pixel_along_pedestrian_ways(
                u_f, v_f, binding=binding, here=here)
        try:
            walk = self._walk_to_pixel_point(
                u_f, v_f, selected=binding)
        except InstanceMoved:
            self._log_refused_pixel(
                (u_f, v_f), "instance_moved", selected=binding)
            return self._refuse(StepOutcome(
                ok=False, code="instance_moved",
                message=("Something went wrong out of your sight and you "
                         "are back at the last junction. Nothing you did "
                         "caused it. Read where you are and go on from "
                         "there."),
            ))
        if walk.resolved.rejection_reason is not None:
            engine_reason = walk.resolved.rejection_reason
            self._log_refused_pixel(
                (u_f, v_f), "unwalkable_pixel",
                engine_rejection_reason=engine_reason,
                selected=binding,
            )
            return self._refuse(StepOutcome(
                ok=False, code="unwalkable_pixel",
                message=_pixel_rejection_message(engine_reason),
            ))
        landed_at = self._here_cm()
        walked_m = math.dist(here, landed_at) / 100.0
        self._spend_stamina(walked_m)
        self.walked_cm += walked_m * 100.0
        if math.dist(here, landed_at) > 1.0:
            self._walked_bearing = bearing_deg(here, landed_at)
        landed, snap_cm = self._match_pose_node(
            self._here_cm(), movement_cm=math.dist(here, landed_at))
        if landed != self.node_id:
            self.arrived_from = self.node_id
            self.node_id = landed
        self._log_pixel_walk((u_f, v_f), walk, start=here,
                             landed=landed, snap_cm=snap_cm,
                             selected=binding)
        if not walk.arrived:
            code = "stuck" if walk.stuck else "walk_timeout"
            if walk.stuck:
                self.blocked_attempts += 1
            return self._refuse(StepOutcome(
                ok=False, code=code, walked_m=walked_m,
                message=(
                    (f"Movement toward the selected pixel was blocked after "
                     f"{walked_m:.0f} m."
                     if walk.stuck else
                     f"Movement toward the selected pixel timed out after "
                     f"{walked_m:.0f} m.")
                ),
            ), seconds=max(walk.sim_seconds, REJECTED_ACTION_SECONDS))
        self.turns += 1
        self.sim_seconds += walk.sim_seconds
        self._issue()
        return StepOutcome(
            ok=True, moved=True, sim_seconds=walk.sim_seconds,
            walked_m=walked_m,
            message=f"You walk {walked_m:.0f} m toward where you pointed.",
        )

    # ---- pedestrian routing: a pixel names a destination, the harness walks --
    def _pedestrian_snap(
        self, point_cm: tuple[float, float], radius_cm: float,
    ) -> tuple[str, float] | None:
        """The certified pedestrian node nearest a point, within a radius.

        The base environment has no certified graph; a subclass with one
        answers and turns ``pedestrian_routing`` on.
        """
        return None

    def _pedestrian_legs(
        self, start_node_id: str, end_node_id: str,
        avoid: Sequence[tuple[str, str]] = (),
    ) -> tuple[bool, list[Any]] | None:
        """(uses a marked crossing, the moves) between two certified nodes
        over the certified legs, never a leg in ``avoid``. Each move has a
        ``stop`` node and point and an ``aim`` node and point: the pawn aims
        at the end of a certified leg and stops at a node on it."""
        return None

    @property
    def max_leg_replans(self) -> int:
        """How many times one walk may go round a leg the engine refused."""
        return 0

    def _ground_z_here(self) -> float:
        if self._ground_z_cm is not None:
            return self._ground_z_cm
        pose = getattr(self, "ue_pose", None)
        if pose is None:
            return 0.0
        return float(pose.z_cm) - PAWN_FEET_BELOW_POSITION_CM

    def _pedestrian_destination(
        self, resolved: ResolvedPixel,
    ) -> StepOutcome | tuple[str, float, str]:
        """Which certified node a resolved pixel means, or why it means none.

        The engine says where the ray landed and on what. A pavement, a
        crossing or a paved island within reach of a certified node is that
        node; a kerb stone, a planter's foot or the base of a wall counts as
        the paving beside it; the carriageway counts only from the gutter;
        anything higher than knee height, the sky, and points far from any
        certified way are refused. The verdict on the straight path is not
        consulted: the walk that follows uses pedestrian ways, crossing on
        the marked crossing, so a point across the road is a destination
        like any other.
        """
        hit = resolved.raw_world_hit_cm
        if hit is None:
            return StepOutcome(
                ok=False, code="unwalkable_pixel",
                message=("The selected point is not on the ground: the sky, or "
                         "something too far up. Aim at paving."))
        surface = _surface_class(resolved.raw_hit_actor)
        height = resolved.raw_world_hit_z_cm
        if surface == "object":
            above = (None if height is None
                     else height - self._ground_z_here())
            if above is None or above > OBJECT_HIT_MAX_ABOVE_GROUND_CM:
                return StepOutcome(
                    ok=False, code="unwalkable_pixel",
                    message=("The selected point is on an object or a wall, "
                             "not on the ground. Aim at plain paving beside it."))
            radius = PEDESTRIAN_SNAP_CM
        elif surface == "road":
            radius = ROAD_SNAP_CM
        else:
            radius = PEDESTRIAN_SNAP_CM
            if height is not None:
                self._ground_z_cm = float(height)
        snap = self._pedestrian_snap(hit, radius)
        if snap is None:
            if surface == "road":
                message = ("That point is on the carriageway. Aim at the "
                           "pavement, or at the pavement or crossing on the "
                           "far side; the walk uses the marked crossing.")
            elif surface == "object":
                message = ("The selected point is on an object beside no "
                           "pavement. Aim at plain paving.")
            else:
                message = ("That point is not on a pedestrian way you can "
                           "reach from here. Aim at the pavement.")
            return StepOutcome(ok=False, code="unwalkable_pixel", message=message)
        node_id, gap = snap
        return node_id, gap, surface

    def _walk_leg(self, move: Any) -> Any:
        """One move through the client: aim at the certified leg's end, stop
        at the move's node; ``None`` when the engine refused the leg."""
        aim = tuple(move.aim_cm)
        stop = tuple(move.stop_cm)
        try:
            return self._ue().walk_world(
                aim[0], aim[1], camera=self.street_camera,
                stop_cm=None if move.stop == move.aim else stop)
        except Exception as error:  # noqa: BLE001 - the leg's verdict, logged
            reason = getattr(error, "reason", None)
            if reason is None:
                raise
            self.embodied_log.append({
                "kind": "pedestrian_leg_refused",
                "aim": move.aim, "stop": move.stop,
                "target_cm": [round(aim[0], 1), round(aim[1], 1)],
                "reason": reason,
            })
            return None

    def _nudge_onto(self, node_id: str) -> bool:
        """Set the pawn back onto its node when it stands a step or more, but
        not far, off it. ``True`` when the pawn is on the node afterwards
        (nudged, or never off it); ``False`` when it is too far off for a
        nudge -- the harness has lost the pawn."""
        anchor = self.position(node_id)
        here = self._here_cm()
        off = math.dist(here, anchor)
        if off <= NUDGE_MIN_CM:
            return True
        client = self._ue()
        if off > NUDGE_MAX_CM or not getattr(client, "can_nudge", False):
            self.embodied_log.append({
                "kind": "pedestrian_off_node", "node": node_id,
                "from_cm": [round(here[0], 1), round(here[1], 1)],
                "distance_cm": round(off, 1),
            })
            return False
        self.ue_pose = client.nudge_world(anchor[0], anchor[1])
        self.pedestrian_nudges += 1
        self.embodied_log.append({
            "kind": "pedestrian_nudge", "node": node_id,
            "from_cm": [round(here[0], 1), round(here[1], 1)],
            "to_cm": [round(anchor[0], 1), round(anchor[1], 1)],
            "distance_cm": round(off, 1),
        })
        return True

    def _execute_pedestrian_legs(
        self, legs: list[Any], *, end_node_id: str,
    ) -> dict[str, Any]:
        """Walk the moves in order, and go round a leg the engine refuses by
        planning again from the node the pawn stands on without it.

        The harness always knows which certified node the pawn stands on:
        the node of the graph before the first move, then each move's stop
        node once the engine reports the walk arrived. The engine's
        controller stops a short way from the point it walks to and its road
        check reads the path from where the pawn really stands, so the pawn
        is set back onto its node before every move (``_nudge_onto``) --
        the legs were certified from the nodes, and this keeps the walk on
        the legs that were certified. A pawn too far off its node to be set
        back, or a walk the engine did not finish, ends the walk where the
        pawn is; the node returned is then the nearest certified one.
        """
        walked_cm = 0.0
        sim_seconds = 0.0
        count = 0
        last_bearing: float | None = None
        complete = True
        node = self.node_id
        refused: list[tuple[str, str]] = []
        replans = 0
        queue = list(legs)
        while queue:
            move = queue.pop(0)
            if not self._nudge_onto(node):
                complete = False
                break
            here = self._here_cm()
            walk = self._walk_leg(move)
            if walk is None:
                refused.append((node, move.aim))
                replans += 1
                plan = (self._pedestrian_legs(node, end_node_id, refused)
                        if replans <= self.max_leg_replans else None)
                if plan is None:
                    complete = False
                    break
                queue = list(plan[1])
                continue
            walked_cm += float(walk.walked_cm)
            sim_seconds += float(walk.sim_seconds)
            count += 1
            last_bearing = bearing_deg(here, tuple(move.stop_cm))
            self.ue_pose = walk.pose
            if not walk.arrived:
                complete = False
                break
            short_by = math.dist(self._here_cm(), tuple(move.stop_cm))
            if short_by > NUDGE_MAX_CM:
                # The engine says it arrived, but not where the move stops.
                # The pawn stands wherever the controller left it: on the
                # nearest certified node within a nudge, if there is one,
                # from which the walk is planned again without this leg;
                # otherwise the harness has lost it and the walk ends here.
                self.embodied_log.append({
                    "kind": "pedestrian_stopped_short", "stop": move.stop,
                    "aim": move.aim, "distance_cm": round(short_by, 1),
                })
                snap = self._pedestrian_snap(self._here_cm(), NUDGE_MAX_CM)
                refused.append((node, move.aim))
                replans += 1
                plan = (self._pedestrian_legs(snap[0], end_node_id, refused)
                        if snap is not None and replans <= self.max_leg_replans else None)
                if plan is None:
                    complete = False
                    break
                node = snap[0]
                queue = list(plan[1])
                continue
            node = move.stop
        # The next observation, and the pixel resolved from it, come from the
        # node the walk ended on -- the pose the legs out of it were certified
        # from -- so the pawn is set back onto it now, not only before the
        # next leg.
        if count and not self._nudge_onto(node):
            complete = False
        return {"walked_cm": walked_cm, "sim_seconds": sim_seconds,
                "legs": count, "complete": complete, "bearing": last_bearing,
                "node": node, "replans": replans, "refused": refused}

    def _walk_to_pixel_along_pedestrian_ways(
        self, u: float, v: float, *, binding: dict[str, Any] | None,
        here: tuple[float, float],
    ) -> StepOutcome:
        """The four-view harness's walk: resolve the pixel, name the certified
        node it means, and walk there along certified ways."""
        try:
            walk = self._walk_to_pixel_point(
                u, v, selected=binding, resolve_only=True)
        except InstanceMoved:
            self._log_refused_pixel((u, v), "instance_moved", selected=binding)
            return self._refuse(StepOutcome(
                ok=False, code="instance_moved",
                message=("Something went wrong out of your sight and you "
                         "are back at the last junction. Nothing you did "
                         "caused it. Read where you are and go on from "
                         "there."),
            ))
        self.ue_pose = walk.pose
        resolved = walk.resolved
        verdict = self._pedestrian_destination(resolved)
        if isinstance(verdict, StepOutcome):
            self._log_refused_pixel(
                (u, v), "unwalkable_pixel",
                engine_rejection_reason=(
                    resolved.rejection_reason
                    or f"harness:{_surface_class(resolved.raw_hit_actor)}"),
                selected=binding)
            return self._refuse(verdict)
        node_id, gap_cm, surface = verdict
        if node_id == self.node_id:
            self._log_refused_pixel((u, v), "already_here", selected=binding)
            return self._refuse(StepOutcome(
                ok=False, code="already_here",
                message=("You are already standing where you pointed. Pick a "
                         "point further along the pavement.")))
        route = self._pedestrian_legs(self.node_id, node_id)
        if route is None:
            self._log_refused_pixel((u, v), "no_pedestrian_way", selected=binding)
            return self._refuse(StepOutcome(
                ok=False, code="no_pedestrian_way",
                message="No pedestrian way leads from here to that point."))
        uses_crossing, legs = route
        run = self._execute_pedestrian_legs(legs, end_node_id=node_id)
        self.pedestrian_replans += run["replans"]
        landed = self._here_cm()
        walked_m = run["walked_cm"] / 100.0
        # The clock is priced at the courier's declared walking speed (the
        # deadlines are). The engine's pawn walks at about 68 cm/s of its
        # own time, half that, so charging its seconds would make every
        # delivery late by construction; the engine's time is recorded
        # beside the charge.
        engine_seconds = run["sim_seconds"]
        charged_seconds = run["walked_cm"] / WALK_SPEED_CM_S
        self._spend_stamina(walked_m)
        self.walked_cm += run["walked_cm"]
        if run["bearing"] is not None:
            self._walked_bearing = run["bearing"]
        # The node the pawn stands on is what the walk tracked: the last
        # move's stop node once the engine said it arrived. Only a walk the
        # engine did not finish leaves the pawn between nodes; then it is
        # the nearest certified node within a nudge, or, further off than
        # that, the node the walk last knew (the next walk will find the
        # pawn off it and say so).
        landed_node = run["node"]
        snap_cm = math.dist(landed, self.position(landed_node))
        if not run["complete"] and snap_cm > NUDGE_MAX_CM:
            snap = self._pedestrian_snap(landed, NUDGE_MAX_CM)
            if snap is not None:
                landed_node, snap_cm = snap
        if landed_node != self.node_id:
            self.arrived_from = self.node_id
            self.node_id = landed_node
        self.embodied_log.append({
            "kind": "pixel_goal",
            "pixel_uv": (round(u, 4), round(v, 4)),
            "raw_world_hit_cm": resolved.raw_world_hit_cm,
            "raw_hit_actor": resolved.raw_hit_actor,
            "surface": surface,
            "destination_node": node_id,
            "destination_gap_cm": round(gap_cm, 1),
            "direct_path_legal": resolved.direct_path_legal,
            "legs": run["legs"],
            "replans": run["replans"],
            "refused_legs": [list(pair) for pair in run["refused"]],
            "uses_marked_crossing": uses_crossing,
            "complete": run["complete"],
            "ticks": run["legs"],
            "sim_seconds": charged_seconds,
            "engine_seconds": round(engine_seconds, 3),
            # ``walked_cm`` keeps the meaning it has in every other movement
            # record, the start-to-end displacement; the legs' length is
            # ``route_cm`` and is what the courier's own totals count.
            "walked_cm": round(math.dist(here, landed), 2),
            "route_cm": round(run["walked_cm"], 2),
            "end_pose": (self.ue_pose.to_dict()
                         if getattr(self, "ue_pose", None) is not None else None),
            "landed_node": landed_node,
            "snap_cm": round(snap_cm, 1),
            # how far the pawn stopped from the certified node it walked to:
            # the same field the engine's own walks report against their
            # NavMesh target, so the summary's max_pose_error_cm reads on
            "pose_error_cm": round(snap_cm, 2),
            "outcome": "arrived" if run["complete"] else "stuck",
            **(binding or {}),
            "controller_status": "arrived" if run["complete"] else "stuck",
        })
        if not run["complete"]:
            # The same contract as an engine walk that stalls: a refusal
            # with code ``stuck``, charged, the metres walked kept.
            self.blocked_attempts += 1
            return self._refuse(StepOutcome(
                ok=False, code="stuck", walked_m=walked_m,
                message=(f"You walk {walked_m:.0f} m along the pavement, but "
                         "could not get all the way to where you pointed. "
                         "Read where you are and go on from there."),
            ), seconds=max(charged_seconds, REJECTED_ACTION_SECONDS))
        self.turns += 1
        self.sim_seconds += charged_seconds
        self._issue()
        crossed = (" You cross the road on the marked crossing."
                   if uses_crossing else "")
        return StepOutcome(
            ok=True, moved=True, sim_seconds=charged_seconds,
            walked_m=walked_m,
            message=(f"You walk {walked_m:.0f} m along the pavement to where "
                     f"you pointed.{crossed}"))

    def _selected_pixel_view(
        self, view: str | None,
    ) -> tuple[ObserveViewsResponse, ObservedView, dict[str, Any]] | StepOutcome:
        pair = self._active_pixel_view_pair
        metadata = self._active_pixel_view_metadata
        if pair is None or metadata is None:
            return StepOutcome(
                ok=False, code="missing_pixel_view",
                message=(
                    "This action needs the photograph label from this turn."),
            )
        if (type(pair) is not ObserveViewsResponse
                or tuple(item.view for item in pair.views) != self.pixel_views
                or set(metadata) != set(self.pixel_views)):
            raise PixelGoalPairIntegrityError("incomplete_view_pair")
        selected = next(item for item in pair.views if item.view == view)
        private = metadata.get(view)
        if not isinstance(private, dict):
            raise PixelGoalPairIntegrityError("incomplete_view_pair")
        if private.get("view") != view:
            raise PixelGoalPairIntegrityError(
                "camera_snapshot_view_mismatch", audit=private)
        selected_group = selected.capture_group_id or pair.capture_group_id
        if private.get("capture_group_id") != selected_group:
            raise PixelGoalPairIntegrityError(
                "camera_snapshot_group_mismatch", audit=private)
        if private.get("camera_snapshot_id") != selected.camera_snapshot_id:
            raise PixelGoalPairIntegrityError(
                "camera_snapshot_stale", audit=private)
        expected = {
            "camera_intrinsics_id": selected.camera_intrinsics_id,
            "width": selected.width,
            "height": selected.height,
        }
        if any(private.get(key) != value for key, value in expected.items()):
            raise PixelGoalPairIntegrityError(
                "incomplete_view_pair", audit=private)
        return pair, selected, {
            "selected_view": view,
            "capture_group_id": selected_group,
            "camera_snapshot_id": selected.camera_snapshot_id,
        }

    def _walk_to_pixel_point(
        self, u: float, v: float, *, selected: dict[str, Any] | None = None,
        resolve_only: bool = False,
    ) -> WalkPixelResponse:
        """One /walk_pixel against the camera the current forward frame was
        taken from.

        Nothing turns the pawn between that /observe and this call, so the
        pose the engine resolves the pixel against is exactly the pose the
        courier's photograph was taken from -- see ``WalkPixelRequest``'s own
        docstring for why the wire needs no snapshot id to make that true.
        """
        request = WalkPixelRequest(
            episode_id=self.episode_id,
            pixel=PixelSpec(u=u, v=v),
            camera=self.street_camera,
            arrive_cm=self.arrive_cm,
            max_sim_seconds=self.max_walk_seconds,
            tick_chunk=self.tick_chunk,
            view=(selected["selected_view"] if selected is not None else None),
            capture_group_id=(selected["capture_group_id"]
                              if selected is not None else None),
            camera_snapshot_id=(selected["camera_snapshot_id"]
                                if selected is not None else None),
            resolve_only=resolve_only,
        )
        try:
            walk = self._ue().walk_pixel(request)
        except RenderServiceError as error:
            if selected is not None:
                # A bound pixel is meaningful only against this immutable
                # pair. Re-opening or moving instances invalidates it, so the
                # action is never replayed.
                raise
            if self._episode_is_lost(error):
                self._reopen_episode(f"walk_pixel: {type(error).__name__}")
                walk = self._ue().walk_pixel(request)
            elif self._instance_is_gone(error) and self._move_instance(
                    f"walk_pixel: {type(error).__name__}"):
                raise InstanceMoved(str(error)) from error
            else:
                raise
        if selected is not None:
            if walk.view != selected["selected_view"]:
                raise PixelGoalPairIntegrityError(
                    "camera_snapshot_view_mismatch",
                    audit=walk.to_dict(),
                )
            if walk.capture_group_id != selected["capture_group_id"]:
                raise PixelGoalPairIntegrityError(
                    "camera_snapshot_group_mismatch",
                    audit=walk.to_dict(),
                )
            if walk.camera_snapshot_id != selected["camera_snapshot_id"]:
                raise PixelGoalPairIntegrityError(
                    "camera_snapshot_stale",
                    audit=walk.to_dict(),
                )
        self.ue_pose = walk.pose
        return walk

    def _log_refused_pixel(
        self,
        pixel: tuple[float, float],
        code: str,
        *,
        engine_rejection_reason: str | None = None,
        selected: dict[str, Any] | None = None,
    ) -> None:
        """A pixel that never became a walk, in the coordinate space's
        ``_log_refused_point`` shape: no ``ticks``, so hop aggregations keep
        counting only hops."""
        record = {
            "kind": "pixel_refused",
            "code": code,
            "pixel_uv": (round(pixel[0], 4), round(pixel[1], 4)),
        }
        if engine_rejection_reason is not None:
            record["engine_rejection_reason"] = engine_rejection_reason
        if selected is not None:
            record.update(selected)
        self.embodied_log.append(record)

    def _log_pixel_walk(self, pixel: tuple[float, float],
                        walk: WalkPixelResponse, *,
                        start: tuple[float, float],
                        landed: str, snap_cm: float,
                        selected: dict[str, Any] | None = None) -> None:
        """One pixel-goal walk, in the same log shape as a hop.

        ``pose_error_cm`` is measured against ``accepted_target_cm`` -- the
        NavMesh-projected point the walk actually steered toward -- not the
        raw raycast hit, the same "measure against what was actually walked
        to" discipline ``_log_point_walk`` keeps.
        """
        accepted = walk.resolved.accepted_target_cm
        record = {
            "kind": "pixel_goal",
            "pixel_uv": (round(pixel[0], 4), round(pixel[1], 4)),
            "raw_world_hit_cm": walk.resolved.raw_world_hit_cm,
            "accepted_target_cm": accepted,
            "navmesh_adjustment_cm": walk.resolved.navmesh_adjustment_cm,
            "ticks": walk.ticks,
            "sim_seconds": walk.sim_seconds,
            "walked_cm": round(math.dist(start, (walk.pose.x_cm, walk.pose.y_cm)), 2),
            "service_walked_cm": walk.walked_cm,
            "end_pose": walk.pose.to_dict(),
            "pose_error_cm": (math.dist((walk.pose.x_cm, walk.pose.y_cm), accepted)
                              if accepted is not None else None),
            "landed_node": landed,
            "snap_cm": round(snap_cm, 1),
            "map_match": (dict(self._last_pose_match)
                          if self._last_pose_match is not None else None),
            "outcome": ("arrived" if walk.arrived else
                        "stuck" if walk.stuck else "timeout"),
            "navigation_request": self._pixel_navigation_request(pixel, walk),
        }
        if selected is not None:
            record.update(selected)
            record["controller_status"] = (
                "arrived" if walk.arrived else
                "stuck" if walk.stuck else "timeout")
        self.embodied_log.append(record)

    def _pixel_navigation_request(
            self, pixel: tuple[float, float],
            walk: WalkPixelResponse) -> dict[str, Any] | None:
        """The design plan §9.4 audit record for this pixel, in the schema
        ``embodiedbench/schemas`` already reserves for ``nav_pixel_goal`` --
        built here rather than left as an aspiration, so a resolved pixel
        walk is provably shaped the way the schema says one must be.

        ``None`` only when the pixel never resolved to a world hit at all;
        the schema's own validator requires ``raw_world_hit`` to record
        something, and there is nothing honest to put there.
        """
        if walk.resolved.raw_world_hit_cm is None:
            return None
        from embodiedbench.schemas.embodiment import DistanceSource, NavigationRequest
        from embodiedbench.schemas.environment import NavigationMode
        from embodiedbench.schemas.geometry import Vec3
        from embodiedbench.schemas.runtime import ImagePoint

        raw_x, raw_y = walk.resolved.raw_world_hit_cm
        accepted = walk.resolved.accepted_target_cm or (raw_x, raw_y)
        # This is an audit artifact, not load-bearing plumbing: a
        # malformed record (e.g. an episode_id with characters StableId's
        # pattern refuses) must not take an otherwise-successful walk down
        # with it. Logged rather than silently dropped, so a run that never
        # gets a navigation_request attached is a visible fact, not a
        # disappearing one.
        try:
            request = NavigationRequest(
                request_id=_stable_id(f"pxg-{self.episode_id}-{self.pixel_walks}"),
                mode=NavigationMode.NAV_PIXEL_GOAL,
                distance_source=DistanceSource.UE_GEOMETRY_TRACE,
                source_image_point=ImagePoint(u_norm=pixel[0], v_norm=pixel[1]),
                target_world=Vec3(x_cm=raw_x, y_cm=raw_y),
                # No camera-snapshot bookkeeping on the wire (see
                # ``WalkPixelRequest``'s docstring) -- this id binds the
                # audit record to the /observe call that produced the frame
                # the pixel was picked on, entirely on the Python side.
                camera_snapshot_id=_stable_id(
                    f"obs-{self.episode_id}-{len(self.frame_yaws)}"),
                camera_intrinsics_id=_stable_id(
                    f"cam-{self.street_camera.width}x{self.street_camera.height}"
                    f"-fov{round(self.street_camera.fov_deg)}"),
                raw_world_hit=Vec3(x_cm=raw_x, y_cm=raw_y),
                projected_target=Vec3(x_cm=accepted[0], y_cm=accepted[1]),
                validated_navigation_target=Vec3(x_cm=accepted[0], y_cm=accepted[1]),
                navmesh_adjustment_cm=walk.resolved.navmesh_adjustment_cm or 0.0,
                controller_path_points=[
                    Vec3(x_cm=point[0], y_cm=point[1], z_cm=point[2])
                    for point in (walk.resolved.controller_path_points_cm or ())
                ],
                controller_path_length_cm=(
                    walk.resolved.controller_path_length_cm),
                controller_path_direct_cm=(
                    walk.resolved.controller_path_direct_cm),
                controller_path_stretch_ratio=(
                    walk.resolved.controller_path_stretch_ratio),
                max_range_m=self.max_step_m,
            )
        except Exception as error:  # noqa: BLE001 — audit-only, never fatal
            logger.warning(
                "nav_pixel_goal audit record could not be built for episode "
                "%s (%s: %s); the walk itself is unaffected",
                self.episode_id, type(error).__name__, error)
            return None
        return request.model_dump(mode="json")

    def _instance_is_gone(self, error: Exception) -> bool:
        """Did this error mean "that instance is not there any more"?

        Different from ``_episode_is_lost``, which is a LIVE service saying it
        does not hold this episode. This is no service at all -- and until now
        it ended the training job, because a lease hands the env one instance
        and the env had nowhere else to go. Twice today: forty minutes in,
        ``ServiceUnreachable: ue-0 /walk unreachable (timed out)``, run over.
        """
        return isinstance(error, ServiceUnreachable) or (
            isinstance(error, RenderServiceError)
            and getattr(error, "code", None) == "engine_down")

    def _move_instance(self, reason: str) -> bool:
        """Give the dead instance back and take a seat on a healthy one.

        Only possible when the env was handed a POOL. With a bare client there
        is exactly one service and nothing to move to, which is the
        single-instance and test setup, so the error propagates as before.

        The walk that failed is NOT retried. ``/walk`` is the one
        non-idempotent endpoint, an unreachable service may or may not have
        moved the pawn, and re-issuing it against a fresh spawn would credit
        the episode with a walk nobody can say happened. The episode re-opens
        at the node the graph believes in and the call that hit the dead
        instance is refused, charged like any refusal.
        """
        if self._pool is None:
            return False
        logger.warning("instance lost under episode %s (%s); re-seating",
                       self.episode_id, reason)
        try:
            self._release_lease()
        except Exception as error:  # noqa: BLE001 — it is already dead
            logger.info("releasing the dead lease raised (%s)", error)
        self._client = None
        self._episode_open = False
        # A fleet of one is the fleet we actually run, and "move to another"
        # has nowhere to go there -- which is how a run kept dying with the
        # move already written. So the same instance is a legitimate
        # destination: `unreachable` is a timed-out or reset SOCKET, not a
        # death certificate, and the engine behind it has been up for hours in
        # every case measured. Give it a moment and re-seat.
        #
        # The lease was dropped above, so the pool decides where the seat
        # comes from -- another instance if one is healthy, this one if it is
        # all there is. The pool has already struck it, so it is picked last.
        time.sleep(self.reseat_pause_s)
        try:
            self._reopen_episode(f"instance lost: {reason}")
        except Exception as error:  # noqa: BLE001
            logger.warning("could not re-open %s on another instance (%s: %s)",
                           self.episode_id, type(error).__name__, error)
            self.embodied_log.append({
                "recovery": "move_failed", "after": reason,
                "error": f"{type(error).__name__}: {error}"})
            return False
        self.embodied_log.append({"recovery": "moved_instance", "after": reason,
                                  "node": self.node_id})
        return True

    def _episode_is_lost(self, error: Exception) -> bool:
        """Did this error mean "your episode is no longer on that instance"?

        Two shapes carry it: the service answering ``bad_request`` because the
        id is not the active one, and the motion backend answering
        ``render_failed`` because no agent is spawned. Both are recoverable --
        the env knows the episode it wants and where the courier stands on the
        graph, so it can re-open rather than take the training run down.
        """
        text = str(error).lower()
        return (
            isinstance(error, BadRequestError) and "not active" in text
        ) or (
            isinstance(error, RenderFailedError) and "no embodied agent" in text
        )

    def _reopen_episode(self, reason: str) -> None:
        """Re-establish the episode at the courier's current graph node."""
        logger.warning("episode %s was lost (%s); re-opening at %s",
                       self.episode_id, reason, self.node_id)
        self._episode_open = False
        node = self.network.nodes[self.node_id]
        response = self._episode_when_free(EpisodeRequest(
            episode_id=self.episode_id,
            map_name=self.network.map_name,
            agent=AgentSpec(
                speed_cm_s=float(self.embodiment.speed_cm_s),
                eye_z_cm=STREET_EYE_CM,
                camera=self.street_camera,
            ),
            spawn=Pose(x_cm=node.x_cm, y_cm=node.y_cm,
                       z_cm=self.spawn_z_cm, yaw_deg=0.0),
        ))
        self.fixed_dt = response.fixed_dt
        self.ue_pose = response.pose
        self._episode_open = True
        self.embodied_log.append({
            "recovery": "reopen", "after": reason, "node": self.node_id,
            "node_xy": (node.x_cm, node.y_cm),
        })

    def _walk_hop(self, toward: str) -> WalkResponse:
        """One /walk to a node's coordinates.

        A lost episode is re-opened once and the walk retried: verl runs
        several env workers in separate processes, so an instance can end up
        serving them in turn, and a single misplaced episode must not end the
        training job. Anything else propagates -- UE owns the physics here, so
        a dead engine really is a dead episode.
        """
        target = self.network.nodes[toward]
        return self._walk_to_point(target.x_cm, target.y_cm)

    def _walk_to_point(self, x_cm: float, y_cm: float) -> WalkResponse:
        """One /walk to an arbitrary world point.

        The wire has always taken coordinates -- a node hop is this with the
        node's own -- so the coordinate action space needed no new endpoint,
        only somewhere to hand a point that is not a node's.
        """
        request = WalkRequest(
            episode_id=self.episode_id,
            target_x_cm=float(x_cm), target_y_cm=float(y_cm),
            arrive_cm=self.arrive_cm,
            max_sim_seconds=self.max_walk_seconds,
            tick_chunk=self.tick_chunk,
        )
        try:
            walk = self._ue().walk(request)
        except RenderServiceError as error:
            if self._episode_is_lost(error):
                self._reopen_episode(f"walk: {type(error).__name__}")
                walk = self._ue().walk(request)
            elif self._instance_is_gone(error) and self._move_instance(
                    f"walk: {type(error).__name__}"):
                # Moved. The walk itself is not re-issued -- see
                # ``_move_instance`` -- so this call is a refusal and the
                # episode carries on from the node it re-opened at.
                raise InstanceMoved(str(error)) from error
            else:
                raise
        self.ue_pose = walk.pose
        return walk

    def _log_hop(self, row: dict[str, Any], walk: WalkResponse,
                 outcome: str) -> None:
        toward = row["node"]
        target = self.network.nodes[toward]
        # What the OFFLINE env would have charged for this same hop: the
        # straight-line graph distance at the courier's current speed. Logged
        # beside the engine's number because Track B changed how movement time
        # is PRICED without changing anything it is spent against -- deadlines,
        # the shift clock, order expiry and the episode budget are all still
        # computed from graph chords at a fixed walking speed. The engine's
        # number is >= this one by construction (a navmesh path is never
        # shorter than the chord it spans), so the gap is a one-directional
        # bias, and the ratio of these two columns is the size of it. Recorded
        # rather than corrected: what to do about it is a decision about the
        # benchmark, not about the plumbing.
        graph_seconds = (row["distance_m"] * 100.0
                         / max(self.travel_speed_cm_s(), 1e-6))
        self.embodied_log.append({
            "graph_seconds": round(graph_seconds, 4),
            "chord_m": round(float(row["distance_m"]), 3),
            "target_node": toward,
            "target_xy": (target.x_cm, target.y_cm),
            "ticks": walk.ticks,
            "sim_seconds": walk.sim_seconds,
            "walked_cm": walk.walked_cm,
            "end_pose": walk.pose.to_dict(),
            "node_xy": (target.x_cm, target.y_cm),
            "pose_error_cm": math.hypot(walk.pose.x_cm - target.x_cm,
                                        walk.pose.y_cm - target.y_cm),
            "outcome": outcome,
        })

    # ── observations ─────────────────────────────────────────────────────────

    def _plain_frame(self, node_id: str, toward: str) -> str | None:
        """The street view under the stock key, captured from the pawn.

        /observe at the agent's CURRENT pose, yawed toward the neighbour
        first, materialised into the album cache; then the stock lookup
        answers from disk. Same first-render-wins idempotency as the live
        env, same containment: busy skips the frame (the miss remains, the
        next look retries), anything deterministic degrades the episode's
        frames -- never its physics -- to album mode.
        """
        if self.front_rear_pixel_goal_mode:
            # ``candidates()`` builds street rows before ``photo_rows()``
            # chooses what this observation mode actually attaches. Letting
            # those discarded rows render would call legacy /observe several
            # times, turn the pawn, and destroy the common-pose pair contract.
            return None
        vantage = self._vantage(node_id)
        key = f"{vantage}/toward_{toward}"
        if not self.live_album.has(key) and not self.live_degraded:
            yaw = bearing_deg(self._camera_at(node_id), self.position(toward))
            request = ObserveRequest(
                episode_id=self.episode_id,
                camera=self.street_camera,
                yaw_deg=yaw,
                return_mode=self.return_mode,
            )
            try:
                try:
                    result = self._ue().observe(request)
                except RenderServiceError as error:
                    # Same recovery as a walk: a lost episode is re-opened
                    # once rather than costing the run its frames.
                    if not self._episode_is_lost(error):
                        raise
                    self._reopen_episode(f"observe: {type(error).__name__}")
                    result = self._ue().observe(request)
                if result.ok:
                    # The service echoed its own (empty) key; the album key
                    # is the caller's to name.
                    self.live_album.store(key, result)
                    if result.pose is not None:
                        self.ue_pose = result.pose
            except ServiceBusy as error:
                logger.info(
                    "observe busy for %s (%s); the next look retries",
                    key, error)
            except (RenderServiceError, ProtocolViolation, OSError) as error:
                # No silent fallback to cached frames. An online run whose
                # frames quietly come from an album is not an online run --
                # walking stays live, the pictures stop being, and the metrics
                # say nothing is wrong. Measured: 11 of 11 episodes finished
                # `degraded` on a path that could never have worked across
                # machines, and the only visible symptom was a flag nobody
                # reads. Let it raise; a broken renderer should stop the run.
                self.live_degraded = True
                if not self.allow_album_fallback:
                    raise
                logger.warning(
                    "observe failed (%s: %s); episode %s continues with "
                    "cached frames only", type(error).__name__, error,
                    self.episode_id)
        path = super()._plain_frame(vantage, toward)
        if path is not None and path not in self.frame_yaws:
            self.frame_yaws[path] = round(bearing_deg(
                self._camera_at(node_id), self.position(toward)), 1)
        return path

    def photo_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One picture of what is in front, when that is what was asked for.

        The row is synthetic on purpose: the caption renderer and the frame
        attacher both walk this list positionally, so handing them a row is
        what keeps a picture and its caption together without either learning
        that this mode exists.
        """
        if self.camera_view == CAMERA_VIEW_FRONT_REAR:
            return self._front_rear_frames()
        if self.camera_view != CAMERA_VIEW_FORWARD:
            return super().photo_rows(rows)
        yaw = self.facing()
        if yaw is None:
            # It has not walked yet, so there is no direction of travel. The
            # pawn spawns facing north and that is the honest answer.
            pose = self.ue_pose
            yaw = float(pose.yaw_deg) % 360.0 if pose is not None else 0.0
        image = self._forward_frame(yaw)
        if image is None:
            return []
        return [{"street": "ahead", "heading": compass_of(yaw), "ahead": True,
                 "image": image, "signal_image": None, "node": None,
                 "bearing": yaw, "distance_m": 0.0}]

    def _front_rear_frames(self) -> list[dict[str, Any]]:
        """Capture and materialise one fresh atomic front/rear observation.

        Unlike the legacy views, these frames are never looked up through the
        first-render-wins cache. Their capture group is part of the path, so
        every call records exactly the pair whose private snapshot identities
        are retained for the next pixel action.
        """
        self._clear_active_pixel_view_pair()
        try:
            pair = self._ue().observe_views(ObserveViewsRequest(
                episode_id=self.episode_id,
                camera=self.street_camera,
                views=self.pixel_views,
                return_mode=self.return_mode,
            ))
            if type(pair) is not ObserveViewsResponse:  # exact wire object
                raise PixelGoalPairIntegrityError("incomplete_view_pair")
            try:
                pair.to_dict()  # validate order, snapshots, pose and timings
            except (TypeError, ValueError) as error:
                raise ProtocolViolation("incomplete_view_pair") from error
            if (not isinstance(pair.capture_group_id, str)
                    or not pair.capture_group_id
                    or Path(pair.capture_group_id).name != pair.capture_group_id
                    or pair.capture_group_id in (".", "..")):
                raise PixelGoalPairIntegrityError("incomplete_view_pair")
            if tuple(view.view for view in pair.views) != self.pixel_views:
                raise PixelGoalPairIntegrityError("incomplete_view_pair")
            if len({view.camera_intrinsics_id for view in pair.views}) != 1:
                raise PixelGoalPairIntegrityError("incomplete_view_pair")
            try:
                finite_pose = all(math.isfinite(value) for value in (
                        pair.pose.x_cm, pair.pose.y_cm, pair.pose.z_cm,
                        pair.pose.yaw_deg,
                        *(view.yaw_offset_deg for view in pair.views)))
            except TypeError as error:
                raise ProtocolViolation("incomplete_view_pair") from error
            if not finite_pose:
                raise PixelGoalPairIntegrityError("incomplete_view_pair")

            common_pose = pair.pose.to_dict()
            rows: list[dict[str, Any]] = []
            metadata: dict[str, dict[str, Any]] = {}
            for view in pair.views:
                row = self._front_rear_row(pair, view, common_pose)
                rows.append(row)
                metadata[view.view] = {
                    "view": view.view,
                    "capture_group_id": view.capture_group_id or pair.capture_group_id,
                    "camera_snapshot_id": view.camera_snapshot_id,
                    "camera_intrinsics_id": view.camera_intrinsics_id,
                    "camera_yaw_deg": row["camera_yaw_deg"],
                    "width": view.width,
                    "height": view.height,
                    "sha256": row["sha256"],
                    "path": row["path"],
                }
            self.ue_pose = pair.pose
            self._active_pixel_view_pair = pair
            self._active_pixel_view_metadata = metadata
            return rows
        except (PixelGoalPairIntegrityError, ProtocolViolation) as error:
            self._record_pixel_view_integrity(error)
            self._clear_active_pixel_view_pair()
            raise

    def _front_rear_row(
        self,
        pair: ObserveViewsResponse,
        view: ObservedView,
        common_pose: dict[str, Any],
    ) -> dict[str, Any]:
        if (view.status != "ok" or not view.camera_snapshot_id
                or not view.camera_intrinsics_id
                or view.width is None or view.width <= 0
                or view.height is None or view.height <= 0):
            raise PixelGoalPairIntegrityError("incomplete_view_pair")
        if view.png_base64 is not None:
            try:
                image_bytes = base64.b64decode(view.png_base64, validate=True)
            except (TypeError, ValueError, binascii.Error) as error:
                raise ProtocolViolation("incomplete_view_pair") from error
        elif view.path is not None:
            try:
                image_bytes = Path(view.path).read_bytes()
            except (TypeError, OSError) as error:
                raise ProtocolViolation("incomplete_view_pair") from error
        else:
            raise PixelGoalPairIntegrityError("incomplete_view_pair")
        if not image_bytes:
            raise PixelGoalPairIntegrityError("incomplete_view_pair")
        digest = hashlib.sha256(image_bytes).hexdigest()
        if view.sha256 is not None and view.sha256 != digest:
            raise PixelGoalPairIntegrityError("incomplete_view_pair")

        group = view.capture_group_id or pair.capture_group_id
        target = (self.live_album.images / self._vantage(self.node_id)
                  / group / f"{view.view}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image_bytes)
        camera_yaw = (
            float(pair.pose.yaw_deg) + float(view.yaw_offset_deg)) % 360.0
        path = str(target)
        self.frame_yaws[path] = round(camera_yaw, 1)
        return {
            "street": _PIXEL_VIEW_STREET.get(view.view, "ahead"),
            "heading": compass_of(camera_yaw),
            "ahead": view.view == "front",
            "image": path,
            "path": path,
            "signal_image": None,
            "node": None,
            "bearing": camera_yaw,
            "distance_m": 0.0,
            "view": view.view,
            "capture_group_id": group,
            "camera_snapshot_id": view.camera_snapshot_id,
            "camera_intrinsics_id": view.camera_intrinsics_id,
            "camera_yaw_deg": camera_yaw,
            "width": view.width,
            "height": view.height,
            "sha256": digest,
            "timing": dict(view.timing or {}),
            "capture_timing": dict(view.timing or {}),
            "pair_timing": dict(pair.timing or {}),
            "common_pose": dict(common_pose),
            "capture_pose": dict(common_pose),
        }

    def _clear_active_pixel_view_pair(self) -> None:
        self._active_pixel_view_pair = None
        self._active_pixel_view_metadata = None

    def _record_pixel_view_integrity(
            self, error: PixelGoalPairIntegrityError | ProtocolViolation) -> None:
        code = getattr(error, "reason", None) or str(error)
        self.embodied_log.append({
            "kind": "pixel_view_integrity",
            "code": code,
            "error_type": type(error).__name__,
            "audit": dict(getattr(error, "audit", {}) or {}),
        })

    def _forward_frame(self, yaw_deg: float) -> str | None:
        """The view along ``yaw_deg`` from where the pawn stands.

        Keyed by vantage and bearing, to the degree: the courier walks a few
        metres a turn and turns as it goes, so a frame is only reusable when
        both the place and the direction repeat.
        """
        key = f"{self._vantage(self.node_id)}/ahead_{round(yaw_deg):+04d}"
        path = self.live_album.path_for(key)
        if not path.exists() and not self.live_degraded:
            request = ObserveRequest(
                episode_id=self.episode_id, camera=self.street_camera,
                yaw_deg=float(yaw_deg), return_mode=self.return_mode)
            try:
                try:
                    result = self._ue().observe(request)
                except RenderServiceError as error:
                    if not self._episode_is_lost(error):
                        raise
                    self._reopen_episode(f"observe: {type(error).__name__}")
                    result = self._ue().observe(request)
                if result.ok:
                    self.live_album.store(key, result)
                    if result.pose is not None:
                        self.ue_pose = result.pose
            except ServiceBusy as error:
                logger.info("observe busy for %s (%s); the next look retries",
                            key, error)
            except (RenderServiceError, ProtocolViolation, OSError) as error:
                self.live_degraded = True
                if not self.allow_album_fallback:
                    raise
                logger.warning(
                    "observe failed (%s: %s); episode %s continues with "
                    "cached frames only", type(error).__name__, error,
                    self.episode_id)
        if not path.exists():
            return None
        self.frame_yaws[str(path)] = round(float(yaw_deg) % 360.0, 1)
        return str(path)

    def _camera_at(self, node_id: str) -> tuple[float, float]:
        """Where the camera stands when photographing from ``node_id``.

        The pawn, when the node in question is the one the courier is at and
        the courier may be standing off it. Street mode is unchanged: there
        the pawn is within ``arrive_cm`` of the node by construction, and
        moving the yaw's origin would change every baked comparison for a
        fraction of a degree.
        """
        if self.pose_tracked_mode and node_id == self.node_id:
            return self._here_cm()
        return super().position(node_id)

    def _vantage(self, node_id: str) -> str:
        """The album key's first component: the place a frame was taken from.

        A node, in the street space, and that is a complete description --
        the courier arrives within ``arrive_cm`` of it every time, so one
        frame per (node, neighbour) can be rendered once and reused, which is
        the idempotency ``observation_media_hash`` and ``FrameAliases`` both
        rest on.

        In the coordinate space the courier stands wherever its last walk
        left it, and the same node can be looked at from anywhere within the
        snap radius. Keyed by node alone, the first visit's photograph would
        be served for every later one -- the observation would stop being of
        where the courier is, and nothing would say so. So the position joins
        the key, rounded to the decimetre: fine enough that two genuinely
        different vantages never share a frame, coarse enough that the same
        one re-rendered is still a cache hit.
        """
        if not self.pose_tracked_mode or node_id != self.node_id:
            return node_id
        x_cm, y_cm = self._here_cm()
        return f"{node_id}@{round(x_cm / 10.0):+d}_{round(y_cm / 10.0):+d}"

    def signal_frame_for(self, node_id: str, toward: str) -> str | None:
        # v1 embodied: hazards off, unconditionally (spec 3b).
        return None

    def obstacle_frame_for(self, node_id: str, toward: str) -> str | None:
        # v1 embodied: hazards off, unconditionally (spec 3b).
        return None

    # ── reporting ────────────────────────────────────────────────────────────

    def album_coverage(self) -> dict[str, Any]:
        """What is actually on disk. Computed directly: the stock method walks
        every edge through ``_plain_frame``, which here would stand the pawn
        at one corner and photograph the entire city from it."""
        total = rendered = 0
        for node_id, node in self.network.nodes.items():
            for neighbour in node.neighbours:
                total += 1
                if self.live_album.has(f"{node_id}/toward_{neighbour}"):
                    rendered += 1
        out = {
            "directed_edges": total, "with_frame": rendered,
            "fraction": round(rendered / total, 4) if total else 0.0,
            "signalised_with_frame": 0,     # hazards off in v1
            "album_root": str(self.album_root) if self.album_root else None,
            "backend": "embodied",
            "degraded": self.live_degraded,
        }
        if self.pose_tracked_mode:
            # Frames are keyed by where the pawn stood, not by node, so
            # "fraction of directed edges covered" counts something this run
            # never renders and would read 0.0 for a fully-photographed
            # episode. Say what is there instead of a ratio that is not one.
            out["with_frame"] = out["fraction"] = None
            out["frames_on_disk"] = sum(
                1 for _ in self.live_album.images.rglob("*.png"))
            out["keyed_by"] = "vantage"
        return out

    def summary(self) -> dict[str, Any]:
        """The stock summary plus the ``embodied`` block -- the I/O evidence
        contract from the spec, aggregated: how many hops UE walked, how much
        engine time they took, how far the pawn's landings sit from the graph
        nodes, and how often it got stuck."""
        out = super().summary()
        # The log carries two entry shapes: hops (walk outcomes) and
        # recoveries (re-spawns after a failed walk). Aggregate them apart.
        hops = [h for h in self.embodied_log if "ticks" in h]
        recoveries = [h for h in self.embodied_log if "recovery" in h]
        out["embodied"] = {
            "hops": len(hops),
            "recoveries": len(recoveries),
            "pedestrian_nudges": self.pedestrian_nudges,
            "pedestrian_replans": self.pedestrian_replans,
            "total_ticks": sum(h["ticks"] for h in hops),
            "total_walk_seconds": round(
                sum(h["sim_seconds"] for h in hops), 6),
            "max_pose_error_cm": round(
                max((h["pose_error_cm"] for h in hops), default=0.0), 2),
            "stuck_count": sum(1 for h in hops if h["outcome"] == "stuck"),
            "walk_timeout_count": sum(
                1 for h in hops if h["outcome"] == "timeout"),
            # Whether this episode stopped seeing live frames partway
            # through. It lived only in album_coverage(), which nothing on
            # the training path calls -- so an episode could switch its
            # observation distribution mid-rollout and say so nowhere a
            # trainer looks.
            "degraded": self.live_degraded,
            "busy_waits": self.busy_waits,
            # Which question this episode was asked. Recorded on every
            # episode, both spaces, because the whole point of the coordinate
            # run is a number compared against the street run's -- and a pair
            # of numbers whose configs are remembered rather than written
            # down is a comparison nobody can check afterwards.
            "action_space": self.action_space,
        }
        if self.pose_tracked_mode:
            matches = [
                h["map_match"] for h in hops
                if isinstance(h.get("map_match"), dict)
            ]
            out["embodied"]["map_matching"] = {
                "method": POSE_MATCH_METHOD,
                "matches": len(matches),
                "node_switches": sum(
                    1 for match in matches
                    if match.get("selected_node") != match.get("previous_node")
                ),
                # The directly measurable failure avoided by the matcher:
                # an unconstrained nearest-node lookup named a different road
                # from the connected, motion-consistent match we retained.
                "prevented_global_jumps": sum(
                    1 for match in matches
                    if match.get("prevented_global_jump") is True
                ),
                "global_recoveries": sum(
                    1 for match in matches
                    if match.get("method") == "global_recovery"
                ),
            }
        if self.coordinate_mode:
            snaps = sorted(h["snap_cm"] for h in hops if "snap_cm" in h)
            out["embodied"].update({
                "max_step_m": self.max_step_m,
                "coordinate_walks": self.coordinate_walks,
                # How often it asked for more than one step buys. A run
                # that is all `too_far` is a policy aiming at the destination
                # every turn, which is a different behaviour from naming
                # reachable points and worth being able to see.
                "coordinate_too_far": sum(
                    1 for h in self.embodied_log
                    if h.get("code") == "too_far"),
                # How far the junction being described sits from where the
                # courier actually is. The number that says whether the
                # observation is about the courier's surroundings.
                "median_snap_cm": (round(snaps[len(snaps) // 2], 1)
                                   if snaps else None),
                "max_snap_cm": round(snaps[-1], 1) if snaps else None,
            })
        elif self.pixel_goal_mode:
            snaps = sorted(h["snap_cm"] for h in hops if "snap_cm" in h)
            out["embodied"].update({
                "pixel_walks": self.pixel_walks,
                # How often a pixel never became a walk target at all --
                # analogous to ``coordinate_too_far``, but there is no
                # single reason to split out: the wire deliberately does not
                # say whether it was "no ground hit" or "off the NavMesh"
                # once it reaches the courier, so neither can this count.
                "pixel_unresolved": sum(
                    1 for h in self.embodied_log
                    if h.get("code") == "unwalkable_pixel"),
                "median_snap_cm": (round(snaps[len(snaps) // 2], 1)
                                   if snaps else None),
                "max_snap_cm": round(snaps[-1], 1) if snaps else None,
            })
            if self.front_rear_pixel_goal_mode:
                out["embodied"].update({
                    "pixel_view_counts": dict(self.pixel_view_counts),
                    "pixel_view_unresolved": {
                        view: sum(
                            1 for row in self.embodied_log
                            if row.get("selected_view") == view
                            and row.get("code") == "unwalkable_pixel")
                        for view in self.pixel_views
                    },
                    "pixel_view_controller_success": {
                        view: sum(
                            1 for row in self.embodied_log
                            if row.get("selected_view") == view
                            and row.get("controller_status") == "arrived")
                        for view in self.pixel_views
                    },
                    "pixel_view_integrity_failures": {
                        code: sum(
                            1 for row in self.embodied_log
                            if row.get("code") == code)
                        for code in (
                            "incomplete_view_pair",
                            "camera_snapshot_stale",
                            "camera_snapshot_view_mismatch",
                            "camera_snapshot_group_mismatch",
                        )
                    },
                })
        return out
