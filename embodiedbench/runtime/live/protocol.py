"""The nav-render/v0 wire protocol, as data.

One dataclass per JSON shape on the wire -- stateless renders, stateful
embodied episodes, and the bound front/rear Pixel Goal calls -- so these
dataclasses are the contract itself, and one serialisation convention for
all of them: two-space indent, sorted keys,
one trailing newline. The convention is not a taste -- the same fixtures live
in this repo and in SimWorld2, and "byte-exact against the golden files" is
the only definition of compatibility that a test can enforce. ``dumps`` here
is the single place the convention is written down; everything that says
"these bytes are the protocol" goes through it.

Optional fields follow the spec's own examples exactly:

* a request item *omits* ``signal``/``obstacle``/``camera`` when unset (the
  street-view example carries none of the three), and omits ``pitch_deg``
  when it is zero -- the fixtures predate the field, and "omitted when
  default" is what keeps them byte-identical;
* an ok result *carries* both ``path`` and ``png_base64``, one of them null,
  because the example does -- the receiver learns the return mode from which
  one is set, not from which key exists;
* a failed result is ``{"key", "status", "error"}`` and nothing else.

Pure stdlib on purpose. This module is imported by the client, the pool, the
tests, and -- copied -- by the SimWorld2 service; a dependency here is a
dependency everywhere the protocol goes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

PROTOCOL = "nav-render/v0"

RETURN_MODE_PATH = "path"
RETURN_MODE_BASE64 = "base64"
RETURN_MODES = (RETURN_MODE_PATH, RETURN_MODE_BASE64)

RENDER_KIND_STREET = "street_view"
RENDER_KIND_LAMP = "lamp"
RENDER_KIND_OBSTACLE = "obstacle"
RENDER_KINDS = (RENDER_KIND_STREET, RENDER_KIND_LAMP, RENDER_KIND_OBSTACLE)

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# The spec's error taxonomy, verbatim. Anything else on the wire is a
# protocol violation, not a new kind of failure.
ERROR_CODES = ("bad_request", "engine_down", "map_mismatch", "render_failed", "busy")


def dumps(payload: Mapping[str, Any]) -> str:
    """The canonical serialisation: 2-space indent, sorted keys, one newline.

    Both repos' golden fixtures are written by exactly this call, which is what
    makes "serialise and compare bytes" a cross-repo compatibility test rather
    than a formatting opinion.
    """
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class ProtocolViolation(ValueError):
    """A payload that does not speak nav-render/v0.

    Raised on parse, not tolerated and guessed around: a service and a client
    that quietly accept each other's malformed messages are a version skew
    nobody finds until the frames are wrong.
    """


def _require(data: Mapping[str, Any], key: str, kind: str) -> Any:
    if key not in data:
        raise ProtocolViolation(f"{kind} is missing {key!r}: {sorted(data)}")
    return data[key]


@dataclass(frozen=True)
class CameraSpec:
    """Width, height and horizontal field of view -- the whole camera."""

    width: int
    height: int
    fov_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {"width": int(self.width), "height": int(self.height),
                "fov_deg": float(self.fov_deg)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraSpec":
        return cls(width=int(_require(data, "width", "camera")),
                   height=int(_require(data, "height", "camera")),
                   fov_deg=float(_require(data, "fov_deg", "camera")))


@dataclass(frozen=True)
class SignalSpec:
    """Which lamp to flip, and to which phase. The caller decided the phase
    from ``sim_seconds``; the service never consults a clock."""

    approach: str          # "node|toward", the album's approach key
    state: str             # "red" | "green"

    def to_dict(self) -> dict[str, Any]:
        return {"approach": self.approach, "state": self.state}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalSpec":
        return cls(approach=str(_require(data, "approach", "signal")),
                   state=str(_require(data, "state", "signal")))


@dataclass(frozen=True)
class ObstacleSpec:
    """What to stand in the street before the capture.

    The *caller* owns which edges have obstacles (ObstacleField is
    deterministic in (map, seed)); the service owns only prop placement
    geometry, which is why the edge endpoints and the street width travel in
    the request rather than living in the service.
    """

    kind: str                       # "road_block" | "slow_pedestrian"
    a_cm: tuple[float, float]       # edge start, UE world cm
    b_cm: tuple[float, float]       # edge end
    street_width_cm: float
    viewpoint: str                  # "carriageway" | "pavement"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "a_cm": [float(self.a_cm[0]), float(self.a_cm[1])],
                "b_cm": [float(self.b_cm[0]), float(self.b_cm[1])],
                "street_width_cm": float(self.street_width_cm),
                "viewpoint": self.viewpoint}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObstacleSpec":
        a = _require(data, "a_cm", "obstacle")
        b = _require(data, "b_cm", "obstacle")
        return cls(kind=str(_require(data, "kind", "obstacle")),
                   a_cm=(float(a[0]), float(a[1])),
                   b_cm=(float(b[0]), float(b[1])),
                   street_width_cm=float(_require(data, "street_width_cm", "obstacle")),
                   viewpoint=str(_require(data, "viewpoint", "obstacle")))


@dataclass(frozen=True)
class RenderItem:
    """One frame to render: a self-contained pose plus any scene dressing.

    Self-contained is the statelessness rule (spec section 3): everything the
    capture needs is in this item, and the service restores the level before
    responding, so one instance can serve many episodes at request
    granularity.
    """

    key: str
    x_cm: float
    y_cm: float
    z_cm: float
    yaw_deg: float
    render_kind: str
    # Camera pitch in UE degrees. Exists so an aimed lamp close-up (camera
    # between lamp and junction, aimed at the lens with computed pitch -- the
    # bake's geometry) becomes expressible the day a lamp_pose export lands;
    # v0 callers send 0. Serialised only when nonzero, so the golden fixtures
    # -- which predate the field -- stay byte-identical.
    pitch_deg: float = 0.0
    signal: SignalSpec | None = None
    obstacle: ObstacleSpec | None = None
    camera: CameraSpec | None = None    # overrides the batch default

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "x_cm": float(self.x_cm), "y_cm": float(self.y_cm),
            "z_cm": float(self.z_cm), "yaw_deg": float(self.yaw_deg),
            "render_kind": self.render_kind,
        }
        if self.pitch_deg:
            out["pitch_deg"] = float(self.pitch_deg)
        if self.signal is not None:
            out["signal"] = self.signal.to_dict()
        if self.obstacle is not None:
            out["obstacle"] = self.obstacle.to_dict()
        if self.camera is not None:
            out["camera"] = self.camera.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderItem":
        signal = data.get("signal")
        obstacle = data.get("obstacle")
        camera = data.get("camera")
        return cls(
            key=str(_require(data, "key", "render item")),
            x_cm=float(_require(data, "x_cm", "render item")),
            y_cm=float(_require(data, "y_cm", "render item")),
            z_cm=float(_require(data, "z_cm", "render item")),
            yaw_deg=float(_require(data, "yaw_deg", "render item")),
            render_kind=str(_require(data, "render_kind", "render item")),
            pitch_deg=float(data.get("pitch_deg", 0.0)),
            signal=SignalSpec.from_dict(signal) if signal is not None else None,
            obstacle=ObstacleSpec.from_dict(obstacle) if obstacle is not None else None,
            camera=CameraSpec.from_dict(camera) if camera is not None else None,
        )


@dataclass(frozen=True)
class RenderBatch:
    """One POST /render body."""

    episode_id: str
    return_mode: str
    camera: CameraSpec
    requests: tuple[RenderItem, ...]
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "return_mode": self.return_mode,
                "camera": self.camera.to_dict(),
                "requests": [item.to_dict() for item in self.requests]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderBatch":
        protocol = str(_require(data, "protocol", "render batch"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"protocol {protocol!r} is not {PROTOCOL!r}; refusing to guess "
                "what a different version means")
        return_mode = str(_require(data, "return_mode", "render batch"))
        if return_mode not in RETURN_MODES:
            raise ProtocolViolation(
                f"return_mode {return_mode!r}; expected one of {RETURN_MODES}")
        return cls(
            episode_id=str(_require(data, "episode_id", "render batch")),
            return_mode=return_mode,
            camera=CameraSpec.from_dict(_require(data, "camera", "render batch")),
            requests=tuple(RenderItem.from_dict(item)
                           for item in _require(data, "requests", "render batch")),
            protocol=protocol,
        )


@dataclass(frozen=True)
class Pose:
    """Where the embodied agent stands: UE world cm plus a yaw.

    The one shape every Track B response shares. Spelled out as its own
    dataclass rather than four loose floats because "pose echo" appears in
    three different messages and they must not be allowed to drift apart.
    """

    x_cm: float
    y_cm: float
    z_cm: float
    yaw_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {"x_cm": float(self.x_cm), "y_cm": float(self.y_cm),
                "z_cm": float(self.z_cm), "yaw_deg": float(self.yaw_deg)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Pose":
        return cls(x_cm=float(_require(data, "x_cm", "pose")),
                   y_cm=float(_require(data, "y_cm", "pose")),
                   z_cm=float(_require(data, "z_cm", "pose")),
                   yaw_deg=float(_require(data, "yaw_deg", "pose")))


@dataclass(frozen=True)
class RenderResult:
    """One frame's outcome. The batch never half-dies: a bad item is a
    ``failed`` result in an otherwise ok response.

    ``pose`` is Track B's addition -- /observe echoes the agent's actual pose
    beside the frame -- and it is serialised only when present, for the same
    reason ``pitch_deg`` is only serialised when nonzero: the Track A golden
    fixtures predate the field and must stay byte-identical.
    """

    key: str
    status: str
    path: str | None = None
    png_base64: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None
    pose: Pose | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        if self.status == STATUS_FAILED:
            return {"key": self.key, "status": self.status,
                    "error": self.error or ""}
        # Both transport keys present, one null -- the spec's own example. The
        # receiver reads the mode off which one is set.
        out = {"key": self.key, "status": self.status,
               "path": self.path, "png_base64": self.png_base64,
               "sha256": self.sha256,
               "width": self.width, "height": self.height}
        if self.pose is not None:
            out["pose"] = self.pose.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderResult":
        pose = data.get("pose")
        return cls(
            key=str(_require(data, "key", "render result")),
            status=str(_require(data, "status", "render result")),
            path=data.get("path"),
            png_base64=data.get("png_base64"),
            sha256=data.get("sha256"),
            width=(None if data.get("width") is None else int(data["width"])),
            height=(None if data.get("height") is None else int(data["height"])),
            error=data.get("error"),
            pose=Pose.from_dict(pose) if pose is not None else None,
        )


@dataclass(frozen=True)
class RenderResponse:
    results: tuple[RenderResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderResponse":
        return cls(results=tuple(RenderResult.from_dict(r)
                                 for r in _require(data, "results", "render response")))


@dataclass(frozen=True)
class Healthz:
    """GET /healthz. ``map_name`` uses DeliveryGym naming ("citycore-paris"),
    so a client can refuse an instance serving the wrong city."""

    status: str
    instance_id: str
    map_name: str
    engine_connected: bool
    episodes_active: int
    uptime_s: float
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "protocol": self.protocol,
                "instance_id": self.instance_id, "map_name": self.map_name,
                "engine_connected": bool(self.engine_connected),
                "episodes_active": int(self.episodes_active),
                "uptime_s": float(self.uptime_s)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Healthz":
        protocol = str(_require(data, "protocol", "healthz"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"healthz speaks {protocol!r}, not {PROTOCOL!r}")
        return cls(
            status=str(_require(data, "status", "healthz")),
            instance_id=str(_require(data, "instance_id", "healthz")),
            map_name=str(_require(data, "map_name", "healthz")),
            engine_connected=bool(_require(data, "engine_connected", "healthz")),
            episodes_active=int(_require(data, "episodes_active", "healthz")),
            uptime_s=float(_require(data, "uptime_s", "healthz")),
            protocol=protocol,
        )


@dataclass(frozen=True)
class WireError:
    """The non-200 body: ``{"error": {"code", "message"}}``."""

    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WireError":
        body = _require(data, "error", "error response")
        return cls(code=str(_require(body, "code", "error body")),
                   message=str(_require(body, "message", "error body")))


# ── Track B: embodied episodes (spec section 3b, stateful) ───────────────────
#
# Track B gives UE ownership of locomotion and locomotion time. These
# endpoints are stateful -- one active embodied episode per instance -- so
# they exist beside the stateless render messages, not instead of them. The
# defaults below are the spec's own example values; a caller that wants
# different numbers says so on the wire.

DEFAULT_ARRIVE_CM = 50.0
DEFAULT_MAX_WALK_SIM_SECONDS = 120.0
DEFAULT_TICK_CHUNK = 10

PIXEL_VIEW_FRONT = "front"
PIXEL_VIEW_REAR = "rear"
PIXEL_VIEW_LEFT = "left"
PIXEL_VIEW_RIGHT = "right"
PIXEL_VIEWS = (PIXEL_VIEW_FRONT, PIXEL_VIEW_REAR)
#: The four-view observation: the pair, plus the same pair taken a quarter
#: turn to the right (its front is the courier's right, its rear the left).
#: Together they see the whole horizon, so a pavement is never in a blind
#: sector because of which way the last walk happened to end.
PIXEL_VIEWS_QUAD = (PIXEL_VIEW_FRONT, PIXEL_VIEW_LEFT, PIXEL_VIEW_RIGHT,
                    PIXEL_VIEW_REAR)
#: Each view's camera yaw relative to the pawn's facing.
PIXEL_VIEW_YAW_OFFSETS = {PIXEL_VIEW_FRONT: 0.0, PIXEL_VIEW_RIGHT: 90.0,
                          PIXEL_VIEW_REAR: 180.0, PIXEL_VIEW_LEFT: 270.0}
_PIXEL_VIEW_SETS = (PIXEL_VIEWS, PIXEL_VIEWS_QUAD)


def _timing(data: Any, kind: str) -> dict[str, float]:
    """Validate the optional capture timing shape shared by view messages."""
    if not isinstance(data, Mapping):
        raise ProtocolViolation(f"{kind} timing must be an object")
    expected = {"capture_read_ms", "encode_ms", "wall_ms"}
    if set(data) != expected:
        raise ProtocolViolation(f"{kind} timing must contain {sorted(expected)}")
    parsed: dict[str, float] = {}
    for key in expected:
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolViolation(f"{kind} timing {key!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ProtocolViolation(
                f"{kind} timing {key!r} must be finite and non-negative")
        parsed[key] = numeric
    return parsed


def _pixel_binding(view: str | None, capture_group_id: str | None,
                   camera_snapshot_id: str | None) -> None:
    binding = (view, capture_group_id, camera_snapshot_id)
    if any(item is not None for item in binding) and not all(binding):
        raise ProtocolViolation(
            "a bound pixel needs view, capture_group_id, and camera_snapshot_id")
    if view is not None and view not in PIXEL_VIEWS_QUAD:
        raise ProtocolViolation(f"unknown pixel view {view!r}")


@dataclass(frozen=True)
class AgentSpec:
    """The embodiment as the service needs it: speed, eye height, camera."""

    speed_cm_s: float
    eye_z_cm: float
    camera: CameraSpec

    def to_dict(self) -> dict[str, Any]:
        return {"speed_cm_s": float(self.speed_cm_s),
                "eye_z_cm": float(self.eye_z_cm),
                "camera": self.camera.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentSpec":
        return cls(
            speed_cm_s=float(_require(data, "speed_cm_s", "agent")),
            eye_z_cm=float(_require(data, "eye_z_cm", "agent")),
            camera=CameraSpec.from_dict(_require(data, "camera", "agent")),
        )


@dataclass(frozen=True)
class EpisodeRequest:
    """POST /episode: spawn (or re-spawn) the agent. Idempotent per
    episode_id; a new episode_id tears down the previous episode's agent."""

    episode_id: str
    map_name: str
    agent: AgentSpec
    spawn: Pose
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "map_name": self.map_name,
                "agent": self.agent.to_dict(),
                "spawn": self.spawn.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeRequest":
        protocol = str(_require(data, "protocol", "episode request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"episode request speaks {protocol!r}, not {PROTOCOL!r}")
        return cls(
            episode_id=str(_require(data, "episode_id", "episode request")),
            map_name=str(_require(data, "map_name", "episode request")),
            agent=AgentSpec.from_dict(_require(data, "agent", "episode request")),
            spawn=Pose.from_dict(_require(data, "spawn", "episode request")),
            protocol=protocol,
        )


@dataclass(frozen=True)
class EpisodeResponse:
    """The service's answer: where the agent actually stands, and the fixed
    dt every subsequent ``sim_seconds`` is a multiple of."""

    episode_id: str
    pose: Pose
    fixed_dt: float

    def to_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id,
                "pose": self.pose.to_dict(),
                "fixed_dt": float(self.fixed_dt)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeResponse":
        return cls(
            episode_id=str(_require(data, "episode_id", "episode response")),
            pose=Pose.from_dict(_require(data, "pose", "episode response")),
            fixed_dt=float(_require(data, "fixed_dt", "episode response")),
        )


@dataclass(frozen=True)
class WalkRequest:
    """POST /walk: MoveTo(target) under lockstep ticks until arrival within
    ``arrive_cm``, no progress (stuck), or ``max_sim_seconds`` of sim time."""

    episode_id: str
    target_x_cm: float
    target_y_cm: float
    arrive_cm: float = DEFAULT_ARRIVE_CM
    max_sim_seconds: float = DEFAULT_MAX_WALK_SIM_SECONDS
    tick_chunk: int = DEFAULT_TICK_CHUNK
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "target": {"x_cm": float(self.target_x_cm),
                           "y_cm": float(self.target_y_cm)},
                "arrive_cm": float(self.arrive_cm),
                "max_sim_seconds": float(self.max_sim_seconds),
                "tick_chunk": int(self.tick_chunk)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WalkRequest":
        protocol = str(_require(data, "protocol", "walk request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"walk request speaks {protocol!r}, not {PROTOCOL!r}")
        target = _require(data, "target", "walk request")
        return cls(
            episode_id=str(_require(data, "episode_id", "walk request")),
            target_x_cm=float(_require(target, "x_cm", "walk target")),
            target_y_cm=float(_require(target, "y_cm", "walk target")),
            arrive_cm=float(data.get("arrive_cm", DEFAULT_ARRIVE_CM)),
            max_sim_seconds=float(data.get("max_sim_seconds",
                                           DEFAULT_MAX_WALK_SIM_SECONDS)),
            tick_chunk=int(data.get("tick_chunk", DEFAULT_TICK_CHUNK)),
            protocol=protocol,
        )


@dataclass(frozen=True)
class WalkResponse:
    """What the walk did. ``sim_seconds`` is ticks * fixed_dt and is the
    authoritative walking time -- the caller adds it to the env clock instead
    of the declared distance/speed arithmetic."""

    arrived: bool
    stuck: bool
    timeout: bool
    ticks: int
    sim_seconds: float
    pose: Pose
    walked_cm: float
    #: Where the walk began, as the service read it at entry.
    #:
    #: Added because ``walked_cm`` could not be checked against anything. It is
    #: a sum of per-chunk displacements and it came back 0.00 on thirteen
    #: consecutive walks whose start and end poses differed by half a metre to
    #: five -- but the only start pose available was the CALLER's idea of where
    #: the pawn was, which ``/observe`` overwrites and other couriers' ticks
    #: move. Twice I called the count impossible against a baseline that was
    #: not the walk's own, and twice that was my error rather than the count's.
    #: With both ends from the same response, path length versus displacement
    #: is an arithmetic identity anyone can check: a path is never shorter than
    #: the line it spans.
    #:
    #: Optional on the wire, and omitted when absent, so every Track B golden
    #: fixture and every service that predates it round-trips unchanged.
    start_pose: Pose | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"arrived": bool(self.arrived), "stuck": bool(self.stuck),
               "timeout": bool(self.timeout),
               "ticks": int(self.ticks),
               "sim_seconds": float(self.sim_seconds),
               "pose": self.pose.to_dict(),
               "walked_cm": float(self.walked_cm)}
        if self.start_pose is not None:
            out["start_pose"] = self.start_pose.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WalkResponse":
        return cls(
            arrived=bool(_require(data, "arrived", "walk response")),
            stuck=bool(_require(data, "stuck", "walk response")),
            timeout=bool(_require(data, "timeout", "walk response")),
            ticks=int(_require(data, "ticks", "walk response")),
            sim_seconds=float(_require(data, "sim_seconds", "walk response")),
            pose=Pose.from_dict(_require(data, "pose", "walk response")),
            walked_cm=float(_require(data, "walked_cm", "walk response")),
            start_pose=(Pose.from_dict(data["start_pose"])
                        if data.get("start_pose") is not None else None),
        )


@dataclass(frozen=True)
class PixelSpec:
    """A normalized image point: where the policy clicked, not where it
    landed. Origin top-left, both components in ``[0, 1]``."""

    u: float
    v: float

    def to_dict(self) -> dict[str, Any]:
        return {"u": float(self.u), "v": float(self.v)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PixelSpec":
        return cls(u=float(_require(data, "u", "pixel")),
                   v=float(_require(data, "v", "pixel")))


@dataclass(frozen=True)
class ResolvedPixel:
    """What the engine made of the pixel before it ever walked anywhere.

    ``raw_world_hit_cm`` is the raycast's ground hit, before NavMesh
    projection. ``accepted_target_cm`` is that hit snapped onto the NavMesh --
    the point the walk actually steers toward -- and ``navmesh_adjustment_cm``
    is the distance between the two. All three are absent together when the
    ray never met the ground at all (sky, a wall): that is a different
    failure from a hit that existed but sat too far from any walkable surface
    to snap, which still reports ``raw_world_hit_cm`` for the record even
    though it, too, could not be walked to. ``rejection_reason`` is set only
    when the pixel could not be turned into a walk target at all -- distinct
    from a target that resolved cleanly and then could not be walked to
    (which is an ordinary ``stuck``/``timeout`` on the walk itself).
    The raw string is diagnostic. A policy-facing caller may translate known
    values into coarse causal classes, but must not echo engine details such
    as actor names, coordinates, or navigation thresholds.
    """

    raw_world_hit_cm: tuple[float, float] | None = None
    accepted_target_cm: tuple[float, float] | None = None
    navmesh_adjustment_cm: float | None = None
    rejection_reason: str | None = None
    controller_path_points_cm: tuple[tuple[float, float, float], ...] | None = None
    controller_path_length_cm: float | None = None
    controller_path_direct_cm: float | None = None
    controller_path_stretch_ratio: float | None = None
    #: The engine's name for the actor the ray hit first, the hit's height,
    #: and whether the engine judged the straight path to the hit legal
    #: (True), illegal because it crosses a carriageway (False), or never got
    #: that far (None). Harness-side classification only; never narrated.
    raw_hit_actor: str | None = None
    raw_world_hit_z_cm: float | None = None
    direct_path_legal: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        def _xy(point: tuple[float, float] | None) -> list[float] | None:
            return None if point is None else [float(point[0]), float(point[1])]
        out = {
            "raw_world_hit_cm": _xy(self.raw_world_hit_cm),
            "accepted_target_cm": _xy(self.accepted_target_cm),
            "navmesh_adjustment_cm": (None if self.navmesh_adjustment_cm is None
                                      else float(self.navmesh_adjustment_cm)),
            "rejection_reason": self.rejection_reason,
        }
        if self.raw_hit_actor is not None:
            out["raw_hit_actor"] = self.raw_hit_actor
        if self.raw_world_hit_z_cm is not None:
            out["raw_world_hit_z_cm"] = float(self.raw_world_hit_z_cm)
        if self.direct_path_legal is not None:
            out["direct_path_legal"] = bool(self.direct_path_legal)
        if self.controller_path_points_cm is not None:
            out.update({
                "controller_path_points_cm": [
                    [float(component) for component in point]
                    for point in self.controller_path_points_cm
                ],
                "controller_path_length_cm": self.controller_path_length_cm,
                "controller_path_direct_cm": self.controller_path_direct_cm,
                "controller_path_stretch_ratio": self.controller_path_stretch_ratio,
            })
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolvedPixel":
        def _point(value: Any) -> tuple[float, float] | None:
            return None if value is None else (float(value[0]), float(value[1]))
        adjustment = data.get("navmesh_adjustment_cm")
        path = data.get("controller_path_points_cm")
        return cls(
            raw_world_hit_cm=_point(data.get("raw_world_hit_cm")),
            accepted_target_cm=_point(data.get("accepted_target_cm")),
            navmesh_adjustment_cm=(None if adjustment is None
                                   else float(adjustment)),
            rejection_reason=data.get("rejection_reason"),
            controller_path_points_cm=(
                None if path is None else tuple(
                    tuple(float(component) for component in point)
                    for point in path)),
            controller_path_length_cm=(
                None if data.get("controller_path_length_cm") is None
                else float(data["controller_path_length_cm"])),
            controller_path_direct_cm=(
                None if data.get("controller_path_direct_cm") is None
                else float(data["controller_path_direct_cm"])),
            controller_path_stretch_ratio=(
                None if data.get("controller_path_stretch_ratio") is None
                else float(data["controller_path_stretch_ratio"])),
            raw_hit_actor=data.get("raw_hit_actor"),
            raw_world_hit_z_cm=(
                None if data.get("raw_world_hit_z_cm") is None
                else float(data["raw_world_hit_z_cm"])),
            direct_path_legal=(
                None if data.get("direct_path_legal") is None
                else bool(data["direct_path_legal"])),
        )


@dataclass(frozen=True)
class WalkPixelRequest:
    """POST /walk_pixel: resolve a screen pixel, then walk it like /walk.

    Legacy requests remain unbound and resolve against the live pawn. A
    front/rear observation binds all three of ``view``, ``capture_group_id``,
    and ``camera_snapshot_id`` so the service resolves the click against the
    exact selected photograph rather than an ambiguous current camera.
    """

    episode_id: str
    pixel: PixelSpec
    camera: CameraSpec
    arrive_cm: float = DEFAULT_ARRIVE_CM
    max_sim_seconds: float = DEFAULT_MAX_WALK_SIM_SECONDS
    tick_chunk: int = DEFAULT_TICK_CHUNK
    protocol: str = PROTOCOL
    view: str | None = None
    capture_group_id: str | None = None
    camera_snapshot_id: str | None = None
    #: Resolve the pixel (ray hit, NavMesh projection, the engine's verdict on
    #: the straight path) and stop there: the pawn does not walk. The caller
    #: then decides where to walk, along certified pedestrian ways.
    resolve_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        _pixel_binding(self.view, self.capture_group_id,
                       self.camera_snapshot_id)
        out = {"protocol": self.protocol,
               "episode_id": self.episode_id,
               "pixel": self.pixel.to_dict(),
               "camera": self.camera.to_dict(),
               "arrive_cm": float(self.arrive_cm),
               "max_sim_seconds": float(self.max_sim_seconds),
               "tick_chunk": int(self.tick_chunk)}
        if self.view is not None:
            out.update({"view": self.view,
                        "capture_group_id": self.capture_group_id,
                        "camera_snapshot_id": self.camera_snapshot_id})
        if self.resolve_only:
            out["resolve_only"] = True
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WalkPixelRequest":
        protocol = str(_require(data, "protocol", "walk_pixel request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"walk_pixel request speaks {protocol!r}, not {PROTOCOL!r}")
        view = data.get("view")
        capture_group_id = data.get("capture_group_id")
        camera_snapshot_id = data.get("camera_snapshot_id")
        _pixel_binding(view, capture_group_id, camera_snapshot_id)
        return cls(
            episode_id=str(_require(data, "episode_id", "walk_pixel request")),
            pixel=PixelSpec.from_dict(_require(data, "pixel", "walk_pixel request")),
            camera=CameraSpec.from_dict(_require(data, "camera", "walk_pixel request")),
            arrive_cm=float(data.get("arrive_cm", DEFAULT_ARRIVE_CM)),
            max_sim_seconds=float(data.get("max_sim_seconds",
                                           DEFAULT_MAX_WALK_SIM_SECONDS)),
            tick_chunk=int(data.get("tick_chunk", DEFAULT_TICK_CHUNK)),
            protocol=protocol,
            view=view,
            capture_group_id=capture_group_id,
            camera_snapshot_id=camera_snapshot_id,
            resolve_only=bool(data.get("resolve_only", False)),
        )


@dataclass(frozen=True)
class WalkPixelResponse:
    """/walk_pixel's answer: the same shape as /walk, plus how the pixel
    resolved. A pixel that never became a target still reports a full
    response -- ``resolved.rejection_reason`` set, ``arrived`` false,
    ``ticks == 0`` -- rather than a wire error, because "that point is not
    walkable" is an ordinary outcome of this endpoint, not a malformed
    request."""

    arrived: bool
    stuck: bool
    timeout: bool
    ticks: int
    sim_seconds: float
    pose: Pose
    walked_cm: float
    resolved: ResolvedPixel
    start_pose: Pose | None = None
    view: str | None = None
    capture_group_id: str | None = None
    camera_snapshot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        _pixel_binding(self.view, self.capture_group_id,
                       self.camera_snapshot_id)
        out = {"arrived": bool(self.arrived), "stuck": bool(self.stuck),
               "timeout": bool(self.timeout),
               "ticks": int(self.ticks),
               "sim_seconds": float(self.sim_seconds),
               "pose": self.pose.to_dict(),
               "walked_cm": float(self.walked_cm),
               "resolved": self.resolved.to_dict()}
        if self.start_pose is not None:
            out["start_pose"] = self.start_pose.to_dict()
        if self.view is not None:
            out.update({"view": self.view,
                        "capture_group_id": self.capture_group_id,
                        "camera_snapshot_id": self.camera_snapshot_id})
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WalkPixelResponse":
        view = data.get("view")
        capture_group_id = data.get("capture_group_id")
        camera_snapshot_id = data.get("camera_snapshot_id")
        _pixel_binding(view, capture_group_id, camera_snapshot_id)
        return cls(
            arrived=bool(_require(data, "arrived", "walk_pixel response")),
            stuck=bool(_require(data, "stuck", "walk_pixel response")),
            timeout=bool(_require(data, "timeout", "walk_pixel response")),
            ticks=int(_require(data, "ticks", "walk_pixel response")),
            sim_seconds=float(_require(data, "sim_seconds", "walk_pixel response")),
            pose=Pose.from_dict(_require(data, "pose", "walk_pixel response")),
            walked_cm=float(_require(data, "walked_cm", "walk_pixel response")),
            resolved=ResolvedPixel.from_dict(
                _require(data, "resolved", "walk_pixel response")),
            start_pose=(Pose.from_dict(data["start_pose"])
                        if data.get("start_pose") is not None else None),
            view=view,
            capture_group_id=capture_group_id,
            camera_snapshot_id=camera_snapshot_id,
        )


@dataclass(frozen=True)
class ObserveRequest:
    """POST /observe: the agent's first-person view at its CURRENT pose,
    optionally yawing to ``yaw_deg`` first (one tick to settle).

    ``yaw_deg`` is serialised even when null because the spec's own example
    carries ``"yaw_deg": null`` -- null means "as the agent stands". The
    response is a /render-item-shaped ``RenderResult`` whose ``pose`` echoes
    where the frame was really taken.
    """

    episode_id: str
    camera: CameraSpec
    yaw_deg: float | None = None
    return_mode: str = RETURN_MODE_PATH
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "camera": self.camera.to_dict(),
                "yaw_deg": (None if self.yaw_deg is None
                            else float(self.yaw_deg)),
                "return_mode": self.return_mode}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObserveRequest":
        protocol = str(_require(data, "protocol", "observe request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"observe request speaks {protocol!r}, not {PROTOCOL!r}")
        return_mode = str(_require(data, "return_mode", "observe request"))
        if return_mode not in RETURN_MODES:
            raise ProtocolViolation(
                f"return_mode {return_mode!r}; expected one of {RETURN_MODES}")
        yaw = _require(data, "yaw_deg", "observe request")
        return cls(
            episode_id=str(_require(data, "episode_id", "observe request")),
            camera=CameraSpec.from_dict(_require(data, "camera", "observe request")),
            yaw_deg=None if yaw is None else float(yaw),
            return_mode=return_mode,
            protocol=protocol,
        )


@dataclass(frozen=True)
class ObservedView:
    """One member of an atomic front/rear observation pair."""

    view: str
    yaw_offset_deg: float
    camera_snapshot_id: str
    camera_intrinsics_id: str
    status: str
    path: str | None = None
    png_base64: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    timing: dict[str, float] | None = None
    #: The engine capture group this view came from when an observation is
    #: assembled from more than one engine pair (the four-view observation
    #: is two pairs). Absent means the response's own group.
    capture_group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "view": self.view,
            "yaw_offset_deg": float(self.yaw_offset_deg),
            "camera_snapshot_id": self.camera_snapshot_id,
            "camera_intrinsics_id": self.camera_intrinsics_id,
            "status": self.status,
            "path": self.path,
            "png_base64": self.png_base64,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }
        if self.timing is not None:
            out["timing"] = _timing(self.timing, "observed view")
        if self.capture_group_id is not None:
            out["capture_group_id"] = self.capture_group_id
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservedView":
        return cls(
            view=str(_require(data, "view", "observed view")),
            yaw_offset_deg=float(_require(data, "yaw_offset_deg", "observed view")),
            camera_snapshot_id=str(_require(data, "camera_snapshot_id", "observed view")),
            camera_intrinsics_id=str(_require(
                data, "camera_intrinsics_id", "observed view")),
            status=str(_require(data, "status", "observed view")),
            path=data.get("path"),
            png_base64=data.get("png_base64"),
            sha256=data.get("sha256"),
            width=(None if data.get("width") is None else int(data["width"])),
            height=(None if data.get("height") is None else int(data["height"])),
            timing=(_timing(data["timing"], "observed view")
                    if "timing" in data else None),
            capture_group_id=data.get("capture_group_id"),
        )


@dataclass(frozen=True)
class ObserveViewsRequest:
    """POST /observe_views: an ordered, common-pose front/rear pair."""

    episode_id: str
    camera: CameraSpec
    views: tuple[str, ...] = PIXEL_VIEWS
    return_mode: str = RETURN_MODE_PATH
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        views = tuple(self.views)
        if views not in _PIXEL_VIEW_SETS:
            raise ProtocolViolation(
                f"observe_views requires one of {_PIXEL_VIEW_SETS}, got {views}")
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "camera": self.camera.to_dict(),
                "views": list(views),
                "return_mode": self.return_mode}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObserveViewsRequest":
        protocol = str(_require(data, "protocol", "observe_views request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"observe_views request speaks {protocol!r}, not {PROTOCOL!r}")
        views = tuple(_require(data, "views", "observe_views request"))
        if views not in _PIXEL_VIEW_SETS:
            raise ProtocolViolation(
                f"observe_views requires one of {_PIXEL_VIEW_SETS}, got {views}")
        return_mode = str(_require(data, "return_mode", "observe_views request"))
        if return_mode not in RETURN_MODES:
            raise ProtocolViolation(
                f"return_mode {return_mode!r}; expected one of {RETURN_MODES}")
        return cls(
            episode_id=str(_require(data, "episode_id", "observe_views request")),
            camera=CameraSpec.from_dict(
                _require(data, "camera", "observe_views request")),
            views=views,
            return_mode=return_mode,
            protocol=protocol,
        )


@dataclass(frozen=True)
class ObserveViewsResponse:
    """The atomic front/rear pair and the pawn pose shared by both views."""

    capture_group_id: str
    pose: Pose
    views: tuple[ObservedView, ...]
    timing: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        order = tuple(view.view for view in self.views)
        if order not in _PIXEL_VIEW_SETS:
            raise ProtocolViolation(
                "observe_views response is not ordered front, rear or "
                "front, left, right, rear")
        if len({view.camera_snapshot_id for view in self.views}) != len(order):
            raise ProtocolViolation("every view needs its own camera snapshot")
        out = {"capture_group_id": self.capture_group_id,
               "pose": self.pose.to_dict(),
               "views": [view.to_dict() for view in self.views]}
        if self.timing is not None:
            out["timing"] = _timing(self.timing, "observe_views response")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObserveViewsResponse":
        views = tuple(ObservedView.from_dict(view)
                      for view in _require(data, "views", "observe_views response"))
        order = tuple(view.view for view in views)
        if order not in _PIXEL_VIEW_SETS:
            raise ProtocolViolation(
                "observe_views response is not ordered front, rear or "
                "front, left, right, rear")
        if len({view.camera_snapshot_id for view in views}) != len(order):
            raise ProtocolViolation("every view needs its own camera snapshot")
        return cls(
            capture_group_id=str(_require(
                data, "capture_group_id", "observe_views response")),
            pose=Pose.from_dict(_require(data, "pose", "observe_views response")),
            views=views,
            timing=(_timing(data["timing"], "observe_views response")
                    if "timing" in data else None),
        )


@dataclass(frozen=True)
class EpisodeEndRequest:
    """POST /episode_end: despawn the agent, keep PIE alive for the next
    episode. The response is ``{"ok": true}`` and carries nothing worth a
    dataclass."""

    episode_id: str
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "episode_id": self.episode_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeEndRequest":
        protocol = str(_require(data, "protocol", "episode_end request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"episode_end request speaks {protocol!r}, not {PROTOCOL!r}")
        return cls(
            episode_id=str(_require(data, "episode_id", "episode_end request")),
            protocol=protocol,
        )
