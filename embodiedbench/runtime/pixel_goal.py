"""Live-UE-only execution boundary for experimental Pixel Goal navigation.

The legacy policy-facing action is a normalized point in one frame; the
front/rear variant also chooses a policy-visible view label. This module binds
the selected point to the immutable UE camera snapshot returned with its RGB
frame and parses the geometry/controller audit produced by UE. It intentionally
has no cached-runtime, graph, depth, or pose-lattice integration.
"""

from __future__ import annotations

import base64
import binascii
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Protocol

from embodiedbench.schemas.embodiment import (
    ControllerResult,
    DistanceSource,
    NavigationRequest,
)
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import ControllerOutcomeCode, NavPixelGoalAction


@dataclass(frozen=True)
class PixelGoalConfig:
    """Configurable Milestone 1A calibration tolerances, expressed in UE cm."""

    max_navmesh_adjustment_cm: float = 10.0
    acceptance_radius_cm: float = 15.0
    trace_distance_cm: float = 100_000.0
    max_ground_slope_deg: float = 30.0
    max_controller_path_stretch_ratio: float = 1.35
    controller_path_detour_allowance_cm: float = 100.0
    # A controller move is judged against UE world time so a slow editor does
    # not turn an otherwise-valid pixel into a policy-facing timeout.
    movement_timeout_sim_s: float = 30.0
    # Host-clock watchdog for a frozen RPC/editor. Reaching it is an
    # infrastructure failure, not a movement result returned to policy.
    execution_timeout_s: float = 30.0
    poll_interval_s: float = 0.05
    fov_degrees: float = 90.0
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        positive = {
            "max_navmesh_adjustment_cm": self.max_navmesh_adjustment_cm,
            "acceptance_radius_cm": self.acceptance_radius_cm,
            "trace_distance_cm": self.trace_distance_cm,
            "max_controller_path_stretch_ratio": (
                self.max_controller_path_stretch_ratio),
            "movement_timeout_sim_s": self.movement_timeout_sim_s,
            "execution_timeout_s": self.execution_timeout_s,
            "poll_interval_s": self.poll_interval_s,
            "fov_degrees": self.fov_degrees,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.max_ground_slope_deg) or not (
            0.0 <= self.max_ground_slope_deg < 90.0
        ):
            raise ValueError("max_ground_slope_deg must be finite and in [0, 90)")
        if self.max_controller_path_stretch_ratio < 1.0:
            raise ValueError(
                "max_controller_path_stretch_ratio must be at least 1")
        if (not math.isfinite(self.controller_path_detour_allowance_cm)
                or self.controller_path_detour_allowance_cm < 0.0):
            raise ValueError(
                "controller_path_detour_allowance_cm must be finite and nonnegative")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in [1, 95]")


@dataclass(frozen=True)
class PixelGoalFrame:
    """RGB plus the private UE calibration identity captured with it."""

    rgb_data_url: str
    camera_snapshot_id: str
    camera_intrinsics_id: str
    width_px: int
    height_px: int
    agent_tag: str | None = None
    view_id: str | None = None
    capture_group_id: str | None = None
    camera_yaw_deg: float | None = None
    capture_timing: dict[str, object] | None = None
    #: Where the camera stood, UE world cm, when the engine reports it. With
    #: the yaw and the fixed intrinsics this is enough to project a world
    #: point back into the picture, which is how the harness walks the pawn
    #: to a certified point it chose itself.
    camera_location_cm: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.rgb_data_url.startswith("data:image/"):
            raise ValueError("Pixel Goal frame must contain an image data URL")
        if not self.camera_snapshot_id:
            raise ValueError("Pixel Goal frame is missing its camera snapshot id")
        if not self.camera_intrinsics_id:
            raise ValueError("Pixel Goal frame is missing its camera intrinsics id")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("Pixel Goal frame dimensions must be positive")
        if (self.view_id is None) != (self.capture_group_id is None):
            raise ValueError(
                "Pixel Goal frame view and capture group must be bound together"
            )
        if self.view_id is not None and self.view_id not in {"front", "rear"}:
            raise ValueError(f"unknown Pixel Goal frame view {self.view_id!r}")
        if self.capture_group_id is not None and not self.capture_group_id:
            raise ValueError("Pixel Goal frame capture group must be non-empty")


@dataclass(frozen=True)
class PixelGoalStarted:
    request: NavigationRequest
    controller_request_result: str
    #: The actor the pixel's ray hit first, as the engine names it. The
    #: harness uses it only to tell a pavement from a carriageway or a kerb
    #: stone; it never reaches the policy.
    raw_hit_actor: str | None = None


@dataclass(frozen=True)
class PixelGoalInProgress:
    request_id: str
    state: str
    feet_position: Vec3 | None = None
    elapsed_sim_s: float | None = None
    distance_travelled_cm: float | None = None


def project_world_point_to_pixel(
    camera_location_cm: tuple[float, float, float],
    camera_yaw_deg: float,
    point_cm: tuple[float, float, float],
    *,
    width_px: int = 640,
    height_px: int = 360,
    hfov_deg: float = 90.0,
) -> tuple[float, float] | None:
    """Where a world point appears in a level, pitch-free capture.

    The inverse of the engine's pixel resolver for the one case the harness
    needs: it has chosen a certified world point to walk to, and must name
    the pixel that hits it. UE yaw turns from +X toward +Y, so the camera's
    right vector is (-sin, cos). ``None`` when the point is not in front of
    the camera; a result outside [0, 1] means it is outside the picture.
    """

    focal_px = (width_px / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    dx = point_cm[0] - camera_location_cm[0]
    dy = point_cm[1] - camera_location_cm[1]
    dz = point_cm[2] - camera_location_cm[2]
    yaw = math.radians(camera_yaw_deg)
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    if forward <= 1e-6:
        return None
    return (0.5 + focal_px * right / (forward * width_px),
            0.5 - focal_px * dz / (forward * height_px))


class PixelGoalEndpoint(Protocol):
    """One reflected UE subsystem reached through an arbitrary SPEAR wrapper."""

    def call(
        self, function_name: str, request: dict[str, object]
    ) -> dict[str, object]: ...


class PixelGoalRejected(Exception):
    """An explicit Pixel Goal resolution or execution rejection."""

    def __init__(
        self,
        reason: str,
        *,
        raw_world_hit: Vec3 | None = None,
        audit: dict[str, object] | None = None,
    ) -> None:
        self.reason = reason
        self.raw_world_hit = raw_world_hit
        self.audit = dict(audit or {})
        super().__init__(reason)


class PixelGoalPairIntegrityError(RuntimeError):
    """A UE pair or bound resolver response cannot name the observed frame."""

    def __init__(
        self, reason: str, *, audit: dict[str, object] | None = None
    ) -> None:
        self.reason = reason
        self.audit = dict(audit or {})
        super().__init__(reason)


class PixelGoalExecutionError(RuntimeError):
    """UE misbehaved AFTER a move was accepted and had started.

    Kept apart from ``PixelGoalRejected`` on purpose. A rejection is a verdict
    on the pixel the policy chose and is narrated back to it; an error while
    polling or cancelling a move already under way says nothing about the
    pixel, may leave the pawn somewhere new, and has to reach the harness as
    an infrastructure failure rather than as "that pixel was unwalkable".
    """


class PixelGoalInfrastructureTimeout(RuntimeError):
    """The host watchdog stopped a move while UE was still non-terminal.

    Unlike ``ControllerOutcomeCode.EXECUTION_TIMEOUT``, this is deliberately
    raised out of the policy step. Treating a frozen or exceptionally slow
    editor as model feedback would mis-score the action and then continue from
    an unexpected partially moved pose.
    """

    def __init__(self, result: ControllerResult) -> None:
        self.result = result
        super().__init__(
            "Pixel Goal infrastructure watchdog expired after "
            f"{result.elapsed_sim_s:.3f} s of UE simulation"
        )


@dataclass(frozen=True)
class PixelGoalViewPair:
    capture_group_id: str
    pose: tuple[float, float, float, float]
    front: PixelGoalFrame
    rear: PixelGoalFrame
    capture_timing: dict[str, object]

    def frame(self, view: str) -> PixelGoalFrame:
        if view == "front":
            return self.front
        if view == "rear":
            return self.rear
        raise PixelGoalRejected("unknown_camera_view")


def _required_string(data: dict[str, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data)
    return value


def _optional_vec3(data: dict[str, object], field: str) -> Vec3 | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data)
    try:
        return Vec3(x_cm=float(value[0]), y_cm=float(value[1]), z_cm=float(value[2]))
    except (TypeError, ValueError) as exc:
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data) from exc


def _required_vec3(data: dict[str, object], field: str) -> Vec3:
    value = _optional_vec3(data, field)
    if value is None:
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data)
    return value


def _optional_vec3_list(data: dict[str, object], field: str) -> list[Vec3]:
    value = data.get(field)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data)
    result: list[Vec3] = []
    for item in value:
        result.append(_required_vec3({field: item}, field))
    return result


def _optional_nonnegative_number(
    data: dict[str, object], field: str,
) -> float | None:
    value = data.get(field)
    if value is None:
        return None
    if isinstance(value, bool):
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise PixelGoalRejected(f"invalid_ue_response:{field}", audit=data)
    return parsed


def _pair_integrity(
    reason: str, response: dict[str, object]
) -> PixelGoalPairIntegrityError:
    return PixelGoalPairIntegrityError(reason, audit=response)


def _pair_number(
    value: object, field: str, response: dict[str, object]
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _pair_integrity(f"invalid_view_pair:{field}", response)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _pair_integrity(f"invalid_view_pair:{field}", response)
    return parsed


def _pair_timing(
    value: object, field: str, response: dict[str, object]
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _pair_integrity(f"invalid_view_pair:{field}", response)
    expected = {"capture_read_ms", "encode_ms", "wall_ms"}
    if set(value) != expected:
        raise _pair_integrity(f"invalid_view_pair:{field}", response)
    parsed: dict[str, object] = {}
    for key in expected:
        number = _pair_number(value[key], f"{field}.{key}", response)
        if number < 0.0:
            raise _pair_integrity(f"invalid_view_pair:{field}.{key}", response)
        parsed[key] = number
    return parsed


def _pair_image_data_url(
    value: object, response: dict[str, object]
) -> str:
    prefix = "data:image/jpeg;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise _pair_integrity("invalid_view_pair:rgb_data_url", response)
    payload = value[len(prefix):]
    if not payload:
        raise _pair_integrity("invalid_view_pair:rgb_data_url", response)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _pair_integrity("invalid_view_pair:rgb_data_url", response) from exc
    if not decoded:
        raise _pair_integrity("invalid_view_pair:rgb_data_url", response)
    return value


def _float32(value: float) -> float:
    """Round once exactly as Task 4's JSON-double -> C++-float cast does."""
    try:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as exc:
        raise ValueError("capture fov is outside float32 range") from exc


class LivePixelGoalRuntime:
    """Associate RGB snapshots with Pixel Goal actions and execute them in UE."""

    def __init__(
        self, endpoint: PixelGoalEndpoint, config: PixelGoalConfig | None = None
    ) -> None:
        self.endpoint = endpoint
        self.config = config or PixelGoalConfig()

    def capture_frame(
        self,
        *,
        agent_tag: str,
        width_px: int = 640,
        height_px: int = 360,
    ) -> PixelGoalFrame:
        if not agent_tag:
            raise ValueError("agent_tag must be non-empty")
        if width_px <= 0 or height_px <= 0:
            raise ValueError("capture dimensions must be positive")
        request: dict[str, object] = {
            "agent_tag": agent_tag,
            "width": width_px,
            "height": height_px,
            "fov_degrees": self.config.fov_degrees,
            "jpeg_quality": self.config.jpeg_quality,
        }
        response = self.endpoint.call("PixelGoal_CaptureFrameJson", request)
        if response.get("success") is not True:
            raise PixelGoalRejected(
                str(response.get("rejection_reason") or response.get("error") or "capture_failed"),
                audit=response,
            )
        try:
            width = int(response.get("width", 0))
            height = int(response.get("height", 0))
        except (TypeError, ValueError) as exc:
            raise PixelGoalRejected("invalid_ue_response:dimensions", audit=response) from exc
        return PixelGoalFrame(
            rgb_data_url=_required_string(response, "rgb_data_url"),
            camera_snapshot_id=_required_string(response, "camera_snapshot_id"),
            camera_intrinsics_id=_required_string(response, "camera_intrinsics_id"),
            width_px=width,
            height_px=height,
            agent_tag=agent_tag,
        )

    def capture_view_pair(
        self,
        *,
        agent_tag: str,
        width_px: int = 640,
        height_px: int = 360,
        fov_degrees: float | None = None,
        commit_snapshot: bool = True,
    ) -> PixelGoalViewPair:
        if not agent_tag:
            raise ValueError("agent_tag must be non-empty")
        if width_px <= 0 or height_px <= 0:
            raise ValueError("capture dimensions must be positive")
        if not isinstance(commit_snapshot, bool):
            raise ValueError("commit_snapshot must be a boolean")
        requested_fov = (
            self.config.fov_degrees if fov_degrees is None else fov_degrees
        )
        if not math.isfinite(requested_fov) or requested_fov <= 0.0:
            raise ValueError("capture fov must be positive and finite")
        requested_fov = _float32(requested_fov)
        request: dict[str, object] = {
            "agent_tag": agent_tag,
            "width": width_px,
            "height": height_px,
            "fov_degrees": requested_fov,
            "jpeg_quality": self.config.jpeg_quality,
            "commit_snapshot": commit_snapshot,
        }
        response = self.endpoint.call("PixelGoal_CaptureViewPairJson", request)
        if not isinstance(response, dict):
            raise PixelGoalPairIntegrityError("invalid_view_pair:response")
        if response.get("success") is not True:
            reason = str(response.get("error") or "view_pair_capture_failed")
            raise _pair_integrity(reason, response)
        if response.get("snapshots_committed") is not commit_snapshot:
            raise _pair_integrity(
                "view_pair_snapshot_commit_mismatch", response)

        group = response.get("capture_group_id")
        if not isinstance(group, str) or not group:
            raise _pair_integrity("invalid_view_pair:capture_group_id", response)
        response_agent = response.get("agent_tag")
        if not isinstance(response_agent, str) or response_agent != agent_tag:
            raise _pair_integrity("view_pair_agent_mismatch", response)

        raw_pose = response.get("pose")
        if not isinstance(raw_pose, dict):
            raise _pair_integrity("invalid_view_pair:pose", response)
        pose = tuple(
            _pair_number(raw_pose.get(field), f"pose.{field}", response)
            for field in ("x_cm", "y_cm", "z_cm", "yaw_deg")
        )

        raw_views = response.get("views")
        if not isinstance(raw_views, list) or len(raw_views) != 2:
            raise _pair_integrity("incomplete_view_pair", response)

        frames: list[PixelGoalFrame] = []
        for expected_view, raw_view in zip(("front", "rear"), raw_views):
            if not isinstance(raw_view, dict):
                raise _pair_integrity("invalid_view_pair:image", response)
            if raw_view.get("success") is not True:
                raise _pair_integrity("invalid_view_pair:image", response)
            if raw_view.get("view_id") != expected_view:
                raise _pair_integrity("view_pair_order_or_view_mismatch", response)
            if raw_view.get("capture_group_id") != group:
                raise _pair_integrity("view_pair_group_mismatch", response)

            raw_location = raw_view.get("loc_cm")
            if not isinstance(raw_location, (list, tuple)) or len(raw_location) != 3:
                raise _pair_integrity("invalid_view_pair:loc_cm", response)
            view_pose = tuple(
                _pair_number(value, "loc_cm", response) for value in raw_location
            )
            view_yaw = _pair_number(raw_view.get("yaw_deg"), "yaw_deg", response)
            if view_pose != pose[:3] or view_yaw != pose[3]:
                raise _pair_integrity("view_pair_pose_mismatch", response)

            width = raw_view.get("width")
            height = raw_view.get("height")
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or isinstance(height, bool)
                or not isinstance(height, int)
                or width != width_px
                or height != height_px
            ):
                raise _pair_integrity("view_pair_intrinsics_mismatch", response)
            fov = _pair_number(
                raw_view.get("camera_horizontal_fov_degrees"),
                "camera_horizontal_fov_degrees",
                response,
            )
            if fov != requested_fov:
                raise _pair_integrity("view_pair_intrinsics_mismatch", response)

            rotation = raw_view.get("camera_rotation_degrees")
            if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
                raise _pair_integrity(
                    "invalid_view_pair:camera_rotation_degrees", response
                )
            camera_rotation = tuple(
                _pair_number(value, "camera_rotation_degrees", response)
                for value in rotation
            )
            timing = _pair_timing(raw_view.get("timing"), "timing", response)
            raw_camera = raw_view.get("camera_location_cm")
            camera_location: tuple[float, float, float] | None = None
            if isinstance(raw_camera, (list, tuple)) and len(raw_camera) == 3:
                camera_location = tuple(  # type: ignore[assignment]
                    _pair_number(value, "camera_location_cm", response)
                    for value in raw_camera
                )
            data_url = _pair_image_data_url(
                raw_view.get("rgb_data_url"), response
            )
            snapshot = raw_view.get("camera_snapshot_id")
            intrinsics = raw_view.get("camera_intrinsics_id")
            if not isinstance(snapshot, str) or not snapshot:
                raise _pair_integrity(
                    "invalid_view_pair:camera_snapshot_id", response
                )
            if not isinstance(intrinsics, str) or not intrinsics:
                raise _pair_integrity(
                    "invalid_view_pair:camera_intrinsics_id", response
                )
            try:
                frames.append(PixelGoalFrame(
                    rgb_data_url=data_url,
                    camera_snapshot_id=snapshot,
                    camera_intrinsics_id=intrinsics,
                    width_px=width,
                    height_px=height,
                    agent_tag=agent_tag,
                    view_id=expected_view,
                    capture_group_id=group,
                    camera_yaw_deg=camera_rotation[1],
                    capture_timing=timing,
                    camera_location_cm=camera_location,
                ))
            except ValueError as exc:
                raise _pair_integrity("invalid_view_pair:image", response) from exc

        if frames[0].camera_snapshot_id == frames[1].camera_snapshot_id:
            raise _pair_integrity("duplicate_view_pair_snapshot", response)
        if frames[0].camera_intrinsics_id != frames[1].camera_intrinsics_id:
            raise _pair_integrity("view_pair_intrinsics_mismatch", response)

        aggregate = response.get("capture_timing")
        expected_aggregate = {
            "total_ms", "lookup_ms", "init_ms", "capture_read_ms", "encode_ms"
        }
        if not isinstance(aggregate, dict) or set(aggregate) != expected_aggregate:
            raise _pair_integrity("invalid_view_pair:capture_timing", response)
        aggregate_numbers = {
            key: _pair_number(value, f"capture_timing.{key}", response)
            for key, value in aggregate.items()
        }
        if any(value < 0.0 for value in aggregate_numbers.values()):
            raise _pair_integrity("invalid_view_pair:capture_timing", response)
        pair_wall_ms = _pair_number(
            response.get("pair_capture_wall_ms"), "pair_capture_wall_ms", response
        )
        if pair_wall_ms < 0.0:
            raise _pair_integrity("invalid_view_pair:pair_capture_wall_ms", response)
        pair_timing: dict[str, object] = {
            "capture_read_ms": aggregate_numbers["capture_read_ms"],
            "encode_ms": aggregate_numbers["encode_ms"],
            "wall_ms": pair_wall_ms,
        }
        return PixelGoalViewPair(
            capture_group_id=group,
            pose=(pose[0], pose[1], pose[2], pose[3]),
            front=frames[0],
            rear=frames[1],
            capture_timing=pair_timing,
        )

    def start(
        self, action: NavPixelGoalAction, frame: PixelGoalFrame
    ) -> PixelGoalStarted:
        request: dict[str, object] = {
            "camera_snapshot_id": frame.camera_snapshot_id,
            "u_norm": action.target.u_norm,
            "v_norm": action.target.v_norm,
            "max_navmesh_adjustment_cm": self.config.max_navmesh_adjustment_cm,
            "acceptance_radius_cm": self.config.acceptance_radius_cm,
            "trace_distance_cm": self.config.trace_distance_cm,
            "max_ground_slope_deg": self.config.max_ground_slope_deg,
            "max_controller_path_stretch_ratio": (
                self.config.max_controller_path_stretch_ratio),
            "controller_path_detour_allowance_cm": (
                self.config.controller_path_detour_allowance_cm),
        }
        if frame.agent_tag is not None:
            request["agent_tag"] = frame.agent_tag
        bound = frame.view_id is not None or frame.capture_group_id is not None
        if bound:
            if frame.view_id is None or frame.capture_group_id is None:
                raise PixelGoalPairIntegrityError("incomplete_frame_binding")
            request["view_id"] = frame.view_id
            request["capture_group_id"] = frame.capture_group_id
        response = self.endpoint.call("PixelGoal_ResolveAndMoveJson", request)
        if bound and not isinstance(response, dict):
            raise PixelGoalPairIntegrityError("invalid_resolver_response")
        if bound and (
            response.get("view_id") != frame.view_id
            or response.get("capture_group_id") != frame.capture_group_id
        ):
            raise PixelGoalPairIntegrityError(
                "resolver_binding_mismatch", audit=response
            )
        raw_hit = _optional_vec3(response, "raw_world_hit_cm")
        if response.get("accepted") is not True:
            raise PixelGoalRejected(
                str(response.get("rejection_reason") or "pixel_goal_rejected"),
                raw_world_hit=raw_hit,
                audit=response,
            )

        response_snapshot = _required_string(response, "camera_snapshot_id")
        if response_snapshot != frame.camera_snapshot_id:
            raise PixelGoalRejected("camera_snapshot_mismatch", audit=response)
        response_intrinsics = _required_string(response, "camera_intrinsics_id")
        if response_intrinsics != frame.camera_intrinsics_id:
            raise PixelGoalRejected("camera_intrinsics_mismatch", audit=response)
        if raw_hit is None:
            raise PixelGoalRejected("invalid_ue_response:raw_world_hit_cm", audit=response)
        validated = _required_vec3(response, "validated_navigation_target_cm")
        try:
            adjustment_cm = float(response["navmesh_adjustment_cm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PixelGoalRejected(
                "invalid_ue_response:navmesh_adjustment_cm", audit=response
            ) from exc

        navigation_request = NavigationRequest(
            request_id=_required_string(response, "request_id"),
            mode=NavigationMode.NAV_PIXEL_GOAL,
            distance_source=DistanceSource.UE_GEOMETRY_TRACE,
            source_image_point=action.target,
            camera_snapshot_id=response_snapshot,
            camera_intrinsics_id=response_intrinsics,
            raw_world_hit=raw_hit,
            validated_navigation_target=validated,
            target_world=raw_hit,
            projected_target=validated,
            navmesh_adjustment_cm=adjustment_cm,
            controller_path_points=_optional_vec3_list(
                response, "controller_path_points_cm"),
            controller_path_length_cm=_optional_nonnegative_number(
                response, "controller_path_length_cm"),
            controller_path_direct_cm=_optional_nonnegative_number(
                response, "controller_path_direct_cm"),
            controller_path_stretch_ratio=_optional_nonnegative_number(
                response, "controller_path_stretch_ratio"),
            max_range_m=self.config.trace_distance_cm / 100.0,
        )
        actor = response.get("raw_hit_actor")
        return PixelGoalStarted(
            request=navigation_request,
            controller_request_result=_required_string(
                response, "controller_request_result"
            ),
            raw_hit_actor=actor if isinstance(actor, str) and actor else None,
        )

    def poll(self, request_id: str) -> ControllerResult | PixelGoalInProgress:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        response = self.endpoint.call(
            "PixelGoal_GetMoveStatusJson", {"request_id": request_id}
        )
        response_id = _required_string(response, "request_id")
        if response_id != request_id:
            raise PixelGoalRejected("controller_request_mismatch", audit=response)
        state = _required_string(response, "state")
        feet = _optional_vec3(response, "final_feet_position_cm")
        if state not in {"completed", "failed"}:
            return PixelGoalInProgress(
                request_id=request_id,
                state=state,
                feet_position=feet,
                elapsed_sim_s=_optional_nonnegative_number(
                    response, "elapsed_sim_s"),
                distance_travelled_cm=_optional_nonnegative_number(
                    response, "distance_travelled_cm"),
            )

        return self._terminal_result(response, request_id=request_id)

    def cancel(
        self,
        request_id: str,
        *,
        reason: str = "execution_timeout",
    ) -> ControllerResult:
        """Abort an in-flight UE move and return its final audited position."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not reason:
            raise ValueError("cancel reason must be non-empty")
        response = self.endpoint.call(
            "PixelGoal_CancelMoveJson",
            {"request_id": request_id, "reason": reason},
        )
        if response.get("cancelled") is not True and response.get("state") in {
            "completed",
            "failed",
        }:
            # The controller may have become terminal between the final poll
            # and the abort call.  Preserve that authoritative result instead
            # of relabelling a completed move as a timeout.
            return self._terminal_result(response, request_id=request_id)
        if response.get("cancelled") is not True:
            raise PixelGoalRejected(
                str(response.get("error") or "controller_cancel_failed"),
                audit=response,
            )
        result = self._terminal_result(
            response,
            request_id=request_id,
            forced_outcome=(
                ControllerOutcomeCode.EXECUTION_TIMEOUT
                if reason == "execution_timeout"
                else ControllerOutcomeCode.CONTROLLER_FAILED
            ),
        )
        if result.failure_reason != reason:
            raise PixelGoalRejected("controller_cancel_reason_mismatch", audit=response)
        return result

    def _terminal_result(
        self,
        response: dict[str, object],
        *,
        request_id: str,
        forced_outcome: ControllerOutcomeCode | None = None,
    ) -> ControllerResult:
        response_id = _required_string(response, "request_id")
        if response_id != request_id:
            raise PixelGoalRejected("controller_request_mismatch", audit=response)
        state = _required_string(response, "state")
        if state not in {"completed", "failed"}:
            raise PixelGoalRejected("controller_not_terminal", audit=response)
        feet = _optional_vec3(response, "final_feet_position_cm")

        accepted_target = _required_vec3(response, "accepted_target_cm")
        final_agent = _required_vec3(response, "final_agent_position_cm")
        if feet is None:
            raise PixelGoalRejected(
                "invalid_ue_response:final_feet_position_cm", audit=response
            )
        planar_error_m = math.hypot(
            feet.x_cm - accepted_target.x_cm,
            feet.y_cm - accepted_target.y_cm,
        ) / 100.0
        error_3d_m = feet.distance_cm(accepted_target) / 100.0
        controller_result = _required_string(response, "controller_result")
        succeeded = state == "completed" and controller_result == "success"
        try:
            yaw_deg = float(response.get("final_yaw_degrees", 0.0))
            elapsed_sim_s = float(response.get("elapsed_sim_s", 0.0))
            distance_travelled_cm = float(response.get("distance_travelled_cm", 0.0))
        except (TypeError, ValueError) as exc:
            raise PixelGoalRejected("invalid_ue_response:controller_metrics", audit=response) from exc
        failure_reason = "" if succeeded else str(
            response.get("failure_reason") or controller_result
        )
        return ControllerResult(
            request_id=request_id,
            outcome=(
                forced_outcome
                or (
                    ControllerOutcomeCode.ACCEPTED
                    if succeeded
                    else ControllerOutcomeCode.CONTROLLER_FAILED
                )
            ),
            requested_target=accepted_target,
            accepted_target=accepted_target,
            final_pose=Pose(
                frame=FrameName.UE_WORLD,
                position=final_agent,
                yaw_deg=yaw_deg,
            ),
            final_feet_position=feet,
            controller_result=controller_result,
            execution_error_planar_m=planar_error_m,
            execution_error_3d_m=error_3d_m,
            elapsed_sim_s=elapsed_sim_s,
            distance_travelled_cm=distance_travelled_cm,
            failure_reason=failure_reason,
        )

    def execute(
        self, action: NavPixelGoalAction, frame: PixelGoalFrame
    ) -> tuple[PixelGoalStarted, ControllerResult]:
        started = self.start(action, frame)
        try:
            return self._drive(started)
        except PixelGoalRejected as error:
            raise PixelGoalExecutionError(
                f"{error.reason} while executing {started.request.request_id}"
            ) from error

    def _drive(self, started: PixelGoalStarted) -> tuple[PixelGoalStarted, ControllerResult]:
        wall_watchdog_deadline = (
            time.monotonic() + self.config.execution_timeout_s)
        while time.monotonic() < wall_watchdog_deadline:
            status = self.poll(started.request.request_id)
            if isinstance(status, ControllerResult):
                return started, status
            if (status.elapsed_sim_s is not None
                    and status.elapsed_sim_s
                    >= self.config.movement_timeout_sim_s):
                return started, self.cancel(
                    started.request.request_id,
                    reason="execution_timeout",
                )
            time.sleep(self.config.poll_interval_s)
        cancelled = self.cancel(
            started.request.request_id,
            reason="wall_watchdog_timeout",
        )
        # Preserve the authoritative terminal result if completion raced the
        # watchdog's final cancellation RPC.
        if cancelled.outcome is ControllerOutcomeCode.ACCEPTED:
            return started, cancelled
        raise PixelGoalInfrastructureTimeout(cancelled)
