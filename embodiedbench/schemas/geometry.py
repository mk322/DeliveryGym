"""Coordinate frames, poses, and transforms.

design plan §6.1 P1 requires explicit UE, bundle-world, map-render, camera, and
embodiment frames, and forbids implicit metre/centimetre conversion. Both rules
are enforced in types rather than left to convention:

- every length carries its unit in the field name (``_m`` or ``_cm``);
- a ``Transform`` names the frame it maps *from* and *to*, and composing two
  transforms whose frames do not meet raises ``FrameError``.

The Paris export is authored in centimetres (design plan §3.2 quotes bounds in cm)
while navigation ranges are specified in metres (design plan §9.4 ``max_range_m``).
Both units therefore exist on purpose; what is forbidden is a bare number whose
unit you have to infer.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from embodiedbench.schemas.base import FrameError, SchemaModel

CM_PER_M = 100.0


class FrameName(str, Enum):
    """The coordinate frames design plan §6.1 P1 names."""

    UE_WORLD = "ue_world"
    BUNDLE_WORLD = "bundle_world"
    MAP_RENDER = "map_render"
    CAMERA = "camera"
    EMBODIMENT = "embodiment"


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Vec3(SchemaModel):
    """A point or vector in centimetres, in some named frame."""

    x_cm: float
    y_cm: float
    z_cm: float = 0.0

    @field_validator("x_cm", "y_cm", "z_cm")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinate must be finite")
        return value

    @classmethod
    def from_m(cls, x_m: float, y_m: float, z_m: float = 0.0) -> "Vec3":
        return cls(x_cm=x_m * CM_PER_M, y_cm=y_m * CM_PER_M, z_cm=z_m * CM_PER_M)

    def as_m(self) -> tuple[float, float, float]:
        return (self.x_cm / CM_PER_M, self.y_cm / CM_PER_M, self.z_cm / CM_PER_M)

    def distance_cm(self, other: "Vec3") -> float:
        return math.dist((self.x_cm, self.y_cm, self.z_cm), (other.x_cm, other.y_cm, other.z_cm))


class Pose(SchemaModel):
    """A position and yaw in a named frame.

    Yaw alone, not a full rotation: every v1 navigation mode terminates at a
    ground pose with a heading (design plan §9.5's lattice is "2.0 m spatial with 8
    quantized headings"). Pitch and roll would be unconstrained degrees of
    freedom that no v1 action can set, and an unconstrained field in a
    conformance hash is a source of false mismatches.
    """

    frame: FrameName
    position: Vec3
    yaw_deg: float = 0.0

    @field_validator("yaw_deg")
    @classmethod
    def _normalize_yaw(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("yaw must be finite")
        # Normalize to [0, 360) so 0 and 360 cannot hash differently.
        return value % 360.0

    def yaw_difference_deg(self, other: "Pose") -> float:
        """Smallest absolute angle between two headings, in degrees."""
        delta = abs(self.yaw_deg - other.yaw_deg) % 360.0
        return min(delta, 360.0 - delta)

    def close_to(self, other: "Pose", *, position_cm: float, yaw_deg: float) -> bool:
        """Whether two poses agree within a stated tolerance.

        the design plan M10 states conformance tolerances as "at most 5 cm and 0.5
        degrees", so the comparison takes both explicitly rather than assuming a
        default a caller might not have thought about.
        """
        if self.frame is not other.frame:
            raise FrameError(f"cannot compare poses across frames: {self.frame} vs {other.frame}")
        return (
            self.position.distance_cm(other.position) <= position_cm
            and self.yaw_difference_deg(other) <= yaw_deg
        )


class CoordinateFrame(SchemaModel):
    """A frame's declared conventions."""

    name: FrameName
    units: str = Field(default="cm", pattern="^(cm|m)$")
    handedness: Handedness = Handedness.LEFT
    up_axis: str = Field(default="z", pattern="^(x|y|z)$")
    description: str = ""


class Transform(SchemaModel):
    """A rigid transform between two named frames.

    Translation-plus-yaw only, matching ``Pose``. Composition checks that the
    frames meet, so a UE->bundle transform cannot be silently applied to a
    camera-frame point.
    """

    from_frame: FrameName
    to_frame: FrameName
    translation: Vec3
    yaw_deg: float = 0.0
    scale: float = 1.0

    @model_validator(mode="after")
    def _distinct_frames(self) -> "Transform":
        if self.from_frame is self.to_frame:
            raise ValueError("a transform must relate two different frames")
        if self.scale <= 0 or not math.isfinite(self.scale):
            raise ValueError("scale must be positive and finite")
        return self

    def apply(self, point: Vec3) -> Vec3:
        radians = math.radians(self.yaw_deg)
        cos, sin = math.cos(radians), math.sin(radians)
        x = (point.x_cm * cos - point.y_cm * sin) * self.scale + self.translation.x_cm
        y = (point.x_cm * sin + point.y_cm * cos) * self.scale + self.translation.y_cm
        z = point.z_cm * self.scale + self.translation.z_cm
        return Vec3(x_cm=x, y_cm=y, z_cm=z)

    def inverse(self) -> "Transform":
        radians = math.radians(-self.yaw_deg)
        cos, sin = math.cos(radians), math.sin(radians)
        inv_scale = 1.0 / self.scale
        tx, ty, tz = (
            -self.translation.x_cm,
            -self.translation.y_cm,
            -self.translation.z_cm,
        )
        x = (tx * cos - ty * sin) * inv_scale
        y = (tx * sin + ty * cos) * inv_scale
        return Transform(
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            translation=Vec3(x_cm=x, y_cm=y, z_cm=tz * inv_scale),
            yaw_deg=-self.yaw_deg % 360.0,
            scale=inv_scale,
        )

    def then(self, other: "Transform") -> "Transform":
        """Compose ``self`` followed by ``other``, requiring the frames to meet."""
        if self.to_frame is not other.from_frame:
            raise FrameError(
                f"cannot compose {self.from_frame}->{self.to_frame} with "
                f"{other.from_frame}->{other.to_frame}"
            )
        combined_translation = other.apply(self.translation)
        return Transform(
            from_frame=self.from_frame,
            to_frame=other.to_frame,
            translation=combined_translation,
            yaw_deg=(self.yaw_deg + other.yaw_deg) % 360.0,
            scale=self.scale * other.scale,
        )


class CameraIntrinsics(SchemaModel):
    """Pinhole intrinsics plus the depth convention (design plan §6.1 P1, P5)."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fx_px: float = Field(gt=0)
    fy_px: float = Field(gt=0)
    cx_px: float = Field(ge=0)
    cy_px: float = Field(ge=0)
    near_cm: float = Field(gt=0)
    far_cm: float = Field(gt=0)
    # design plan §9.3 forbids relative depth: the unprojection maths is only valid
    # if depth is metric and its meaning is stated.
    depth_encoding: str = Field(default="metric_distance_cm", pattern="^metric_(distance|z)_cm$")

    @model_validator(mode="after")
    def _planes_ordered(self) -> "CameraIntrinsics":
        if self.far_cm <= self.near_cm:
            raise ValueError("far plane must exceed near plane")
        if self.cx_px > self.width_px or self.cy_px > self.height_px:
            raise ValueError("principal point lies outside the image")
        return self

    def unproject(self, u_norm: float, v_norm: float, distance_cm: float) -> Vec3:
        """Camera-frame point for a normalized image coordinate and metric range.

        This is the shared step design plan §9.4 puts at the head of both point
        modes, so both modes use one implementation rather than two that drift.
        """
        if not (0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0):
            raise ValueError(f"normalized image point out of range: ({u_norm}, {v_norm})")
        if not math.isfinite(distance_cm) or distance_cm <= 0:
            raise ValueError("distance must be positive and finite")
        u_px = u_norm * self.width_px
        v_px = v_norm * self.height_px
        dx = (u_px - self.cx_px) / self.fx_px
        dy = (v_px - self.cy_px) / self.fy_px
        norm = math.sqrt(dx * dx + dy * dy + 1.0)
        if self.depth_encoding == "metric_distance_cm":
            scale = distance_cm / norm
        else:
            scale = distance_cm
        return Vec3(x_cm=dx * scale, y_cm=dy * scale, z_cm=scale)

    def project(self, point: "Vec3") -> tuple[float, float, float] | None:
        """Normalized image coordinates and range for a camera-frame point.

        The exact inverse of :meth:`unproject`, which is what makes the design plan
        9.7.2's oracle possible: take a known world point, project it into the
        image, feed that image point back through the point-navigation chain,
        and check the chain returns where it started. A transform that is wrong
        but self-consistent survives every test that only runs one direction.

        Returns ``None`` when the point is behind the camera or falls outside
        the sensor, since neither can be expressed as an image coordinate and
        clamping would fabricate a point the camera never saw.
        """
        if point.z_cm <= 0:
            return None
        u_px = point.x_cm / point.z_cm * self.fx_px + self.cx_px
        v_px = point.y_cm / point.z_cm * self.fy_px + self.cy_px
        u_norm = u_px / self.width_px
        v_norm = v_px / self.height_px
        if not (0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0):
            return None
        distance_cm = math.sqrt(point.x_cm**2 + point.y_cm**2 + point.z_cm**2)
        if self.depth_encoding != "metric_distance_cm":
            distance_cm = point.z_cm
        return (u_norm, v_norm, distance_cm)


def require_frame(value: Any, expected: FrameName, *, what: str) -> None:
    """Raise ``FrameError`` unless ``value`` is expressed in ``expected``."""
    observed = getattr(value, "frame", None)
    if observed is not expected:
        raise FrameError(f"{what} must be in {expected.value}, got {getattr(observed, 'value', observed)!r}")
