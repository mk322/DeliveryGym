"""Fail-closed readiness checks for policy-visible Pixel Goal RGB pairs.

The UE cameras keep rendering continuously, but a teleport or a longer walk can
temporarily expose incomplete streaming or an adapting eye-exposure history.
This module samples non-actionable preview pairs until both views are usable and
stable, then verifies that the one committed pair shown to the policy matches
the converged preview.  It never edits pixels or changes model input.
"""

from __future__ import annotations

import base64
import binascii
import copy
import io
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalFrame,
    PixelGoalViewPair,
)


CAPTURE_READINESS_VERSION = "stable-preview-pair-v1"
VULKAN_OOM_PATTERNS = (
    "out of memory on vulkan",
    "failed to allocate device memory",
    "vk_error_out_of_device_memory",
)


class CaptureReadinessError(RuntimeError):
    """The renderer did not produce a trustworthy policy observation."""


@dataclass(frozen=True)
class CaptureReadinessConfig:
    max_wait_s: float = 45.0
    preview_interval_s: float = 0.35
    stable_transitions_required: int = 2
    stability_mae_max: float = 0.02
    committed_mae_max: float = 0.04
    min_luma_p95: float = 12.0
    max_black_fraction: float = 0.98
    max_white_fraction: float = 0.98
    analysis_width: int = 160
    analysis_height: int = 90
    engine_idle_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        positive = {
            "max_wait_s": self.max_wait_s,
            "preview_interval_s": self.preview_interval_s,
            "stability_mae_max": self.stability_mae_max,
            "committed_mae_max": self.committed_mae_max,
            "min_luma_p95": self.min_luma_p95,
            "engine_idle_timeout_s": self.engine_idle_timeout_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.stable_transitions_required < 1:
            raise ValueError("stable_transitions_required must be positive")
        if self.analysis_width < 16 or self.analysis_height < 16:
            raise ValueError("analysis dimensions must be at least 16 pixels")
        for name in ("stability_mae_max", "committed_mae_max"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most 1")
        for name in ("max_black_fraction", "max_white_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_luma_p95 > 255.0:
            raise ValueError("min_luma_p95 must be at most 255")


def _histogram_percentile(histogram: list[int], percentile: float) -> float:
    total = sum(histogram)
    if total <= 0:
        raise CaptureReadinessError("capture_readiness_empty_image")
    rank = max(0.0, min(1.0, percentile)) * (total - 1)
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative > rank:
            return float(value)
    return 255.0


def _decode_luma(frame: PixelGoalFrame) -> Image.Image:
    prefix = "data:image/jpeg;base64,"
    if not frame.rgb_data_url.startswith(prefix):
        raise CaptureReadinessError("capture_readiness_invalid_image_data_url")
    try:
        payload = base64.b64decode(
            frame.rgb_data_url[len(prefix):], validate=True)
    except (binascii.Error, ValueError) as error:
        raise CaptureReadinessError(
            "capture_readiness_invalid_image_base64") from error
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.size != (frame.width_px, frame.height_px):
                raise CaptureReadinessError(
                    "capture_readiness_image_dimensions_mismatch")
            return image.convert("L")
    except CaptureReadinessError:
        raise
    except Exception as error:
        raise CaptureReadinessError(
            "capture_readiness_invalid_jpeg") from error


def analyze_frame(
    frame: PixelGoalFrame,
    config: CaptureReadinessConfig,
) -> tuple[dict[str, float | bool], bytes]:
    """Return auditable luma metrics plus a private blurred signature."""

    luma = _decode_luma(frame)
    histogram = luma.histogram()
    pixels = float(luma.width * luma.height)
    statistics = ImageStat.Stat(luma)
    metrics: dict[str, float | bool] = {
        "mean": round(float(statistics.mean[0]), 4),
        "stddev": round(math.sqrt(float(statistics.var[0])), 4),
        "p50": _histogram_percentile(histogram, 0.50),
        "p90": _histogram_percentile(histogram, 0.90),
        "p95": _histogram_percentile(histogram, 0.95),
        "p99": _histogram_percentile(histogram, 0.99),
        "black_fraction_lte_8": round(sum(histogram[:9]) / pixels, 6),
        "white_fraction_gte_247": round(sum(histogram[247:]) / pixels, 6),
    }
    metrics["usable"] = bool(
        metrics["p95"] >= config.min_luma_p95
        and metrics["black_fraction_lte_8"] <= config.max_black_fraction
        and metrics["white_fraction_gte_247"] <= config.max_white_fraction
    )
    signature_image = luma.resize(
        (config.analysis_width, config.analysis_height),
        Image.Resampling.BILINEAR,
    ).filter(ImageFilter.GaussianBlur(radius=1.0))
    return metrics, signature_image.tobytes()


def normalized_mae(first: bytes, second: bytes) -> float:
    if not first or len(first) != len(second):
        raise CaptureReadinessError("capture_readiness_signature_mismatch")
    return sum(abs(a - b) for a, b in zip(first, second)) / (
        len(first) * 255.0)


class VulkanOomMonitor:
    """Scan only the current launcher log and fail on new Vulkan OOM evidence."""

    def __init__(self, log_path: Path | str | None) -> None:
        self.log_path = Path(log_path).resolve() if log_path else None
        self._offset = 0
        self._checks = 0
        self._matched_pattern: str | None = None

    def assert_healthy(self) -> None:
        self._checks += 1
        if self._matched_pattern is not None:
            raise CaptureReadinessError(
                f"renderer_vulkan_oom:{self._matched_pattern}")
        if self.log_path is None or not self.log_path.is_file():
            return
        size = self.log_path.stat().st_size
        if size < self._offset:
            self._offset = 0
        with self.log_path.open("r", encoding="utf-8", errors="replace") as log:
            log.seek(self._offset)
            chunk = log.read()
            self._offset = log.tell()
        lowered = chunk.lower()
        self._matched_pattern = next(
            (pattern for pattern in VULKAN_OOM_PATTERNS if pattern in lowered),
            None,
        )
        if self._matched_pattern is not None:
            raise CaptureReadinessError(
                f"renderer_vulkan_oom:{self._matched_pattern}")

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.log_path is not None,
            "log_path": str(self.log_path) if self.log_path else None,
            "checks": self._checks,
            "matched_pattern": self._matched_pattern,
            "healthy": self._matched_pattern is None,
        }


class CaptureReadinessGate:
    """Converge previews and return only a matching committed view pair."""

    def __init__(
        self,
        config: CaptureReadinessConfig | None = None,
        *,
        oom_monitor: VulkanOomMonitor | None = None,
        engine_idle_waiter: Callable[[float], Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CaptureReadinessConfig()
        self.oom_monitor = oom_monitor or VulkanOomMonitor(None)
        self.engine_idle_waiter = engine_idle_waiter
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self._checkpoints: list[dict[str, Any]] = []
        self._settle_requests = 0
        self._preview_pairs = 0
        self._committed_pairs = 0
        self._discarded_committed_pairs = 0
        self._engine_idle_waits: list[Any] = []

    def capture(
        self,
        runtime: LivePixelGoalRuntime,
        *,
        agent_tag: str,
        width_px: int,
        height_px: int,
        fov_degrees: float,
    ) -> PixelGoalViewPair:
        self._settle_requests += 1
        settle_index = self._settle_requests
        started = self.monotonic_fn()
        previous_signatures: dict[str, bytes] | None = None
        stable_transitions = 0
        preview_attempt = 0
        waited_for_idle = False
        last_checkpoint: dict[str, Any] | None = None

        while self.monotonic_fn() - started < self.config.max_wait_s:
            self.oom_monitor.assert_healthy()
            preview_attempt += 1
            preview = runtime.capture_view_pair(
                agent_tag=agent_tag,
                width_px=width_px,
                height_px=height_px,
                fov_degrees=fov_degrees,
                commit_snapshot=False,
            )
            self._preview_pairs += 1
            metrics, signatures = self._analyze_pair(preview)
            transition = (
                {
                    view: round(normalized_mae(
                        previous_signatures[view], signatures[view]), 6)
                    for view in ("front", "rear")
                }
                if previous_signatures is not None else None
            )
            usable = all(bool(metrics[view]["usable"])
                         for view in ("front", "rear"))
            stable = bool(
                transition is not None
                and max(transition.values()) <= self.config.stability_mae_max
            )
            stable_transitions = (
                stable_transitions + 1 if usable and stable else 0)
            last_checkpoint = {
                "settle_index": settle_index,
                "preview_attempt": preview_attempt,
                "kind": "preview",
                "capture_group_id": preview.capture_group_id,
                "elapsed_s": round(self.monotonic_fn() - started, 4),
                "metrics": metrics,
                "transition_mae": transition,
                "usable": usable,
                "stable_transition": stable,
                "stable_transitions": stable_transitions,
            }
            self._checkpoints.append(last_checkpoint)
            previous_signatures = signatures

            # The first camera read requests the current streaming region.
            # Only then can SPEAR's engine-idle service authoritatively wait on
            # the same world/resources that the policy camera needs.
            if not waited_for_idle and self.engine_idle_waiter is not None:
                remaining = self.config.max_wait_s - (
                    self.monotonic_fn() - started)
                if remaining <= 1e-6:
                    break
                wait_result = self.engine_idle_waiter(min(
                    remaining, self.config.engine_idle_timeout_s))
                self._engine_idle_waits.append(copy.deepcopy(wait_result))
                waited_for_idle = True

            if stable_transitions >= self.config.stable_transitions_required:
                committed = runtime.capture_view_pair(
                    agent_tag=agent_tag,
                    width_px=width_px,
                    height_px=height_px,
                    fov_degrees=fov_degrees,
                    commit_snapshot=True,
                )
                self._committed_pairs += 1
                committed_metrics, committed_signatures = self._analyze_pair(
                    committed)
                committed_delta = {
                    view: round(normalized_mae(
                        signatures[view], committed_signatures[view]), 6)
                    for view in ("front", "rear")
                }
                committed_usable = all(
                    bool(committed_metrics[view]["usable"])
                    for view in ("front", "rear")
                )
                committed_matches = bool(
                    committed_usable
                    and max(committed_delta.values())
                    <= self.config.committed_mae_max
                )
                self._checkpoints.append({
                    "settle_index": settle_index,
                    "preview_attempt": preview_attempt,
                    "kind": "committed",
                    "capture_group_id": committed.capture_group_id,
                    "elapsed_s": round(self.monotonic_fn() - started, 4),
                    "metrics": committed_metrics,
                    "preview_to_committed_mae": committed_delta,
                    "usable": committed_usable,
                    "matches_converged_preview": committed_matches,
                })
                if committed_matches:
                    self.oom_monitor.assert_healthy()
                    return committed
                self._discarded_committed_pairs += 1
                previous_signatures = committed_signatures
                stable_transitions = 0

            remaining = self.config.max_wait_s - (
                self.monotonic_fn() - started)
            if remaining > 1e-6:
                self.sleep_fn(min(self.config.preview_interval_s, remaining))
            else:
                break

        self.oom_monitor.assert_healthy()
        reason = (
            "capture_readiness_timeout"
            if last_checkpoint is None
            else "capture_readiness_timeout:"
                 f"usable={last_checkpoint['usable']},"
                 f"stable_transitions={last_checkpoint['stable_transitions']}"
        )
        raise CaptureReadinessError(reason)

    def _analyze_pair(
        self, pair: PixelGoalViewPair,
    ) -> tuple[dict[str, dict[str, float | bool]], dict[str, bytes]]:
        metrics: dict[str, dict[str, float | bool]] = {}
        signatures: dict[str, bytes] = {}
        for view, frame in (("front", pair.front), ("rear", pair.rear)):
            metrics[view], signatures[view] = analyze_frame(frame, self.config)
        return metrics, signatures

    def report(self) -> dict[str, Any]:
        return {
            "version": CAPTURE_READINESS_VERSION,
            "enabled": True,
            "configuration": asdict(self.config),
            "settle_requests": self._settle_requests,
            "preview_pairs": self._preview_pairs,
            "committed_pairs": self._committed_pairs,
            "discarded_committed_pairs": self._discarded_committed_pairs,
            "engine_idle_waits": copy.deepcopy(self._engine_idle_waits),
            "checkpoints": copy.deepcopy(self._checkpoints),
            "vulkan_oom_monitor": self.oom_monitor.report(),
        }
