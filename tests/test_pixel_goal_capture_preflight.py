from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from embodiedbench.runtime.pixel_goal import PixelGoalFrame, PixelGoalViewPair
from tools.pixel_goal_capture_preflight import (
    CaptureReadinessConfig,
    CaptureReadinessError,
    CaptureReadinessGate,
    VulkanOomMonitor,
    analyze_frame,
)


def jpeg_data_url(*, dark: bool = False, variant: int = 0) -> str:
    image = Image.new("RGB", (64, 36), (2, 2, 2) if dark else (70, 90, 110))
    if not dark:
        draw = ImageDraw.Draw(image)
        draw.rectangle((8 + variant, 8, 34 + variant, 30), fill=(180, 150, 90))
        draw.rectangle((40, 3, 61, 18), fill=(25, 45, 65))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(
        encoded.getvalue()).decode("ascii")


def frame(view: str, group: str, data_url: str) -> PixelGoalFrame:
    return PixelGoalFrame(
        rgb_data_url=data_url,
        camera_snapshot_id=f"{group}-{view}",
        camera_intrinsics_id="intrinsics",
        width_px=64,
        height_px=36,
        agent_tag="agent",
        view_id=view,
        capture_group_id=group,
        camera_yaw_deg=0.0 if view == "front" else 180.0,
        capture_timing={
            "capture_read_ms": 1.0,
            "encode_ms": 1.0,
            "wall_ms": 2.0,
        },
    )


def pair(index: int, data_url: str) -> PixelGoalViewPair:
    group = f"pair-{index}"
    return PixelGoalViewPair(
        capture_group_id=group,
        pose=(1.0, 2.0, 90.0, 45.0),
        front=frame("front", group, data_url),
        rear=frame("rear", group, data_url),
        capture_timing={
            "capture_read_ms": 2.0,
            "encode_ms": 2.0,
            "wall_ms": 4.0,
        },
    )


class PairRuntime:
    def __init__(self, previews: list[str], committed: list[str]) -> None:
        self.previews = list(previews)
        self.committed = list(committed)
        self.calls: list[bool] = []

    def capture_view_pair(self, *, commit_snapshot: bool, **_kwargs):
        self.calls.append(commit_snapshot)
        source = self.committed if commit_snapshot else self.previews
        if not source:
            raise AssertionError("no scripted capture remains")
        return pair(len(self.calls), source.pop(0))


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def readiness_config(**overrides) -> CaptureReadinessConfig:
    values = {
        "max_wait_s": 2.0,
        "preview_interval_s": 0.1,
        "stable_transitions_required": 2,
        "stability_mae_max": 0.01,
        "committed_mae_max": 0.02,
        "analysis_width": 32,
        "analysis_height": 18,
        "engine_idle_timeout_s": 1.0,
    }
    values.update(overrides)
    return CaptureReadinessConfig(**values)


def test_frame_metrics_reject_old_storefront_level_underexposure():
    config = readiness_config()

    dark_metrics, _ = analyze_frame(frame(
        "front", "dark", jpeg_data_url(dark=True)), config)
    usable_metrics, _ = analyze_frame(frame(
        "front", "good", jpeg_data_url()), config)

    assert dark_metrics["p95"] < 12.0
    assert dark_metrics["usable"] is False
    assert usable_metrics["p95"] >= 12.0
    assert usable_metrics["usable"] is True


def test_gate_returns_only_stable_matching_committed_pair():
    good = jpeg_data_url()
    incomplete = jpeg_data_url(dark=True)
    runtime = PairRuntime(
        previews=[incomplete, good, good, good],
        committed=[good],
    )
    clock = Clock()
    idle_waits: list[float] = []
    gate = CaptureReadinessGate(
        readiness_config(),
        engine_idle_waiter=lambda timeout: idle_waits.append(timeout) or {
            "success": True,
        },
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    result = gate.capture(
        runtime, agent_tag="agent", width_px=64, height_px=36,
        fov_degrees=90.0,
    )

    assert result.capture_group_id == "pair-5"
    assert runtime.calls == [False, False, False, False, True]
    assert len(idle_waits) == 1
    report = gate.report()
    assert report["preview_pairs"] == 4
    assert report["committed_pairs"] == 1
    assert report["discarded_committed_pairs"] == 0
    assert report["checkpoints"][-1]["matches_converged_preview"] is True


def test_gate_discards_changed_committed_pair_and_reconverges():
    first = jpeg_data_url(variant=0)
    changed = jpeg_data_url(variant=18)
    runtime = PairRuntime(
        previews=[first, first, first, changed, changed, changed],
        committed=[changed, changed],
    )
    clock = Clock()
    gate = CaptureReadinessGate(
        readiness_config(committed_mae_max=0.005),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    result = gate.capture(
        runtime, agent_tag="agent", width_px=64, height_px=36,
        fov_degrees=90.0,
    )

    assert result.front.rgb_data_url == changed
    assert gate.report()["discarded_committed_pairs"] == 1
    assert runtime.calls.count(True) == 2


def test_gate_fails_closed_when_dark_frames_never_become_usable():
    dark = jpeg_data_url(dark=True)
    runtime = PairRuntime(previews=[dark] * 10, committed=[])
    clock = Clock()
    gate = CaptureReadinessGate(
        readiness_config(max_wait_s=0.25),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    with pytest.raises(CaptureReadinessError, match="usable=False"):
        gate.capture(
            runtime, agent_tag="agent", width_px=64, height_px=36,
            fov_degrees=90.0,
        )

    assert True not in runtime.calls


def test_vulkan_monitor_fails_on_current_log_oom(tmp_path: Path):
    log = tmp_path / "current-ue.log"
    log.write_text("normal startup\n", encoding="utf-8")
    monitor = VulkanOomMonitor(log)
    monitor.assert_healthy()
    with log.open("a", encoding="utf-8") as output:
        output.write("LogVulkanRHI: Error: Failed to allocate Device Memory\n")

    with pytest.raises(CaptureReadinessError, match="renderer_vulkan_oom"):
        monitor.assert_healthy()

    assert monitor.report()["healthy"] is False
