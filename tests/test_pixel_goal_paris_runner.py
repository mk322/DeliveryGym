from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_pixel_goal_1b_poc as paris_runner

from embodiedbench.runtime.pixel_goal_paris_poc import (
    PARIS_POC_SCENE,
    RUE_DE_RIVOLI_SIDEWALK,
)
from tools.run_pixel_goal_1b_poc import (
    AttachedParisGameSession,
    INVALID_GOALS,
    MANUAL_GOALS,
    _build_argument_parser,
    build_paris_setup_request,
    validate_mount_inputs,
)


class _MutableConfig(SimpleNamespace):
    def defrost(self):
        pass

    def freeze(self):
        pass


def test_attached_game_session_binds_game_world_without_editor_or_pie(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _MutableConfig(
        SPEAR=_MutableConfig(LAUNCH_MODE="editor"),
        SP_SERVICES=_MutableConfig(
            RPC_SERVICE=_MutableConfig(RPC_SERVER_PORT=1)
        ),
    )
    game = object()

    class FakeInstance:
        def __init__(self, *, config):
            self.config = config
            self.closed_with = None

        def get_game(self):
            return game

        def close(self, *, force):
            self.closed_with = force

    fake_spear = SimpleNamespace(
        get_config=lambda **kwargs: config,
        configure_system=lambda *, config: None,
        Instance=FakeInstance,
    )
    monkeypatch.setitem(sys.modules, "spear", fake_spear)
    monkeypatch.setenv("SIMWORLD_RPC_PORT", "30128")

    session = AttachedParisGameSession("runtime.yaml")
    assert config.SPEAR.LAUNCH_MODE == "none"
    assert config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT == 30128
    assert session._game is None

    session.begin_play()
    assert session._game is game
    session.end_play()
    assert session._game is None
    session.shutdown()
    assert session._instance.closed_with is False


def test_attached_session_requires_two_bounded_engine_idle_observations():
    calls: list[tuple[float, float]] = []

    class AsyncLoadingService:
        def wait_for_engine_idle(
            self, *, max_time_seconds: float, sleep_time_seconds: float,
        ) -> None:
            calls.append((max_time_seconds, sleep_time_seconds))

    clock = SimpleNamespace(value=0.0)
    session = object.__new__(AttachedParisGameSession)
    session._game = SimpleNamespace(async_loading_service=AsyncLoadingService())

    result = session.wait_for_engine_idle(
        2.0,
        poll_interval_s=0.25,
        sleep_fn=lambda seconds: setattr(clock, "value", clock.value + seconds),
        monotonic_fn=lambda: clock.value,
    )

    assert len(calls) == 2
    assert calls[0] == pytest.approx((2.0, 0.25))
    assert calls[1] == pytest.approx((1.75, 0.25))
    assert result == {
        "success": True,
        "consecutive_idle_observations": 2,
        "wall_ms": pytest.approx(250.0),
    }


def test_runner_uses_only_the_approved_real_paris_scene_and_local_region():
    request = build_paris_setup_request(RUE_DE_RIVOLI_SIDEWALK)

    assert request == {
        "scene": PARIS_POC_SCENE,
        "region_name": "rue_de_rivoli_sidewalk",
        "agent_spawn_cm": pytest.approx([-26805.8209, 9437.5276, 90.0]),
        "agent_yaw_deg": pytest.approx(165.0056),
        "nav_bounds_center_cm": pytest.approx([-27385.3767, 9592.8117, 100.0]),
        "nav_bounds_extent_cm": pytest.approx([1800.0, 700.0, 300.0]),
    }
    assert "spawn_floor" not in request
    assert "graph" not in request


def test_runner_has_five_distinct_normalized_only_actions_and_two_invalids():
    assert MANUAL_GOALS == [
        (0.50, 0.82),
        (0.62, 0.94),
        (0.50, 0.88),
        (0.48, 0.96),
        (0.52, 0.98),
    ]
    assert len(set(MANUAL_GOALS)) == 5
    assert len(INVALID_GOALS) == 2
    assert all(0.0 <= value <= 1.0 for uv in MANUAL_GOALS for value in uv)
    assert all(
        0.0 <= value <= 1.0 for _, uv in INVALID_GOALS for value in uv
    )
    assert len({u for u, _ in MANUAL_GOALS}) >= 3
    assert len({v for _, v in MANUAL_GOALS}) >= 2


def test_runner_keeps_m1a_controller_defaults_for_comparability():
    args = _build_argument_parser().parse_args([])

    assert args.acceptance_radius_cm == pytest.approx(15.0)
    assert args.max_navmesh_adjustment_cm == pytest.approx(10.0)
    assert getattr(args, "capture_warmup_s", None) == pytest.approx(6.0)
    assert args.scene == PARIS_POC_SCENE


def test_runner_warms_persistent_capture_once_for_configured_duration():
    sleeps: list[float] = []
    monotonic_values = iter((100.0, 106.125))

    warmup_ms = paris_runner._warm_up_paris_capture(
        6.0,
        sleep_fn=sleeps.append,
        monotonic_fn=lambda: next(monotonic_values),
    )

    assert sleeps == [6.0]
    assert warmup_ms == pytest.approx(6125.0)


def test_runner_resets_each_calibration_trial_and_waits_for_navmesh_readiness():
    class FakeEndpoint:
        def __init__(self):
            self.calls = []

        def call(self, function, request):
            self.calls.append((function, request))
            if function == "PixelGoal_ResetParisPocTrialJson":
                return {
                    "success": True,
                    "controller_stopped": True,
                    "camera_snapshots_invalidated": True,
                }
            return {"poc_ready": True, "agent_on_navmesh": True}

    endpoint = FakeEndpoint()
    result = paris_runner._reset_paris_poc_trial(endpoint, timeout_s=0.1)

    assert endpoint.calls == [
        ("PixelGoal_ResetParisPocTrialJson", {}),
        ("PixelGoal_GetParisPocStatusJson", {}),
    ]
    assert result == {
        "success": True,
        "controller_stopped": True,
        "camera_snapshots_invalidated": True,
        "poc_ready": True,
        "agent_on_navmesh": True,
    }


def test_mount_preflight_requires_real_paris_content_and_simworld_project(
    tmp_path: Path,
):
    content = tmp_path / "CityCore_Paris"
    scenes = content / "Scenes"
    scenes.mkdir(parents=True)
    (scenes / "ParisCity_FinalBlueprints.umap").write_bytes(b"test-map")
    project = tmp_path / "SimWorld.uproject"
    project.write_text("{}", encoding="utf-8")

    validate_mount_inputs(content, project)

    with pytest.raises(FileNotFoundError, match="CityCore_Paris"):
        validate_mount_inputs(tmp_path / "missing", project)
    with pytest.raises(FileNotFoundError, match="SimWorld.uproject"):
        validate_mount_inputs(content, tmp_path / "missing.uproject")
    with pytest.raises(FileNotFoundError, match="ParisCity_FinalBlueprints.umap"):
        validate_mount_inputs(tmp_path, project)


def test_live_launcher_uses_read_only_paris_content_and_shared_editor_attach():
    launcher = (
        Path(__file__).resolve().parents[1] / "tools" / "run_pixel_goal_1b_poc.sh"
    ).read_text(encoding="utf-8")
    config = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "pixel_goal_1b_poc_spear.yaml"
    ).read_text(encoding="utf-8")

    assert "unshare --user --map-root-user --mount" not in launcher
    assert '[[ -w "$citycore_source" ]]' in launcher
    assert 'ln -s "$citycore_source" "$citycore_target"' in launcher
    assert 'PIXEL_GOAL_UNREAL_EDITOR:?' in launcher
    assert "  -game \\" in launcher
    assert "bEnableAsyncStaticMeshCompilation=False" in launcher
    assert "PixelGoalRunId" in launcher
    assert "Load map complete /Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints" in launcher
    assert "--launch-mode attach" in launcher
    assert "--shutdown-attached-editor" in launcher
    assert "spear.get_config" in launcher
    assert '-sp-config-file="$runtime_config"' in launcher
    assert 'export PYTHONPATH="$repo_root:$SIMWORLD_SPEAR_PYTHON' in launcher
    assert "run_pixel_goal_1b_poc.py" in launcher
    assert "LAUNCH_MODE: none" in config
