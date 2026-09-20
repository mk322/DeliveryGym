"""Contracts for the manual closed-loop Paris Pixel Goal demo."""

from __future__ import annotations

import base64
import os
import subprocess
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import tools.run_pixel_goal_1b_closed_loop as closed_loop_runner
from embodiedbench.runtime.pixel_goal import PixelGoalFrame, PixelGoalRejected
from embodiedbench.schemas.embodiment import ControllerResult
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import ControllerOutcomeCode
from tools.run_pixel_goal_1b_closed_loop import (
    ManualSequenceTerminated,
    _build_argument_parser,
    _prepare_paris_loop,
    build_sequence_summary,
    execute_manual_sequence,
    parse_manual_uv,
    prompt_manual_uv,
    run,
    validate_closed_loop_report,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.50 0.82", (0.50, 0.82)), ("0.25,0.90", (0.25, 0.90))],
)
def test_parse_manual_uv_accepts_two_normalized_coordinates(text, expected):
    assert parse_manual_uv(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0.5",
        "0.5 0.8 0.9",
        "-0.1 0.5",
        "1.1 0.5",
        "nan 0.5",
        "inf 0.5",
    ],
)
def test_parse_manual_uv_rejects_malformed_or_non_normalized_input(text):
    with pytest.raises(ValueError):
        parse_manual_uv(text)


def _step(index, initial_x, final_x, error_m=0.1):
    return {
        "index": index,
        "navigation_mode": "nav_pixel_goal",
        "camera_snapshot_id": f"snapshot-{index}",
        "camera_intrinsics_id": "perspective-640x360-hfov90",
        "post_move_camera_snapshot_id": f"snapshot-{index + 1}",
        "post_move_camera_intrinsics_id": "perspective-640x360-hfov90",
        "controller_result": "success",
        "initial_feet_position_cm": [initial_x, 0.0, 0.0],
        "final_feet_position_cm": [final_x, 0.0, 0.0],
        "execution_error_planar_m": error_m,
        "before_frame": f"fpv_{index - 1:03d}.png",
        "after_frame": f"fpv_{index:03d}.png",
    }


def test_sequence_summary_builds_continuous_trajectory_and_counts_resets():
    steps = [_step(1, 0.0, 100.0), _step(2, 100.0, 180.0)]
    calls = [
        ("PixelGoal_SetupParisPocJson", {}, {}),
        ("PixelGoal_ResolveAndMoveJson", {}, {}),
        ("PixelGoal_ResolveAndMoveJson", {}, {}),
    ]

    summary = build_sequence_summary(steps, calls)

    assert summary["trajectory_feet_cm"] == [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [180.0, 0.0, 0.0],
    ]
    assert summary["continuity_errors_cm"] == pytest.approx([0.0])
    assert summary["reset_calls"] == 0
    assert summary["resolve_calls"] == 2
    assert summary["mean_execution_error_planar_m"] == pytest.approx(0.1)
    assert summary["maximum_execution_error_planar_m"] == pytest.approx(0.1)


def test_report_acceptance_rejects_reset_or_broken_frame_chain():
    steps = [_step(1, 0.0, 100.0), _step(2, 100.0, 180.0)]
    report = {
        "navigation_mode": "nav_pixel_goal",
        "requested_move_count": 2,
        "successful_steps": steps,
        "summary": build_sequence_summary(steps, []),
    }
    assert validate_closed_loop_report(report, expected_moves=2)["passed"] is True

    report["summary"]["reset_calls"] = 1
    with pytest.raises(ValueError, match="reset"):
        validate_closed_loop_report(report, expected_moves=2)

    report["summary"]["reset_calls"] = 0
    report["successful_steps"][1]["before_frame"] = "unrelated.png"
    with pytest.raises(ValueError, match="frame chain"):
        validate_closed_loop_report(report, expected_moves=2)


def test_report_acceptance_rejects_broken_camera_snapshot_chain():
    steps = [_step(1, 0.0, 100.0), _step(2, 100.0, 180.0)]
    steps[0]["post_move_camera_snapshot_id"] = "unrelated-snapshot"
    report = {
        "navigation_mode": "nav_pixel_goal",
        "successful_steps": steps,
        "summary": build_sequence_summary(steps, []),
    }

    with pytest.raises(ValueError, match="snapshot chain"):
        validate_closed_loop_report(report, expected_moves=2)


def _jpeg_data_url() -> str:
    stream = BytesIO()
    Image.new("RGB", (2, 2), (80, 120, 160)).save(stream, format="JPEG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _frame(index: int) -> PixelGoalFrame:
    return PixelGoalFrame(
        rgb_data_url=_jpeg_data_url(),
        camera_snapshot_id=f"snapshot-{index}",
        camera_intrinsics_id="perspective-2x2-hfov90",
        width_px=2,
        height_px=2,
        agent_tag="PixelGoalParisPocAgent",
    )


class _FakeEndpoint:
    def __init__(self):
        self.calls = []


class _FakeRuntime:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.config = SimpleNamespace(
            execution_timeout_s=1.0, poll_interval_s=0.0
        )
        self.frames = [_frame(0), _frame(1), _frame(2)]
        self.capture_count = 0
        self.started_snapshot_ids = []
        self.started_actions = []
        self._move_index = 0

    def capture_frame(self, *, agent_tag):
        frame = self.frames[self.capture_count]
        self.capture_count += 1
        return frame

    def start(self, action, frame):
        self._move_index += 1
        self.started_actions.append(action)
        self.started_snapshot_ids.append(frame.camera_snapshot_id)
        initial_x = float((self._move_index - 1) * 100)
        target = Vec3(x_cm=initial_x + 100.0, y_cm=0.0, z_cm=0.0)
        response = {
            "initial_feet_position_cm": [initial_x, 0.0, 0.0],
            "raw_world_hit_cm": [target.x_cm, target.y_cm, target.z_cm],
            "validated_navigation_target_cm": [
                target.x_cm,
                target.y_cm,
                target.z_cm,
            ],
        }
        self.endpoint.calls.append(
            ("PixelGoal_ResolveAndMoveJson", {}, response)
        )
        return SimpleNamespace(
            request=SimpleNamespace(
                request_id=f"move-{self._move_index}",
                camera_snapshot_id=frame.camera_snapshot_id,
                camera_intrinsics_id=frame.camera_intrinsics_id,
                raw_world_hit=target,
                validated_navigation_target=target,
                navmesh_adjustment_cm=0.0,
            ),
            controller_request_result="accepted",
        )

    def poll(self, request_id):
        final_x = float(self._move_index * 100)
        final_feet = Vec3(x_cm=final_x, y_cm=0.0, z_cm=0.0)
        return ControllerResult(
            request_id=request_id,
            outcome=ControllerOutcomeCode.ACCEPTED,
            requested_target=final_feet,
            accepted_target=final_feet,
            final_pose=Pose(
                frame=FrameName.UE_WORLD,
                position=Vec3(x_cm=final_x, y_cm=0.0, z_cm=90.0),
                yaw_deg=0.0,
            ),
            final_feet_position=final_feet,
            controller_result="success",
            execution_error_planar_m=0.0,
            execution_error_3d_m=0.0,
        )


def test_manual_sequence_chains_post_move_frame_without_reset(tmp_path):
    endpoint = _FakeEndpoint()
    runtime = _FakeRuntime(endpoint)

    steps, rejected = execute_manual_sequence(
        runtime,
        endpoint,
        tmp_path,
        move_count=2,
        select_uv=lambda index, path: [(0.5, 0.8), (0.6, 0.9)][
            index - 1
        ],
    )

    assert rejected == []
    assert runtime.capture_count == 3
    assert [step["camera_snapshot_id"] for step in steps] == [
        "snapshot-0",
        "snapshot-1",
    ]
    assert steps[0]["after_frame"] == steps[1]["before_frame"]
    assert not any(
        function == "PixelGoal_ResetParisPocTrialJson"
        for function, _, _ in endpoint.calls
    )
    assert (tmp_path / steps[0]["before_frame"]).is_file()
    assert (tmp_path / steps[1]["after_frame"]).is_file()


def test_selection_context_is_logged_but_never_sent_to_ue(tmp_path):
    from tools.run_pixel_goal_1b_closed_loop import ManualPixelSelection

    endpoint = _FakeEndpoint()
    runtime = _FakeRuntime(endpoint)

    steps, rejected = execute_manual_sequence(
        runtime,
        endpoint,
        tmp_path,
        move_count=1,
        select_uv=lambda index, path: ManualPixelSelection(
            uv=(0.5, 0.70), visual_context="clear_straight_long"
        ),
    )

    assert rejected == []
    assert steps[0]["visual_context"] == "clear_straight_long"
    assert steps[0][
        "initial_feet_to_validated_target_planar_m"
    ] == pytest.approx(1.0)
    action_payload = runtime.started_actions[0].model_dump(mode="json")
    assert action_payload == {
        "type": "nav_pixel_goal",
        "frame": "camera",
        "target": {"u_norm": 0.5, "v_norm": 0.70, "distance_m": None},
    }
    assert "visual_context" not in str(action_payload)


def test_accepted_controller_failure_stops_before_next_capture(tmp_path):
    endpoint = _FakeEndpoint()

    class FailedRuntime(_FakeRuntime):
        def poll(self, request_id):
            final_feet = Vec3(x_cm=20.0, y_cm=0.0, z_cm=0.0)
            return ControllerResult(
                request_id=request_id,
                outcome=ControllerOutcomeCode.CONTROLLER_FAILED,
                requested_target=Vec3(x_cm=100.0, y_cm=0.0, z_cm=0.0),
                accepted_target=Vec3(x_cm=100.0, y_cm=0.0, z_cm=0.0),
                final_pose=Pose(
                    frame=FrameName.UE_WORLD,
                    position=Vec3(x_cm=20.0, y_cm=0.0, z_cm=90.0),
                    yaw_deg=0.0,
                ),
                final_feet_position=final_feet,
                controller_result="blocked",
                execution_error_planar_m=0.8,
                execution_error_3d_m=0.8,
                failure_reason="blocked",
            )

    runtime = FailedRuntime(endpoint)
    with pytest.raises(ManualSequenceTerminated) as raised:
        execute_manual_sequence(
            runtime,
            endpoint,
            tmp_path,
            move_count=2,
            select_uv=lambda index, path: (0.5, 0.8),
        )

    assert raised.value.reason == "controller_failed"
    assert raised.value.successful_steps == []
    assert raised.value.terminal["controller_result"] == "blocked"
    assert runtime.capture_count == 1
    assert not any(
        function == "PixelGoal_ResetParisPocTrialJson"
        for function, _, _ in endpoint.calls
    )


def test_execution_timeout_cancels_controller_and_stops_sequence(tmp_path):
    endpoint = _FakeEndpoint()

    class TimeoutRuntime(_FakeRuntime):
        def __init__(self, value):
            super().__init__(value)
            self.config.execution_timeout_s = 0.0
            self.cancelled = []

        def cancel(self, request_id, *, reason):
            self.cancelled.append((request_id, reason))
            final_feet = Vec3(x_cm=15.0, y_cm=0.0, z_cm=0.0)
            return ControllerResult(
                request_id=request_id,
                outcome=ControllerOutcomeCode.EXECUTION_TIMEOUT,
                requested_target=Vec3(x_cm=100.0, y_cm=0.0, z_cm=0.0),
                accepted_target=Vec3(x_cm=100.0, y_cm=0.0, z_cm=0.0),
                final_pose=Pose(
                    frame=FrameName.UE_WORLD,
                    position=Vec3(x_cm=15.0, y_cm=0.0, z_cm=90.0),
                    yaw_deg=0.0,
                ),
                final_feet_position=final_feet,
                controller_result="aborted",
                execution_error_planar_m=0.85,
                execution_error_3d_m=0.85,
                failure_reason="execution_timeout",
            )

    runtime = TimeoutRuntime(endpoint)
    with pytest.raises(ManualSequenceTerminated) as raised:
        execute_manual_sequence(
            runtime,
            endpoint,
            tmp_path,
            move_count=1,
            select_uv=lambda index, path: (0.5, 0.8),
        )

    assert runtime.cancelled == [("move-1", "execution_timeout")]
    assert raised.value.reason == "execution_timeout"
    assert raised.value.terminal["controller_result"] == "aborted"
    assert runtime.capture_count == 1


def test_rejected_attempt_is_recorded_and_reprompts_same_step(tmp_path):
    endpoint = _FakeEndpoint()

    class RejectOnceRuntime(_FakeRuntime):
        def __init__(self, value):
            super().__init__(value)
            self.start_attempts = 0

        def start(self, action, frame):
            self.start_attempts += 1
            if self.start_attempts == 1:
                raise PixelGoalRejected(
                    "hit_not_walkable_ground",
                    audit={
                        "accepted": False,
                        "rejection_reason": "hit_not_walkable_ground",
                    },
                )
            return super().start(action, frame)

    runtime = RejectOnceRuntime(endpoint)
    selections = []

    def select_uv(index, path):
        selections.append((index, path.name))
        return (0.5, 0.2) if len(selections) == 1 else (0.5, 0.9)

    steps, rejected = execute_manual_sequence(
        runtime, endpoint, tmp_path, move_count=1, select_uv=select_uv
    )

    assert len(steps) == 1
    assert len(rejected) == 1
    assert rejected[0]["step_index"] == 1
    assert rejected[0]["requested_uv"] == pytest.approx([0.5, 0.2])
    assert rejected[0]["rejection_reason"] == "hit_not_walkable_ground"
    assert rejected[0]["camera_snapshot_id"] == "snapshot-0"
    assert [selection[0] for selection in selections] == [1, 1]
    assert runtime.capture_count == 3
    assert rejected[0]["selected_frame"] != steps[0]["selected_frame"]
    assert (tmp_path / rejected[0]["selected_frame"]).is_file()
    assert not any(
        function == "PixelGoal_ResetParisPocTrialJson"
        for function, _, _ in endpoint.calls
    )


def test_prompt_manual_uv_reprompts_after_invalid_input(tmp_path):
    responses = iter(("not-a-point", "0.4 0.85"))
    output = []

    result = prompt_manual_uv(
        3,
        tmp_path / "fpv_002.png",
        input_fn=lambda prompt: next(responses),
        output_fn=lambda message, **kwargs: output.append(message),
    )

    assert result == pytest.approx((0.4, 0.85))
    assert output[0].startswith("STEP 3 BEFORE_FPV")
    assert output[1].startswith("INVALID_INPUT")


def test_prepare_paris_loop_sets_up_and_warms_once_without_reset():
    class Endpoint:
        def __init__(self):
            self.calls = []

        def call(self, function, request):
            response = {
                "success": True,
                "synthetic_floor_spawned": False,
                "dynamic_runtime_generation": True,
                "supports_runtime_generation": True,
                "navmesh_has_valid_data": True,
                "active_navmesh_tiles": 8,
                "rgb_capture_mode": "agent_native",
                "rgb_capture_render_state_persistent": True,
            }
            self.calls.append((function, request, response))
            return response

    endpoint = Endpoint()
    waits = []
    warmups = []
    result = _prepare_paris_loop(
        endpoint,
        {"scene": "CityCore_Paris"},
        readiness_timeout_s=30.0,
        capture_warmup_s=6.0,
        wait_fn=lambda value, timeout: waits.append((value, timeout))
        or {"poc_ready": True},
        warm_up_fn=lambda duration: warmups.append(duration) or 6000.0,
    )

    assert [call[0] for call in endpoint.calls] == [
        "PixelGoal_SetupParisPocJson"
    ]
    assert waits == [(endpoint, 30.0)]
    assert warmups == [6.0]
    assert result["capture_warmup_ms"] == pytest.approx(6000.0)
    assert result["status"] == {"poc_ready": True}
    assert not any(
        call[0] == "PixelGoal_ResetParisPocTrialJson"
        for call in endpoint.calls
    )


@pytest.mark.parametrize(
    "override",
    [
        {"success": False},
        {"synthetic_floor_spawned": True},
        {"dynamic_runtime_generation": False},
        {"supports_runtime_generation": False},
        {"navmesh_has_valid_data": False},
        {"active_navmesh_tiles": 0},
        {"rgb_capture_mode": "one_shot"},
        {"rgb_capture_render_state_persistent": False},
    ],
)
def test_prepare_paris_loop_rejects_unready_navmesh_or_nonpersistent_rgb(
    override,
):
    setup = {
        "success": True,
        "synthetic_floor_spawned": False,
        "dynamic_runtime_generation": True,
        "supports_runtime_generation": True,
        "navmesh_has_valid_data": True,
        "active_navmesh_tiles": 8,
        "rgb_capture_mode": "agent_native",
        "rgb_capture_render_state_persistent": True,
    }
    setup.update(override)

    class Endpoint:
        calls = []

        def call(self, function, request):
            self.calls.append((function, request, setup))
            return setup

    warmups = []
    with pytest.raises(RuntimeError, match="Paris Pixel Goal setup"):
        _prepare_paris_loop(
            Endpoint(),
            {"scene": "CityCore_Paris"},
            readiness_timeout_s=30.0,
            capture_warmup_s=6.0,
            wait_fn=lambda endpoint, timeout: {"poc_ready": True},
            warm_up_fn=lambda duration: warmups.append(duration),
        )
    assert warmups == []


def test_closed_loop_cli_keeps_approved_v0_defaults():
    args = _build_argument_parser().parse_args([])

    assert args.move_count == 6
    assert args.max_navmesh_adjustment_cm == pytest.approx(10.0)
    assert args.acceptance_radius_cm == pytest.approx(15.0)
    assert args.capture_warmup_s == pytest.approx(6.0)
    assert args.launch_mode == "attach"


def test_run_writes_accepted_single_session_report_and_cleans_up(
    tmp_path, monkeypatch
):
    simworld_root = tmp_path / "simworld"
    simworld_root.mkdir()
    (simworld_root / "SimWorld.uproject").write_text("{}", encoding="utf-8")
    citycore = tmp_path / "CityCore_Paris"
    scenes = citycore / "Scenes"
    scenes.mkdir(parents=True)
    (scenes / "ParisCity_FinalBlueprints.umap").write_bytes(b"map")
    output = tmp_path / "evidence"

    class FakeSession:
        instances = []

        def __init__(self, spear_config):
            self.spear_config = spear_config
            self.began = False
            self.ended = False
            self.shutdown_called = False
            self.__class__.instances.append(self)

        def begin_play(self):
            self.began = True

        def end_play(self):
            self.ended = True

        def shutdown(self):
            self.shutdown_called = True

    class FakeEndpoint:
        instances = []

        def __init__(self, session, event_log):
            self.session = session
            self.event_log = event_log
            self.calls = []
            self.__class__.instances.append(self)

        def call(self, function, request):
            if function == "PixelGoal_SetupParisPocJson":
                response = {
                    "success": True,
                    "synthetic_floor_spawned": False,
                    "dynamic_runtime_generation": True,
                    "supports_runtime_generation": True,
                    "navmesh_has_valid_data": True,
                    "active_navmesh_tiles": 8,
                    "rgb_capture_mode": "agent_native",
                    "rgb_capture_render_state_persistent": True,
                }
            elif function == "PixelGoal_GetParisPocStatusJson":
                response = {"poc_ready": True, "agent_on_navmesh": True}
            else:
                raise AssertionError(f"unexpected call: {function}")
            self.calls.append((function, request, response))
            return response

    class FakeRuntime:
        def __init__(self, endpoint, config):
            self.endpoint = endpoint
            self.config = config

    steps = [_step(1, 0.0, 100.0), _step(2, 100.0, 180.0)]
    monkeypatch.setattr(
        closed_loop_runner, "AttachedParisGameSession", FakeSession
    )
    monkeypatch.setattr(
        closed_loop_runner, "SpearPixelGoalEndpoint", FakeEndpoint
    )
    monkeypatch.setattr(closed_loop_runner, "LivePixelGoalRuntime", FakeRuntime)
    monkeypatch.setattr(
        closed_loop_runner,
        "execute_manual_sequence",
        lambda runtime, endpoint, output_dir, move_count, select_uv: (steps, []),
    )
    monkeypatch.setattr(
        closed_loop_runner,
        "_warm_up_paris_capture",
        lambda duration: 6000.0,
    )

    args = _build_argument_parser().parse_args(
        [
            "--simworld-root",
            str(simworld_root),
            "--citycore-content",
            str(citycore),
            "--output",
            str(output),
            "--move-count",
            "2",
        ]
    )
    report = run(args)

    assert report["acceptance"]["passed"] is True
    assert report["trial_protocol"] == {
        "single_live_session": True,
        "reset_between_moves": False,
        "post_move_frame_is_next_observation": True,
    }
    assert report["config"]["acceptance_radius_cm"] == pytest.approx(15.0)
    assert report["config"]["max_navmesh_adjustment_cm"] == pytest.approx(
        10.0
    )
    assert [call[0] for call in FakeEndpoint.instances[0].calls] == [
        "PixelGoal_SetupParisPocJson",
        "PixelGoal_GetParisPocStatusJson",
    ]
    assert FakeSession.instances[0].began is True
    assert FakeSession.instances[0].ended is True
    assert FakeSession.instances[0].shutdown_called is False
    persisted = (output / "closed_loop_report.json").read_text(
        encoding="utf-8"
    )
    assert '"successful_move_count": 2' in persisted


def test_run_persists_partial_trajectory_when_controller_terminates(
    tmp_path, monkeypatch
):
    output = tmp_path / "failure-evidence"

    class FakeSession:
        def __init__(self, spear_config):
            self.ended = False

        def begin_play(self):
            pass

        def end_play(self):
            self.ended = True

        def shutdown(self):
            pass

    class FakeEndpoint:
        def __init__(self, session, event_log):
            self.calls = []

        def call(self, function, request):
            if function == "PixelGoal_SetupParisPocJson":
                response = {
                    "success": True,
                    "synthetic_floor_spawned": False,
                    "dynamic_runtime_generation": True,
                    "supports_runtime_generation": True,
                    "navmesh_has_valid_data": True,
                    "active_navmesh_tiles": 8,
                    "rgb_capture_mode": "agent_native",
                    "rgb_capture_render_state_persistent": True,
                }
            else:
                response = {"poc_ready": True, "agent_on_navmesh": True}
            self.calls.append((function, request, response))
            return response

    completed = [_step(1, 0.0, 100.0)]
    termination = ManualSequenceTerminated(
        "controller_failed",
        successful_steps=completed,
        rejected_attempts=[
            {
                "step_index": 2,
                "requested_uv": [0.2, 0.2],
                "rejection_reason": "hit_not_walkable_ground",
            }
        ],
        terminal={
            "step_index": 2,
            "controller_result": "blocked",
            "failure_reason": "blocked",
        },
    )
    monkeypatch.setattr(closed_loop_runner, "validate_mount_inputs", lambda *a: None)
    monkeypatch.setattr(
        closed_loop_runner, "AttachedParisGameSession", FakeSession
    )
    monkeypatch.setattr(
        closed_loop_runner, "SpearPixelGoalEndpoint", FakeEndpoint
    )
    monkeypatch.setattr(
        closed_loop_runner,
        "LivePixelGoalRuntime",
        lambda endpoint, config: SimpleNamespace(endpoint=endpoint, config=config),
    )
    monkeypatch.setattr(
        closed_loop_runner, "_warm_up_paris_capture", lambda duration: 6000.0
    )
    monkeypatch.setattr(
        closed_loop_runner,
        "execute_manual_sequence",
        lambda *args, **kwargs: (_ for _ in ()).throw(termination),
    )

    args = _build_argument_parser().parse_args(
        [
            "--simworld-root",
            str(tmp_path / "simworld"),
            "--citycore-content",
            str(tmp_path / "CityCore_Paris"),
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ManualSequenceTerminated):
        run(args)

    failure = (output / "closed_loop_failure.json").read_text(
        encoding="utf-8"
    )
    assert '"reason": "controller_failed"' in failure
    assert '"controller_result": "blocked"' in failure
    assert '"index": 1' in failure
    assert '"reset_calls": 0' in failure


def test_closed_loop_launcher_passes_real_paris_args_and_keeps_stdin(
    tmp_path,
):
    source_root = Path(__file__).resolve().parents[1]
    repo_root = tmp_path / "repo"
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True)
    launcher = tools_dir / "run_pixel_goal_1b_closed_loop.sh"
    launcher.write_bytes(
        (source_root / "tools" / launcher.name).read_bytes()
    )
    launcher.chmod(0o755)
    (tools_dir / "pixel_goal_1b_poc_spear.yaml").write_text(
        "SP_SERVICES:\n  RPC_SERVICE:\n    RPC_SERVER_PORT: 30128\n",
        encoding="utf-8",
    )
    simworld_root = repo_root / ".simworld-ue"
    simworld_root.mkdir()
    (simworld_root / "SimWorld.uproject").write_text("{}", encoding="utf-8")
    citycore = tmp_path / "CityCore_Paris"
    citycore_scenes = citycore / "Scenes"
    citycore_scenes.mkdir(parents=True)
    (citycore_scenes / "ParisCity_FinalBlueprints.umap").write_bytes(b"map")
    citycore_scenes.chmod(0o555)
    citycore.chmod(0o555)
    trace = tmp_path / "launch-trace.txt"
    fake_python = tmp_path / "fake-python"
    fake_editor = tmp_path / "fake-editor"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-" && "$#" -eq 3 ]]; then
  mkdir -p "$(dirname "$3")"
  printf 'fake-config\n' > "$3"
  exit 0
fi
if [[ "$1" == "-" ]]; then
  exit 0
fi
{
  printf 'RUNNER_ARGS'
  printf ' <%s>' "$@"
  printf '\nRPC=%s\n' "${SIMWORLD_RPC_PORT:-missing}"
  IFS= read -r manual_input || true
  printf 'STDIN=%s\n' "$manual_input"
} >> "$PIXEL_GOAL_LAUNCH_TRACE"
""",
        encoding="utf-8",
    )
    fake_editor.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'EDITOR_ARGS'
  printf ' <%s>' "$@"
  printf '\n'
} >> "$PIXEL_GOAL_LAUNCH_TRACE"
trap 'exit 0' TERM INT
while :; do sleep 1; done
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_editor.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PIXEL_GOAL_PYTHON": str(fake_python),
            "PIXEL_GOAL_UNREAL_EDITOR": str(fake_editor),
            "PIXEL_GOAL_LAUNCH_TRACE": str(trace),
            "SIMWORLD_RPC_PORT": "30133",
            "CITYCORE_PARIS_CONTENT": str(citycore),
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher), "--move-count", "1"],
        cwd=repo_root,
        env=env,
        input="0.50 0.82\n",
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    launched = trace.read_text(encoding="utf-8")
    assert "</Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints>" in launched
    assert "<-game>" in launched
    assert "<-RenderOffScreen>" in launched
    assert "<-sp-config-file=" in launched
    runner_arg = f"<{repo_root / 'tools/run_pixel_goal_1b_closed_loop.py'}>"
    assert runner_arg in launched
    assert "<--move-count> <6>" in launched
    assert "<--output> <" + str(
        repo_root / "artifacts/pixel_goal_1b_closed_loop"
    ) + ">" in launched
    assert "RPC=30133" in launched
    assert "STDIN=0.50 0.82" in launched
