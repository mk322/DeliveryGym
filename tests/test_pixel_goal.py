"""Milestone 1A contracts for live-UE Pixel Goal navigation."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

import embodiedbench.runtime.pixel_goal as pixel_goal_module
from embodiedbench.runtime.pixel_goal import (
    LivePixelGoalRuntime,
    PixelGoalConfig,
    PixelGoalFrame,
    PixelGoalInfrastructureTimeout,
    PixelGoalInProgress,
    PixelGoalPairIntegrityError,
    PixelGoalRejected,
)
from embodiedbench.schemas import NavPixelGoalAction as PublicNavPixelGoalAction
from embodiedbench.schemas.base import CapabilityError
from embodiedbench.schemas.embodiment import (
    ControllerResult,
    DistanceSource,
    NavigationRequest,
)
from embodiedbench.schemas.environment import EnvironmentBundle, NavigationMode
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    ControllerOutcomeCode,
    NavPixelGoalAction,
    RuntimeCapabilities,
    RuntimeMode,
)


def pixel_goal_action() -> NavPixelGoalAction:
    return NavPixelGoalAction(target={"u_norm": 0.25, "v_norm": 0.75})


def pixel_goal_request() -> NavigationRequest:
    return NavigationRequest(
        request_id="pixel-move-1",
        mode=NavigationMode.NAV_PIXEL_GOAL,
        distance_source=DistanceSource.UE_GEOMETRY_TRACE,
        source_image_point={"u_norm": 0.25, "v_norm": 0.75},
        camera_snapshot_id="pixel-snapshot-1",
        camera_intrinsics_id="perspective-640x360-hfov90",
        raw_world_hit=Vec3(x_cm=400.0, y_cm=50.0, z_cm=50010.0),
        validated_navigation_target=Vec3(
            x_cm=404.0, y_cm=53.0, z_cm=50010.0
        ),
        target_world=Vec3(x_cm=400.0, y_cm=50.0, z_cm=50010.0),
        projected_target=Vec3(x_cm=404.0, y_cm=53.0, z_cm=50010.0),
        navmesh_adjustment_cm=5.0,
        max_range_m=1000.0,
    )


def pixel_goal_controller_result() -> ControllerResult:
    return ControllerResult(
        request_id="pixel-move-1",
        outcome=ControllerOutcomeCode.ACCEPTED,
        requested_target=Vec3(x_cm=100.0, y_cm=200.0, z_cm=10.0),
        accepted_target=Vec3(x_cm=100.0, y_cm=200.0, z_cm=10.0),
        final_pose=Pose(
            frame=FrameName.UE_WORLD,
            position=Vec3(x_cm=106.0, y_cm=205.3, z_cm=101.0),
            yaw_deg=0.0,
        ),
        final_feet_position=Vec3(x_cm=106.0, y_cm=205.3, z_cm=10.0),
        controller_result="success",
        execution_error_planar_m=0.08,
        execution_error_3d_m=0.08,
    )


class TestPixelGoalActionContract:
    def test_action_is_exported_from_the_public_schema_api(self):
        assert PublicNavPixelGoalAction is NavPixelGoalAction

    def test_action_is_only_a_normalized_image_point(self):
        action = pixel_goal_action()

        assert action.type == "nav_pixel_goal"
        assert action.target.distance_m is None
        assert ActionEnvelope(
            episode_id="calibration", step_index=0, action=action
        ).navigation_mode() is NavigationMode.NAV_PIXEL_GOAL

    def test_distance_is_rejected(self):
        with pytest.raises(ValidationError, match="must not carry distance_m"):
            NavPixelGoalAction(
                target={"u_norm": 0.5, "v_norm": 0.8, "distance_m": 3.0}
            )

    def test_normalized_bounds_are_inherited(self):
        with pytest.raises(ValidationError):
            NavPixelGoalAction(target={"u_norm": 1.01, "v_norm": 0.5})

    def test_pixel_goal_does_not_require_a_pose_lattice(self):
        bundle = EnvironmentBundle(
            environment_id="pixel-calibration",
            version="0.1.0",
            base_world_id="pixel-calibration",
            base_world_version="0.1.0",
            supported_navigation_modes=[NavigationMode.NAV_PIXEL_GOAL],
            supported_runtimes=["live"],
        )

        assert bundle.sensors.pose_lattice_spacing_cm is None
        assert bundle.sensors.pose_lattice_headings is None

    @pytest.mark.parametrize("mode", [RuntimeMode.TEXT, RuntimeMode.CACHED])
    def test_non_live_runtime_cannot_advertise_pixel_goal(self, mode):
        with pytest.raises(ValidationError, match="live runtime"):
            RuntimeCapabilities(
                mode=mode,
                navigation_modes=[NavigationMode.NAV_PIXEL_GOAL],
            )

    @pytest.mark.parametrize("runtime", ["text", "cached"])
    def test_non_live_environment_cannot_declare_pixel_goal(self, runtime):
        with pytest.raises(ValidationError, match="live runtime"):
            EnvironmentBundle(
                environment_id="pixel-calibration",
                version="0.1.0",
                base_world_id="pixel-calibration",
                base_world_version="0.1.0",
                supported_navigation_modes=[NavigationMode.NAV_PIXEL_GOAL],
                supported_runtimes=[runtime],
            )

    def test_additive_mode_preserves_existing_exact_wire_revision(self):
        """Pixel Goal must not invalidate cached/text 0.1 artifacts."""

        assert ActionEnvelope.SCHEMA_VERSION == "0.1.0"
        assert RuntimeCapabilities.SCHEMA_VERSION == "0.1.0"
        assert EnvironmentBundle.SCHEMA_VERSION == "0.1.0"
        assert NavigationRequest.SCHEMA_VERSION == "0.1.0"
        assert ControllerResult.SCHEMA_VERSION == "0.1.0"


class TestPixelGoalAuditContract:
    def test_request_records_geometry_without_quantization(self):
        request = pixel_goal_request()

        assert request.distance_source is DistanceSource.UE_GEOMETRY_TRACE
        assert request.camera_snapshot_id == "pixel-snapshot-1"
        assert request.raw_world_hit == Vec3(
            x_cm=400.0, y_cm=50.0, z_cm=50010.0
        )
        assert request.validated_navigation_target is not None
        assert request.quantized_target is None

    def test_request_requires_exact_camera_snapshot(self):
        data = pixel_goal_request().model_dump()
        data["camera_snapshot_id"] = None

        with pytest.raises(ValidationError, match="camera snapshot"):
            NavigationRequest.model_validate(data)

    def test_request_forbids_quantization(self):
        data = pixel_goal_request().model_dump()
        data["quantized_target"] = {
            "frame": FrameName.UE_WORLD,
            "position": {"x_cm": 400.0, "y_cm": 50.0, "z_cm": 50010.0},
            "yaw_deg": 0.0,
        }

        with pytest.raises(ValidationError, match="must not be quantized"):
            NavigationRequest.model_validate(data)

    def test_controller_result_records_final_feet_and_errors(self):
        result = pixel_goal_controller_result()

        assert result.controller_result == "success"
        assert result.final_feet_position is not None
        assert result.execution_error_planar_m == pytest.approx(0.08)
        assert result.execution_error_3d_m == pytest.approx(0.08)


class ScriptedEndpoint:
    def __init__(
        self,
        *,
        capture: dict[str, object] | None = None,
        view_pair: dict[str, object] | None = None,
        resolve: dict[str, object] | None = None,
        statuses: list[dict[str, object]] | None = None,
        cancel: dict[str, object] | None = None,
    ) -> None:
        self.capture = capture or {}
        self.view_pair = view_pair or {}
        self.resolve = resolve or {}
        self.statuses = list(statuses or [])
        self.cancel = cancel or {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(
        self, function_name: str, request: dict[str, object]
    ) -> dict[str, object]:
        self.calls.append((function_name, request))
        if function_name == "PixelGoal_CaptureFrameJson":
            return self.capture
        if function_name == "PixelGoal_CaptureViewPairJson":
            return self.view_pair
        if function_name == "PixelGoal_ResolveAndMoveJson":
            return self.resolve
        if function_name == "PixelGoal_GetMoveStatusJson":
            if not self.statuses:
                raise AssertionError("no scripted Pixel Goal status remains")
            return self.statuses.pop(0)
        if function_name == "PixelGoal_CancelMoveJson":
            return self.cancel
        raise AssertionError(f"unexpected endpoint function {function_name}")


def pixel_goal_frame(
    snapshot_id: str = "pixel-snapshot-7",
    *,
    view: str | None = None,
    group: str | None = None,
) -> PixelGoalFrame:
    return PixelGoalFrame(
        rgb_data_url="data:image/jpeg;base64,AA==",
        camera_snapshot_id=snapshot_id,
        camera_intrinsics_id="perspective-640x360-hfov90",
        width_px=640,
        height_px=360,
        view_id=view,
        capture_group_id=group,
    )


def frame_json(
    view: str,
    snapshot_id: str,
    camera_yaw_deg: float,
    *,
    group: str = "view-pair-4",
    intrinsics: str = "perspective-640x360-hfov90",
    fov_degrees: float = 90.0,
) -> dict[str, object]:
    return {
        "success": True,
        "view_id": view,
        "capture_group_id": group,
        "rgb_data_url": "data:image/jpeg;base64,AA==",
        "camera_snapshot_id": snapshot_id,
        "camera_intrinsics_id": intrinsics,
        "width": 640,
        "height": 360,
        "loc_cm": [1.0, 2.0, 90.0],
        "yaw_deg": 45.0,
        "camera_rotation_degrees": [0.0, camera_yaw_deg, 0.0],
        "camera_horizontal_fov_degrees": fov_degrees,
        "timing": {"capture_read_ms": 1.0, "encode_ms": 2.0, "wall_ms": 3.0},
    }


def view_pair_json(
    *, fov_degrees: float = 90.0, snapshots_committed: bool = True,
) -> dict[str, object]:
    return {
        "success": True,
        "snapshots_committed": snapshots_committed,
        "agent_tag": "agent",
        "capture_group_id": "view-pair-4",
        "pose": {"x_cm": 1.0, "y_cm": 2.0, "z_cm": 90.0, "yaw_deg": 45.0},
        "views": [
            frame_json(
                "front", "front-snapshot", 45.0, fov_degrees=fov_degrees),
            frame_json(
                "rear", "rear-snapshot", 225.0, fov_degrees=fov_degrees),
        ],
        "pair_capture_wall_ms": 8.0,
        "capture_timing": {
            "total_ms": 7.0,
            "lookup_ms": 0.5,
            "init_ms": 0.5,
            "capture_read_ms": 2.0,
            "encode_ms": 4.0,
        },
    }


class TestLivePixelGoalRuntime:
    def test_capture_view_pair_keeps_both_snapshot_identities(self):
        endpoint = ScriptedEndpoint(view_pair=view_pair_json())

        pair = LivePixelGoalRuntime(endpoint).capture_view_pair(agent_tag="agent")

        assert pair.capture_group_id == "view-pair-4"
        assert pair.pose == (1.0, 2.0, 90.0, 45.0)
        assert pair.front.camera_snapshot_id == "front-snapshot"
        assert pair.rear.camera_snapshot_id == "rear-snapshot"
        assert pair.front.camera_yaw_deg == 45.0
        assert pair.rear.camera_yaw_deg == 225.0
        assert pair.capture_timing == {
            "capture_read_ms": 2.0,
            "encode_ms": 4.0,
            "wall_ms": 8.0,
        }
        assert endpoint.calls == [(
            "PixelGoal_CaptureViewPairJson",
            {
                "agent_tag": "agent",
                "width": 640,
                "height": 360,
                "fov_degrees": 90.0,
                "jpeg_quality": 90,
                "commit_snapshot": True,
            },
        )]

    def test_capture_view_pair_requests_noncommitting_preview_explicitly(self):
        endpoint = ScriptedEndpoint(view_pair=view_pair_json(
            snapshots_committed=False))

        pair = LivePixelGoalRuntime(endpoint).capture_view_pair(
            agent_tag="agent", commit_snapshot=False)

        assert pair.capture_group_id == "view-pair-4"
        assert endpoint.calls[0][1]["commit_snapshot"] is False

    @pytest.mark.parametrize("echo", [None, 1, True])
    def test_capture_view_pair_requires_exact_snapshot_commit_echo(self, echo):
        response = view_pair_json(snapshots_committed=False)
        if echo is None:
            response.pop("snapshots_committed")
        else:
            response["snapshots_committed"] = echo

        with pytest.raises(
            PixelGoalPairIntegrityError,
            match="view_pair_snapshot_commit_mismatch",
        ):
            LivePixelGoalRuntime(
                ScriptedEndpoint(view_pair=response)
            ).capture_view_pair(agent_tag="agent", commit_snapshot=False)

    @pytest.mark.parametrize(
        ("case", "mutate"),
        [
            ("incomplete", lambda pair: pair["views"].pop()),
            ("reversed", lambda pair: pair["views"].reverse()),
            ("duplicate snapshot", lambda pair: pair["views"][1].update(
                camera_snapshot_id="front-snapshot")),
            ("common pose", lambda pair: pair["views"][1].update(
                loc_cm=[9.0, 2.0, 90.0])),
            ("common intrinsics", lambda pair: pair["views"][1].update(
                camera_intrinsics_id="different-intrinsics")),
            ("view echo", lambda pair: pair["views"][1].update(view_id="front")),
            ("group echo", lambda pair: pair["views"][1].update(
                capture_group_id="view-pair-5")),
            ("image data", lambda pair: pair["views"][0].update(
                rgb_data_url="not-an-image")),
            ("per-view timing", lambda pair: pair["views"][0].update(
                timing={"capture_read_ms": 1.0, "encode_ms": 2.0})),
        ],
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_capture_view_pair_rejects_malformed_atomic_pairs(self, case, mutate):
        response = copy.deepcopy(view_pair_json())
        mutate(response)

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(
                ScriptedEndpoint(view_pair=response)
            ).capture_view_pair(agent_tag="agent")

    @pytest.mark.parametrize(
        "data_url",
        [
            "data:image/jpeg;base64AA==",
            "data:image/jpeg;base64,",
            "data:image/jpeg;base64,@@==",
            "data:image/jpeg;base64,A===",
            "data:image/png;base64,AA==",
            17,
        ],
        ids=[
            "missing-comma",
            "empty-payload",
            "invalid-alphabet",
            "invalid-padding",
            "wrong-mime",
            "non-string",
        ],
    )
    def test_capture_view_pair_rejects_non_task4_image_data_urls(self, data_url):
        response = view_pair_json()
        response["views"][0]["rgb_data_url"] = data_url

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(
                ScriptedEndpoint(view_pair=response)
            ).capture_view_pair(agent_tag="agent")

    @pytest.mark.parametrize(
        "agent_echo",
        [None, 17, "different-agent"],
        ids=["missing", "wrong-type", "wrong-value"],
    )
    def test_capture_view_pair_requires_the_exact_task4_agent_echo(
        self, agent_echo
    ):
        response = view_pair_json()
        if agent_echo is None:
            response.pop("agent_tag")
        else:
            response["agent_tag"] = agent_echo

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(
                ScriptedEndpoint(view_pair=response)
            ).capture_view_pair(agent_tag="agent")

    def test_capture_view_pair_binds_requested_fov_to_rpc_and_response(self):
        response = view_pair_json()
        endpoint = ScriptedEndpoint(view_pair=response)

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(endpoint).capture_view_pair(
                agent_tag="agent", width_px=640, height_px=360,
                fov_degrees=75.0,
            )

        assert endpoint.calls == [(
            "PixelGoal_CaptureViewPairJson",
            {
                "agent_tag": "agent",
                "width": 640,
                "height": 360,
                "fov_degrees": 75.0,
                "jpeg_quality": 90,
                "commit_snapshot": True,
            },
        )]

    @pytest.mark.parametrize("explicit_fov", [False, True])
    def test_capture_view_pair_normalizes_non_exact_fov_to_task4_float32(
        self, explicit_fov
    ):
        normalized_fov = 90.0999984741211
        endpoint = ScriptedEndpoint(view_pair=view_pair_json(
            fov_degrees=normalized_fov))
        runtime = LivePixelGoalRuntime(
            endpoint, PixelGoalConfig(fov_degrees=90.1))
        kwargs = {"fov_degrees": 90.1} if explicit_fov else {}

        pair = runtime.capture_view_pair(agent_tag="agent", **kwargs)

        assert pair.front.camera_intrinsics_id == "perspective-640x360-hfov90"
        assert endpoint.calls[0][1]["fov_degrees"] == normalized_fov

    def test_capture_view_pair_still_rejects_a_different_float32_fov(self):
        endpoint = ScriptedEndpoint(view_pair=view_pair_json(
            fov_degrees=90.10000610351562))

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(endpoint).capture_view_pair(
                agent_tag="agent", fov_degrees=90.1)

    def test_capture_view_pair_requires_a_successful_complete_response(self):
        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(ScriptedEndpoint(view_pair={
                "success": False,
                "error": "incomplete_view_pair",
            })).capture_view_pair(agent_tag="agent")

    def test_capture_parses_the_snapshot_attached_to_rgb(self):
        endpoint = ScriptedEndpoint(
            capture={
                "success": True,
                "rgb_data_url": "data:image/jpeg;base64,AA==",
                "camera_snapshot_id": "pixel-snapshot-9",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
                "width": 640,
                "height": 360,
            }
        )

        frame = LivePixelGoalRuntime(endpoint).capture_frame(
            agent_tag="PixelGoalCalibrationAgent", width_px=640, height_px=360
        )

        assert frame.camera_snapshot_id == "pixel-snapshot-9"
        assert frame.width_px == 640
        assert endpoint.calls == [
            (
                "PixelGoal_CaptureFrameJson",
                {
                    "agent_tag": "PixelGoalCalibrationAgent",
                    "width": 640,
                    "height": 360,
                    "fov_degrees": 90.0,
                    "jpeg_quality": 90,
                },
            )
        ]

    def test_start_binds_action_to_the_exact_frame_snapshot(self):
        endpoint = ScriptedEndpoint(
            resolve={
                "accepted": True,
                "request_id": "pixel-move-7",
                "camera_snapshot_id": "pixel-snapshot-7",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
                "raw_world_hit_cm": [500.0, 20.0, 50010.0],
                "validated_navigation_target_cm": [504.0, 23.0, 50010.0],
                "navmesh_adjustment_cm": 5.0,
                "controller_request_result": "request_successful",
            }
        )
        runtime = LivePixelGoalRuntime(endpoint)

        started = runtime.start(pixel_goal_action(), pixel_goal_frame())

        assert endpoint.calls[-1][1]["camera_snapshot_id"] == "pixel-snapshot-7"
        assert endpoint.calls[-1][1]["u_norm"] == 0.25
        assert endpoint.calls[-1][1]["v_norm"] == 0.75
        assert started.request.quantized_target is None
        assert started.request.raw_world_hit == Vec3(
            x_cm=500.0, y_cm=20.0, z_cm=50010.0
        )
        assert "view_id" not in endpoint.calls[-1][1]
        assert "capture_group_id" not in endpoint.calls[-1][1]

    def test_start_records_the_exact_controller_polyline_and_detour_limits(self):
        endpoint = ScriptedEndpoint(resolve={
            "accepted": True,
            "request_id": "pixel-move-path",
            "camera_snapshot_id": "pixel-snapshot-7",
            "camera_intrinsics_id": "perspective-640x360-hfov90",
            "raw_world_hit_cm": [500.0, 20.0, 10.0],
            "validated_navigation_target_cm": [504.0, 23.0, 10.0],
            "navmesh_adjustment_cm": 5.0,
            "controller_path_points_cm": [
                [0.0, 0.0, 10.0],
                [250.0, 10.0, 10.0],
                [504.0, 23.0, 10.0],
            ],
            "controller_path_length_cm": 504.2,
            "controller_path_direct_cm": 504.0,
            "controller_path_stretch_ratio": 1.0004,
            "controller_request_result": "request_successful",
        })

        started = LivePixelGoalRuntime(endpoint).start(
            pixel_goal_action(), pixel_goal_frame())

        request = endpoint.calls[-1][1]
        assert request["max_controller_path_stretch_ratio"] == 1.35
        assert request["controller_path_detour_allowance_cm"] == 100.0
        assert [point.x_cm for point in started.request.controller_path_points] == [
            0.0, 250.0, 504.0]
        assert started.request.controller_path_length_cm == pytest.approx(504.2)
        assert started.request.controller_path_stretch_ratio == pytest.approx(1.0004)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_controller_path_stretch_ratio": 0.99}, "at least 1"),
            ({"controller_path_detour_allowance_cm": -1.0}, "nonnegative"),
        ],
    )
    def test_controller_path_detour_config_fails_closed(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PixelGoalConfig(**kwargs)

    def test_start_sends_and_checks_view_group_binding(self):
        endpoint = ScriptedEndpoint(resolve={
            "accepted": True,
            "request_id": "pixel-move-7",
            "camera_snapshot_id": "rear-snapshot",
            "camera_intrinsics_id": "perspective-640x360-hfov90",
            "view_id": "rear",
            "capture_group_id": "view-pair-4",
            "raw_world_hit_cm": [500.0, 20.0, 50010.0],
            "validated_navigation_target_cm": [504.0, 23.0, 50010.0],
            "navmesh_adjustment_cm": 5.0,
            "controller_request_result": "request_successful",
        })
        frame = pixel_goal_frame(
            "rear-snapshot", view="rear", group="view-pair-4")

        LivePixelGoalRuntime(endpoint).start(pixel_goal_action(), frame)

        assert endpoint.calls[-1][1]["view_id"] == "rear"
        assert endpoint.calls[-1][1]["capture_group_id"] == "view-pair-4"

    @pytest.mark.parametrize(
        ("field", "wrong"),
        [("view_id", "front"), ("capture_group_id", "view-pair-5")],
    )
    def test_accepted_resolver_binding_mismatch_is_an_integrity_error(
        self, field, wrong
    ):
        response = {
            "accepted": True,
            "request_id": "pixel-move-7",
            "camera_snapshot_id": "rear-snapshot",
            "camera_intrinsics_id": "perspective-640x360-hfov90",
            "view_id": "rear",
            "capture_group_id": "view-pair-4",
            "raw_world_hit_cm": [500.0, 20.0, 50010.0],
            "validated_navigation_target_cm": [504.0, 23.0, 50010.0],
            "navmesh_adjustment_cm": 5.0,
            "controller_request_result": "request_successful",
        }
        response[field] = wrong

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(ScriptedEndpoint(resolve=response)).start(
                pixel_goal_action(),
                pixel_goal_frame(
                    "rear-snapshot", view="rear", group="view-pair-4"),
            )

    @pytest.mark.parametrize(
        ("field", "wrong"),
        [("view_id", "front"), ("capture_group_id", "view-pair-5")],
    )
    def test_rejected_resolver_binding_mismatch_preempts_policy_rejection(
        self, field, wrong
    ):
        response = {
            "accepted": False,
            "rejection_reason": "no_geometry_hit",
            "camera_snapshot_id": "rear-snapshot",
            "camera_intrinsics_id": "perspective-640x360-hfov90",
            "view_id": "rear",
            "capture_group_id": "view-pair-4",
        }
        response[field] = wrong

        with pytest.raises(PixelGoalPairIntegrityError):
            LivePixelGoalRuntime(ScriptedEndpoint(resolve=response)).start(
                pixel_goal_action(),
                pixel_goal_frame(
                    "rear-snapshot", view="rear", group="view-pair-4"),
            )

    def test_rejected_resolver_with_exact_binding_remains_an_ordinary_rejection(self):
        endpoint = ScriptedEndpoint(resolve={
            "accepted": False,
            "rejection_reason": "no_geometry_hit",
            "camera_snapshot_id": "rear-snapshot",
            "camera_intrinsics_id": "perspective-640x360-hfov90",
            "view_id": "rear",
            "capture_group_id": "view-pair-4",
        })

        with pytest.raises(PixelGoalRejected, match="no_geometry_hit"):
            LivePixelGoalRuntime(endpoint).start(
                pixel_goal_action(),
                pixel_goal_frame(
                    "rear-snapshot", view="rear", group="view-pair-4"),
            )

    @pytest.mark.parametrize(
        "malformed_response",
        [
            pytest.param(None, id="accepted-none"),
            pytest.param([], id="accepted-list"),
            pytest.param("not-an-object", id="accepted-string"),
            pytest.param(17, id="accepted-number"),
            pytest.param(True, id="accepted-bool"),
            pytest.param(None, id="rejected-none"),
            pytest.param([], id="rejected-list"),
            pytest.param("not-an-object", id="rejected-string"),
            pytest.param(17, id="rejected-number"),
            pytest.param(False, id="rejected-bool"),
        ],
    )
    def test_bound_resolver_non_object_response_is_an_integrity_error(
        self, malformed_response
    ):
        class NonObjectEndpoint:
            def call(self, function_name, request):
                assert function_name == "PixelGoal_ResolveAndMoveJson"
                return malformed_response

        with pytest.raises(
            PixelGoalPairIntegrityError, match="invalid_resolver_response"
        ):
            LivePixelGoalRuntime(NonObjectEndpoint()).start(
                pixel_goal_action(),
                pixel_goal_frame(
                    "rear-snapshot", view="rear", group="view-pair-4"),
            )

    def test_snapshot_mismatch_is_rejected(self):
        endpoint = ScriptedEndpoint(
            resolve={
                "accepted": True,
                "request_id": "pixel-move-7",
                "camera_snapshot_id": "different-snapshot",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
                "raw_world_hit_cm": [500.0, 20.0, 50010.0],
                "validated_navigation_target_cm": [500.0, 20.0, 50010.0],
                "navmesh_adjustment_cm": 0.0,
                "controller_request_result": "request_successful",
            }
        )

        with pytest.raises(PixelGoalRejected, match="camera_snapshot_mismatch"):
            LivePixelGoalRuntime(endpoint).start(
                pixel_goal_action(), pixel_goal_frame()
            )

    def test_rejection_is_explicit_and_preserves_first_hit(self):
        endpoint = ScriptedEndpoint(
            resolve={
                "accepted": False,
                "rejection_reason": "hit_not_walkable_ground",
                "raw_world_hit_cm": [400.0, 0.0, 50120.0],
                "camera_snapshot_id": "pixel-snapshot-2",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
            }
        )
        runtime = LivePixelGoalRuntime(endpoint)

        with pytest.raises(PixelGoalRejected) as caught:
            runtime.start(pixel_goal_action(), pixel_goal_frame("pixel-snapshot-2"))

        assert caught.value.reason == "hit_not_walkable_ground"
        assert caught.value.raw_world_hit == Vec3(
            x_cm=400.0, y_cm=0.0, z_cm=50120.0
        )

    def test_poll_computes_planar_and_3d_error_from_final_feet(self):
        endpoint = ScriptedEndpoint(
            statuses=[
                {
                    "request_id": "pixel-move-1",
                    "state": "completed",
                    "controller_result": "success",
                    "accepted_target_cm": [100.0, 200.0, 10.0],
                    "final_agent_position_cm": [106.0, 208.0, 101.0],
                    "final_feet_position_cm": [106.0, 208.0, 12.0],
                    "final_yaw_degrees": 30.0,
                    "elapsed_sim_s": 2.5,
                }
            ]
        )

        result = LivePixelGoalRuntime(endpoint).poll("pixel-move-1")

        assert isinstance(result, ControllerResult)
        assert result.execution_error_planar_m == pytest.approx(0.10)
        assert result.execution_error_3d_m == pytest.approx(
            (104.0**0.5) / 100.0
        )
        assert result.final_feet_position == Vec3(
            x_cm=106.0, y_cm=208.0, z_cm=12.0
        )

    def test_poll_preserves_in_progress_ue_simulation_metrics(self):
        endpoint = ScriptedEndpoint(statuses=[{
            "request_id": "pixel-move-1",
            "state": "moving",
            "final_feet_position_cm": [10.0, 20.0, 0.0],
            "elapsed_sim_s": 7.5,
            "distance_travelled_cm": 412.0,
        }])

        status = LivePixelGoalRuntime(endpoint).poll("pixel-move-1")

        assert isinstance(status, PixelGoalInProgress)
        assert status.elapsed_sim_s == pytest.approx(7.5)
        assert status.distance_travelled_cm == pytest.approx(412.0)

    def test_execute_sim_timeout_aborts_controller_and_returns_typed_result(self):
        endpoint = ScriptedEndpoint(
            resolve={
                "accepted": True,
                "request_id": "pixel-move-timeout",
                "camera_snapshot_id": "pixel-snapshot-7",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
                "raw_world_hit_cm": [500.0, 20.0, 50010.0],
                "validated_navigation_target_cm": [500.0, 20.0, 50020.0],
                "navmesh_adjustment_cm": 10.0,
                "controller_request_result": "request_successful",
            },
            statuses=[{
                "request_id": "pixel-move-timeout",
                "state": "moving",
                "final_feet_position_cm": [10.0, 0.0, 50010.0],
                "elapsed_sim_s": 0.1,
                "distance_travelled_cm": 10.0,
            }],
            cancel={
                "cancelled": True,
                "request_id": "pixel-move-timeout",
                "state": "failed",
                "controller_result": "aborted",
                "failure_reason": "execution_timeout",
                "accepted_target_cm": [500.0, 20.0, 50020.0],
                "final_agent_position_cm": [10.0, 0.0, 50101.0],
                "final_feet_position_cm": [10.0, 0.0, 50010.0],
                "final_yaw_degrees": 0.0,
                "elapsed_sim_s": 0.1,
                "distance_travelled_cm": 10.0,
            },
        )
        runtime = LivePixelGoalRuntime(
            endpoint,
            PixelGoalConfig(
                movement_timeout_sim_s=0.05,
                execution_timeout_s=10.0,
                poll_interval_s=1e-9,
            ),
        )

        _started, result = runtime.execute(pixel_goal_action(), pixel_goal_frame())

        assert result.outcome is ControllerOutcomeCode.EXECUTION_TIMEOUT
        assert result.controller_result == "aborted"
        assert result.failure_reason == "execution_timeout"
        assert endpoint.calls[-1] == (
            "PixelGoal_CancelMoveJson",
            {
                "request_id": "pixel-move-timeout",
                "reason": "execution_timeout",
            },
        )

    def test_execute_lets_ue_finish_past_the_old_45_second_wall_limit(
        self, monkeypatch,
    ):
        endpoint = ScriptedEndpoint(
            resolve={
                "accepted": True,
                "request_id": "pixel-move-slow-host",
                "camera_snapshot_id": "pixel-snapshot-7",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
                "raw_world_hit_cm": [500.0, 20.0, 50010.0],
                "validated_navigation_target_cm": [500.0, 20.0, 50020.0],
                "navmesh_adjustment_cm": 10.0,
                "controller_request_result": "request_successful",
            },
            statuses=[
                {
                    "request_id": "pixel-move-slow-host",
                    "state": "moving",
                    "final_feet_position_cm": [250.0, 10.0, 50010.0],
                    "elapsed_sim_s": 9.0,
                    "distance_travelled_cm": 250.0,
                },
                {
                    "request_id": "pixel-move-slow-host",
                    "state": "completed",
                    "controller_result": "success",
                    "failure_reason": "",
                    "accepted_target_cm": [500.0, 20.0, 50020.0],
                    "final_agent_position_cm": [500.0, 20.0, 50111.0],
                    "final_feet_position_cm": [500.0, 20.0, 50020.0],
                    "final_yaw_degrees": 0.0,
                    "elapsed_sim_s": 18.0,
                    "distance_travelled_cm": 500.0,
                },
            ],
        )
        host_times = iter((0.0, 46.0, 60.0))
        monkeypatch.setattr(
            pixel_goal_module.time, "monotonic", lambda: next(host_times))
        monkeypatch.setattr(pixel_goal_module.time, "sleep", lambda _s: None)
        runtime = LivePixelGoalRuntime(
            endpoint,
            PixelGoalConfig(
                movement_timeout_sim_s=45.0,
                execution_timeout_s=300.0,
            ),
        )

        _started, result = runtime.execute(
            pixel_goal_action(), pixel_goal_frame())

        assert result.outcome is ControllerOutcomeCode.ACCEPTED
        assert result.elapsed_sim_s == pytest.approx(18.0)
        assert not any(
            name == "PixelGoal_CancelMoveJson"
            for name, _request in endpoint.calls
        )

    def test_execute_wall_watchdog_is_an_infrastructure_failure(self):
        endpoint = ScriptedEndpoint(
            resolve={
                "accepted": True,
                "request_id": "pixel-move-watchdog",
                "camera_snapshot_id": "pixel-snapshot-7",
                "camera_intrinsics_id": "perspective-640x360-hfov90",
                "raw_world_hit_cm": [500.0, 20.0, 50010.0],
                "validated_navigation_target_cm": [500.0, 20.0, 50020.0],
                "navmesh_adjustment_cm": 10.0,
                "controller_request_result": "request_successful",
            },
            cancel={
                "cancelled": True,
                "request_id": "pixel-move-watchdog",
                "state": "failed",
                "controller_result": "aborted",
                "failure_reason": "wall_watchdog_timeout",
                "accepted_target_cm": [500.0, 20.0, 50020.0],
                "final_agent_position_cm": [10.0, 0.0, 50101.0],
                "final_feet_position_cm": [10.0, 0.0, 50010.0],
                "final_yaw_degrees": 0.0,
                "elapsed_sim_s": 0.1,
                "distance_travelled_cm": 10.0,
            },
        )
        runtime = LivePixelGoalRuntime(
            endpoint,
            PixelGoalConfig(execution_timeout_s=1e-9, poll_interval_s=1e-9),
        )

        with pytest.raises(PixelGoalInfrastructureTimeout) as caught:
            runtime.execute(pixel_goal_action(), pixel_goal_frame())

        assert caught.value.result.failure_reason == "wall_watchdog_timeout"
        assert endpoint.calls[-1] == (
            "PixelGoal_CancelMoveJson",
            {
                "request_id": "pixel-move-watchdog",
                "reason": "wall_watchdog_timeout",
            },
        )
