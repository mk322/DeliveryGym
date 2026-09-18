"""Contract tests for every protocol schema (the design plan M1).

M1 accepts when "JSON Schema/Pydantic round-trip, unknown-field/version, and
invalid-fixture tests pass for every contract" and "invalid versions, coordinate
frames, capabilities, and action payloads fail with typed errors".

The suite enumerates the schema registry rather than naming contracts, so a new
contract without a fixture fails ``test_every_versioned_schema_has_a_fixture``
instead of being silently untested.
"""

from __future__ import annotations

import json

import pytest

from embodiedbench.artifacts.hashing import canonical_json
from embodiedbench.schemas import SCHEMA_REGISTRY
from embodiedbench.schemas.base import (
    CapabilityError,
    FrameError,
    PayloadError,
    SchemaError,
    SchemaVersion,
    VersionError,
)
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.fixtures import (
    invalid_fixtures,
    iter_valid,
    uncovered_schemas,
    valid_fixtures,
)
from embodiedbench.schemas.geometry import (
    CameraIntrinsics,
    FrameName,
    Pose,
    Transform,
    Vec3,
)
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    NavWaypointAction,
    RuntimeCapabilities,
    RuntimeMode,
    StepResult,
    TerminationReason,
)

VALID = list(iter_valid())
VALID_IDS = [schema_id for schema_id, _ in VALID]


# ─────────────────────────────────────────────────────────────────────────────
# Coverage
# ─────────────────────────────────────────────────────────────────────────────


def test_every_versioned_schema_has_a_fixture():
    """A contract without a fixture is a contract nothing below this tests."""
    assert uncovered_schemas() == set()


def test_registry_is_non_trivial():
    assert len(SCHEMA_REGISTRY()) >= 19


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_round_trip_preserves_value(schema_id, model):
    assert model.round_trip().to_dict() == model.to_dict()


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_round_trip_preserves_content_hash(schema_id, model):
    assert model.round_trip().content_hash() == model.content_hash()


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_serialization_is_json_and_canonical(schema_id, model):
    payload = model.to_dict()
    encoded = canonical_json(payload)
    assert json.loads(encoded) == payload


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_envelope_declares_its_schema_and_version(schema_id, model):
    payload = model.to_dict()
    assert payload["schema"] == schema_id
    assert payload["schema_version"] == type(model).SCHEMA_VERSION


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_all_schemas_are_v0_before_m12(schema_id, model):
    """the design plan M1: all schemas remain v0.x with no compatibility promise."""
    assert SchemaVersion.parse(type(model).SCHEMA_VERSION).major == 0


# ─────────────────────────────────────────────────────────────────────────────
# Unknown fields and versions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_unknown_field_is_rejected(schema_id, model):
    payload = model.to_dict()
    payload["a_field_from_the_future"] = 1
    with pytest.raises(PayloadError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_wrong_major_version_is_a_version_error(schema_id, model):
    payload = model.to_dict()
    payload["schema_version"] = "9.0.0"
    with pytest.raises(VersionError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_differing_minor_is_rejected_while_v0(schema_id, model):
    """the design plan M1: early consumers pin exact schema revisions."""
    payload = model.to_dict()
    payload["schema_version"] = "0.99.0"
    with pytest.raises(VersionError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_wrong_schema_id_is_a_version_error(schema_id, model):
    payload = model.to_dict()
    payload["schema"] = "embodiedbench/something_else"
    with pytest.raises(VersionError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_missing_version_is_a_version_error(schema_id, model):
    payload = model.to_dict()
    del payload["schema_version"]
    with pytest.raises(VersionError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_malformed_version_is_a_version_error(schema_id, model):
    payload = model.to_dict()
    payload["schema_version"] = "not-a-version"
    with pytest.raises(VersionError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("schema_id,model", VALID, ids=VALID_IDS)
def test_non_object_payload_is_a_payload_error(schema_id, model):
    with pytest.raises(PayloadError):
        type(model).from_dict(["not", "an", "object"])


# ─────────────────────────────────────────────────────────────────────────────
# Invalid fixtures
# ─────────────────────────────────────────────────────────────────────────────

INVALID = invalid_fixtures()


@pytest.mark.parametrize("fixture", INVALID, ids=[f.name for f in INVALID])
def test_invalid_fixture_is_rejected(fixture):
    """Each of these would let a real defect through if the schema allowed it."""
    with pytest.raises(SchemaError) as caught:
        fixture.model.from_dict(fixture.payload)
    assert isinstance(caught.value, SchemaError), fixture.rule


def test_invalid_fixtures_cover_distinct_rules():
    rules = [f.rule for f in INVALID]
    assert len(rules) == len(set(rules)), "two invalid fixtures cite the same rule"


def test_invalid_fixture_set_is_substantial():
    assert len(INVALID) >= 25


# ─────────────────────────────────────────────────────────────────────────────
# Typed errors: versions, frames, capabilities, action payloads
# ─────────────────────────────────────────────────────────────────────────────


def test_version_compatibility_rule():
    assert SchemaVersion.parse("0.1.0").can_read(SchemaVersion.parse("0.1.0"))
    assert not SchemaVersion.parse("0.1.0").can_read(SchemaVersion.parse("0.2.0"))
    assert not SchemaVersion.parse("1.0.0").can_read(SchemaVersion.parse("2.0.0"))
    # From 1.0 on, a reader accepts an older or equal minor.
    assert SchemaVersion.parse("1.3.0").can_read(SchemaVersion.parse("1.2.0"))
    assert not SchemaVersion.parse("1.2.0").can_read(SchemaVersion.parse("1.3.0"))


def test_comparing_poses_across_frames_is_a_frame_error():
    here = Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0))
    there = Pose(frame=FrameName.CAMERA, position=Vec3(x_cm=0, y_cm=0))
    with pytest.raises(FrameError):
        here.close_to(there, position_cm=5.0, yaw_deg=0.5)


def test_composing_transforms_that_do_not_meet_is_a_frame_error():
    ue_to_world = Transform(
        from_frame=FrameName.UE_WORLD,
        to_frame=FrameName.BUNDLE_WORLD,
        translation=Vec3(x_cm=10, y_cm=0),
    )
    camera_to_agent = Transform(
        from_frame=FrameName.CAMERA,
        to_frame=FrameName.EMBODIMENT,
        translation=Vec3(x_cm=0, y_cm=0),
    )
    with pytest.raises(FrameError):
        ue_to_world.then(camera_to_agent)


def test_transform_round_trips_through_its_inverse():
    transform = Transform(
        from_frame=FrameName.UE_WORLD,
        to_frame=FrameName.BUNDLE_WORLD,
        translation=Vec3(x_cm=1500, y_cm=-250, z_cm=10),
        yaw_deg=37.0,
    )
    point = Vec3(x_cm=123.5, y_cm=-88.25, z_cm=4.0)
    back = transform.inverse().apply(transform.apply(point))
    assert back.distance_cm(point) < 1e-6


def test_transform_between_identical_frames_is_rejected():
    with pytest.raises(ValueError):
        Transform(
            from_frame=FrameName.UE_WORLD,
            to_frame=FrameName.UE_WORLD,
            translation=Vec3(x_cm=0, y_cm=0),
        )


def test_capability_error_when_runtime_lacks_the_navigation_mode():
    capabilities = RuntimeCapabilities(
        mode=RuntimeMode.TEXT, navigation_modes=[NavigationMode.NAV_WAYPOINT]
    )
    envelope = ActionEnvelope(
        episode_id="ep_1",
        step_index=0,
        action={"type": "nav_point_3d", "target": {"u_norm": 0.5, "v_norm": 0.5, "distance_m": 6.0}},
    )
    with pytest.raises(CapabilityError):
        capabilities.require_action(envelope)


def test_capability_check_passes_for_an_offered_mode():
    capabilities = RuntimeCapabilities(
        mode=RuntimeMode.TEXT, navigation_modes=[NavigationMode.NAV_WAYPOINT]
    )
    envelope = ActionEnvelope(
        episode_id="ep_1", step_index=0, action=NavWaypointAction(target_node="int_2")
    )
    capabilities.require_action(envelope)


def test_embodiment_capability_error_is_typed():
    from embodiedbench.schemas.fixtures import _capabilities

    with pytest.raises(CapabilityError):
        _capabilities().require(NavigationMode.NAV_POINT_3D)


# ─────────────────────────────────────────────────────────────────────────────
# terminated vs truncated: separately tested causes (the design plan M1)
# ─────────────────────────────────────────────────────────────────────────────


def _step(**overrides):
    payload = valid_fixtures()["embodiedbench/step_result"].to_dict()
    payload.update(overrides)
    return StepResult.from_dict(payload)


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.TASK_SUCCESS,
        TerminationReason.TASK_FAILURE,
        TerminationReason.UNRECOVERABLE_STATE,
    ],
)
def test_termination_causes_set_terminated(reason):
    result = _step(terminated=True, truncated=False, termination_reason=reason.value)
    assert result.terminated and not result.truncated
    assert reason.is_termination


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.STEP_BUDGET_EXHAUSTED,
        TerminationReason.SIM_TIME_BUDGET_EXHAUSTED,
        TerminationReason.TOOL_CALL_BUDGET_EXHAUSTED,
        TerminationReason.TOKEN_BUDGET_EXHAUSTED,
        TerminationReason.INFRASTRUCTURE_FAILURE,
    ],
)
def test_truncation_causes_set_truncated(reason):
    result = _step(terminated=False, truncated=True, termination_reason=reason.value)
    assert result.truncated and not result.terminated
    assert not reason.is_termination


def test_termination_cause_cannot_be_used_for_truncation():
    with pytest.raises(SchemaError):
        _step(terminated=False, truncated=True, termination_reason="task_success")


def test_truncation_cause_cannot_be_used_for_termination():
    with pytest.raises(SchemaError):
        _step(terminated=True, truncated=False, termination_reason="step_budget_exhausted")


def test_infrastructure_failure_is_a_truncation_not_a_task_failure():
    """design plan §12.2 records evaluator infrastructure timeouts separately."""
    assert not TerminationReason.INFRASTRUCTURE_FAILURE.is_termination


# ─────────────────────────────────────────────────────────────────────────────
# Privileged state never reaches the agent
# ─────────────────────────────────────────────────────────────────────────────


def test_observation_has_no_field_that_can_carry_privileged_state():
    from embodiedbench.schemas.runtime import Observation

    forbidden = {"privileged_state", "privileged_state_ref", "private_state", "world_state"}
    assert forbidden & set(Observation.model_fields) == set()


def test_step_result_keeps_privileged_state_out_of_the_observation():
    result = valid_fixtures()["embodiedbench/step_result"]
    assert result.privileged_state_ref is not None
    assert "private://" not in canonical_json(result.observation.to_dict()).decode()


def test_episode_spec_agent_view_hides_the_order_schedule():
    spec = valid_fixtures()["embodiedbench/episode_spec"]
    visible = canonical_json(spec.agent_visible()).decode()
    assert spec.order_schedule, "fixture should have orders to hide"
    for order in spec.order_schedule:
        assert order.order_id not in visible


def test_courier_observable_summary_covers_every_outcome_relevant_field():
    profile = valid_fixtures()["embodiedbench/courier_profile"]
    assert set(profile.observable_summary()) == set(profile.outcome_relevant_fields)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: units are explicit (design plan §6.1 P1)
# ─────────────────────────────────────────────────────────────────────────────


def test_metre_and_centimetre_conversion_is_explicit():
    assert Vec3.from_m(1.0, 2.0, 3.0).x_cm == 100.0
    assert Vec3(x_cm=250.0, y_cm=0.0).as_m()[0] == 2.5


def test_non_finite_coordinates_are_rejected():
    with pytest.raises(ValueError):
        Vec3(x_cm=float("nan"), y_cm=0.0)
    with pytest.raises(ValueError):
        Vec3(x_cm=float("inf"), y_cm=0.0)


def test_yaw_is_normalized_so_0_and_360_agree():
    a = Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0), yaw_deg=0.0)
    b = Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0), yaw_deg=360.0)
    assert a.content_hash() == b.content_hash()


def test_yaw_difference_takes_the_short_way_round():
    a = Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0), yaw_deg=350.0)
    b = Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0), yaw_deg=10.0)
    assert a.yaw_difference_deg(b) == pytest.approx(20.0)


def test_unprojection_is_centred_and_scales_with_distance():
    intrinsics = CameraIntrinsics(
        width_px=640, height_px=480, fx_px=320.0, fy_px=320.0,
        cx_px=320.0, cy_px=240.0, near_cm=10.0, far_cm=100000.0,
    )
    centre = intrinsics.unproject(0.5, 0.5, 1000.0)
    assert centre.x_cm == pytest.approx(0.0)
    assert centre.y_cm == pytest.approx(0.0)
    assert centre.z_cm == pytest.approx(1000.0)
    farther = intrinsics.unproject(0.5, 0.5, 2000.0)
    assert farther.z_cm == pytest.approx(2 * centre.z_cm)


def test_unprojection_rejects_out_of_frame_and_bad_distance():
    intrinsics = CameraIntrinsics(
        width_px=640, height_px=480, fx_px=320.0, fy_px=320.0,
        cx_px=320.0, cy_px=240.0, near_cm=10.0, far_cm=100000.0,
    )
    with pytest.raises(ValueError):
        intrinsics.unproject(1.5, 0.5, 100.0)
    with pytest.raises(ValueError):
        intrinsics.unproject(0.5, 0.5, -1.0)


def test_far_plane_must_exceed_near_plane():
    with pytest.raises(ValueError):
        CameraIntrinsics(
            width_px=640, height_px=480, fx_px=320.0, fy_px=320.0,
            cx_px=320.0, cy_px=240.0, near_cm=100.0, far_cm=10.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers used by later milestones
# ─────────────────────────────────────────────────────────────────────────────


def test_certified_road_coverage_excludes_excluded_edges():
    graph = valid_fixtures()["embodiedbench/nav_graph"]
    certified = 1850.0 + 400.0
    assert graph.certified_length_cm() == pytest.approx(certified)
    assert graph.certified_road_coverage() == pytest.approx(certified / (certified + 4528.0))


def test_neighbors_ignore_excluded_edges_by_default():
    """the design plan M2: excluded edges are unavailable to episode generation."""
    graph = valid_fixtures()["embodiedbench/nav_graph"]
    assert graph.neighbors("int_1") == ["int_2"]
    assert "dock_1" in graph.neighbors("int_1", certified_only=False)
