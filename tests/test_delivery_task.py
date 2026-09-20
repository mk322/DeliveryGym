"""Tests for the Delivery task layer on ``EnvSpec``.

The point of building the task on a compiled spec is that it can answer "can I
run here?" before doing any work, and refuse with a reason instead of failing
mid-episode. These tests are mostly about that refusal being correct, and about
the reward being something anyone can recompute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodiedbench.schemas.env_spec import (
    AffordanceInventory,
    EnvSpec,
    GraphSummary,
    NavigationStyle,
    ObservationSupport,
    SolvabilityEvidence,
)
from embodiedbench.schemas.environment import CertificationGrade, NavigationMode
from embodiedbench.schemas.runtime import ActionResult, ActionStatus, Observation
from embodiedbench.schemas.trajectory import Trajectory, TrajectoryTurn
from embodiedbench.tasks.delivery import COURIER_PROFILES, DeliveryTask
from embodiedbench.tasks.profiles import (
    PRESETS,
    BudgetProfile,
    ConstraintProfile,
    DeliveryTaskConfig,
    ObservationProfile,
    OrderProfile,
    RewardProfile,
    TransportProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_env(**overrides) -> EnvSpec:
    base = dict(
        env_id="demo-env",
        map_name="demo",
        navigation_style=NavigationStyle.GRAPH,
        navigation_modes=[NavigationMode.NAV_WAYPOINT],
        enabled_actions=["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"],
        navigation_rationale="only 23.1% of edges are near-cardinal",
        graph=GraphSummary(
            node_count=1162, edge_count=5649, mean_degree=9.72, max_degree=35,
            cardinal_fraction=0.231, largest_component_fraction=1.0,
        ),
        affordances=AffordanceInventory(counts={"building": 414, "restaurant": 18, "store": 11}),
        grade=CertificationGrade.B,
        quality_flags=[],
        solvability=SolvabilityEvidence(
            episodes=5, delivered_episodes=5, solvability_rate=1.0, mean_steps=21.8
        ),
    )
    base.update(overrides)
    return EnvSpec(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration composition (design plan §8.2)
# ─────────────────────────────────────────────────────────────────────────────


def test_every_preset_is_valid():
    for name, config in PRESETS.items():
        assert config.validate() == [], f"{name}: {config.validate()}"


def test_invalid_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError, match="invalid delivery configuration"):
        DeliveryTask(DeliveryTaskConfig(order=OrderProfile(order_count=0)))


def test_validation_reports_every_problem_not_just_the_first():
    config = DeliveryTaskConfig(
        order=OrderProfile(order_count=0),
        budget=BudgetProfile(steps=0),
        reward=RewardProfile(delivery=0.0),
    )
    problems = config.validate()
    assert len(problems) >= 3, problems


def test_battery_without_a_scooter_is_incoherent():
    config = DeliveryTaskConfig(
        constraint=ConstraintProfile(battery=True),
        transport=TransportProfile(modes=("walk",), initial_mode="walk"),
    )
    assert any("battery requires a scooter" in p for p in config.validate())


def test_initial_mode_must_be_available():
    config = DeliveryTaskConfig(
        transport=TransportProfile(modes=("walk",), initial_mode="scooter")
    )
    assert any("initial_mode" in p for p in config.validate())


def test_step_budget_must_plausibly_cover_the_orders():
    config = DeliveryTaskConfig(
        order=OrderProfile(order_count=20), budget=BudgetProfile(steps=50)
    )
    assert any("cannot plausibly cover" in p for p in config.validate())


def test_image_observations_require_the_rgb_channel():
    config = DeliveryTaskConfig(
        observation=ObservationProfile(channels=("text",), include_fpv=True)
    )
    assert any("require the rgb channel" in p for p in config.validate())


def test_battery_requires_a_charging_station_affordance():
    config = DeliveryTaskConfig(constraint=ConstraintProfile(battery=True))
    assert config.required_affordances().get("charging_station") == 1


def test_bus_transport_requires_bus_stops():
    config = PRESETS["multi_modal"]
    needs = config.required_affordances()
    assert needs.get("bus_station", 0) >= 2
    assert needs.get("car_rental", 0) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility: refuse with a reason, before doing work
# ─────────────────────────────────────────────────────────────────────────────


def test_standard_delivery_runs_on_a_paris_like_environment():
    """Paris has no `customer` POIs; delivery must still run there."""
    task = DeliveryTask("standard")
    compatibility = task.check_environment(make_env())
    assert compatibility.can_run, compatibility.reasons


def test_multi_modal_is_refused_where_bus_stops_do_not_exist():
    task = DeliveryTask("multi_modal")
    compatibility = task.check_environment(make_env())
    assert not compatibility.can_run
    assert "bus_station" in compatibility.deficit
    assert any("missing affordances" in r for r in compatibility.reasons)


def test_unusable_environment_is_refused():
    task = DeliveryTask("standard")
    env = make_env(
        usable=False,
        failure_code="no_pois",
        failure_explanation="no POIs",
        grade=CertificationGrade.FAIL,
        solvability=None,
    )
    assert not task.check_environment(env).can_run


def test_vision_task_is_refused_without_an_album():
    config = DeliveryTaskConfig(
        observation=ObservationProfile(channels=("text", "rgb"), include_fpv=True)
    )
    task = DeliveryTask(config)
    compatibility = task.check_environment(make_env())
    assert not compatibility.can_run
    assert any("no cached album" in r for r in compatibility.reasons)


def test_vision_task_accepted_when_an_album_exists():
    config = DeliveryTaskConfig(
        observation=ObservationProfile(channels=("text", "rgb"), include_fpv=True)
    )
    env = make_env(
        observation=ObservationSupport(
            channels=["text", "rgb"], has_cached_album=True, album_waypoints=136,
            album_headings=4,
        )
    )
    assert DeliveryTask(config).check_environment(env).can_run


def test_generation_refuses_an_incompatible_environment():
    with pytest.raises(ValueError, match="cannot run"):
        DeliveryTask("multi_modal").generate(make_env(), seed=1)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism (the design plan M4)
# ─────────────────────────────────────────────────────────────────────────────


def test_same_seed_produces_an_identical_episode():
    task = DeliveryTask("standard")
    env = make_env()
    a = task.generate(env, seed=42)
    b = task.generate(env, seed=42)
    assert a.content_hash() == b.content_hash()


def test_different_seeds_produce_different_episodes():
    task = DeliveryTask("standard")
    env = make_env()
    assert task.generate(env, seed=1).instance_id != task.generate(env, seed=2).instance_id


def test_episode_carries_the_environment_hash():
    env = make_env(world_bundle_sha256="b" * 64)
    spec = DeliveryTask("standard").generate(env, seed=7)
    assert spec.environment_sha256 == "b" * 64


# ─────────────────────────────────────────────────────────────────────────────
# The agent-facing contract
# ─────────────────────────────────────────────────────────────────────────────


def test_instruction_describes_the_actual_action_space():
    """A Paris instruction must not tell the agent to use directional MOVE."""
    graph_env = make_env()
    instruction = DeliveryTask("standard").instruction(graph_env)
    assert "MOVE_TO" in instruction
    assert 'MOVE(direction' not in instruction

    cardinal_env = make_env(
        navigation_style=NavigationStyle.CARDINAL_AND_GRAPH,
        enabled_actions=["VIEW_ORDERS", "MOVE_TO", "MOVE"],
        navigation_rationale="100% cardinal",
        graph=GraphSummary(
            node_count=136, edge_count=153, mean_degree=2.25, max_degree=4,
            cardinal_fraction=1.0, largest_component_fraction=1.0,
        ),
    )
    assert "MOVE(direction" in DeliveryTask("standard").instruction(cardinal_env)


def test_action_schema_comes_from_the_environment():
    env = make_env()
    assert DeliveryTask("standard").action_schema(env) == env.enabled_actions


def test_reset_observation_exposes_every_outcome_relevant_field():
    """design plan §17: hidden attributes make instances unfair and uninterpretable."""
    task = DeliveryTask("constrained")
    observation = task.reset_observation(make_env())
    for key in ("carrying_capacity", "battery", "deadlines", "transport_modes"):
        assert key in observation, key
    courier = COURIER_PROFILES[task.config.courier_profile]
    for field_name in courier.outcome_relevant_fields:
        assert field_name in observation["courier"], field_name


# ─────────────────────────────────────────────────────────────────────────────
# Verifiable reward
# ─────────────────────────────────────────────────────────────────────────────


def _trajectory(turns: int = 3, invalid: int = 0) -> Trajectory:
    records = []
    for index in range(turns):
        status = ActionStatus.REJECTED if index < invalid else ActionStatus.ACCEPTED
        records.append(
            TrajectoryTurn(
                step_index=index,
                observation=Observation(episode_id="ep", step_index=index),
                action_result=ActionResult(
                    status=status,
                    error_code="vendor_action_error" if status is ActionStatus.REJECTED else None,
                ),
                reward=0.0,
            )
        )
    return Trajectory(
        episode_id="ep", instance_id="inst", environment_id="demo-env",
        environment_version="0.1.0", turns=records, total_reward=0.0,
    )


def test_reward_is_recomputable_from_recorded_facts():
    task = DeliveryTask("standard")
    trajectory = _trajectory(turns=10, invalid=2)
    privileged = {"delivered_count": 2, "on_time_count": 1}
    components = task.reward_components(privileged, trajectory)
    weights = task.config.reward
    assert components["delivery"] == pytest.approx(weights.delivery * 2)
    assert components["on_time_bonus"] == pytest.approx(weights.on_time_bonus * 1)
    assert components["late_penalty"] == pytest.approx(-weights.late_penalty * 1)
    assert components["invalid_action_penalty"] == pytest.approx(
        -weights.invalid_action_penalty * 2
    )


def test_more_deliveries_score_higher():
    task = DeliveryTask("standard")
    trajectory = _trajectory(turns=5)
    low = sum(task.reward_components({"delivered_count": 1}, trajectory).values())
    high = sum(task.reward_components({"delivered_count": 3}, trajectory).values())
    assert high > low


def test_invalid_actions_are_penalised():
    task = DeliveryTask("standard")
    clean = task.reward_components({"delivered_count": 1}, _trajectory(turns=5, invalid=0))
    messy = task.reward_components({"delivered_count": 1}, _trajectory(turns=5, invalid=5))
    assert sum(messy.values()) < sum(clean.values())


def test_score_report_declines_to_invent_the_primary_metric():
    report = DeliveryTask("standard").evaluate(_trajectory(), {"delivered_count": 1})
    assert report.normalized_utility_vs_upper_bound is None
    assert report.upper_bound_undefined_reason
    assert report.success is True


def test_zero_deliveries_is_not_success():
    report = DeliveryTask("standard").evaluate(_trajectory(), {"delivered_count": 0})
    assert report.success is False


def test_training_shaping_is_reported_separately_from_the_score():
    """design plan §11.3: shaping must never enter the benchmark score."""
    report = DeliveryTask("standard").evaluate(_trajectory(), {"delivered_count": 1})
    assert "progress" in report.training_shaping
    assert not any(name.startswith("shaping") for name in report.metrics)


def test_score_report_round_trips():
    report = DeliveryTask("standard").evaluate(_trajectory(), {"delivered_count": 1})
    assert report.round_trip().to_dict() == report.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Against real compiled environments
# ─────────────────────────────────────────────────────────────────────────────

SPEC_DIR = REPO_ROOT / "artifacts" / "verification" / "MAPS"
REAL = sorted(SPEC_DIR.glob("*.envspec.json")) if SPEC_DIR.exists() else []


@pytest.mark.skipif(not REAL, reason="no compiled EnvSpecs on disk")
@pytest.mark.parametrize("path", REAL, ids=lambda p: p.name.split(".")[0])
def test_standard_delivery_runs_on_every_compiled_map(path):
    env = EnvSpec.from_dict(json.loads(path.read_text()))
    compatibility = DeliveryTask("standard").check_environment(env)
    assert compatibility.can_run, f"{env.map_name}: {compatibility.reasons}"


@pytest.mark.skipif(not REAL, reason="no compiled EnvSpecs on disk")
def test_paris_refuses_multi_modal_for_the_right_reason():
    paris = [p for p in REAL if "citycore-paris" in p.name]
    if not paris:
        pytest.skip("Paris not compiled")
    env = EnvSpec.from_dict(json.loads(paris[0].read_text()))
    compatibility = DeliveryTask("multi_modal").check_environment(env)
    assert not compatibility.can_run
    # design plan §3.3.7's missing Paris affordances, surfaced as a deficit.
    assert "bus_station" in compatibility.deficit


def test_the_step_floor_matches_what_a_leg_actually_costs():
    """A validator that passes an unfinishable configuration certifies it.

    The old floor was 5 steps an order. A delivery leg on the compiled Paris
    carriageway is a median 530 m over 18 m edges, so a shortest-path oracle
    spends ~23 turns an order stepping junction by junction and ~11 using
    ``follow_street``. At 5 the validator happily accepted the 120-step budget
    the benchmark actually ran, under which the oracle could deliver 3.1 of 10.
    """
    from embodiedbench.tasks.profiles import (
        MIN_STEPS_PER_ORDER,
        BudgetProfile,
        DeliveryTaskConfig,
        OrderProfile,
    )

    assert MIN_STEPS_PER_ORDER >= 11
    tight = DeliveryTaskConfig(
        order=OrderProfile(order_count=10),
        budget=BudgetProfile(steps=10 * MIN_STEPS_PER_ORDER - 1),
    )
    assert any("cannot plausibly cover" in p for p in tight.validate())
    fine = DeliveryTaskConfig(
        order=OrderProfile(order_count=10),
        budget=BudgetProfile(steps=10 * MIN_STEPS_PER_ORDER),
    )
    assert not any("cannot plausibly cover" in p for p in fine.validate())
