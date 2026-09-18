"""``delivery@1`` task plugin, built on ``EnvSpec`` (design plan §8).

The task reads a compiled ``EnvSpec`` rather than a map name. That is what lets
one environment host many tasks: a task states what it needs, checks it against
the published affordance inventory, and refuses with a reason if the environment
cannot host it. Nothing here re-derives a fact the compiler already measured.

Three properties are load-bearing.

**Determinism.** the design plan M4 requires the same ``(environment, config, seed)`` to
produce a byte-identical ``EpisodeSpec`` in two clean processes, so generation
uses an explicitly seeded ``random.Random`` over sorted inputs. No set
iteration, no dict ordering, no wall-clock.

**Verifiable reward.** Every component is computed from recorded environment
facts -- deliveries, lateness, invalid actions -- so anyone can recompute a
score from a trajectory. No model judges anything. design plan §11.3 keeps training
shaping out of the benchmark score, so shaping is reported in its own block and
never summed in.

**Honest metrics.** ``normalized_utility_vs_upper_bound`` stays undefined with a
stated reason, because design plan §12.5 defines it against an upper-bound policy
that does not exist before M7. Reporting a ratio to nothing would be worse than
reporting nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.schemas.env_spec import EnvSpec
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.episode import (
    Budgets,
    CourierProfile,
    EpisodeSpec,
    ScheduledOrder,
    TaskConfigRef,
)
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import RuntimeMode
from embodiedbench.schemas.trajectory import MetricValue, ScoreReport, Trajectory
from embodiedbench.tasks.core import SolvabilityVerdict, TaskRequirements
from embodiedbench.tasks.profiles import (
    MIN_STEPS_PER_ORDER,
    PRESETS,
    DeliveryTaskConfig,
)

# The M0-frozen v1 courier profile set (ADR-0004). scooter_veteran is
# deliberately absent: design plan §8.2 conditions it on a scientific justification
# that does not exist.
COURIER_PROFILES: dict[str, CourierProfile] = {
    "walker_novice": CourierProfile(
        profile_id="walker_novice",
        transport_modes=["walk"],
        carrying_capacity=1,
        owns_scooter=False,
        battery_enabled=False,
        outcome_relevant_fields=["transport_modes", "carrying_capacity", "owns_scooter"],
    ),
    "scooter_standard": CourierProfile(
        profile_id="scooter_standard",
        transport_modes=["walk", "scooter"],
        carrying_capacity=3,
        owns_scooter=True,
        battery_enabled=True,
        outcome_relevant_fields=[
            "transport_modes", "carrying_capacity", "owns_scooter", "battery_enabled",
        ],
    ),
    "multi_modal_courier": CourierProfile(
        profile_id="multi_modal_courier",
        transport_modes=["walk", "scooter", "bus", "car"],
        carrying_capacity=4,
        owns_scooter=True,
        battery_enabled=True,
        bus_access=True,
        car_rental_access=True,
        outcome_relevant_fields=[
            "transport_modes", "carrying_capacity", "owns_scooter", "battery_enabled",
            "bus_access", "car_rental_access",
        ],
    ),
}


@dataclass
class Compatibility:
    """Whether a task can run in an environment, and what is missing."""

    can_run: bool
    reasons: list[str] = field(default_factory=list)
    deficit: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"can_run": self.can_run, "reasons": self.reasons, "deficit": self.deficit}


class DeliveryTask:
    """The Delivery task plugin."""

    id = "delivery"
    version = "0.2.0"

    def __init__(self, config: DeliveryTaskConfig | str = "standard"):
        if isinstance(config, str):
            if config not in PRESETS:
                raise ValueError(f"unknown preset {config!r}; have {sorted(PRESETS)}")
            config = PRESETS[config]
        problems = config.validate()
        if problems:
            raise ValueError("invalid delivery configuration: " + "; ".join(problems))
        self.config = config

    # ── requirements and compatibility ───────────────────────────────────────

    def requirements(self) -> TaskRequirements:
        return TaskRequirements(affordances=self.config.required_affordances())

    def check_environment(self, env: EnvSpec) -> Compatibility:
        """Whether this environment can host this task, with reasons.

        Checked against the published spec rather than by attempting an episode
        and seeing what breaks, so the answer is available before any work.
        """
        reasons: list[str] = []
        if not env.usable:
            reasons.append(f"environment is unusable: {env.failure_code}")
        needed = self.config.required_affordances()
        deficit = env.affordances.deficit(needed)
        if deficit:
            reasons.append(
                "missing affordances: "
                + ", ".join(f"{name} (need {count} more)" for name, count in sorted(deficit.items()))
            )
        if "MOVE_TO" not in env.enabled_actions:
            reasons.append("environment does not offer graph navigation")
        if self.config.observation.include_fpv and not env.supports_vision():
            reasons.append(
                "task asks for first-person images but the environment has no cached album"
            )
        if env.solvability is not None and env.solvability.solvability_rate <= 0.0:
            reasons.append("environment has no demonstrated solvable episode")
        return Compatibility(can_run=not reasons, reasons=reasons, deficit=deficit)

    # ── generation ───────────────────────────────────────────────────────────

    def generate(self, env: EnvSpec, seed: int, *, world: Any = None) -> EpisodeSpec:
        """Produce a frozen episode for this environment and seed."""
        compatibility = self.check_environment(env)
        if not compatibility.can_run:
            raise ValueError(
                f"delivery cannot run in {env.env_id!r}: " + "; ".join(compatibility.reasons)
            )

        rng = random.Random(seed)
        courier = COURIER_PROFILES[self.config.courier_profile]

        # Spawn and order sites come from the world bundle when one is supplied;
        # otherwise the episode pins only the schedule and the runtime picks its
        # own spawn, which is still deterministic for a given seed.
        spawn = Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0.0, y_cm=0.0))
        site_ids: list[str] = []
        if world is not None:
            sites = sorted(world.interaction_sites, key=lambda s: s.site_id)
            site_ids = [s.site_id for s in sites]
            nodes = sorted(world.nav_graph.nodes, key=lambda n: n.node_id)
            if nodes:
                chosen = nodes[rng.randrange(len(nodes))]
                spawn = Pose(
                    frame=FrameName.BUNDLE_WORLD,
                    position=Vec3(
                        x_cm=chosen.position.x_cm,
                        y_cm=chosen.position.y_cm,
                        z_cm=chosen.position.z_cm,
                    ),
                )

        orders: list[ScheduledOrder] = []
        if len(site_ids) >= 2:
            for index in range(self.config.order.order_count):
                orders.append(
                    ScheduledOrder(
                        order_id=f"o_{index}",
                        available_from_sim_time_s=index * self.config.order.interval_s,
                        pickup_site=site_ids[rng.randrange(len(site_ids))],
                        dropoff_site=site_ids[rng.randrange(len(site_ids))],
                    )
                )

        return EpisodeSpec(
            instance_id=f"{env.map_name}_{self.id}_{seed:06d}",
            environment_id=env.env_id,
            environment_version="0.1.0",
            environment_sha256=env.world_bundle_sha256,
            task=TaskConfigRef(plugin=self.id, version=self.version),
            embodiment_profile="abstract_courier_v1",
            courier_profile=courier,
            navigation_mode=NavigationMode.NAV_WAYPOINT,
            runtime_track=(
                RuntimeMode.CACHED if self.config.observation.include_fpv else RuntimeMode.TEXT
            ),
            seed=seed,
            spawn=spawn,
            order_schedule=orders,
            observable_instruction=self.instruction(env),
            budgets=Budgets(
                steps=self.config.budget.steps,
                tool_calls=self.config.budget.tool_calls,
                sim_s=self.config.budget.sim_s,
                output_tokens=self.config.budget.output_tokens,
            ),
            evaluator_id="delivery_score",
            evaluator_version=self.version,
            private_state_ref=f"private://{env.map_name}/{seed}",
        )

    def check_solvable(self, spec: EpisodeSpec, env: EnvSpec) -> SolvabilityVerdict:
        """design plan §8.3's rejection rules, as far as a spec can answer them."""
        reasons: list[str] = []
        if spec.budgets.steps < self.config.order.order_count * MIN_STEPS_PER_ORDER:
            reasons.append("step budget cannot plausibly cover the order count")
        if not env.usable:
            reasons.append("environment is unusable")
        if env.solvability is None or env.solvability.solvability_rate <= 0.0:
            reasons.append("environment has no demonstrated solvable episode")
        # design plan §8.3: success must not depend on an unvalidated synthetic edge.
        if env.graph.largest_component_fraction < 0.5:
            reasons.append("graph is too fragmented for reliable routing")
        return SolvabilityVerdict(feasible=not reasons, reasons=reasons)

    # ── the agent-facing contract ────────────────────────────────────────────

    def instruction(self, env: EnvSpec) -> str:
        """The observable instruction, derived from the actual action space."""
        movement = (
            "Move with MOVE_TO(k) to step to a numbered neighbouring waypoint."
            if "MOVE" not in env.enabled_actions
            else "Move with MOVE(direction=...) or MOVE_TO(k)."
        )
        constraints = []
        if self.config.constraint.deadlines:
            constraints.append("orders have deadlines")
        if self.config.constraint.battery:
            constraints.append("your scooter has a battery that depletes")
        if self.config.constraint.carrying_capacity:
            constraints.append(
                f"you can carry at most {self.config.constraint.carrying_capacity} order(s)"
            )
        tail = ("Constraints: " + "; ".join(constraints) + ".") if constraints else ""
        return (
            f"You are a delivery courier in {env.map_name}. Accept orders, collect them "
            f"from restaurants, and deliver them to their destinations. {movement} {tail}"
        ).strip()

    def action_schema(self, env: EnvSpec) -> list[str]:
        """The actions available here, taken from the environment, not assumed."""
        return list(env.enabled_actions)

    def reset_observation(self, env: EnvSpec) -> dict[str, Any]:
        """Everything the agent is entitled to at reset (design plan §8.2, 10.1)."""
        courier = COURIER_PROFILES[self.config.courier_profile]
        return {
            "instruction": self.instruction(env),
            "actions": self.action_schema(env),
            "navigation_style": env.navigation_style.value,
            **self.config.observable_summary(courier),
        }

    # ── evaluation ───────────────────────────────────────────────────────────

    def reward_components(self, privileged: dict[str, Any], trajectory: Trajectory) -> dict[str, float]:
        """Rule-based, recomputable reward components (design plan §11.3)."""
        weights = self.config.reward
        delivered = float(privileged.get("delivered_count") or 0)
        on_time = float(privileged.get("on_time_count") or delivered)
        late = max(0.0, delivered - on_time)
        invalid = float(
            sum(1 for turn in trajectory.turns if turn.action_result.status.value != "accepted")
        )
        steps = float(len(trajectory.turns))
        return {
            "delivery": weights.delivery * delivered,
            "on_time_bonus": weights.on_time_bonus * on_time,
            "late_penalty": -weights.late_penalty * late,
            "invalid_action_penalty": -weights.invalid_action_penalty * invalid,
            "step_cost": -weights.step_cost * steps,
        }

    def evaluate(self, trajectory: Trajectory, privileged: dict[str, Any]) -> ScoreReport:
        components = self.reward_components(privileged, trajectory)
        task_score = sum(components.values())
        delivered = float(privileged.get("delivered_count") or 0)
        invalid = float(
            sum(1 for turn in trajectory.turns if turn.action_result.status.value != "accepted")
        )

        metrics = {
            "task_score": MetricValue(value=task_score),
            "deliveries": MetricValue(value=delivered),
            "orders_offered": MetricValue(value=float(self.config.order.order_count)),
            "delivery_rate": MetricValue(
                value=delivered / max(1.0, float(self.config.order.order_count))
            ),
            "net_earnings": MetricValue(value=float(privileged.get("earnings_total") or 0.0)),
            "steps_taken": MetricValue(value=float(len(trajectory.turns))),
            "invalid_actions": MetricValue(value=invalid),
            "total_reward": MetricValue(value=float(trajectory.total_reward)),
        }
        for name, value in components.items():
            metrics[f"reward.{name}"] = MetricValue(value=value)

        return ScoreReport(
            instance_id=trajectory.instance_id,
            episode_id=trajectory.episode_id,
            evaluator_id="delivery_score",
            evaluator_version=self.version,
            success=delivered >= 1,
            normalized_utility_vs_upper_bound=None,
            upper_bound_undefined_reason=(
                "no documented upper-bound policy exists before M7; design plan §12.5 defines the "
                "primary metric relative to one, so it is reported as undefined rather than "
                "estimated"
            ),
            metrics=metrics,
            # design plan §11.3: shaping is reported separately and never ranked on.
            training_shaping={"progress": self.config.reward.shaping_progress},
            costs={
                "environment_steps": float(len(trajectory.turns)),
                "output_tokens": float(sum(t.tokens.response_tokens for t in trajectory.turns)),
            },
            trajectory_sha256=trajectory.content_hash(),
        )
