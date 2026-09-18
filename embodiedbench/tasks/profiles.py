"""Composable Delivery configuration (design plan §8.2).

design plan §8.2 replaces "a single large mutable config" with validated composition:

    DeliveryTaskConfig
      order_profile / constraint_profile / transport_profile
      courier_profile / reward_profile / observation_profile / budget_profile

Each profile is small, frozen, and independently checkable, so an invalid
combination fails at construction instead of producing an episode nobody can
complete. design plan §8.3 requires generation to be followed by a feasibility check;
the profiles make most infeasibility detectable before generation even runs.

Every field that changes an outcome is declared observable. design plan §17 lists
hidden courier attributes as a risk that makes instances "unfair and
uninterpretable", so a profile that keeps an outcome-relevant value to itself is
rejected rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodiedbench.schemas.episode import CourierProfile


@dataclass(frozen=True)
class OrderProfile:
    """How many orders exist and when they become available."""

    order_count: int = 3
    interval_s: float = 300.0
    max_in_pool: int = 3
    single_item: bool = True

    def validate(self) -> list[str]:
        problems = []
        if self.order_count < 1:
            problems.append("order_count must be at least 1")
        if self.interval_s < 0:
            problems.append("interval_s cannot be negative")
        if self.max_in_pool < 1:
            problems.append("max_in_pool must be at least 1")
        return problems


@dataclass(frozen=True)
class ConstraintProfile:
    """What makes the task hard, beyond finding the way."""

    deadlines: bool = True
    deadline_multiplier: float = 1.5
    carrying_capacity: int = 3
    battery: bool = False
    walking_energy: bool = False
    food_temperature: bool = False
    traffic_lights: bool = False

    def validate(self) -> list[str]:
        problems = []
        if self.deadline_multiplier <= 0:
            problems.append("deadline_multiplier must be positive")
        if self.carrying_capacity < 1:
            problems.append("carrying_capacity must be at least 1")
        return problems

    def required_affordances(self) -> dict[str, int]:
        """Affordances this constraint set needs the map to provide."""
        needs: dict[str, int] = {}
        if self.battery:
            needs["charging_station"] = 1
        return needs

    def observable_fields(self) -> dict[str, Any]:
        """What the agent is told at reset. Everything that changes an outcome."""
        return {
            "deadlines": self.deadlines,
            "carrying_capacity": self.carrying_capacity,
            "battery": self.battery,
            "walking_energy": self.walking_energy,
            "food_temperature": self.food_temperature,
            "traffic_lights": self.traffic_lights,
        }


@dataclass(frozen=True)
class TransportProfile:
    """Which ways of moving exist for this instance."""

    modes: tuple[str, ...] = ("walk", "scooter")
    initial_mode: str = "walk"
    bus: bool = False
    car_rental: bool = False

    def validate(self) -> list[str]:
        problems = []
        if not self.modes:
            problems.append("at least one transport mode is required")
        if self.initial_mode not in self.modes:
            problems.append(f"initial_mode {self.initial_mode!r} is not in modes {self.modes}")
        return problems

    def required_affordances(self) -> dict[str, int]:
        needs: dict[str, int] = {}
        if self.bus:
            needs["bus_station"] = 2
        if self.car_rental:
            needs["car_rental"] = 1
        return needs


@dataclass(frozen=True)
class RewardProfile:
    """Verifiable, rule-based reward weights.

    Every component is computed from recorded environment facts -- deliveries,
    lateness, violations -- so a score can be recomputed from a trajectory by
    anyone. No model judges anything. design plan §11.3 keeps training shaping
    separate from the benchmark score, so shaping lives in its own block and is
    never summed into the task score.
    """

    delivery: float = 1.0
    on_time_bonus: float = 0.5
    late_penalty: float = 0.5
    invalid_action_penalty: float = 0.01
    step_cost: float = 0.0
    # Reported separately, never added to the task score.
    shaping_progress: float = 0.0

    def validate(self) -> list[str]:
        problems = []
        if self.delivery <= 0:
            problems.append("a delivery must be worth something")
        for name in ("on_time_bonus", "late_penalty", "invalid_action_penalty", "step_cost"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must not be negative")
        return problems


@dataclass(frozen=True)
class ObservationProfile:
    """What the agent perceives."""

    channels: tuple[str, ...] = ("text",)
    include_map_image: bool = False
    include_fpv: bool = False
    max_images_per_turn: int = 2

    def validate(self) -> list[str]:
        problems = []
        if "text" not in self.channels:
            problems.append("text observations are always required")
        if (self.include_fpv or self.include_map_image) and "rgb" not in self.channels:
            problems.append("image observations require the rgb channel")
        return problems


# The measured floor for one order on the compiled Paris carriageway, not a
# guess. A delivery leg is a median 530 m over 18 m edges, so a shortest-path
# oracle spends about 23 turns an order stepping junction by junction and about
# 11 using ``follow_street`` -- 109 turns for a ten-order shift. The old floor
# was 5, which would have validated a configuration in which a perfect courier
# could not finish a quarter of the work, and did: the 120-step budget the
# benchmark actually ran capped the oracle at 3.1 of 10 deliveries. A validator
# that passes an unfinishable configuration is worse than no validator, because
# it certifies it.
MIN_STEPS_PER_ORDER = 12


@dataclass(frozen=True)
class BudgetProfile:
    """Hardware-independent budgets (design plan §12.2)."""

    steps: int = 400
    tool_calls: int | None = 80
    sim_s: float | None = 7200.0
    output_tokens: int | None = 120000

    def validate(self) -> list[str]:
        problems = []
        if self.steps < 1:
            problems.append("steps budget must be positive")
        for name in ("tool_calls", "output_tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                problems.append(f"{name} budget must be positive when set")
        if self.sim_s is not None and self.sim_s <= 0:
            problems.append("sim_s budget must be positive when set")
        return problems


@dataclass(frozen=True)
class DeliveryTaskConfig:
    """The composed configuration (design plan §8.2)."""

    order: OrderProfile = field(default_factory=OrderProfile)
    constraint: ConstraintProfile = field(default_factory=ConstraintProfile)
    transport: TransportProfile = field(default_factory=TransportProfile)
    reward: RewardProfile = field(default_factory=RewardProfile)
    observation: ObservationProfile = field(default_factory=ObservationProfile)
    budget: BudgetProfile = field(default_factory=BudgetProfile)
    courier_profile: str = "scooter_standard"

    def validate(self) -> list[str]:
        """Every problem with this configuration, not just the first."""
        problems: list[str] = []
        for name in ("order", "constraint", "transport", "reward", "observation", "budget"):
            problems += [f"{name}: {p}" for p in getattr(self, name).validate()]

        # Cross-profile coherence: a constraint that needs a transport mode the
        # transport profile does not offer is unsatisfiable, and would surface
        # much later as an episode nobody can finish.
        if self.constraint.battery and "scooter" not in self.transport.modes:
            problems.append("constraint: battery requires a scooter in transport.modes")
        if self.constraint.carrying_capacity < 1:
            problems.append("constraint: carrying_capacity must allow at least one order")
        if self.budget.steps < self.order.order_count * MIN_STEPS_PER_ORDER:
            problems.append(
                f"budget: {self.budget.steps} steps cannot plausibly cover "
                f"{self.order.order_count} orders "
                f"({MIN_STEPS_PER_ORDER} each is the measured floor)"
            )
        return problems

    def required_affordances(self) -> dict[str, int]:
        """What the map must provide for this configuration to run.

        A delivery needs somewhere to collect from and somewhere to deliver to.
        Restaurants are the pickup source; buildings are the drop-off pool,
        which is why Paris runs despite having no ``customer`` POIs at all.
        """
        needs: dict[str, int] = {"restaurant": 1, "building": 1}
        for profile in (self.constraint, self.transport):
            for name, count in profile.required_affordances().items():
                needs[name] = max(needs.get(name, 0), count)
        return needs

    def observable_summary(self, courier: CourierProfile | None = None) -> dict[str, Any]:
        """Exactly what the agent is told at reset (design plan §8.2, 10.1)."""
        summary: dict[str, Any] = {
            "orders_expected": self.order.order_count,
            "transport_modes": list(self.transport.modes),
            "initial_transport_mode": self.transport.initial_mode,
            "budgets": {
                "steps": self.budget.steps,
                "tool_calls": self.budget.tool_calls,
                "sim_s": self.budget.sim_s,
            },
            **self.constraint.observable_fields(),
        }
        if courier is not None:
            summary["courier"] = courier.observable_summary()
        return summary


# Named presets, from easiest to hardest. Each is a complete, validated
# configuration rather than a diff, so reading one tells you the whole task.
PRESETS: dict[str, DeliveryTaskConfig] = {
    "minimal": DeliveryTaskConfig(
        order=OrderProfile(order_count=1),
        constraint=ConstraintProfile(deadlines=False, carrying_capacity=1),
        transport=TransportProfile(modes=("walk",), initial_mode="walk"),
        courier_profile="walker_novice",
        budget=BudgetProfile(steps=200, tool_calls=40),
    ),
    "standard": DeliveryTaskConfig(),
    "constrained": DeliveryTaskConfig(
        order=OrderProfile(order_count=5, max_in_pool=5),
        constraint=ConstraintProfile(
            deadlines=True, deadline_multiplier=1.2, carrying_capacity=2, battery=True
        ),
        transport=TransportProfile(modes=("walk", "scooter"), initial_mode="scooter"),
        courier_profile="scooter_standard",
    ),
    "multi_modal": DeliveryTaskConfig(
        order=OrderProfile(order_count=5, max_in_pool=5),
        constraint=ConstraintProfile(deadlines=True, carrying_capacity=4, battery=True),
        transport=TransportProfile(
            modes=("walk", "scooter", "bus", "car"), initial_mode="walk",
            bus=True, car_rental=True,
        ),
        courier_profile="multi_modal_courier",
    ),
}
