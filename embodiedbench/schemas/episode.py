"""``EpisodeSpec`` and benchmark instances (design plan §8.2, 12.2).

design plan §12.2 is explicit that "a seed alone is not a benchmark instance": an
instance pins environment, task, embodiment, runtime track, spawn, order
schedule, budgets, and evaluator, each with a hash. It also requires the order
schedule to fix ``available_from_sim_time`` per order, so two divergent policies
see the same stream.

design plan §8.2 requires every profile field that changes outcomes to be visible to
the agent at reset and recorded in the ``EpisodeSpec``, trajectory metadata, and
evaluator output — hence ``outcome_relevant_fields`` on the courier profile, and
a validator that refuses a profile which hides one.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from embodiedbench.schemas.base import SchemaModel
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.geometry import Pose
from embodiedbench.schemas.runtime import RuntimeMode
from embodiedbench.schemas.world import Sha256, StableId


class Budgets(SchemaModel):
    """Hardware-independent budgets (design plan §12.2).

    Wall-clock is deliberately absent: design plan §12.2 makes it a reported cost
    plus a non-scoring safety timeout, not a scoring budget, so it does not
    belong in the structure an evaluator scores against.
    """

    steps: int = Field(gt=0)
    tool_calls: int | None = Field(default=None, ge=0)
    sim_s: float | None = Field(default=None, gt=0.0)
    output_tokens: int | None = Field(default=None, gt=0)

    def as_remaining(self, used: dict[str, float] | None = None) -> dict[str, float]:
        used = used or {}
        out: dict[str, float] = {"steps": self.steps - used.get("steps", 0)}
        for name in ("tool_calls", "sim_s", "output_tokens"):
            limit = getattr(self, name)
            if limit is not None:
                out[name] = limit - used.get(name, 0)
        return out


class CourierProfile(SchemaModel):
    """Task-side courier configuration (design plan §8.2), distinct from embodiment."""

    SCHEMA_ID = "embodiedbench/courier_profile"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    profile_id: str = Field(min_length=1)
    version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    transport_modes: list[str] = Field(min_length=1)
    carrying_capacity: int = Field(ge=0)
    owns_scooter: bool = False
    battery_enabled: bool = False
    bus_access: bool = False
    car_rental_access: bool = False
    # Every field here changes outcomes, so every one must reach the agent at
    # reset (design plan §8.2, and design plan §17's "hidden courier attributes" risk).
    outcome_relevant_fields: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_hidden_bonuses(self) -> "CourierProfile":
        declared = set(self.outcome_relevant_fields)
        known = {
            "transport_modes",
            "carrying_capacity",
            "owns_scooter",
            "battery_enabled",
            "bus_access",
            "car_rental_access",
        }
        unknown = declared - known - set(self.extra)
        if unknown:
            raise ValueError(f"outcome_relevant_fields names unknown fields: {sorted(unknown)}")
        # Anything in `extra` is by definition not a schema field the agent can
        # discover, so it must be declared outcome-relevant or it is a hidden
        # bonus of exactly the kind design plan §17 forbids.
        undeclared_extra = set(self.extra) - declared
        if undeclared_extra:
            raise ValueError(
                "extra courier fields must be declared outcome-relevant so the agent sees "
                f"them at reset: {sorted(undeclared_extra)}"
            )
        return self

    def observable_summary(self) -> dict[str, Any]:
        """Exactly what the agent is shown at reset."""
        out: dict[str, Any] = {}
        for name in self.outcome_relevant_fields:
            out[name] = self.extra[name] if name in self.extra else getattr(self, name, None)
        return out


class ScheduledOrder(SchemaModel):
    """One pinned order in the frozen stream (design plan §12.2)."""

    order_id: StableId
    available_from_sim_time_s: float = Field(ge=0.0)
    pickup_site: StableId
    dropoff_site: StableId
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskConfigRef(SchemaModel):
    plugin: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    config_sha256: Sha256 | None = None


class EpisodeSpec(SchemaModel):
    """A frozen, reproducible episode (design plan §8.1 generate(), 12.2)."""

    SCHEMA_ID = "embodiedbench/episode_spec"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    instance_id: StableId
    environment_id: StableId
    environment_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    environment_sha256: Sha256 | None = None
    task: TaskConfigRef
    embodiment_profile: str = Field(min_length=1)
    courier_profile: CourierProfile | None = None
    navigation_mode: NavigationMode
    runtime_track: RuntimeMode
    seed: int
    spawn: Pose
    order_schedule: list[ScheduledOrder] = Field(default_factory=list)
    observable_instruction: str = ""
    budgets: Budgets
    safety_timeout_s: float = Field(default=3600.0, gt=0.0)
    evaluator_id: str = ""
    evaluator_version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    evaluator_sha256: Sha256 | None = None
    private_state_ref: str | None = None

    @model_validator(mode="after")
    def _schedule_is_deterministic(self) -> "EpisodeSpec":
        ids = [o.order_id for o in self.order_schedule]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate order ids in the frozen schedule")
        times = [o.available_from_sim_time_s for o in self.order_schedule]
        if times != sorted(times):
            # A schedule that is not time-ordered invites an implementation to
            # reorder it, which is how two divergent policies come to see
            # different streams (design plan §12.6's release check).
            raise ValueError("order schedule must be sorted by available_from_sim_time_s")
        return self

    def agent_visible(self) -> dict[str, Any]:
        """The reset-time facts the agent is entitled to (design plan §10.1).

        Deliberately excludes the order schedule and every private reference:
        which orders exist and when is privileged until the environment
        surfaces them.
        """
        out: dict[str, Any] = {
            "instance_id": self.instance_id,
            "instruction": self.observable_instruction,
            "embodiment_profile": self.embodiment_profile,
            "navigation_mode": self.navigation_mode.value,
            "runtime_track": self.runtime_track.value,
            "budgets": self.budgets.model_dump(mode="json", exclude_none=True),
            "spawn": self.spawn.model_dump(mode="json"),
        }
        if self.courier_profile is not None:
            out["courier_profile"] = self.courier_profile.observable_summary()
        return out
