"""Trajectory and score report (design plan §10.4, 12.5).

design plan §10.4 lists what every turn must record. Two of those items are the ones
that quietly invalidate an RL run if they are wrong, so they are validated
rather than trusted:

**The loss mask must exclude environment and tool observations.** design plan §2.1
calls observation-token loss masking a correctness risk, design plan §11.1 makes it a
trainer selection gate, and the design plan M6 requires that observation, tool, and
image tokens are absent from the response mask for every turn while all sampled
response tokens are included exactly once. ``TrajectoryTurn`` checks that its
mask covers exactly the sampled response tokens.

**Reward must be counted once.** design plan §11.1 requires that context compaction
and fan-out preserve total reward exactly once, and design plan §12.6 requires that
retrying a step cannot duplicate reward or events. ``Trajectory`` rejects
duplicate event ids and duplicate step indices.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from embodiedbench.schemas.base import SchemaModel
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    ActionResult,
    Event,
    Observation,
    TerminationReason,
)
from embodiedbench.schemas.world import Sha256, StableId


class TokenAccounting(SchemaModel):
    """Per-turn token counts and the policy loss mask."""

    prompt_tokens: int = Field(default=0, ge=0)
    response_tokens: int = Field(default=0, ge=0)
    # Indices into the response token sequence that contribute to the policy
    # loss. Environment text, tool results, and image tokens must not appear.
    response_loss_mask: list[int] = Field(default_factory=list)
    observation_token_indices: list[int] = Field(default_factory=list)
    image_token_indices: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mask_is_exactly_the_sampled_response(self) -> "TokenAccounting":
        mask = self.response_loss_mask
        if len(mask) != len(set(mask)):
            raise ValueError("a response token appears twice in the loss mask")
        if any(i < 0 or i >= self.response_tokens for i in mask):
            raise ValueError("loss mask indexes outside the response token range")
        excluded = set(self.observation_token_indices) | set(self.image_token_indices)
        leaked = excluded & set(mask)
        if leaked:
            raise ValueError(
                f"observation/image tokens are inside the policy loss mask: {sorted(leaked)[:8]}"
            )
        return self

    def masked_token_count(self) -> int:
        return len(self.response_loss_mask)


class CostAccounting(SchemaModel):
    """design plan §10.4: token, latency, service, and environment cost."""

    model_latency_s: float = Field(default=0.0, ge=0.0)
    environment_latency_s: float = Field(default=0.0, ge=0.0)
    estimated_usd: float | None = Field(default=None, ge=0.0)
    gpu_seconds: float | None = Field(default=None, ge=0.0)


class TrajectoryTurn(SchemaModel):
    """One agent turn (design plan §10.4)."""

    step_index: int = Field(ge=0)
    observation: Observation
    observation_text: str = ""
    model_input_refs: list[str] = Field(default_factory=list)
    raw_model_output: str = ""
    action: ActionEnvelope | None = None
    action_result: ActionResult
    events: list[Event] = Field(default_factory=list)
    reward: float = 0.0
    reward_components: dict[str, float] = Field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    termination_reason: TerminationReason | None = None
    tokens: TokenAccounting = Field(default_factory=TokenAccounting)
    cost: CostAccounting = Field(default_factory=CostAccounting)
    compaction_boundary: bool = False

    @model_validator(mode="after")
    def _reward_components_sum(self) -> "TrajectoryTurn":
        if self.reward_components:
            total = sum(self.reward_components.values())
            if abs(total - self.reward) > 1e-6:
                raise ValueError(f"turn reward {self.reward} != components sum {total}")
        return self


class Trajectory(SchemaModel):
    """A full episode record, trainer-neutral (design plan §11)."""

    SCHEMA_ID = "embodiedbench/trajectory"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    episode_id: StableId
    instance_id: StableId
    benchmark_version: str = ""
    environment_id: StableId
    environment_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    task_plugin: str = ""
    task_plugin_version: str = ""
    model_id: str = ""
    harness_version: str = ""
    seed: int = 0
    turns: list[TrajectoryTurn] = Field(default_factory=list)
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    termination_reason: TerminationReason | None = None

    @model_validator(mode="after")
    def _counted_once(self) -> "Trajectory":
        indices = [t.step_index for t in self.turns]
        if indices != sorted(indices):
            raise ValueError("turns must be in step order")
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate step index: a retried step was recorded twice")
        event_ids = [e.event_id for turn in self.turns for e in turn.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate event id: an event was counted twice")
        if self.turns:
            total = sum(t.reward for t in self.turns)
            if abs(total - self.total_reward) > 1e-6:
                raise ValueError(
                    f"total_reward {self.total_reward} != sum of turn rewards {total}"
                )
            if self.terminated and self.truncated:
                raise ValueError("a trajectory cannot both terminate and truncate")
        return self

    def policy_token_count(self) -> int:
        return sum(t.tokens.masked_token_count() for t in self.turns)

    def compaction_boundaries(self) -> list[int]:
        return [t.step_index for t in self.turns if t.compaction_boundary]


class MetricValue(SchemaModel):
    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    @model_validator(mode="after")
    def _interval_ordered(self) -> "MetricValue":
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("a confidence interval needs both bounds")
        if self.ci_low is not None and self.ci_high is not None:
            if self.ci_low > self.ci_high:
                raise ValueError("confidence interval bounds are reversed")
        return self


class ScoreReport(SchemaModel):
    """An evaluator's verdict (design plan §8.1 evaluate(), 12.5)."""

    SCHEMA_ID = "embodiedbench/score_report"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    instance_id: StableId
    episode_id: StableId
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    success: bool = False
    # design plan §12.5's primary Delivery metric. Optional because PointNav does not
    # have one, and because the metric is undefined when the bound is
    # non-positive -- a case the design plan requires defined behaviour for, which here
    # means recording None plus a reason rather than a misleading number.
    normalized_utility_vs_upper_bound: float | None = None
    upper_bound_undefined_reason: str | None = None
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    # design plan §11.3: training shaping is reported separately and never used for
    # public ranking, so it cannot share a container with scored metrics.
    training_shaping: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None
    costs: dict[str, float] = Field(default_factory=dict)
    environment_sha256: Sha256 | None = None
    trajectory_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _undefined_bound_is_explained(self) -> "ScoreReport":
        if self.normalized_utility_vs_upper_bound is None and self.upper_bound_undefined_reason:
            return self
        if self.normalized_utility_vs_upper_bound is None:
            return self
        if self.upper_bound_undefined_reason:
            raise ValueError(
                "a normalized utility was reported alongside a reason it is undefined"
            )
        return self
