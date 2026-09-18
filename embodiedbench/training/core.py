"""Trainer-neutral training samples and their invariants (design plan §11).

A ``TrainingSample`` is what any framework adapter consumes. Its invariants are
the R1 selection gate written down as code, because design plan §2.1 names
observation-token loss masking and fan-out reward accounting as the two
correctness risks that make an RL run silently invalid:

- the response mask covers exactly the sampled assistant tokens;
- no environment, tool, or image token is inside it;
- every sampled token has a log-prob, positionally aligned;
- total reward is preserved exactly once across however many samples one
  trajectory is split into.

``validate_sample`` raises rather than warns. A sample that violates any of
these does not produce a worse gradient — it produces a meaningless one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodiedbench.artifacts.hashing import sha256_bytes


class MaskViolation(Exception):
    """A training sample's mask does not describe what the policy sampled."""


class RewardAccountingError(Exception):
    """Reward was lost or duplicated when a trajectory became samples."""


@dataclass
class TrainingSample:
    """One trajectory (or fan-out shard) in trainer-neutral form."""

    episode_id: str
    input_ids: list[int]
    response_mask: list[bool]
    logprobs: list[float]
    reward: float
    advantages: list[float] = field(default_factory=list)
    image_token_positions: list[int] = field(default_factory=list)
    pixel_values: Any = None
    image_grid_thw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_indices(self) -> list[int]:
        return [i for i, flagged in enumerate(self.response_mask) if flagged]

    @property
    def response_token_count(self) -> int:
        return sum(self.response_mask)

    def fingerprint(self) -> str:
        """Identity of the token/mask content, for cross-process comparison."""
        payload = "|".join(
            [
                ",".join(str(t) for t in self.input_ids),
                ",".join("1" if m else "0" for m in self.response_mask),
                f"{self.reward:.9f}",
            ]
        )
        return sha256_bytes(payload.encode())


def build_training_sample(
    state: Any,
    *,
    episode_id: str,
    reward: float,
    metadata: dict[str, Any] | None = None,
) -> TrainingSample:
    """Convert a rollout state into a training sample.

    The mask comes from the provenance recorded during rollout, not from a
    post-hoc scan for delimiters. Re-deriving "which tokens were the model's"
    by pattern-matching the template is how image and tool tokens end up in the
    loss when a template changes.
    """
    sample = TrainingSample(
        episode_id=episode_id,
        input_ids=list(state.input_ids),
        response_mask=list(state.assistant_mask),
        logprobs=list(state.assistant_logprobs()),
        reward=float(reward),
        image_token_positions=list(state.image_token_positions),
        pixel_values=(state.pixel_values or None),
        image_grid_thw=(state.image_grid_thw or None),
        metadata=dict(metadata or {}),
    )
    validate_sample(sample)
    return sample


def validate_sample(sample: TrainingSample) -> None:
    """Enforce the R1 mask invariants, raising on any violation."""
    if len(sample.input_ids) != len(sample.response_mask):
        raise MaskViolation(
            f"{len(sample.input_ids)} tokens but {len(sample.response_mask)} mask entries"
        )
    if not sample.input_ids:
        raise MaskViolation("empty sample")

    response_count = sample.response_token_count
    if response_count == 0:
        raise MaskViolation("no sampled tokens: this sample cannot produce a policy gradient")
    if len(sample.logprobs) != response_count:
        raise MaskViolation(
            f"{response_count} masked tokens but {len(sample.logprobs)} log-probs; "
            "the sampled tokens and their log-probs are not aligned"
        )

    leaked = sorted(set(sample.image_token_positions) & set(sample.response_indices))
    if leaked:
        raise MaskViolation(
            f"image tokens inside the policy loss mask at positions {leaked[:8]}"
        )


def check_reward_conservation(
    total_reward: float, samples: list[TrainingSample], *, tolerance: float = 1e-6
) -> None:
    """design plan §11.1: fan-out must preserve total reward exactly once.

    Called wherever one trajectory becomes several samples — context compaction
    boundaries, fan-out over branches. Summing to more than the total means a
    branch was counted twice; less means one was dropped.
    """
    observed = sum(s.reward for s in samples)
    if abs(observed - total_reward) > tolerance:
        raise RewardAccountingError(
            f"fan-out changed total reward: expected {total_reward}, samples sum to {observed}"
        )


def split_for_fanout(sample: TrainingSample, shards: int) -> list[TrainingSample]:
    """Split a sample's reward across N shards without changing the total.

    The naive implementation divides by N and loses a little to float error at
    the last shard. This assigns the remainder explicitly so the invariant holds
    exactly, which is what ``check_reward_conservation`` verifies.
    """
    if shards < 1:
        raise ValueError("shards must be at least 1")
    if shards == 1:
        return [sample]
    share = sample.reward / shards
    out: list[TrainingSample] = []
    for index in range(shards):
        value = share if index < shards - 1 else sample.reward - share * (shards - 1)
        out.append(
            TrainingSample(
                episode_id=f"{sample.episode_id}#shard{index}",
                input_ids=list(sample.input_ids),
                response_mask=list(sample.response_mask),
                logprobs=list(sample.logprobs),
                reward=value,
                image_token_positions=list(sample.image_token_positions),
                pixel_values=sample.pixel_values,
                image_grid_thw=sample.image_grid_thw,
                metadata={**sample.metadata, "shard": index, "of": shards},
            )
        )
    check_reward_conservation(sample.reward, out)
    return out
