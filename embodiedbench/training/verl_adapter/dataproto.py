"""Convert a trainer-neutral rollout into verl's ``DataProto``.

verl's actor expects a prompt/response split: ``input_ids`` is the whole
sequence, ``responses`` is its tail, and ``response_mask`` marks which tokens in
that tail contribute to the loss. A multi-turn agent rollout maps onto this
cleanly, and the mapping is the whole point of the adapter:

    [ system + first observation ][ assistant | observation | assistant | ... ]
    <---------- prompt ---------><------------- response ------------------->

The response region starts at the first token the policy sampled and runs to the
end. Environment turns *inside* that region — later observations, their image
tokens, tool results — stay in ``responses`` because they are part of the
sequence, but their ``response_mask`` entries are zero. That zeroing is exactly
what design plan §11.1 requires the trainer to honour, and what R1 then verifies verl
actually does.

Nothing here computes a loss or a gradient. It only states, in verl's own
vocabulary, which tokens the policy produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RolloutLayout:
    """Where the prompt ends and what the response region contains."""

    prompt_length: int
    response_length: int
    masked_token_count: int
    environment_tokens_in_response: int
    image_tokens_in_response: int

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "masked_token_count": self.masked_token_count,
            "environment_tokens_in_response": self.environment_tokens_in_response,
            "image_tokens_in_response": self.image_tokens_in_response,
        }


def describe_layout(sample: Any) -> RolloutLayout:
    """Compute the prompt/response split for a training sample."""
    mask = list(sample.response_mask)
    if not any(mask):
        raise ValueError("sample has no sampled tokens; nothing to train on")
    prompt_length = mask.index(True)
    response_length = len(mask) - prompt_length
    response_region = mask[prompt_length:]
    image_positions = set(getattr(sample, "image_token_positions", []) or [])
    return RolloutLayout(
        prompt_length=prompt_length,
        response_length=response_length,
        masked_token_count=sum(response_region),
        environment_tokens_in_response=response_length - sum(response_region),
        image_tokens_in_response=sum(1 for p in image_positions if p >= prompt_length),
    )


def _mrope_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Build Qwen3-VL mRoPE position ids using verl's own helper.

    verl's actor branches on ``position_ids.dim() == 3`` to detect mRoPE
    (dp_actor.py:284). Computing these with verl's helper rather than our own
    means the adapter cannot disagree with the trainer about token positions,
    which for an interleaved image/text sequence is a subtle way to corrupt a
    run without any error.
    """
    from verl.models.transformers.qwen2_vl import get_rope_index

    return get_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )


def build_dataproto(
    samples: list[Any],
    *,
    processor: Any,
    advantages: list[float] | None = None,
    temperature: float = 1.0,
    pad_token_id: int = 0,
    device: str = "cpu",
) -> tuple[Any, list[RolloutLayout]]:
    """Assemble a verl ``DataProto`` from training samples.

    Sequences are right-padded to a common length; padded positions get
    ``attention_mask`` 0 and ``response_mask`` 0, so they can influence neither
    the forward pass nor the loss.
    """
    from verl.protocol import DataProto

    if not samples:
        raise ValueError("no samples")
    layouts = [describe_layout(s) for s in samples]

    # verl requires one prompt length across the batch, so the split is taken at
    # the smallest prompt; anything before it is prompt for every sample.
    prompt_length = min(layout.prompt_length for layout in layouts)
    max_total = max(len(s.input_ids) for s in samples)
    response_length = max_total - prompt_length

    batch_input_ids: list[torch.Tensor] = []
    batch_attention: list[torch.Tensor] = []
    batch_responses: list[torch.Tensor] = []
    batch_response_mask: list[torch.Tensor] = []
    batch_old_logprobs: list[torch.Tensor] = []
    batch_advantages: list[torch.Tensor] = []
    batch_position_ids: list[torch.Tensor] = []
    multi_modal_inputs: list[dict[str, Any]] = []

    for order, sample in enumerate(samples):
        ids = list(sample.input_ids)
        mask = list(sample.response_mask)
        pad = max_total - len(ids)

        padded_ids = ids + [pad_token_id] * pad
        attention = [1] * len(ids) + [0] * pad
        padded_mask = mask + [False] * pad

        input_ids = torch.tensor(padded_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention, dtype=torch.long)

        response_mask = torch.tensor(
            [1 if flag else 0 for flag in padded_mask[prompt_length:]], dtype=torch.long
        )

        # Place each sampled token's rollout log-prob at its own position;
        # unmasked positions carry zero and are never read, because verl
        # multiplies by response_mask.
        old_logprobs = torch.zeros(response_length, dtype=torch.float32)
        rollout_logprobs = list(sample.logprobs)
        cursor = 0
        for offset in range(response_length):
            if response_mask[offset] and cursor < len(rollout_logprobs):
                old_logprobs[offset] = rollout_logprobs[cursor]
                cursor += 1
        if cursor != len(rollout_logprobs):
            raise ValueError(
                f"{len(rollout_logprobs)} rollout log-probs but only {cursor} masked "
                "positions to place them at; the mask and the rollout disagree"
            )

        advantage_value = (
            advantages[order] if advantages is not None else float(sample.reward)
        )
        advantage = torch.full((response_length,), float(advantage_value), dtype=torch.float32)
        advantage = advantage * response_mask

        pixel_values = getattr(sample, "pixel_values", None)
        grid = getattr(sample, "image_grid_thw", None)
        if pixel_values:
            merged_pixels = torch.cat([p for p in pixel_values], dim=0)
            merged_grid = torch.cat([g for g in grid], dim=0)
            multi_modal_inputs.append(
                {"pixel_values": merged_pixels, "image_grid_thw": merged_grid}
            )
        else:
            merged_grid = None
            multi_modal_inputs.append({})

        position_ids = _mrope_position_ids(
            processor,
            input_ids=input_ids,
            image_grid_thw=merged_grid,
            attention_mask=attention_mask,
        )

        batch_input_ids.append(input_ids)
        batch_attention.append(attention_mask)
        batch_responses.append(input_ids[prompt_length:])
        batch_response_mask.append(response_mask)
        batch_old_logprobs.append(old_logprobs)
        batch_advantages.append(advantage)
        batch_position_ids.append(position_ids)

    import numpy as np

    tensors = {
        "input_ids": torch.stack(batch_input_ids).to(device),
        "attention_mask": torch.stack(batch_attention).to(device),
        "position_ids": torch.stack(batch_position_ids).to(device),
        "responses": torch.stack(batch_responses).to(device),
        "response_mask": torch.stack(batch_response_mask).to(device),
        "old_log_probs": torch.stack(batch_old_logprobs).to(device),
        "advantages": torch.stack(batch_advantages).to(device),
    }
    non_tensors = {"multi_modal_inputs": np.array(multi_modal_inputs, dtype=object)}
    data = DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={"temperature": temperature, "pad_token_id": pad_token_id},
    )
    return data, layouts
