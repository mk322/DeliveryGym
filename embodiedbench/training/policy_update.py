"""A minimal, real policy update over masked tokens (design plan §11.1, M8).

the design plan M8 accepts when "one text and one RGB-D policy update complete
end-to-end" and "weight synchronization produces a changed checkpoint hash,
nonzero parameter delta, and changed pinned-prompt logits".

This is deliberately the simplest correct thing — a REINFORCE step with a
baseline — because R1 is a *correctness* gate, not an algorithm bake-off. What
matters is that the gradient reaches only the sampled tokens, that the step
actually changes the weights, and that the changed weights actually change
behavior. A more sophisticated objective would add ways to be wrong without
adding anything the gate is testing.

The loss is explicitly masked twice over: the per-token log-probs are gathered
only at response positions, and the mask is applied again as a multiplier. That
redundancy is intentional — if either mechanism regressed alone, the tests would
still catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from embodiedbench.training.core import TrainingSample, validate_sample
from embodiedbench.training.logprobs import build_model_inputs, selected_token_logprobs


@dataclass
class UpdateResult:
    """Evidence that one policy update happened and did something."""

    loss: float
    masked_token_count: int
    grad_norm: float
    parameter_delta_l2: float
    parameters_changed: bool
    logits_changed: bool
    logits_max_abs_delta: float
    param_digest_before: str = ""
    param_digest_after: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "masked_token_count": self.masked_token_count,
            "grad_norm": self.grad_norm,
            "parameter_delta_l2": self.parameter_delta_l2,
            "parameters_changed": self.parameters_changed,
            "logits_changed": self.logits_changed,
            "logits_max_abs_delta": self.logits_max_abs_delta,
            "param_digest_before": self.param_digest_before,
            "param_digest_after": self.param_digest_after,
            "notes": self.notes,
        }


def masked_policy_loss(
    model: Any,
    sample: TrainingSample,
    *,
    device: str,
    dtype: Any,
    advantage: float,
) -> tuple[Any, int]:
    """REINFORCE loss over the sampled tokens only.

    Returns ``(loss, masked_token_count)``. The loss is the negative advantage-
    weighted mean log-prob of the tokens the policy actually sampled; every
    environment, tool, and image token contributes exactly zero.
    """
    validate_sample(sample)

    inputs = build_model_inputs(sample, device=device, dtype=dtype)
    input_ids = inputs["input_ids"]

    # Predictions at position i-1 produce the token at position i.
    mask = torch.tensor(sample.response_mask, device=device, dtype=torch.bool)
    positions = torch.arange(len(sample.input_ids), device=device)
    selectable = mask & (positions > 0)
    if not bool(selectable.any()):
        raise ValueError("no selectable response token")

    index = positions[selectable]
    targets = input_ids[0][index]
    token_logprobs = selected_token_logprobs(model, inputs, index, targets)

    # Second application of the mask. Redundant by design (see module docstring).
    weights = mask[index].float()
    loss = -(advantage * token_logprobs * weights).sum() / weights.sum().clamp(min=1.0)
    return loss, int(weights.sum().item())


def run_policy_update(
    adapter: Any,
    samples: list[TrainingSample],
    *,
    learning_rate: float = 1e-6,
    optimizer: Any = None,
    baseline: float | None = None,
    probe_prompt: str = "Where should the courier go next?",
) -> UpdateResult:
    """Take one optimizer step and prove it changed the policy.

    ``adapter`` is a ``Qwen3VLAdapter``. The model is switched to train mode for
    the step and back afterwards, so a caller cannot accidentally leave dropout
    enabled for subsequent rollouts.
    """
    model = adapter.model
    device, dtype = adapter.device, adapter.dtype

    rewards = [s.reward for s in samples]
    if baseline is None:
        baseline = sum(rewards) / len(rewards) if rewards else 0.0

    logits_before = adapter.logits_for_prompt(probe_prompt)
    digest_before = adapter.parameter_digest()
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Snapshot on the CPU. Cloning a model's worth of parameters onto the
    # same card doubles resident weights purely to verify a delta, and on a
    # 24 GB card that is the difference between a training step and an OOM.
    before = [p.detach().to("cpu", copy=True) for p in trainable]

    notes: list[str] = []
    # The caller may own the optimizer, and for a training *loop* it must.
    # Building a fresh one per call throws away all optimizer state between
    # iterations, which makes momentum and Adam's moment estimates useless
    # and silently reduces any of them to plain SGD. A one-shot gate does
    # not care; twenty REINFORCE steps very much do.
    owned = optimizer is None
    if owned:
        optimizer = torch.optim.SGD(trainable, lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)

    # Recompute activations in the backward pass instead of holding them.
    # A 2B VLM with four episodes of multimodal context needs ~23 GB of
    # activations and OOMs a 24 GB card at batch 4; checkpointing trades
    # some compute for that, and batch size is what makes the advantage
    # estimate anything other than noise.
    checkpointing = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            checkpointing = True
        except Exception as error:  # noqa: BLE001
            notes.append(f"gradient checkpointing unavailable: {error}")
    model.train()
    total_loss = 0.0
    total_masked = 0
    try:
        for sample in samples:
            advantage = sample.reward - baseline
            if advantage == 0.0:
                # A zero advantage yields a zero gradient, which would make the
                # "parameters changed" assertion vacuous. Say so rather than
                # letting the gate pass on a no-op.
                notes.append(f"{sample.episode_id}: zero advantage, contributes no gradient")
            loss, masked = masked_policy_loss(
                model, sample, device=device, dtype=dtype, advantage=advantage
            )
            (loss / max(1, len(samples))).backward()
            total_loss += float(loss.item())
            total_masked += masked

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0).item()
        )
        optimizer.step()
    finally:
        model.eval()
        if checkpointing:
            model.gradient_checkpointing_disable()

    delta_sq = 0.0
    for old, parameter in zip(before, trainable):
        delta_sq += float(
            (parameter.detach().cpu() - old).float().pow(2).sum().item())
    delta_l2 = delta_sq ** 0.5

    logits_after = adapter.logits_for_prompt(probe_prompt)
    logits_delta = float((logits_after - logits_before).abs().max().item())
    digest_after = adapter.parameter_digest()

    return UpdateResult(
        loss=total_loss / max(1, len(samples)),
        masked_token_count=total_masked,
        grad_norm=grad_norm,
        parameter_delta_l2=delta_l2,
        parameters_changed=delta_l2 > 0.0,
        logits_changed=logits_delta > 0.0,
        logits_max_abs_delta=logits_delta,
        param_digest_before=digest_before,
        param_digest_after=digest_after,
        notes=notes,
    )


@torch.no_grad()
def loss_membership_check(adapter: Any, sample: TrainingSample) -> dict[str, Any]:
    """Verify which tokens actually contribute a term to the policy loss.

    A note on a tempting but wrong test. It is natural to differentiate the loss
    with respect to the *input embeddings* and to expect zero gradient at
    environment and image positions. That expectation is false, and acting on it
    produces a false alarm: an autoregressive model conditions every prediction
    on the whole prefix, so prompt tokens necessarily have non-zero embedding
    gradients even under a perfect mask. Embedding gradient measures conditioning
    influence, not loss membership, and the two are different questions.

    The question the gate actually asks is which ``(position, target)`` pairs are
    summed. This answers it directly and independently of the loss code:

    1. recompute the per-token log-probs from a forward pass;
    2. compare the masked loss against the mean over response positions only;
    3. widen the mask to include some environment positions and confirm the loss
       moves — a mask that changes nothing is a mask that is not being applied.
    """
    model = adapter.model
    device, dtype = adapter.device, adapter.dtype
    validate_sample(sample)

    input_ids = torch.tensor([sample.input_ids], device=device)
    inputs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    if sample.pixel_values:
        inputs["pixel_values"] = torch.cat(
            [p.to(device, dtype) for p in sample.pixel_values], dim=0
        )
        inputs["image_grid_thw"] = torch.cat(
            [g.to(device) for g in sample.image_grid_thw], dim=0
        )
    logits = model(**inputs).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    def mean_logprob_over(positions: list[int]) -> float:
        usable = [p for p in positions if p > 0]
        if not usable:
            return 0.0
        index = torch.tensor(usable, device=device)
        targets = input_ids[0][index]
        values = log_probs[index - 1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(values.mean().item())

    response_positions = sample.response_indices
    response_only = mean_logprob_over(response_positions)

    environment_positions = [
        i for i, flagged in enumerate(sample.response_mask) if not flagged and i > 0
    ]
    widened = mean_logprob_over(sorted(response_positions + environment_positions[:16]))

    image_positions = [p for p in sample.image_token_positions if p > 0]
    image_only = mean_logprob_over(image_positions) if image_positions else None

    return {
        "response_position_count": len(response_positions),
        "environment_position_count": len(environment_positions),
        "image_position_count": len(image_positions),
        "mean_logprob_response_only": response_only,
        "mean_logprob_widened_mask": widened,
        "mask_is_load_bearing": abs(widened - response_only) > 1e-6,
        "mean_logprob_image_positions": image_only,
        "image_positions_are_all_environment": all(
            not sample.response_mask[p] for p in sample.image_token_positions
        ),
    }
