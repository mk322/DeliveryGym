"""Qwen3-VL rollout adapter with exact multi-turn token accounting.

This exists to satisfy the parts of design plan §11.1's selection gate that are easy
to get subtly wrong and fatal when wrong:

    sampled tokens and log probabilities are preserved without lossy text
    re-tokenization; observation/image/tool tokens are excluded from policy loss

The usual way both break is the same: build each turn's prompt by decoding the
conversation back to a string and re-tokenizing it. Re-tokenization is not
guaranteed to reproduce the token ids that were actually sampled — a merge can
span the boundary between generated text and the environment text appended after
it — so the ids the trainer optimizes stop matching the ids the policy emitted,
and the log-probs no longer correspond to anything.

So this adapter never re-tokenizes. It maintains one growing token sequence:

- environment turns are tokenized once and appended, marked ``assistant=False``;
- generated turns keep the exact ids ``generate`` returned, marked
  ``assistant=True``, alongside the transition scores from the same call.

Image tokens are tracked explicitly by id, because they live inside environment
turns and must be excluded from the policy loss for a second, independent
reason: they are not sampled text at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# Qwen chat-template fragments, tokenized once and spliced. Splicing pre-tokenized
# fragments is what lets the response ids survive untouched.
# How many elements of a parameter to reduce at once when digesting.
# 2**22 float32 values is 16 MB of temporary, which is nothing next to
# a 7B model and bounded no matter how large a single weight is.
_DIGEST_CHUNK = 1 << 22

IM_END = "<|im_end|>\n"
USER_OPEN = "<|im_start|>user\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"
SYSTEM_OPEN = "<|im_start|>system\n"


@dataclass
class TurnSpan:
    """One contiguous span of the running sequence."""

    role: str  # "system" | "user" | "assistant"
    start: int
    end: int
    is_assistant: bool
    logprobs: list[float] = field(default_factory=list)
    image_positions: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class RolloutState:
    """The full token sequence for one episode, with provenance per token."""

    input_ids: list[int] = field(default_factory=list)
    assistant_mask: list[bool] = field(default_factory=list)
    spans: list[TurnSpan] = field(default_factory=list)
    # Vision inputs, concatenated across turns in the order their placeholder
    # tokens appear in input_ids.
    pixel_values: list[Any] = field(default_factory=list)
    image_grid_thw: list[Any] = field(default_factory=list)
    image_token_positions: list[int] = field(default_factory=list)

    def response_token_indices(self) -> list[int]:
        return [i for i, is_assistant in enumerate(self.assistant_mask) if is_assistant]

    def observation_token_indices(self) -> list[int]:
        return [i for i, is_assistant in enumerate(self.assistant_mask) if not is_assistant]

    def assistant_logprobs(self) -> list[float]:
        out: list[float] = []
        for span in self.spans:
            if span.is_assistant:
                out.extend(span.logprobs)
        return out

    def summary(self) -> dict[str, int]:
        return {
            "total_tokens": len(self.input_ids),
            "assistant_tokens": sum(self.assistant_mask),
            "environment_tokens": len(self.input_ids) - sum(self.assistant_mask),
            "image_tokens": len(self.image_token_positions),
            "turns": len(self.spans),
        }


class Qwen3VLAdapter:
    """Loads Qwen3-VL and drives a multi-turn multimodal rollout."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: Any = None,
        attn_implementation: str | None = None,
        lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        from transformers import AutoProcessor

        self.model_path = model_path
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        kwargs: dict[str, Any] = {"dtype": self.dtype, "device_map": device}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        # Loaded by class *capability* rather than by name. This was pinned to
        # ``Qwen3VLForConditionalGeneration``, which meant the whole training
        # path could only ever run against one checkpoint -- and the one it
        # named is not on every host. Every Qwen-family VLM shares the
        # ``<|image_pad|>`` placeholder convention this adapter is written
        # around, so the smaller ones are usable as gate models: a correctness
        # gate wants the cheapest checkpoint that exercises the path, not the
        # best one.
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM
        self.model = _AutoVLM.from_pretrained(model_path, **kwargs)
        self.model.eval()

        # LoRA, when asked for. What it buys here is not speed but *size*:
        # a full fine-tune keeps parameters, gradients and two Adam moments
        # for every weight, so a 7B model needs about 84 GB and does not fit
        # a 24 GB card at any batch. With adapters only the adapters carry
        # gradients and optimizer state, which is a few tens of MB, and the
        # frozen base costs its weights and nothing else.
        #
        # Nothing downstream needs to know. ``run_policy_update`` already
        # trains ``[p for p in model.parameters() if p.requires_grad]``, and
        # PEFT freezes the base, so the same code trains adapters instead.
        self.lora = lora
        if lora:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.model = get_peft_model(self.model, config)
            self.model.eval()

        # The id whose occurrences mark where visual embeddings are spliced in.
        config = self.model.config
        self.image_token_id = int(
            getattr(config, "image_token_id", None)
            or getattr(getattr(config, "vision_config", object()), "image_token_id", 0)
            or self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        )

    # ── tokenization helpers ─────────────────────────────────────────────────

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _encode_with_images(self, text: str, images: list[Any]) -> tuple[list[int], Any, Any]:
        """Tokenize a chunk containing image placeholders, expanded by the processor."""
        if not images:
            return self._encode_text(text), None, None
        batch = self.processor(text=[text], images=images, return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        return ids, batch["pixel_values"], batch["image_grid_thw"]

    def _append(
        self,
        state: RolloutState,
        ids: list[int],
        *,
        role: str,
        is_assistant: bool,
        logprobs: list[float] | None = None,
        pixel_values: Any = None,
        image_grid_thw: Any = None,
    ) -> TurnSpan:
        start = len(state.input_ids)
        state.input_ids.extend(ids)
        state.assistant_mask.extend([is_assistant] * len(ids))
        image_positions = [
            start + offset for offset, token in enumerate(ids) if token == self.image_token_id
        ]
        state.image_token_positions.extend(image_positions)
        if pixel_values is not None:
            state.pixel_values.append(pixel_values)
            state.image_grid_thw.append(image_grid_thw)
        span = TurnSpan(
            role=role,
            start=start,
            end=len(state.input_ids),
            is_assistant=is_assistant,
            logprobs=list(logprobs or []),
            image_positions=image_positions,
        )
        state.spans.append(span)
        return span

    # ── rollout ──────────────────────────────────────────────────────────────

    def start_episode(self, system_prompt: str) -> RolloutState:
        state = RolloutState()
        ids = self._encode_text(SYSTEM_OPEN + system_prompt + IM_END)
        self._append(state, ids, role="system", is_assistant=False)
        return state

    def observe(self, state: RolloutState, text: str, images: list[Any] | None = None) -> TurnSpan:
        """Append an environment turn and open the assistant turn.

        Environment text, image tokens, and the assistant-open marker are all
        marked non-assistant: none of them were sampled by the policy, so none
        may enter the policy loss.
        """
        images = images or []
        placeholder = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in images)
        chunk = USER_OPEN + placeholder + text + IM_END + ASSISTANT_OPEN
        ids, pixel_values, grid = self._encode_with_images(chunk, images)
        return self._append(
            state,
            ids,
            role="user",
            is_assistant=False,
            pixel_values=pixel_values,
            image_grid_thw=grid,
        )

    def _model_inputs(self, state: RolloutState) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "input_ids": torch.tensor([state.input_ids], device=self.device),
            "attention_mask": torch.ones(1, len(state.input_ids), dtype=torch.long, device=self.device),
        }
        if state.pixel_values:
            inputs["pixel_values"] = torch.cat(
                [p.to(self.device, self.dtype) for p in state.pixel_values], dim=0
            )
            inputs["image_grid_thw"] = torch.cat(
                [g.to(self.device) for g in state.image_grid_thw], dim=0
            )
        return inputs

    @torch.no_grad()
    def generate(
        self,
        state: RolloutState,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> tuple[str, TurnSpan]:
        """Sample one assistant turn, keeping its exact ids and log-probs."""
        inputs = self._model_inputs(state)
        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temperature > 0.0
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=None,
            top_k=None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        prompt_length = len(state.input_ids)
        response_ids = output.sequences[0][prompt_length:].tolist()

        # Log-probs of the tokens that were actually sampled, from the same call
        # that sampled them. Recomputing them later from decoded text is exactly
        # the lossy path this adapter exists to avoid.
        transition = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )[0]
        logprobs = [float(v) for v in transition[: len(response_ids)].tolist()]

        span = self._append(
            state, response_ids, role="assistant", is_assistant=True, logprobs=logprobs
        )
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return text, span

    def close_assistant_turn(self, state: RolloutState) -> None:
        """Append the assistant end marker as environment tokens.

        ``generate`` stops at ``<|im_end|>`` or the token cap. The separator that
        follows belongs to the template, not to the policy's output, so it is
        appended here rather than left inside the sampled span.
        """
        if state.input_ids and state.input_ids[-1] == self.tokenizer.eos_token_id:
            return
        self._append(state, self._encode_text(IM_END), role="separator", is_assistant=False)

    # ── verification ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def recompute_logprobs(self, state: RolloutState) -> list[float]:
        """Teacher-forced log-probs of every assistant token in the sequence.

        A single forward pass over the final sequence, used to check that the
        rollout's stored log-probs correspond to the ids actually in the
        sequence. If splicing had corrupted a boundary, these would disagree.
        """
        inputs = self._model_inputs(state)
        logits = self.model(**inputs).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        out: list[float] = []
        for index in state.response_token_indices():
            if index == 0:
                continue
            token = state.input_ids[index]
            out.append(float(log_probs[index - 1, token]))
        return out

    @torch.no_grad()
    def logits_for_prompt(self, prompt: str) -> Any:
        """Logits for a fixed prompt, for the weight-sync check (the design plan M8)."""
        ids = self._encode_text(SYSTEM_OPEN + "You are a test probe." + IM_END + USER_OPEN + prompt + IM_END + ASSISTANT_OPEN)
        inputs = {
            "input_ids": torch.tensor([ids], device=self.device),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=self.device),
        }
        return self.model(**inputs).logits[0, -1].float().cpu()

    @torch.no_grad()
    def parameter_digest(self) -> str:
        """A checkpoint hash sensitive enough to detect one small update.

        the design plan M8 requires weight synchronization to produce a *changed*
        checkpoint hash. A digest that samples a few elements per tensor and
        rounds to six decimals does not: a single step at lr=1e-5 moves 4.4B
        parameters by a total L2 of about 6e-7, so almost every sampled element
        is unchanged at that precision and the hash falsely reports "no change".

        This reduces over every element instead, accumulating in float64 and
        formatting at full double precision, so an update that moves any weight
        at all changes the digest.
        """
        from embodiedbench.artifacts.hashing import sha256_bytes

        chunks: list[bytes] = []
        for name, parameter in sorted(self.model.named_parameters()):
            if parameter.numel() == 0:
                continue
            # Reduced in bounded chunks. ``parameter.detach().double()``
            # allocates a copy at twice the weight's size -- 4 GB for a 7B
            # model's embedding matrix, which OOMs a 24 GB card before the first
            # training step -- and ``sum(dtype=float64)`` upcasts internally and
            # costs the same. Chunking keeps the temporary bounded regardless of
            # how large the parameter is, and the arithmetic is identical.
            flat = parameter.detach().reshape(-1)
            total = energy = 0.0
            for start in range(0, flat.numel(), _DIGEST_CHUNK):
                piece = flat[start:start + _DIGEST_CHUNK].float()
                total += float(piece.sum().item())
                energy += float(piece.pow(2).sum().item())
            chunks.append(f"{name}:{total!r}:{energy!r}".encode())
        return sha256_bytes(b"|".join(chunks))
