"""Drop the images a truncated trajectory no longer contains.

``vagen/agent_loop/gym_agent_loop.py`` ends an over-long rollout like this::

    prompt_ids=prompt_ids[-self.prompt_length:],
    response_ids=response_ids[: self.response_length],
    ...
    multi_modal_data = {"image": agent_data.image_data}

Both ends of the token sequence are cut -- the prompt from the left, the
response from the right -- and the image list is passed through whole. In a
multi-turn rollout only the system prompt and the first observation are in the
prompt segment; every later turn's observation, and so nearly every image, is
in the response segment. So the cut lands among the images, and it can land
*inside* one: the ids keep some of that image's ``<|image_pad|>`` tokens and
lose the rest, while ``image_grid_thw`` still describes all of it.

Nothing compares the two. ``get_rope_index`` builds position ids from the
grids, they are used to index a batch built from the ids, and a phase later the
run dies on

    RuntimeError: shape mismatch: value tensor of shape [3, 17239] cannot be
    broadcast to indexing result of shape [3, 17201]

which says nothing about images or truncation. The run that produced those
numbers had 3680 pad tokens' worth of grid -- 46 frames at 80 tokens each --
against 3402 in the ids: 42 whole frames and 42 tokens of the 43rd.

This moves each cut to the nearest image boundary and passes on only the images
that survive it. A truncated trajectory loses its last turns, which is what
truncation means; it stops losing the agreement between the ids and the grids.

Sizing the lengths so this rarely fires is a separate matter and belongs in the
launch script, where the number that matters is ``prompt_length +
response_length``: ``vllm_async_server.py`` assigns exactly that to the
engine's ``max_model_len`` and discards ``rollout.max_model_len``.

This patches the vendored checkout in place. It lives here rather than as an
edit to ``vendor/`` because ``vendor/`` is gitignored, so an edit made there is
invisible to git and lost on the next clone.

Idempotent; run it after ``git submodule update``:

    python -m embodiedbench.training.vagen.patches.agent_loop_image_safe_truncation
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TARGET = REPO / "vendor/vagen/vagen/agent_loop/gym_agent_loop.py"

ANCHOR = """        output = AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
"""

REPLACEMENT = """        # Patched by embodiedbench/training/vagen/patches/: cutting the ids
        # without cutting the images leaves image_grid_thw describing frames
        # that are no longer in the sequence.
        keep_prompt, keep_response, images = self._cut_on_image_boundaries(
            prompt_ids, response_ids, agent_data.image_data)
        multi_modal_data = {"image": images} if images else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[len(prompt_ids) - keep_prompt:],
            response_ids=response_ids[:keep_response],
            response_mask=agent_data.response_mask[:keep_response],
"""

# The two remaining `[: self.response_length]` slices in the same call.
ANCHOR_TAIL = """            response_logprobs=(
                agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
            ),
"""

REPLACEMENT_TAIL = """            response_logprobs=(
                agent_data.response_logprobs[:keep_response] if agent_data.response_logprobs else None
            ),
"""

METHOD = '''
    def _cut_on_image_boundaries(self, prompt_ids, response_ids, images):
        """How much of each segment to keep so no image is half in it.

        Returns (prompt_kept, response_kept, images_kept). Falls back to the
        plain lengths whenever the sequence cannot be read as one image block
        per image, which is the text-only case and any processor whose pad
        token is not <|image_pad|>.
        """
        prompt_kept = min(self.prompt_length, len(prompt_ids))
        response_kept = min(self.response_length, len(response_ids))
        if not images or self.processor is None:
            return prompt_kept, response_kept, images
        if prompt_kept == len(prompt_ids) and response_kept == len(response_ids):
            return prompt_kept, response_kept, images

        pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if pad_id is None or pad_id < 0:
            return prompt_kept, response_kept, images
        # An image is written <|vision_start|> <|image_pad|>... <|vision_end|>,
        # and get_rope_index reads the token after every <|vision_start|> it
        # finds. Ending the sequence on one indexes off the end of it, so the
        # opening token goes with the image it opens.
        start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

        sequence = list(prompt_ids) + list(response_ids)
        blocks, index, length = [], 0, len(sequence)
        while index < length:
            if sequence[index] == pad_id:
                end = index
                while end < length and sequence[end] == pad_id:
                    end += 1
                blocks.append((index, end))
                index = end
            else:
                index += 1
        # One contiguous run of pad tokens per image, or this is not a
        # sequence we know how to cut.
        if len(blocks) != len(images):
            return prompt_kept, response_kept, images

        low = len(prompt_ids) - prompt_kept
        high = len(prompt_ids) + response_kept

        first = 0
        for position, (start, end) in enumerate(blocks):
            if end <= low:
                first = position + 1
            elif start < low:
                low = end          # the cut split this one; drop it whole
                first = position + 1
            else:
                break

        last = len(blocks)
        for position in range(len(blocks) - 1, -1, -1):
            start, end = blocks[position]
            if start >= high:
                last = position
            elif end > high:
                high = start       # same, from the other end
                last = position
            else:
                break
        while high > len(prompt_ids) and sequence[high - 1] == start_id:
            high -= 1

        return (len(prompt_ids) - low, high - len(prompt_ids),
                images[first:last])
'''

MARKER = "_cut_on_image_boundaries"


def apply(target: Path = TARGET) -> str:
    if not target.exists():
        raise FileNotFoundError(f"vendored VAGEN not found at {target}")
    source = target.read_text()
    if MARKER in source:
        return "already patched"
    for anchor in (ANCHOR, ANCHOR_TAIL):
        if anchor not in source:
            raise RuntimeError(
                f"{target} does not contain the expected AgentLoopOutput call; "
                "VAGEN has moved and this patch needs rewriting")
    patched = source.replace(ANCHOR, REPLACEMENT, 1)
    patched = patched.replace(ANCHOR_TAIL, REPLACEMENT_TAIL, 1)

    # The method goes on the same class, just above the method that used to do
    # the cutting.
    hook = "    async def _handle_pending_state("
    if hook not in patched:
        raise RuntimeError(f"{target} has no _handle_pending_state to sit above")
    patched = patched.replace(hook, METHOD.lstrip("\n") + "\n" + hook, 1)
    target.write_text(patched)
    return "patched"


if __name__ == "__main__":
    print(f"{apply()}: {TARGET}")
