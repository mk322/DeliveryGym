"""Never cut a multi-turn trajectory through an image's token block.

``verl/workers/rollout/schemas.py`` ends an over-long trajectory with a blind
slice::

    self.input_ids = self.input_ids[..., : self.max_model_len]

which is fine for text and wrong the moment there are images in it. In a
multi-turn rollout only the system prompt and the first observation are in the
prompt segment; every later turn's observation -- and so every later image --
is appended to the response segment. A trajectory that outruns its budget is
therefore cut among the images, and the cut can land inside one, leaving fewer
``<|image_pad|>`` tokens in ``input_ids`` than ``image_grid_thw`` still says are
there. Nothing checks the two against each other. ``get_rope_index`` builds
position ids from the grids, the batch is indexed with the ids, and the run
dies a phase later on

    RuntimeError: shape mismatch: value tensor of shape [3, 9355] cannot be
    broadcast to indexing result of shape [3, 9017]

with nothing in the message about images or truncation. The 338-token gap is
one 320x240 frame and a bit.

So pull the cut back to the start of the first image it would have split, and
drop that image and everything after it from ``image_grid_thw`` and
``pixel_values`` as well. The trajectory loses its last turns, which is what
truncation means; it does not lose the agreement between the ids and the grids.

Sizing the lengths so this rarely fires is a separate matter and belongs in the
launch script: the cap that bites is ``prompt_length + response_length``,
because ``vllm_async_server.py`` assigns exactly that to ``max_model_len`` and
discards ``rollout.max_model_len``.

This patches the vendored checkout in place. It lives here rather than as an
edit to ``vendor/`` because ``vendor/`` is gitignored, so an edit made there is
invisible to git and lost on the next clone.

Idempotent; run it after ``git submodule update``:

    python -m embodiedbench.training.vagen.patches.multiturn_image_safe_truncation
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TARGET = REPO / "vendor/vagen/verl/verl/workers/rollout/schemas.py"

ANCHOR = """    def truncate_output_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> None:
        self.input_ids = self.input_ids[..., : self.max_model_len]
"""

METHOD = '''    def _pull_cut_back_off_images(self, processing_class) -> None:
        """Move an over-long trajectory's cut off an image block.

        Patched in by embodiedbench/training/vagen/patches/. Later turns'
        observations live in the response segment, so the slice below can land
        inside an image and leave fewer <|image_pad|> tokens than
        image_grid_thw describes; get_rope_index then computes positions for a
        sequence that no longer exists.
        """
        multi_modal = self.multi_modal_inputs or {}
        grids = multi_modal.get("image_grid_thw")
        if grids is None or len(grids) == 0:
            return

        # Both caps can cut: the trajectory's own, and the response's.
        limit = min(self.max_model_len, self.prompt_ids.shape[-1] + self.max_response_len)
        if self.input_ids.shape[-1] <= limit:
            return

        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_processor = getattr(processing_class, "image_processor", None)
        merge = int(getattr(image_processor, "merge_size", 2) or 2)

        pad_at = (self.input_ids.reshape(-1) == pad_id).nonzero(as_tuple=True)[0]
        if pad_at.numel() == 0:
            return

        per_image = [int(g[0]) * int(g[1]) * int(g[2]) // (merge * merge) for g in grids]
        survived = int((pad_at < limit).sum())

        kept = seen = 0
        for count in per_image:
            if seen + count > survived:
                break
            seen += count
            kept += 1

        cut = limit
        if kept < len(per_image):
            # `seen` pads belong to whole images, so the next one is the first
            # pad of the image the cut would have split. End the sequence there.
            cut = int(pad_at[seen])

        self.input_ids = self.input_ids[..., :cut]
        self.attention_mask = self.attention_mask[..., :cut]
        self.position_ids = self.position_ids[..., :cut]
        self.loss_mask = self.loss_mask[..., :cut]

        multi_modal["image_grid_thw"] = grids[:kept]
        pixels = multi_modal.get("pixel_values")
        if pixels is not None:
            rows = sum(int(g[0]) * int(g[1]) * int(g[2]) for g in grids[:kept])
            multi_modal["pixel_values"] = pixels[:rows]

'''

REPLACEMENT = METHOD + """    def truncate_output_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> None:
        self._pull_cut_back_off_images(processing_class)
        self.input_ids = self.input_ids[..., : self.max_model_len]
"""

MARKER = "_pull_cut_back_off_images"


def apply(target: Path = TARGET) -> str:
    if not target.exists():
        raise FileNotFoundError(f"vendored verl not found at {target}")
    source = target.read_text()
    if MARKER in source:
        return "already patched"
    if ANCHOR not in source:
        raise RuntimeError(
            f"{target} does not contain the expected truncate_output_ids body; "
            "verl has moved and this patch needs rewriting")
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    return "patched"


if __name__ == "__main__":
    print(f"{apply()}: {TARGET}")
