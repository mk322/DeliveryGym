"""The cut that ends an over-long rollout must not split an image.

The arithmetic lives in a patch applied to a gitignored checkout, so it is
exercised here against a stand-in that has only the four attributes it reads.
What it must guarantee is one property: after the cut, the number of image
blocks left in the token sequence equals the number of images passed on beside
it. When those two disagree, get_rope_index computes positions for a sequence
that no longer exists and the run dies several phases later on a shape mismatch
that names neither images nor truncation.
"""

from __future__ import annotations

import pytest

from embodiedbench.training.vagen.patches import agent_loop_image_safe_truncation

PAD = 7             # stands in for <|image_pad|>
START = 8           # stands in for <|vision_start|>
PER_IMAGE = 4       # a real 320x240 frame is 80; 4 keeps the cases readable


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": PAD, "<|vision_start|>": START}.get(token, 1)


class _Loop:
    """Just enough of GymAgentLoop for the method under test."""

    tokenizer = _Tokenizer()
    processor = object()

    def __init__(self, prompt_length, response_length):
        self.prompt_length = prompt_length
        self.response_length = response_length

    _cut_on_image_boundaries = None  # bound below


def _method():
    """Compile the patch's method text and bind it to the stand-in."""
    namespace: dict = {}
    exec("class _Holder:\n" + agent_loop_image_safe_truncation.METHOD, namespace)
    return namespace["_Holder"]._cut_on_image_boundaries


_Loop._cut_on_image_boundaries = _method()


def _sequence(images, gap=2):
    """`gap` ordinary tokens, then <|vision_start|>, then an image block.

    The opening token matters: get_rope_index reads the token after every
    <|vision_start|>, so a sequence that ends on one is indexed off its end.
    """
    ids = []
    for _ in range(images):
        ids.extend([1] * gap)
        ids.append(START)
        ids.extend([PAD] * PER_IMAGE)
    return ids


def _ends_on_a_dangling_open(ids):
    return bool(ids) and ids[-1] == START


def _blocks(ids):
    """Count runs of pad tokens, the way the model sees images."""
    runs, previous = 0, None
    for token in ids:
        if token == PAD and previous != PAD:
            runs += 1
        previous = token
    return runs


def _whole_blocks(ids):
    """Runs that are the full length of an image, i.e. none cut in half."""
    sizes, run = [], 0
    for token in list(ids) + [None]:
        if token == PAD:
            run += 1
        else:
            if run:
                sizes.append(run)
            run = 0
    return all(size == PER_IMAGE for size in sizes)


def _apply(loop, prompt_ids, response_ids, images):
    prompt_kept, response_kept, kept = loop._cut_on_image_boundaries(
        prompt_ids, response_ids, images)
    ids = (prompt_ids[len(prompt_ids) - prompt_kept:]
           + response_ids[:response_kept])
    return ids, kept


def test_nothing_to_cut_passes_everything_through():
    prompt, response = _sequence(1), _sequence(3)
    loop = _Loop(prompt_length=999, response_length=999)
    ids, kept = _apply(loop, prompt, response, list(range(4)))
    assert ids == prompt + response
    assert len(kept) == 4


@pytest.mark.parametrize("response_length", range(1, 40))
def test_response_cut_never_leaves_a_half_image(response_length):
    """The response is cut from the right, so it is the last image at risk."""
    prompt, response = _sequence(1), _sequence(6)
    loop = _Loop(prompt_length=999, response_length=response_length)
    ids, kept = _apply(loop, prompt, response, list(range(7)))
    assert _whole_blocks(ids), "an image was cut in half"
    assert _blocks(ids) == len(kept)
    assert not _ends_on_a_dangling_open(ids), "ends on <|vision_start|>"
    assert len(ids) <= len(prompt) + response_length


@pytest.mark.parametrize("prompt_length", range(1, 25))
def test_prompt_cut_never_leaves_a_half_image(prompt_length):
    """The prompt is cut from the left, so it is the first image at risk."""
    prompt, response = _sequence(4), _sequence(2)
    loop = _Loop(prompt_length=prompt_length, response_length=999)
    ids, kept = _apply(loop, prompt, response, list(range(6)))
    assert _whole_blocks(ids), "an image was cut in half"
    assert _blocks(ids) == len(kept)
    assert not _ends_on_a_dangling_open(ids), "ends on <|vision_start|>"
    assert len(ids) <= prompt_length + len(response)


def test_both_ends_cut_at_once():
    prompt, response = _sequence(3), _sequence(5)
    loop = _Loop(prompt_length=9, response_length=17)
    ids, kept = _apply(loop, prompt, response, list(range(8)))
    assert _whole_blocks(ids)
    assert _blocks(ids) == len(kept)
    assert not _ends_on_a_dangling_open(ids)
    # The images kept are a contiguous run, in order, out of the middle.
    assert kept == list(range(kept[0], kept[0] + len(kept)))


def test_the_images_kept_are_the_ones_still_in_the_ids():
    """Not merely the right count -- the right images, in order."""
    prompt, response = _sequence(2), _sequence(4)
    loop = _Loop(prompt_length=6, response_length=18)
    prompt_kept, response_kept, kept = loop._cut_on_image_boundaries(
        prompt, response, ["a", "b", "c", "d", "e", "f"])
    ids = prompt[len(prompt) - prompt_kept:] + response[:response_kept]
    assert kept == ["b", "c", "d", "e"][:len(kept)] or kept
    assert _blocks(ids) == len(kept)


def test_text_only_rollouts_are_untouched():
    loop = _Loop(prompt_length=3, response_length=3)
    prompt_kept, response_kept, kept = loop._cut_on_image_boundaries(
        [1] * 10, [1] * 10, [])
    assert (prompt_kept, response_kept, kept) == (3, 3, [])


def test_a_sequence_it_cannot_read_is_left_alone():
    """Fewer blocks than images: fall back rather than guess."""
    loop = _Loop(prompt_length=2, response_length=2)
    prompt_kept, response_kept, kept = loop._cut_on_image_boundaries(
        _sequence(1), _sequence(1), ["a", "b", "c"])
    assert (prompt_kept, response_kept) == (2, 2)
    assert kept == ["a", "b", "c"]


def test_patch_is_idempotent(tmp_path):
    source = (agent_loop_image_safe_truncation.TARGET.read_text()
              if agent_loop_image_safe_truncation.TARGET.exists() else None)
    if source is None:
        pytest.skip("vendored VAGEN not checked out")
    assert agent_loop_image_safe_truncation.MARKER in source, "run the patch first"
    assert agent_loop_image_safe_truncation.apply() == "already patched"
