"""Keep only the most recent frames in the context, not every frame ever.

DO NOT APPLY YET -- it kills the run. Trimming at this seam is wrong.

Applied on 2026-08-13 it took vLLM's EngineCore down with
``IndexError: list index out of range`` inside a minute. By the time this hook
runs the prompt is already tokenised: ``agent_data.prompt_ids`` carries the
``<image>`` tokens, and rewriting the text of ``agent_data.messages``
afterwards changes nothing about them. So the ids kept N placeholders while
``image_data`` had been cut to three, the processor paired them by position,
and it indexed past the end -- the same ids-versus-grids failure the
image-safe truncation patch exists for, arriving from the opposite side.

The idea is right and the seam is not. A window has to be applied where the
ids are built, dropping the placeholder TOKENS with the images, which is the
same place ``_cut_on_image_boundaries`` already works. Left here, tested at
the function level, and not wired up.

``gym_agent_loop.py`` accumulates images for the whole episode::

    agent_data.image_data.extend(new_images)

and never drops any. So a forty-turn episode carries forty turns' worth of
pictures into every later prompt, the sequence grows without bound, and the
run ends on the shape mismatch the image-safe truncation patch exists to make
survivable. Survivable is not the same as intended: truncation throws away the
OLDEST turns' text as well, which is the memory the courier needs, in order to
keep frames it can no longer act on.

A courier does not need the photograph it took nine junctions ago. It needs
the ones in front of it now and enough of the recent past to notice it is
going in circles. So the window is on the IMAGES, explicitly, and the text of
every turn stays.

The count is deliberately small. Each frame is a few hundred tokens of a
budget the observation text also has to fit in, and the whole reason the live
configs were cut from the offline task's forty turns to four was this budget:
holding the last few frames instead of all of them is what buys those turns
back.

``<image>`` placeholders in older turns are rewritten to the word the model
would read anyway, so the text still says a photograph was there and the
count of placeholders still matches the count of images -- a mismatch between
them is the same class of failure as the truncation bug, arriving from the
other side.

This patches the vendored checkout in place, because ``vendor/`` is gitignored
and an edit made there is invisible to git and lost on the next clone. Run it
after ``git submodule update``, alongside the other patch in this directory::

    python -m embodiedbench.training.vagen.patches.recent_images_only
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TARGET = REPO / "vendor/vagen/vagen/agent_loop/gym_agent_loop.py"

#: How many of the most recent frames survive into the next prompt.
#: Overridable so a run can measure the axis rather than inherit it.
ENV_VAR = "VAGEN_RECENT_IMAGES"
DEFAULT_KEEP = 3

MARKER = "_keep_recent_images"

ANCHOR = "        agent_data.image_data.extend(new_images)"

REPLACEMENT = '''        agent_data.image_data.extend(new_images)
        # Patched by embodiedbench/training/vagen/patches/: keep only the most
        # recent frames. Unbounded, a forty-turn episode carries forty turns of
        # pictures into every later prompt and the sequence overruns -- and the
        # truncation that catches it drops the OLDEST TEXT, which is the memory
        # the courier needs, to keep frames it can no longer act on.
        from embodiedbench.training.vagen.patches.recent_images_only import (
            _keep_recent_images,
        )
        _keep_recent_images(agent_data)'''

HELPER = ""


def _keep_recent_images(agent_data, keep: int | None = None) -> None:
    """Drop all but the last ``keep`` frames, and the placeholders with them.

    The placeholder count and the image count are one contract: the processor
    pairs them by position, so removing an image without removing its
    ``<image>`` marker shifts every later pairing by one. Older markers become
    the words they stood for, so the text still records that a photograph was
    taken and the prompt stays readable.

    Defined here and imported by the injected line rather than copied into the
    vendored file, so it is a function a test can call.
    """
    if keep is None:
        keep = int(os.environ.get(ENV_VAR, DEFAULT_KEEP))
    if keep <= 0:
        return
    images = agent_data.image_data
    drop = len(images) - keep
    if drop <= 0:
        return
    del images[:drop]
    messages = getattr(agent_data, "messages", None)
    if not messages:
        return
    # From the front: the oldest placeholders are the ones retired.
    remaining = drop
    for message in messages:
        if remaining <= 0:
            break
        content = message.get("content")
        if isinstance(content, str) and "<image>" in content:
            take = min(content.count("<image>"), remaining)
            message["content"] = content.replace("<image>", "(a photograph)", take)
            remaining -= take


def patch(text: str) -> str:
    if MARKER in text:
        return text
    if ANCHOR not in text:
        raise SystemExit(
            "anchor not found -- the vendored agent loop has moved; find "
            "where image_data is extended and re-point this patch")
    return text.replace(ANCHOR, REPLACEMENT, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"vendored VAGEN not checked out at {TARGET}")
    before = TARGET.read_text()
    after = patch(before)
    if after == before:
        print(f"already patched: {TARGET}")
        return
    TARGET.write_text(after)
    keep = os.environ.get(ENV_VAR, DEFAULT_KEEP)
    print(f"patched: {TARGET} (keeping the most recent {keep} frames)")


if __name__ == "__main__":
    main()
