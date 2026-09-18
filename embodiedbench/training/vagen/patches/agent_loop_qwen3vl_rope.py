"""Make VAGEN's multi-turn agent loop pick the right mrope for Qwen3-VL.

``verl/utils/dataset/rl_dataset.py`` already dispatches on the processor::

    if "Qwen3VLProcessor" in self.processor.__class__.__name__:
        from verl.models.transformers.qwen3_vl import get_rope_index
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index

``verl/experimental/agent_loop/agent_loop.py`` does not -- it imports the
Qwen2-VL version unconditionally. That is the path VAGEN's multi-turn training
takes, so a Qwen3-VL run there gets Qwen2-VL position ids. Qwen3-VL's image
processor is ``Qwen2VLImageProcessorFast``, so the outer guard still passes and
nothing complains: the positions are simply wrong, and a run trains to a worse
policy for a reason no log line mentions.

This patches the vendored checkout in place. It lives here rather than as an
edit to ``vendor/`` because ``vendor/`` is gitignored, so an edit made there is
invisible to git and lost on the next clone -- the one line that makes Qwen3-VL
correct would be the one line nobody can review.

Idempotent; run it after ``git submodule update``:

    python -m embodiedbench.training.vagen.patches.agent_loop_qwen3vl_rope
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TARGET = REPO / "vendor/vagen/verl/verl/experimental/agent_loop/agent_loop.py"

BEFORE = "                from verl.models.transformers.qwen2_vl import get_rope_index\n"
AFTER = (
    "                # Qwen3-VL has its own mrope; rl_dataset.py dispatches on\n"
    "                # the processor and this path did not. Patched by\n"
    "                # embodiedbench/training/vagen/patches/.\n"
    '                if "Qwen3VLProcessor" in self.processor.__class__.__name__:\n'
    "                    from verl.models.transformers.qwen3_vl import get_rope_index\n"
    "                else:\n"
    "                    from verl.models.transformers.qwen2_vl import get_rope_index\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"not found: {TARGET}\nrun: git submodule update --init --recursive")
        return 1
    source = TARGET.read_text()
    if "Qwen3VLProcessor" in source:
        print("already patched")
        return 0
    if BEFORE not in source:
        print("anchor not found -- verl changed; re-check agent_loop.py by hand")
        return 1
    TARGET.write_text(source.replace(BEFORE, AFTER, 1))
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
