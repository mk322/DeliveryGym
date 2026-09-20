"""Give PolicyLossConfig the field the bypass writes into.

verl's rollout-correction bypass skips recomputing ``old_log_probs`` and uses
the logprobs the engine already produced while sampling -- 207 s of a 1424 s
step here, because a second forward pass over forty 40k-token trajectories is
not cheap. It is the documented feature:
https://verl.readthedocs.io/en/latest/algo/rollout_corr.html

This pin has it on the algorithm side and in the helper, but not on the
config it writes to. ``apply_rollout_correction`` does

    policy_loss_config["rollout_correction"] = rollout_corr_config

and ``PolicyLossConfig`` has no such field, so omegaconf's struct mode refuses
the assignment:

    omegaconf.errors.ConfigKeyError: Key 'rollout_correction' is not in struct
    full_key: actor_rollout_ref.actor.policy_loss.rollout_correction

The run reaches validation, then dies at the first training step. Adding the
field -- which is what upstream verl carries -- is the whole of the fix; the
default of None leaves every run that does not enable the bypass exactly as
it was.

Worth being explicit about what the bypass changes, since it is not free: the
PPO ratio becomes pi_theta/pi_rollout rather than pi_theta/pi_old. That is the
correct correction when the rollout policy and the old policy are the same
checkpoint, which is the case in this synchronous loop -- the batch is
sampled and then immediately trained on.

This patches the vendored checkout in place. It lives here rather than as an
edit to ``vendor/`` because ``vendor/`` is gitignored, so an edit made there
is invisible to git and lost on the next clone.

Idempotent; run it after ``git submodule update``:

    python -m embodiedbench.training.vagen.patches.policy_loss_rollout_correction_field
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TARGET = REPO / "vendor/vagen/verl/verl/workers/config/actor.py"

ANCHOR = """    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
"""

REPLACEMENT = """    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    # Written at runtime by apply_rollout_correction, which cannot add a key
    # omegaconf's struct mode does not already know. Patched in by
    # embodiedbench/training/vagen/patches/. None leaves every run that does
    # not enable the bypass untouched.
    rollout_correction: Any = None
"""

MARKER = "rollout_correction: Any = None"


def apply(target: Path = TARGET) -> str:
    if not target.exists():
        raise FileNotFoundError(f"vendored verl not found at {target}")
    source = target.read_text()
    if MARKER in source:
        return "already patched"
    if ANCHOR not in source:
        raise RuntimeError(
            f"{target} does not contain the expected PolicyLossConfig fields; "
            "verl has moved and this patch needs rewriting")
    patched = source.replace(ANCHOR, REPLACEMENT, 1)
    if "from typing import Any" not in patched and "\nfrom typing import" not in patched:
        patched = patched.replace("from dataclasses import", "from typing import Any\nfrom dataclasses import", 1)
    elif "Any" not in patched.split("from typing import")[1].split("\n")[0]:
        head = patched.split("from typing import")[1].split("\n")[0]
        patched = patched.replace(f"from typing import{head}",
                                  f"from typing import Any,{head}", 1)
    target.write_text(patched)
    return "patched"


if __name__ == "__main__":
    print(f"{apply()}: {TARGET}")
