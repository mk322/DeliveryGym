# What is in vendor/, and why it is committed

`vendor/` holds two third-party checkouts that the training path imports
directly, plus the compiled city maps. The Python packages are installed
editable (`pip install -e vendor/vagen/verl -e vendor/vagen`), so
`import vagen` and `import verl` resolve here and nowhere else.

| path | upstream | branch | commit |
|---|---|---|---|
| `vendor/vagen` | https://github.com/ymzhang0303/VAGEN.git | `dev/env_v2` | `b93deaacba0c27eef4f26ffadc06710a5221c180` |
| `vendor/vagen/verl` | https://github.com/JamesKrW/verl.git | `vagen-lite` | `3fe0a29975e1b02ae2bd1dec249f7807dd7966f5` |
| `vendor/vagen/vagen/envs/deliverybench/maps/` | the ten compiled city maps (nodes, streets, addresses, buildings) | — | — |

The nesting is upstream's: VAGEN's own install clones verl inside the VAGEN
checkout, and the layout is kept so its imports keep working.

## Why the source is committed rather than cloned

The convention — verl, VAGEN, SLIME, OpenRLHF — is to pin a commit and clone
it at install time. That assumes the machine has a network, and it assumes
upstream still has the pinned code: `verl.experimental.dataset` no longer
exists on upstream `main`, and VAGEN's entrypoint moved. A clean clone of
this repository must import and train without either, so the bytes travel
with the branch. **Do not replace `vendor/` with an upstream checkout.**

## Local modifications

The vendored trees differ from the pinned upstream commits in a handful of
files; every change is in this repository's history (`git log -- vendor/`).
`vendor/LOCAL_PATCHES.diff` records the original five:

- `vagen/agent_loop/gym_agent_loop.py` — `reward_extra_info` carries the
  courier's money and violation metrics (`earnings`, `delivered`,
  `red_crossings`, `blocked_attempts`, `earnings_at_{20..100}`, the
  constraint-compliance keys) with a constant key set on every trajectory
- `vagen/agent_loop/gym_agent_loop_no_concat.py` — the same metrics on the
  other loop variant
- `verl/experimental/agent_loop/agent_loop.py` — reward-extra keys are the
  UNION across the batch, read with `.get(key, 0.0)`
- `verl/models/transformers/qwen3_vl.py`, `verl/workers/config/actor.py`,
  `verl/workers/rollout/schemas.py` — Qwen3-VL and rollout fixes
- `vagen/gym_agent_dataset.py` — warns when a limit-1 seed range is not
  exactly the block's `n_envs` (a subset would otherwise be drawn silently)

Everything else is applied at runtime as a monkeypatch module under
`embodiedbench/training/vagen/patches/` (see `scripts/train_grpo.sh`).
Prefer adding a patch module over editing vendored source: a patch module
survives a vendor bump, an edit does not.

## What is not here

Two directories are excluded from git because they are large and nothing in
the training or evaluation path reads them: the engine's own first-person
frame set (`vagen/vagen/envs/deliverybench/deliverybench_fpv/`) and its
per-episode scratch (`vagen/outputs/`). The benchmark's photographs are the
albums (`docs/INSTALL.md`).

To bump a vendored dependency, clone the upstream fresh, re-apply the
modifications above, and update the commits in the table.
