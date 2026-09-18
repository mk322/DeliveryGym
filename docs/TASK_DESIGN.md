# The courier task, as a benchmark: what is measured, and what is known to be wrong

This is the design review of the RL environment, written against the code as
it stands and the measurements the runs have produced. Its audience is anyone
deciding whether a number from this benchmark means what it claims to mean —
including us. Sections marked **⚠ open** are known weaknesses with owners of
their own; nothing here is hidden in a comment.

## 1. The task

One courier, on foot, in a rendered Paris street network (`citycore-paris`,
an Unreal Engine scene; frames are UE renders baked into albums by
`embodiedbench/compiler/fpv_render.py`, one frame per
street-position-heading — an earlier revision of this document said "real
photographs", which was wrong; the pedestrian lamps are the one asset
photographed from life).
An **ENDLESS shift**: the dispatcher issues one order at a time and refills
after every delivery; the shift ends on the simulated clock or at the turn
budget (60 turns in training, 100 in validation), whichever first. The courier sees, each turn:

- the street-view photograph(s) of up to 3 candidate streets at the junction
  (covers 93% of junctions fully; median junction has 2 streets)
- a pedestrian signal frame when the street carries one
- the phone: an SVG map rasterised to PNG, the order card, earnings so far

and acts through fenced tool calls (`navigate`, street choice by name +
direction, waypoint interaction). One THOUGHT line and one call per turn,
~80 tokens of text; the images dominate the token budget.

**Why route choice cannot be solved from text**: the street names shown are
the real ones, but which street is *walkable, faster, signalled* is only in
the photographs — this is the axis the `narration` setting moves, and the
benchmark setting keeps the facts in the images.

## 2. The reward

Two numbers exist. They must never be conflated:

| name | definition | role |
|---|---|---|
| **earnings** (`reward_basis: earnings`) | the fee the job pays: 3.00 + 0.01/metre of the order, full on time, a fraction late. Zero unless a delivery completes | **trained on and reported.** Externally defined by the task — no constant in this repo can tune it |
| `env_return` | +1.0 delivery, ±0.5 punctuality, +0.1 collection, −1.0 red-light violation | carried in `info` for diagnosis; four constants with no external justification; never optimised, never reported |

Design decisions that were *measured*, not chosen:

- **Total earnings under a fixed clock**, not earnings/hour: the rate was
  gamed by shrinking the denominator (held-out fell 2.16 → 0.70 over twenty
  steps while entropy tripled). A total under a cap has no denominator.
- **Progress shaping off** (`progress_weight: 0.0`): at 0.2 it was 96.7% of
  the gradient; training score rose to its maximum while held-out earnings
  fell 90% and deliveries went 17.2% → 1.6%. The policy learned exactly what
  it was paid for. It stays available for cold-starting a policy that never
  delivers, and it is *never* on in the validation config.
- **ENDLESS with queue_depth 1**, because GRPO needs within-group variance:
  a solo order pays the same fee to any successful route, so groups scored
  identically and `grad_norm` was 0. Under a fixed clock, two deliveries beat
  one beats none — the variance is the objective.

## 3. The splits, exactly

| set | spec | expands to | decoding |
|---|---|---|---|
| train | `seed: [0, 799, 1]`, n=800 (the experiment arms: `[0, 999]` + `[1200, 4199]`) | exactly 0–799 | T=1.0 sampled, GROUP=8 |
| **val** | `seed: [1000, 1063, 1]`, n=64 | **exactly** 1000–1063 | greedy, n=1 |
| train-probe | `seed: [0, 31, 1]`, n=32 | **exactly** seeds 0–31 (32 values, limit 1 — fixed by arithmetic) | greedy, n=1 |

- The ranges are inclusive and sized to their `n_envs`, so the loader never
  draws a subset. (It used to: `[1000, 1064]` with `n_envs 64` sampled 64 of
  65, seeded by the block's position in the file — the RL harness and the
  API evaluator scored sets that differed by one seed. Fixed in the release
  branch; numbers logged before it are on 1001–1064.)
- The same seed yields the same shift, byte-identical first observation —
  verified directly, `reset(42) == reset(42)`, `reset(42) != reset(43)`.
- Ranges are disjoint by construction (≤800 < 1000); no future edit can
  overlap them without moving a range boundary past the other.
- The 64 val seeds are **immutable**: every number this project has reported
  is on exactly these shifts.

**Why the per-step training reward cannot show learning, and what to read
instead.** A step draws 6 of the 800 training seeds; shifts differ in what
they can possibly pay (order distances differ); the H100 run's per-step means
ranged 6.3–16.8 with no trend *while held-out rose monotonically 7.205 →
8.670*. That curve is the draw, not the policy. The **train-probe** block
exists for this: the same 32 training shifts, greedily decoded, at every
validation pass — `val-core/courier_paris_trainprobe/...` is the
apples-to-apples "is training-set performance rising" curve. And the gap
between probe and val-core is the memorisation meter, which a fixed pool of
800 shifts makes worth watching.

## 4. Success, and why it read 0.0% for 110 steps

`info["success"]` had a definition — "delivered everything issued" — that is
**unsatisfiable under ENDLESS**: the dispatcher refills after every delivery,
so an open order exists at every instant and `delivered >= orders_issued` is
false by construction. The 0.0% success rate across a run whose earnings rose
20% was measuring the dispatcher, not the courier.

Now mode-aware (`vagen_courier_env.py`):

- **solo**: delivered everything issued (unchanged) — also legitimately ends
  the episode early, which is its second job in the agent loop.
- **ENDLESS**: the shift *ended* having delivered at least once — judged only
  at `done`, because `success=True` is also the early-termination gate and
  ending an endless shift at the first delivery would destroy the
  total-earnings objective.

The headline metric for ENDLESS remains **earnings**; success is a floor
("the shift was not a zero"), not the target. **Deliveries-per-shift now
reaches the metrics**: `delivered` rides `reward_extra_info` beside
`traj_success`, so validation reports the count, not only the floor.

## 5. Input/output honesty — what has been audited, what is open

Audited and holding (each has a test or a measured check):

- Same prompt builder for training and evaluation — 0 mismatches in 833 rows.
- Truncation falls on image boundaries; images dropped from ids are dropped
  from `multi_modal_data` (the mrope crash class). 69 tests.
- Prompt budget holds the system prompt *and* the whole first observation
  (4864 vs measured max 4611); response budget holds a 60-turn rich shift
  (57344 vs observed 54110 overflow at 53248).
- Training and evaluation see the same albums — the silently-missing-album
  class now raises with the resolved paths in the message.
- The oracle (banner-following scripted policy) delivers 24/24 — the task is
  solvable from what the observations show.
- **296 dumped trajectories audited end to end** (7 training steps + 2
  validations of the A5000 soak): 98.6% of 13,855 assistant turns are exactly
  one well-formed action; 21 genuine env rejections total; action verbs span
  the whole interface (10.8k `walk_to`, 1.4k `navigate`, 558 `collect`, 431
  `hand_over`, 256 `wait` — and the waits occur in trajectories that saw a
  pedestrian light); photograph captions match `<image>` counts; notes
  accumulate and deadlines count down correctly turn over turn. The two
  malformed tails found were the old RESP_LEN clip landing mid-image in the
  *dump text* — the training ids are protected by the truncation patches, and
  the current 57344 budget makes the clip itself vanish.
- The API evaluator (`embodiedbench.eval`) drives the same `CourierGymEnv`
  the trainer does, so its observations are the trainer's observations by
  construction; infrastructure-ended episodes are excluded from its means.

**⚠ open — ranked by how much they threaten the "first open-source embodied
benchmark" claim:**

1. **The headline number is one city.** The standard validation column is
   `citycore-paris` shifts, so held-out there means *unseen shifts*, not
   *unseen streets*. Nine procedural cities are now baked and the
   multi-city arms (`experiments/README.md`, RQ3) validate on
   `large-city-30`, which no arm trains on — that column is the transfer
   number. Provenance note for release: frames are renders of the CityCore
   UE scene; the licence of that asset (Marketplace terms for rendered
   output) must be confirmed before frames ship publicly.
2. **Entropy telemetry is not understood** (watch item from the H100 run:
   0.044 → 9.602 across 110 steps, near ln|V|, while val rose — those two
   facts do not cohere; either the metric's aggregation changed or the
   sampled policy is far more random than its greedy self). Do not tune
   `ENTROPY_COEF` until the metric itself is explained.
3. ~~Image budget vs red-light charging~~ — **done, verified**: the charge
   gates on the album's certification AND on served size (`image_max_side` →
   `served_long_edge`; lamp area scales quadratically with resize; below
   4 px² nothing is charged; 49/49 legible approaches carry `lamp_px`
   metadata). Pinned by
   `test_a_lamp_too_small_at_the_served_size_is_not_charged`.
4. **Anti-lost affordances** (task #15) — the four-piece kit is pending;
   long shifts still lose some policies to wandering, which depresses
   earnings variance late in shifts.
5. **success under ENDLESS is a floor, not a target** — fine for RL, but a
   leaderboard would want deliveries-per-shift or earnings percentiles as
   the headline. Decide before publishing.

## 6. The adaptive curriculum (opt-in; off is the benchmark, exactly)

`adaptive: true` in a training yaml — and only there; nothing in the
process environment can switch it on — closes a loop: episodes log a
failure taxonomy (delivered/late/expired, red crossings, detour ratio) →
`tools/adaptive_profile.py` distils the **validation** lines (seeds in
`[1000, 1200)`, the window every training pool jumps over) into sampling
weights (late→`len_long`, red→`signalled`, detour→`junction_dense`,
all-zero→`len_short`) → the dispatcher multiplies order-acceptance
probabilities by them, floored at 0.15 so the support never narrows, budget
widened by the same factor so shifts never come up short. Weights are
`1 + 2·rate`, so the curriculum anneals itself away as the failures do. The
diagnosis comes from held-out greedy evaluation, never training reward, so
the knob the optimiser could game is not the knob that steers -- this is the
PLR/ADR family with an offline signal.

With the switch off the dispatcher takes its pre-existing branch and consumes
the RNG identically; `tests/test_adaptive_env.py` pins byte-identical shifts,
distribution shift under bias, support preservation, and the builder's
mapping. The experiment suite comparing it against control is
`experiments/README.md`.

## 7. Multi-city

Env blocks each carry their own `map_dir` and `album_root`, so "train on
these cities, validate on that one, identical settings" is a yaml edit.
Ten cities are baked (`citycore-paris` and nine procedural ones; the album
layout is in `docs/INSTALL.md`); `experiments/train_m8_maps.yaml` rotates
eight of them and `val_courier_rq3v2.yaml` judges on the tenth.

## 8. Protocol for changing any of this

The val seeds, the val config, and the fee definition are load-bearing: every
comparison across runs assumes them. The rule the yaml comments already
carry, restated once: **change the training set freely, never the val block;
a new question gets a new `data_source` name, not an edit to an old one.**
