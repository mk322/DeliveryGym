# The experiment arms

Every arm is the base recipe — `examples/train_single_node.sh`, GRPO on
Qwen3-VL-4B, earnings as the reward — plus one training yaml and, for some,
a validation yaml or the curriculum sidecar. Nothing else differs between
arms; that is what makes them comparable. Launch one by name:

```bash
MODEL_PATH=... ALBUMS_DIR=... bash experiments/run_arm.sh base2
bash experiments/run_arm.sh --list          # every arm and its settings
```

The arm table lives in `experiments/run_arm.sh` and is the same table the
reported runs were launched from. All arms run with `ENTROPY_COEF=0`
(the entropy bonus is off; see `docs/TASK_DESIGN.md` §5 for why it is not
tuned) and 1000 steps unless `TOTAL_STEPS` says otherwise.

## What each arm asks

| RQ | arm(s) | the question | judged on |
|---|---|---|---|
| — | `base2` | the control every arm is read against: earnings, no shaping, no curriculum | `val_courier.yaml` (64 Paris shifts, seeds 1000–1063) |
| **RQ1** reward | `r1-red`, `r1-all`, `r1-shape` | does a fine for crossing on red / walking into a barrier / tiny progress shaping change what the policy learns to do, and at what cost in earnings? | same |
| **RQ2** evolving env, one city | `a-fail`, `a-frontier`, `a-hard`, `a-ladder` | four curriculum signals through one dispatcher: failure-targeted, GRPO-group-variance ("frontier", the learnability signal of the PLR family), failure at double dose, and a self-paced ladder (the any-schedule control) | same |
| **RQ2** evolving env, eight cities | `m8u` (static control), `m8c` (city learnability), `m8l` (small-to-large ladder), `m8n` (upweight the poorest city), `m8x` (`m8c` at 0.6 mix instead of 0.3) | where the curriculum has leverage: the dispatcher chooses **which city** a shift is in. `m8n` is the naive heuristic learnability has to beat; `m8x` is the dose-response check | `val_courier_rq3v2.yaml`: unseen **large-city-30** plus the Paris column |
| **RQ3** map scaling | `m1v2`, `m2v2`, `m4v2`, `m8u` | 1/2/4/8 training cities, identical recipe: does breadth buy transfer to a city none of them saw? | same |
| **RQ4** task scaling | `t60`, `t200`, `t600` | pools of 60/200/600 seeds: how much does repetition hurt? (at 6 seeds/step the pools repeat 10×/3×/1× in 100 steps) | `val_courier.yaml` |
| **RQ5** constraints | `c1` (fee jitter + food temperature + notes), `c2` (+ phone battery), `c3` (+ power bank), `c3f` (`c3` with the skill-gated curriculum), `q2` / `f-cat` (queue depth 2 without / with food categories), `d-tight` (tight deadlines) | which constraints a policy actually learns to comply with, and how reward shapes behaviour once compliance is priced | `val_courier_constraints.yaml`, `val_courier_recharge.yaml`, `val_courier_categories.yaml` — each begins with the untouched standard block so cross-arm comparison holds |
| **RQ-V** vision | `v-route`, `v-all` | the narration ladder: the route in text, then everything in text. The gap to `base2` on the standard column is what reading the pictures is worth, in dollars | `val_courier_v_route.yaml`, `val_courier_v_all.yaml` |

## Seeds, and why the pools are 4000

Seeds are consumed at `TRAIN_BS` per step — `GROUP=8` samples the *same*
seed, so a group costs one. The standard pool holds 4000 seeds
(`[0, 999]` + `[1200, 4199]`, jumping over the validation window
`[1000, 1200)`), so 1000 steps at `TRAIN_BS=6` see every seed at most twice.
Every arm uses the same pool; arms differ in reward or curriculum, never in
data. The `t*` arms are the exception on purpose.

**Validation is frozen.** Every validation block names its seeds exactly
(`seed: [1000, 1063, 1]` with `n_envs: 64` — the range is inclusive and sized
to the block, so no subset is ever drawn). The rule: change a training yaml
freely; never edit a validation block — a new question gets a new
`data_source` name.

## The curriculum sidecar

Adaptive arms carry `adaptive: true` in their training yaml (`*_cur.yaml`,
`train_e4b_gentle.yaml`, `train_a_hard.yaml`) and run
`tools/adaptive_profile.py --mode $ADAPTIVE_MODE` beside the trainer
(`examples/train_adaptive_city.sh` starts it). The sidecar reads the episode
log, keeps only **validation** episodes (seeds in `[1000, 1200)`) for the
diagnosis, and writes `adaptive_profile.json` — order-class weights (`bias`)
and/or `city_weights` — which the dispatcher re-reads on change. Modes:

| mode | signal | what it steers |
|---|---|---|
| `fail` | held-out failure taxonomy (late → long orders, red crossings → signalled, detours → dense junctions) | order-class bias, `1 + 2·rate`, floored at 0.15 |
| `frontier` | GRPO group variance per order class (learnability) | order-class bias |
| `hard` | `fail` at double dose | order-class bias |
| `ladder` | fixed self-paced schedule | order-class bias |
| `city` | group variance per city | `city_weights` |
| `city_need` | lowest held-out earnings per city | `city_weights` |
| `city_ladder` | small-to-large, unlocked by held-out delivery rate | `city_weights` |
| `skill_battery` | phone-alive rate below 0.90 → lean into long/signalled shifts | order-class bias |

The curriculum freezes when held-out earnings fall to 70 % of their peak
(training earnings for the order-class modes, held-out for the city modes,
which lower training earnings by design). Every decision is replayable: the
same seed draws the same shift, city choice included, and the profile's
history is in `$EXPERIMENT_DIR/adaptive_profile.log`.

## Reading a run

The held-out curve is `val-core/courier_paris_val/earnings_at_100` (the
mean earnings of 64 greedy shifts at turn 100), logged every `TEST_FREQ`
steps. Per-step training reward is the draw, not the policy: six seeds a
step differ in what they can pay. Compare arms on the held-out column, and
compare two checkpoints seed-by-seed with `embodiedbench-eval compare`.
