# DeliveryBench — Protocol (human overview)

A short, high-level guide to how DeliveryBench works now, with the **changes
from the previous version** called out. For the full contract see
`PROTOCOL_AGENT.md`.

---

## The task

A courier earns money by delivering food in a simulated city within a 2-hour
shift. Each order: **view → accept → walk to the restaurant → pick up → walk to
the customer → drop off**. Score = profit (earnings − $100 start), usually
reported as **profit per simulated hour**; an episode is a "success" once at
least one delivery completes.

The episode ends at the 2-hour sim-time limit (not on its own otherwise). The
agent sees, each step, a first-person photo and a top-down Google-Maps-style
city map, plus a text state block (where it is, adjacent waypoints, active
orders, nearby POIs).

---

## What changed vs the previous version

### 1. Movement is **direction-based & first-person** (the big one)
- You walk with **`MOVE(direction="forward"|"left"|"right"|"backward")`**,
  relative to the way you are **facing**. Moving turns you: right → you now face
  90° to the right, backward → you turn around, left → 90° left.
- You are always on the road graph: each `MOVE` goes to the next
  intersection/door in that direction. The **`### orientation`** block says which
  of the four directions has a road right now ("forward → 108 River Ave…",
  others "(blocked)"); trying a blocked direction just fails with a clear message.
- The **first-person image** is now a single panel showing all four views at
  once — **FRONT** (centre, biggest), **BACK** (top), **LEFT/RIGHT** (sides),
  labelled. FRONT is where `MOVE("forward")` goes. The top-down map draws your
  position with a **little arrow for your facing** (North is up) so you don't get
  lost. You infer which way to go from these — like a person reading the street.
- *(An earlier iteration used `STEP_TO("<waypoint_id>")`; both that and the old
  coordinate `MOVE(x, y)` are now removed — movement is purely direction-based.)*

### 2. The **navigation tool** gives a map + estimate + live next move
- **`NAVIGATE(target="restaurant 1", mode="walk")`** is a *query-only* tool: it
  draws the route on the city-map image **and reports the travel time / energy /
  battery / (bus) fare estimate** — without moving you.
- It **does not spell out waypoint ids or a full route chain**, but it now keeps
  a live `next_move:` hint in the text observation. The hint is one action-shaped
  instruction: `move forward`, `move backward`, `turn left`, `turn right`, or
  `you have arrived`. This is recomputed after every move from your current
  waypoint and facing.
- The **transport mode is a parameter** (`walk` default / `e-scooter` / `bus`);
  the stage only offers modes it has unlocked. This one tool replaced the old six
  (`NAVIGATE_WALK`/`_ESCOOTER`/`_BUS` + `VISUAL_*`).
- The drawn route + a green **source** dot and red **destination** pin **stay on
  the map (under your blue dot) on every later step**. On arrival the route line
  disappears, but the navigation text remains long enough to say
  `next_move: you have arrived`.
- For accepted orders, address targets are resolved against the order's actual
  pickup/dropoff node first, before the global map address book. So
  `NAVIGATE(target="200 Cherry St")` means the active order endpoint when that
  address appears in `active_orders`.

### 3. Curriculum stages
Difficulty is dialled in via feature flags:
- **Stage 1** — walk only, one restaurant + one customer, single-item orders,
  no battery/energy/temperature/bag mechanics. Actions: view/accept/`MOVE`/
  pickup/drop-off/wait + `NAVIGATE` (walk mode only here).
- **Stage 2** — adds battery + walking-energy + e-scooter (charging, resting,
  buying packs, switching modes, scooter navigation). *(Bag compartments are
  intentionally left for Stage 3: with no food-temperature/fragility yet, the
  bag has no effect, so it would just be busywork here.)*
- **Stage 3** — the full game: insulated bag, food temperature/fragility,
  delivery methods, buses & car rental, and multi-agent help/coordination.

### 4. Clearer order display & dock-based pickup
- The state line shows **`Active orders: #0`** / **`Carrying: #0`** (with a `#`)
  so an order whose id is `0` isn't misread as "zero orders."
- **PICKUP / DROP_OFF now succeed once you're standing on the order's dock** (its
  access point on the road). Previously the target was the building *door*, which
  for big buildings could sit farther than the 6 m tolerance from any reachable
  spot — so some orders were impossible to pick up. Fixed.

### 5. Multi-order pools and feasibility filtering
- Multi-order support was already in the environment: `VIEW_ORDERS()` shows the
  current pool, `ACCEPT_ORDER([id1, id2, ...])` can accept several, and the pool
  refills after accepted orders leave it. The new multi-order configs make that
  choice real by increasing the pool and removing the old one-restaurant /
  one-customer restriction.
- New order-pool filter flags can make the visible pool all feasible or mixed:

```
enable_feasible_orders: true
enable_infeasible_orders: false
feasible_order_step_budget: 20
feasible_order_non_move_actions: 5
```

This is the recommended **feasible-only** setting. The mixed setting is
`enable_feasible_orders: true` and `enable_infeasible_orders: true`. The filter
runs when the agent calls `VIEW_ORDERS()`, using the agent's current position,
not just the original spawn.
- The oracle logic is shared with
  `python -m vagen.envs.deliverybench.tools.analyze_order_feasibility`. On
  `small-city-11`, from spawn `[-17.0, 256.58]`, a full 25-step episode budget
  gives `88 / 352` raw candidate orders that are structurally deliverable.

### 6. Housekeeping
- The six navigation tools were **merged into one `NAVIGATE(target, mode)`**
  (map image + cost estimate; mode is a parameter; no turn-by-turn chain).
- Removed actions: the old id-based `STEP_TO`, the coordinate `MOVE(x, y)`,
  `STEP_FORWARD`, `TURN_AROUND`, and unused internal nav helpers — so the action
  list shown to the model matches what actually runs.

`hand_to_customer` (Stage 3) also works from the dock now: the first DROP_OFF
there knocks, the customer takes ~30 s to come out, so you `WAIT` then DROP_OFF
again to hand it over. (This come-out delay didn't exist before — it's new.)

---

## Mental model for one delivery (Stage 1)

```
VIEW_ORDERS  ->  ACCEPT_ORDER(0)
NAVIGATE(target="146 Church Ave")        # use the exact Pickup address from active_orders
MOVE("backward") -> MOVE("forward") -> ...   # follow next_move until "you have arrived"
PICKUP(orders=[0])                       # once standing on the restaurant's dock
NAVIGATE(target="108 Sycamore St")       # use the exact Dropoff address from active_orders
MOVE(...) -> MOVE(...) -> ...
DROP_OFF(oid=0)                          # once on the customer's dock → payout, repeat
```

Because each leg is several `MOVE`s, a delivery takes more decision steps
than the old single-jump flow — but it exercises real navigation and the road
network, which is the point.

---

## Running it

- Reference rollout: `OPENROUTER_API_KEY=… python -m vagen.envs.deliverybench.rollout_qwen`
  (or `BACKEND=vllm …`, or `BACKEND=sglang …` against a local server). Outputs
  land in `outputs/rollout_<timestamp>/`; replay them with
  `python scripts/view_rollout.py <…>/log.jsonl`.
- Debug interactively: `python -m vagen.envs.deliverybench.cli`.

### Local serving notes (Qwen3.5-VL-9B via sglang) — updated 2026-06-17
- **Two environments, one HTTP link.** The *model server* runs in its own env
  (here `qwen_env`, sglang) on the GPU; the *rollout client* runs in the `vagen`
  env (it owns DeliveryBench + the renderer, and needs `PyQt5` and `pyqtgraph`).
  The client makes API calls only — it never touches the GPU. So you can't run
  the rollout from the server env, nor the server from `vagen`.
- **Current server.** The active server is `qwen3.5-9b` at
  `http://127.0.0.1:30002/v1`, in tmux session `qwen35_sglang_gpu2`, launched by
  `scripts/start_sglang_qwen35_9b_gpu2.sh`.
- **Old-driver workarounds.** On our current box the GPU driver is older than the
  server env's CUDA build, so the server is launched with two extra flags
  (`FLASHINFER_USE_CUDA_NORM=1`, `CUDA_HOME=<env>`), and JSON output is enforced
  by the prompt instead of a decoding grammar (the grammar path crashes that
  stack). See `start_sglang.sh` and PROTOCOL_AGENT §15 — a machine with an
  up-to-date driver needs none of this.
- **Context window.** With images on, the rollout aggressively compresses its own
  history to stay inside the server's token window. Current visual configs use
  `context_token_threshold: 26000`, `image_turns_kept: 1`, and
  `tokens_per_image: 3000` against the 32768-token server limit.


## 14. Constraint & action matrix (per stage)

Legend: ✓ = active/available, · = off/unavailable. S1 = Stage 1, S2 = Stage 2,
S3 = Stage 3 (full).

### 14a. Constraints — what each tests and how to satisfy it

| Constraint (config) | S1 | S2 | S3 | Capacity tested | How the agent satisfies it |
|---|:--:|:--:|:--:|---|---|
| 2-hour shift (`time_limit_hours`) | ✓ | ✓ | ✓ | Global time budgeting | Maximise deliveries per sim-hour; don't spend steps on infeasible orders. |
| Per-order deadline (`Time Left`) | ✓ | ✓ | ✓ | Time estimation & prioritisation | Accept only orders you can pick up **and** drop off before the deadline (`NAVIGATE` reports the time estimate); deliver before it hits 0. |
| Direction movement + facing | ✓ | ✓ | ✓ | Egocentric spatial reasoning | `MOVE(forward/left/right/backward)` relative to facing; read the 4-way views + map arrow + `### orientation` to head toward the `NAVIGATE` route. |
| Single item / 1 restaurant+customer / pool≤3 (`require_single_item`, `num_*`, `max_orders_in_pool`) | ✓ | · | · | The bare delivery loop | Just learn view → accept → pickup → dropoff. |
| Multi-item orders | · | ✓ | ✓ | Carrying/packing | Carry all of an order's items; in S3 place each in a suitable compartment. |
| Earning jitter (`enable_earning_jitter`) | · | ✓ | ✓ | Value estimation under noise | Judge expected payout vs travel cost; don't over-fit to exact dollar amounts. |
| Food prep time (`enable_prep_time`) | · | ✓ | ✓ | Scheduling | If food isn't ready, do other useful work or `WAIT`; don't idle at the door. |
| Walking energy (`enable_walking_energy`) | · | ✓ | ✓ | Stamina budgeting | Keep energy > 0: `REST` at a rest_area, or `BUY`+`USE_ENERGY_DRINK`. Energy 0 → hospitalised (money + long idle). |
| E-scooter battery (`enable_battery`) | · | ✓ | ✓ | Resource budgeting + mode use | Ride for speed but watch battery: `CHARGE` at a charging_station, `USE_BATTERY_PACK`, or `SWITCH(to="walk")` before depletion (depleted → slow towing). |
| Food temperature (`enable_food_temperature`) | · | · | ✓ | Quality optimisation | Minimise transit time; keep hot/cold items in the right compartment; `USE_ICE_PACK`/`USE_HEAT_PACK` to hold temperature. |
| Food fragility (`enable_food_fragility`) | · | · | ✓ | Speed-vs-quality tradeoff | Don't use `pace="accel"` with fragile items — speed damages them and lowers the rating. |
| Food smell/odor (`enable_food_smell`) | · | · | ✓ | Packing | Separate strong-smelling items into different compartments. |
| Bag compartments (`enable_bag_compartments`) | · | · | ✓ | Structured packing | After `PICKUP`, `PLACE_FOOD_IN_BAG` to assign each item to a compartment meeting its temp/smell needs. |
| Delivery methods (`enable_delivery_methods`) | · | ✓ | ✓ | Instruction following | `DROP_OFF` with the required method (`leave_at_door`/`knock`/`call`/`hand_to_customer`). |
| Special notes (`enable_special_notes`) | · | · | ✓ | Constraint satisfaction | Read the order note and obey its delivery-method requirement. |
| Advanced transport — car/bus (`enable_advanced_transport`) | · | · | ✓ | Mode selection under cost | `RENT_CAR` for speed ($/min) or `BOARD_BUS` ($1) when faster/cheaper than walking/scooter for a leg. |
| Multi-agent help (`enable_multi_agent`) | · | · | ✓ | Coordination / comms | Only meaningful with ≥2 agents: `VIEW_HELP_BOARD`/`ACCEPT_HELP` for bounties, `POST_HELP`+`PLACE_TEMP_BOX` when you need help, `SAY` to coordinate. |

Always-on baseline (every stage): walk-start, the 2-hour clock, per-order
deadlines, the waypoint graph, and the reward = Δearnings / success = ≥1
delivery contract.

### 14b. Action space (state-changing) — availability by stage

| Action | S1 | S2 | S3 | Purpose |
|---|:--:|:--:|:--:|---|
| `VIEW_ORDERS` / `ACCEPT_ORDER` | ✓ | ✓ | ✓ | See the pool / accept (one or a list). |
| `MOVE(direction=...)` | ✓ | ✓ | ✓ | Step forward/left/right/backward of your facing; turns you. The only locomotion. |
| `PICKUP` / `DROP_OFF` | ✓ | ✓ | ✓ | Collect ready food / deliver — while standing on the order's dock. |
| `WAIT` | ✓ | ✓ | ✓ | Pass time (prep/charge) — last resort. |
| `CHARGE` / `REST` | · | ✓ | ✓ | Recharge scooter / restore energy at a POI. |
| `BUY` / `USE_ENERGY_DRINK` / `USE_BATTERY_PACK` | · | ✓ | ✓ | Buy at a store; consume for energy/battery. |
| `SWITCH` | · | ✓ | ✓ | Change transport mode (walk ↔ e-scooter[/car]). |
| `PLACE_FOOD_IN_BAG` / `VIEW_BAG` | · | · | ✓ | Arrange/inspect the insulated bag. |
| `USE_ICE_PACK` / `USE_HEAT_PACK` | · | · | ✓ | Hold a compartment's temperature. |
| `RENT_CAR` / `RETURN_CAR` / `BOARD_BUS` / `VIEW_BUS_SCHEDULE` | · | · | ✓ | Car & bus transport. |
| `VIEW_HELP_BOARD` / `POST_HELP` / `ACCEPT_HELP` / `EDIT_HELP` / `PLACE_TEMP_BOX` / `TAKE_FROM_TEMP_BOX` / `REPORT_HELP_FINISHED` / `SAY` | · | · | ✓ | Multi-agent help & coordination. |

### 14c. Tools (query-only) — availability by stage

One tool; the transport mode is a parameter, gated by stage flags.

| Tool | S1 | S2 | S3 | `mode=` options by stage | Purpose |
|---|:--:|:--:|:--:|---|---|
| `NAVIGATE(target, mode)` | ✓ | ✓ | ✓ | S1: `walk` · S2: `walk`,`e-scooter` · S3: `walk`,`e-scooter`,`bus` | Draw the route on the map, report time/energy/battery/(bus) fare estimate, and keep a live `next_move` hint. No waypoint-id chain. |

---

## 15. Is the environment / curriculum well-defined?

**Core environment: yes, well-defined.** The gym contract (reset/step/
system_prompt/close), reward (Δearnings) and success (≥1 delivery), done
conditions (2-h limit / failed-action repeat-guard), units (cm→m), action
parsing, the waypoint graph + direction-based `MOVE` with persistent facing, the
query-only `NAVIGATE` tool, the `info["action_error"]` error contract, and
per-action flag gating are all
consistent and verified. The advanced mechanics (energy, battery, food quality,
transport, settlement payout) are real and wired (§13).

**Curriculum: Stage 1 is clean; Stages 2–3 are coarse and bundle too many axes
per step.**

- **Stage 1 — well-defined.** A minimal, single-axis stage: the pure delivery
  loop on foot with loosened deadlines (×1.5). Good starting curriculum.
- **Stage 2 — under-isolated.** It is meant to add "energy + battery +
  e-scooter," but because it leaves the order-shaping fields at their defaults
  it *also* flips, in one step: `require_single_item` F (multi-item),
  `num_restaurants`/`num_customers` → unbounded (whole map), `enable_prep_time`
  T, `enable_earning_jitter` T, `max_orders_in_pool` → unbounded, and
  `deadline_multiplier` 1.5 → 1.0 (tighter). So a single curriculum step changes
  ~6 difficulty axes at once. **Recommended:** carry Stage 1's order-shaping
  (1 restaurant/customer, single item, pool 3, jitter off, prep off, deadline
  ×1.5) into Stage 2 so it isolates resource management, then ramp order
  complexity in a later step.
- **Stage 3 — a single giant jump.** Bare defaults turn on temperature + smell +
  fragility + bag + delivery-methods + special-notes + car/bus + multi-agent all
  at once, plus full-map orders and e-scooter spawn. Two specific concerns:
  (a) `enable_multi_agent=True` adds empty help-board / pickables sections in a
  single-agent rollout (pointless unless actually running multiple agents).
  **Recommended:** split Stage 3 into 2–3
  sub-stages (food-quality+bag → delivery-methods+notes → advanced-transport,
  and gate multi-agent only for genuinely multi-agent runs).
- **Cross-cutting (deadline pricing).** Order deadlines are priced on the
  pickup→dropoff distance only; the approach leg (current position → pickup) is
  unbudgeted. Stage 1's ×1.5 absorbs this, but Stages 2–3 use ×1.0, so with the
  unbudgeted approach leg deadlines can be quite tight on foot — worth a
  feasibility check (or a ≥1.2 multiplier) when tuning those stages.

---

## 16. Visual curriculum search with Qwen3-VL-8B (2026-06-15)

**What we're doing.** Running many *visual* DeliveryBench rollouts with
Qwen3-VL-8B (served locally via SGLang) to find which environment settings make
the task feel trivial / easy / medium / hard for the model — the empirical basis
for a future curriculum. The model's own settings are kept fixed; difficulty is
meant to come from the environment, not from handicapping the model.

**What was broken (and made early results misleading).** Our first "trivial"
calibration run (band **B0**: loosened deadlines, one restaurant, one customer,
a single order) returned **0 deliveries**. Taken at face value that looks like
the model failing the easiest possible task. It wasn't — three separate problems
were stacked underneath:

1. **Answer format.** The model often wrote moves without quotes
   (`STEP_TO(int_10)`) or with JavaScript-style `text=true`; the parser rejected
   both. The broken `text=true` was the worst, because it silently disabled the
   navigation tool, so the agent never received turn-by-turn directions and
   wandered.
2. **Navigation usage.** Without working directions the agent looped over the
   same intersections and never reached the restaurant.
3. **Pickup reachability (an environment bug).** This was the real blocker. An
   order's pickup/drop-off point is a *building door*, which sits a few metres off
   the street where the agent can actually stand. The "you're at the door"
   tolerance was 6 m, but a survey of the map showed doors are **up to ~8 m** from
   the nearest reachable spot — so **~71% of orders were impossible to pick up or
   drop off for anyone**, model or human, given the Stage-1 moveset.

**Why B0's 0-delivery result was misleading.** It conflated three failures —
formatting, navigation, and a genuine map/tolerance bug — and attributed all of
them to the model. Most of the loss was the environment being unwinnable, not the
model being weak.

**What we fixed and why the first delivery matters.** In the rollout harness we
added tolerant input handling (auto-quoting waypoints, accepting `text=true`) —
purely forgiving the model's formatting slips, not telling it what to do. In the
environment we raised the door arrival tolerance to **10 m** (covers 100% of the
map's doors with margin) — the agent still has to navigate to the nearest street
point, so this fixes feasibility without trivialising navigation. With those in
place, the same B0 config produced its **first successful delivery** (`DROP_OFF`
at step 36, +$7.97). That's the milestone: it proves the whole loop — see →
accept → navigate → pick up → drop off → get paid — now works end to end with
visual input, so any remaining difficulty is real task difficulty we can measure.

**Prompt policy (fixed per experiment).** Older B0 scans used
`Scaffolded Task-Navigation Prompt v1`, restored from source history because it was
the first version to complete B0 (run `B0_optA`: `DROP_OFF` at step 36, +$7.97).
The current Qwen3.5-VL-9B visual runs use
`Scaffolded Task-Navigation Prompt v2.1`. It keeps the same workflow scaffold but
adds stronger navigation discipline: once an order's Pickup/Dropoff address is
known, call `NAVIGATE` with the actual address string from `active_orders` before
moving; follow `next_move` exactly until `you have arrived`; then call
`PICKUP`/`DROP_OFF`. It also explicitly warns not to copy placeholder text like
`<exact Pickup address>`.

We tried stripping this scaffold down and it **failed B0 every time** — those
minimal variants are ablations, not the operational prompt:
- *Minimal Interface* (interface-only): the model picked an order up but **never
  attempted a drop-off**, as if picking up were the whole job. 0 deliveries.
- *Minimal Task-Interface v1.1* (added the delivery definition): the model then
  headed for the customer after pickup but got **cut off by the repeat-guard**
  (three identical no-progress moves) before arriving. 0 deliveries.
- *Protocol-Aware v1* (added the env's hard protocol facts but still no workflow
  scaffold): survived much longer but **circled by dead-reckoning**, abandoned the
  NAVIGATE tool after pickup, and never got within the drop-off tolerance. 0
  deliveries — a navigation/localization failure.

So the scaffolded prompt is the baseline and the minimal/protocol-aware prompts are
recorded as failed ablations. Crucially, **within a difficulty comparison, keep the
prompt fixed and log its version** — otherwise a changing prompt contaminates the
difficulty signal. The formatting tolerance above stays, but it is logged (raw vs
normalized action, plus a rewrite count) so we can always see how often the model's
output needed fixing.

**Where the difficulty now lives.** Prompt v2.x largely fixed the "agent never
calls NAVIGATE" failure mode. The remaining failures are mostly graph-step budget:
`MOVE` advances one waypoint edge per model turn. `e-scooter` reduces simulated
travel time and battery cost, but it does not reduce the number of decisions
needed to traverse a long route.

**Latest text-nav sweep (formerly B2; `B2 = text_nav`, 2026-06-17,
Qwen3.5-VL-9B, prompt v2.1 settings).**
Current configs use `max_steps: 25`, `context_token_threshold: 26000`,
`concurrency: 2`, one restaurant, one customer, single item, prep/jitter/food
quality/bag/multi-agent/advanced transport off.
Older notes may still use `B2`/`b2`; those are text-navigation baseline runs
where `NAVIGATE` exposes textual `next_move` guidance. Current config and run
names use the shorter `textnav_*` convention.

| Mode | Seeds | NAVIGATE called | Pickup reached | Delivered |
|---|---:|---:|---:|---:|
| walk | 80-87 | 8/8 | 5/8 | 3/8 |
| e-scooter | 90-97 | 8/8 | 6/8 | 3/8 |

Run outputs live under
`experiments/rollout_qwen35_sglang_visual/runs/`. Interpretation: the agent now
uses NAVIGATE reliably, but 25 steps is still tight for long pickup/dropoff routes
such as `200 Cherry St`; several failures reach the pickup at step 25 with no
remaining step for `PICKUP`.

**Order feasibility estimate.** The oracle estimator now lives in
`vagen/envs/deliverybench/tools/analyze_order_feasibility.py`. It counts
shortest-path waypoint moves plus fixed non-move workflow actions. The runtime
`VIEW_ORDERS()` filter uses the same estimator from the agent's current
location, with `feasible_order_step_budget: 20` as the current default knob.

---

## 17. Visual route-following training target (Qwen3.5-VL-9B)

The current research target is to train a Qwen3.5-VL-9B checkpoint that can
complete DeliveryBench deliveries by reading the rendered navigation map. This
is a VLM reasoning skill: convert the highlighted visual route and the agent's
current facing into a valid primitive action such as
`MOVE(direction="forward")`.

**Why this is ready for training.** The surrounding delivery workflow is now
mostly controlled. In visual route-following mode the agent reliably:

- views and accepts the available order;
- calls `NAVIGATE` with the pickup address;
- receives a map image with a highlighted blue route;
- no longer receives textual `next_move` guidance in the policy observation.

The remaining failure is concentrated in visual route following: the model often
chooses the wrong `MOVE(direction=...)` from the route image.

**Current visual-mode contract.**

- Keep the full delivery workflow: `VIEW_ORDERS` → `ACCEPT_ORDER` →
  `NAVIGATE(pickup)` → visual `MOVE` steps → `PICKUP` →
  `NAVIGATE(dropoff)` → visual `MOVE` steps → `DROP_OFF`.
- Do not replace this with a vacuum route-image-to-action toy task for final
  training/evaluation.
- Hide textual `next_move`, `oracle_next_move`, and full oracle route from the
  policy prompt/observation.
- Keep oracle next action in `info` for evaluation, reward shaping, debugging,
  and SFT label generation.
- Use `[pickup_hint]` and `[dropoff_hint]` as the reliable pickup/dropoff gates.
- Treat `planned_route_*` fields as stale route-plan estimates, not as live
  arrival/progress signals.

**Recent prompt and renderer cleanup.**

- Fixed the JSON output example from double braces to a normal JSON object.
- Shortened the reasoning/future-plan instructions to reduce long reflective
  responses.
- Kept the long-horizon courier objective and order-selection framing.
- Shortened the visual-mode `NAVIGATE` action description while preserving the
  workflow rules.
- Clarified that the blue line is the route, blue-line arrows may be absent on
  short/final segments, and the orange arrow only shows the agent's facing.
- Renamed stale navigation estimate fields from `initial_*` to
  `planned_route_*`.
- Improved route/map readability with compact legends, clearer active-order
  labels, route waypoint labels, and safer navigation crop margins.

**Controlled rollout comparison.**

The same visual single-order seed range `[100, 107]` has been used as a probe.
The environment settings are stable across these runs: walk mode, feasible-only
orders, single-order pool, max 25 steps, local SGLang Qwen3.5-VL-9B, grammar off
except for the failed grammar run.

| Run | Main difference | Result |
| --- | --- | --- |
| `visnav_walk_single_feasible_s100-107_25_v0` | early visual prompt | NAV 8/8, pickup 2/8, delivered 0/8 |
| `visnav_walk_single_feasible_s100-107_25_nogrammar` | grammar disabled, old prompt | NAV 8/8, pickup 2/8, delivered 0/8 |
| `visnav_walk_single_feasible_s100-107_25_blueline` | emphasized blue route line | NAV 8/8, pickup 3/8, delivered 1/8 |
| `visnav_walk_single_feasible_s100-107_25_arrowlegend_20260622_120636` | clarified blue route arrows and orange facing arrow | NAV 8/8, pickup 0/8, delivered 0/8 |
| `visnav_walk_single_feasible_s100-107_25_promptfix_plannedroute_20260622_130308` | JSON/prompt cleanup and `planned_route_*` fields | NAV 8/8, pickup 1/8, delivered 0/8 |

Latest controlled run details:

- called `NAVIGATE`: 8/8 = 100%
- arrived pickup: 1/8 = 12.5%
- delivered: 0/8 = 0%
- correct MOVE: 81/169 = 47.9%
- failed MOVE: 12
- parser no-valid: 0

**Interpretation.** The prompt and environment changes removed many non-target
failure modes, especially missing navigation calls, textual oracle leakage,
malformed JSON, and stale distance wording. They did not solve the intended
weakness. The model still struggles to interpret the route image into the right
primitive MOVE. This is the training target.

**Training direction.** Start with SFT data from realistic DeliveryBench
trajectories:

- input: policy-visible observation text plus rendered navigation image;
- label: `oracle_next_action_before_action` on oracle MOVE turns;
- target: `MOVE(direction=...)`.

Then use RL rewards for correct visual MOVE, route progress, pickup/dropoff
success, and penalties for failed MOVE or premature pickup/dropoff. Evaluation
should remain in the full delivery workflow.

---

## 18. Minimal SFT Export Implementation

The first training-pipeline component is now an oracle visual SFT-data exporter:

- `vagen/envs/deliverybench/tools/generate_visual_sft_data.py`
- `vagen/envs/deliverybench/test_visual_sft_export.py`

It generates parquet rows containing:

- `messages`: system/user/assistant training messages;
- `images`: absolute paths to pre-action rendered observations;
- `seed`, `turn_index`, `stage`, and `action` metadata.

The exporter follows the real delivery workflow rather than a toy isolated
route task:

`VIEW_ORDERS -> ACCEPT_ORDER -> NAVIGATE(pickup) -> MOVE... -> PICKUP -> NAVIGATE(dropoff) -> MOVE... -> DROP_OFF`

Key safeguards:

- pre-action images are saved directly from the policy-visible observation;
- the number of `<image>` placeholders must match the saved image count;
- textual `next_move`, `oracle_next_move`, and `oracle_next_action` are checked
  so they do not leak into policy messages;
- oracle actions remain available only as labels/debug/evaluation metadata;
- `--move-only` can create a navigation-focused subset while preserving
  realistic trajectory context;
- `--move-repeat` can upsample visual MOVE decisions.

Focused verification:

```bash
python -m vagen.envs.deliverybench.test_visual_sft_export
python -m vagen.envs.deliverybench.test_visual_route_following
python -m py_compile vagen/envs/deliverybench/tools/generate_visual_sft_data.py
```

Current training-readiness notes:

- Qwen3.5-VL LoRA SFT is not yet fully ready inside the current `vagen`
  environment because the installed Transformers stack does not correctly load
  `qwen3_5`;
- `qwen_env` can load Qwen3.5-VL model/processor classes but still needs the
  training dependencies aligned;
- actual GPU availability must be checked before Qwen3.5-VL-9B LoRA SFT;
- GRPO/RL should be deferred until after SFT data export and one LoRA SFT smoke
  run work end to end.

Minimal next experiment:

1. Generate a tiny parquet from a small seed range.
2. Smoke-train Qwen3.5-VL-9B LoRA after environment/GPU alignment.
3. Evaluate on the same full workflow visual single-order config.
4. Add RL only after SFT establishes that the model can better convert visual
   route images into `MOVE(direction=...)`.
