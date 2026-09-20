# DeliveryBench Environment — Developer Documentation

> Location: `vagen/envs/deliverybench/`
> Status: Active development. Intended to be promoted to a "formal" VAGEN
> environment alongside `alfworld` / `webshop` / `sokoban` / `frozenlake`.

This document describes how the DeliveryBench environment is wired into
VAGEN, what its underlying simulator (`vlm_delivery`) provides, and what
hooks/configuration knobs exist so future work can extend it cleanly.

> **Authoritative action/observation contract:** the live, verified contract for
> the action space, observation sections, navigation/movement wiring, error
> contract, and per-stage curriculum lives in **`PROTOCOL_AGENT.md`** (engineer
> reference) and **`PROTOCOL_HUMAN.md`** (high-level overview). When this file and
> the protocols disagree, the protocols win. This file is the wider developer
> map (wrapper internals, simulator internals, config reference, offline tools).
>
> **Movement model v2 (current):** locomotion is **direction-based**
> `MOVE(direction="forward"|"left"|"right"|"backward")` on a waypoint graph,
> relative to a persistent compass facing. Navigation is the single query-only
> tool `NAVIGATE(target, mode="walk"|"e-scooter"|"bus")`. The earlier
> coordinate `MOVE(x, y)`, the id-based `STEP_TO`, `STEP_FORWARD`/`TURN_AROUND`,
> and the six separate nav tools (`NAVIGATE_WALK/_ESCOOTER/_BUS` + `VISUAL_*`)
> have all been **deleted/merged** — do not reference them.

---

## 1. High-level picture

DeliveryBench is a **single (or multi-) agent text/vision RL environment** in
which a courier earns money by accepting food orders, picking them up at
restaurants, transporting them through a procedurally-generated city, and
dropping them at customers' addresses within a 2-hour simulated window.

```
┌──────────────────────────────────────────────────────────────────────┐
│ VAGEN training loop  (RayPPOTrainer / AgentLoopManager)              │
│                                                                      │
│   env_registry.yaml  ──► DeliveryBench (GymImageEnv)                 │
│                                  │                                   │
│                                  ▼                                   │
│              vagen/envs/deliverybench/deliverybench_env.py           │
│              (async wrapper, prompt building, reward shaping)        │
│                                  │                                   │
│                                  ▼                                   │
│            vlm_delivery/gym_like_interface/text_env.py               │
│              DeliveryBenchGymEnvText  (sync gym-like API)            │
│                                  │                                   │
│                ┌─────────────────┼────────────────┐                  │
│                ▼                 ▼                ▼                  │
│           Map (graph)   OrderManager / Bus    DeliveryMan agent      │
│           POI / roads    StoreManager          + InsulatedBag        │
│           VirtualClock   Comms (multi-agent)   + EScooter / Car      │
└──────────────────────────────────────────────────────────────────────┘
```

Two execution backends are shipped in `vlm_delivery/gym_like_interface/`:

| Class | Module | Purpose |
|---|---|---|
| `DeliveryBenchGymEnvText` | `text_env.py` | **Used by VAGEN.** Pure-Python, no UE; supports optional Qt or PIL map image rendering for vision mode. |
| `DeliveryBenchGymEnvQtRouteA` | `gym_like_interface.py` | Optional Qt + UnrealEngine backend with viewer, sim/VLM-pump timers, BlockingQueuedConnection invoker. Not used in training; useful for visual debugging. |

VAGEN talks **only** to `DeliveryBenchGymEnvText`. The Qt/UE variant is kept
for offline visualization and for the original SimWorld pipeline.

---

## 2. File layout

```
vagen/envs/deliverybench/
├── __init__.py                    # exports DeliveryBench, DeliveryBenchEnvConfig
├── deliverybench_env.py           # VAGEN-facing GymImageEnv wrapper (THIS FILE)
├── test_deliverybench.py          # smoke tests (reset/step/system_prompt)
├── test_error_in_context.py       # checks that error text reaches the agent
├── utils/
│   ├── prompt.py                  # (legacy) simple prompt templates
│   └── utils.py                   # parse_response() — JSON / <think><answer> / raw
├── maps/                          # 9 pre-generated city maps
│   ├── small-city-{11,13,15}/
│   ├── medium-city-{18,20,22}/    # default = medium-city-22
│   └── large-city-{26,28,30}/
│       └── {roads, progen_world_enriched, buildings, elements, routes}.json
├── tools/                         # offline CLI helpers (NOT used inside the env step loop)
│   ├── RENDERING.md               # ⭐ runbook: how to render a map's FPV images
│   ├── render_fpv_dataset_ue.py   # unified UE renderer: plain + traffic-light + obstacle
│   ├── render_waypoint_pedestrian_light_ue.py / render_dock_obstacle_ue.py  # per-kind reruns
│   ├── visualize_waypoints.py     # waypoint-graph debug viewer
│   └── render_gmaps.py            # Google-Maps-style scenario PNG renderer (Pillow)
└── vlm_delivery/                  # the actual simulator (copied from original repo)
    ├── base/        # Vector, Node, Edge, Road, Graph, VirtualClock, DMAction, DMActionKind, TransportMode
    ├── map/         # Map, MapObserver, MapExportor (Qt + PIL), drawing helpers
    ├── entities/    # DeliveryMan, Order/OrderManager, Store/StoreManager,
    │                # InsulatedBag, EScooter, Car, Bus/BusManager, TempBox
    ├── actions/     # one file per action handler (move.py, pick_up_food.py, navigate.py, ...)
    ├── gameplay/    # action_space (parser & spec), prompt (system prompt), comms,
    │                # help (multi-agent), settlement, run_recorder
    ├── input/       # JSON game configs (see §6)
    ├── gym_like_interface/  # text_env.py + gym_like_interface.py
    ├── utils/       # vlm_prompt (observation builder), action_runtime,
    │                # transport, trajectory_recorder, viewer, etc.
    ├── vlm/         # base_model.py (OpenAI/OpenRouter client; only used in standalone runs)
    ├── communicator/        # UE comms (only used by Qt backend)
    ├── agentgym_http_server.py     # standalone AgentGym-compatible HTTP wrapper
    └── agentgym_gateway_server.py
```

---

## 3. The VAGEN-facing API: `DeliveryBench`

`DeliveryBench` (in `deliverybench_env.py`) inherits from
`vagen/envs/gym_image_env.py::GymImageEnv` and implements the four required
async methods.

### 3.1 `__init__(env_config)`
- Accepts either a `DeliveryBenchEnvConfig` dataclass or a plain dict.
- Does **not** create the underlying simulator yet (lazy init in `reset`).
- Initialises bookkeeping fields: `total_reward`, `deliveries_completed`,
  `_prev_earnings`, and a small queue `_recent_parsed_actions` (size
  `_repeat_limit = 3`) for repetition-based early termination.

### 3.2 `async system_prompt() -> {"obs_str": str}`
- Calls `vlm_delivery.gameplay.prompt.get_system_prompt(cfg)` to render the
  English system prompt with concrete numbers (walk speed, e-scooter
  drain, car cost, charging price, etc.) pulled from
  `game_mechanics_config.json`.
- Then conditionally appends static sections to the system prompt so they
  don't have to be repeated every turn:
  - `### static_poi_directory` if `map_poi_in_system_prompt=True`
  - `### store_catalog` if `store_catalog_in_system_prompt=True` (default)
- Lines mentioning disabled features are filtered out (e.g. all
  `charging_station` references disappear when `enable_battery=False`).
  The same filtering is reapplied to every step observation in
  `_build_observation`.

### 3.3 `async reset(seed) -> (obs, info)`
1. Builds a fresh `DeliveryBenchGymEnvText` via `_create_env()` (passing
   `enable_vlm=False` because VAGEN supplies actions externally).
2. Calls the underlying `env.reset(seed)` inside `asyncio.to_thread`
   (the sim is synchronous Python).
3. Re-applies feature flags onto `dm.cfg` and on the bag's runtime flags
   so they propagate after the simulator's own defaults are loaded.
4. If `fixed_spawn_position` is set, overrides `dm.x / dm.y` (in cm).
5. Re-seeds and regenerates the order pool via `OrderManager.fill_pool`
   so that:
   - the seed actually controls which orders appear,
   - `enable_food_temperature`, `enable_prep_time` and
     `enable_earning_jitter` flags are honored on every order,
   - `max_orders_in_pool`, `num_restaurants`, `num_customers` and
     `require_single_item` are applied (these are the curriculum knobs
     used by `STAGE_1_CONFIG`).
6. Builds the first observation via `_build_observation(init_obs=True)`.
7. Returns `info = {"seed": seed, "raw_info": raw_info}`.

### 3.4 `async step(action_str) -> (obs, reward, done, info)`
1. **Parse** the model's response with `utils.utils.parse_response`,
   which accepts (in order of preference):
   - a raw JSON object `{"reasoning_and_reflection": ..., "action": "...", "future_plan": ...}` — the `free_think` format.
   - a `<think>...</think><answer>...</answer>` block — the `wm` format.
   - any line that matches a known `ACTION_NAME(...)` regex.
2. **Repetition guard**: if the same parsed action appears `_repeat_limit`
   (=3) times in a row, the episode is force-terminated with reward 0 and
   `action_error = "Terminated: repeated '<action>' 3 consecutive times"`.
3. **Execute** the parsed action through
   `DeliveryBenchGymEnvText.step(action_str)`, which:
   - calls `parse_action` to build a `DMAction`,
   - runs `dm._start_action(act)` synchronously,
   - then `dm.poll_time_events()` so virtual time-driven updates apply.
4. **Reward** is computed by `_compute_reward`:
   ```
   reward = (cur_earnings − prev_earnings)        # delta cash, can be negative
          + format_reward  (if format_correct and no engine error)
   ```
   The delta-earnings term naturally bundles delivery payouts, hospital
   fees, charging fees, item purchases, etc.
5. **Metrics** populate `info["metrics"]`:
   ```
   turn_metrics:
     action_is_valid       : format_correct AND no engine error
     action_is_effective   : no engine error
   traj_metrics:
     success               : sim time exceeded time_limit_hours (see §3.5)
     deliveries_completed  : len(dm.completed_orders)
     sim_hours             : dm.clock.now_sim()/3600
     time_limit_reached    : sim_hours >= time_limit_hours
   ```
6. `info["is_tool"]` (bool) is set on every step: `True` when the
   executed call was a query-only tool (determined by `TOOL_ACTION_KINDS`),
   `False` for primitive actions, parse failures, and exception paths.
7. Returns the next observation, reward, and done. `done = terminated or truncated` from the underlying env, **plus** the repetition early-termination above.

### 3.5 Episode termination
The env has two orthogonal stopping conditions:
- `max_steps` (default 100): caught by `DeliveryBenchGymEnvText` as
  `truncated=True` once `elapsed_steps >= max_steps`.
- `time_limit_hours` (default 2.0): when `dm.clock.now_sim()/3600` exceeds
  this, the wrapper sets `done=True` (the underlying env never terminates
  on its own — its `_is_terminated()` always returns False).
  `traj_metrics["time_limit_reached"]` reports this condition.

`success` (in `traj_metrics` and `info["success"]`) is
`deliveries_completed >= 1`. The denser performance signals are
`deliveries_completed` and cumulative `total_reward` (= earnings delta).

**Error feedback contract**: handler-level failures (PICKUP not ready /
too far, DROP_OFF not carrying, MOVE direction blocked, ...) are written to
`dm.vlm_errors` by the simulator; `build_state_observation()` deliberately
omits them. The wrapper copies them into `info["action_error"]` (and clears
`dm.vlm_errors`). **Rollout/training harnesses must relay
`info["action_error"]` to the model** (e.g. prepend to the next user
message) or the agent gets zero feedback on failed actions.

### 3.6 `async close()` — joins `env.close()` on a worker thread.

### 3.7 Observation construction (`_build_observation`)
Calls `dm.build_state_observation()` (preferred) or `dm.build_vlm_input()`
(fallback) to get a structured text block with `###` sections:

| Section | Content |
|---|---|
| `### agent_state` | id, time, transport mode, position (m), speed, pace, energy%, earnings, **`Active orders: #<id>`** / **`Carrying: #<id>`** (`#`-prefixed so id 0 ≠ a count of zero), inventory, scooter/charging status. |
| `### orientation` | (when `enable_waypoints`, default on) "You are at: `<addr>` … Facing: `<N/E/S/W>`" plus the four facing-relative `MOVE` directions (forward/left/right/backward → destination name + distance + road, or `(blocked)`). Id-free; pairs with `MOVE`. |
| `### waypoint_marks` | (when `enable_waypoint_marks`) Ephemeral numbered candidate list, re-derived every step: `- MOVE_TO(k): <waypoint id> (<name>), <dist> m, in your FRONT VIEW` (panel wording with FPV) or `… forward of you` (MOVE-direction wording without). The SAME `enumerate_candidates()` call feeds the FPV marker overlay, this text, and the `MOVE_TO` validator, so number k means one waypoint everywhere. Never phrased in compass bearings (the graph bearing axis is flipped vs the rendered map's rose). |
| `### store_catalog` | Items + prices + effect descriptions. Hidden if `store_catalog_in_system_prompt=True` (moved to system prompt). |
| `### active_orders` | One block per accepted, undelivered order via `Order.to_text()` (pickup/dropoff XY + road + addr + dock id, time-left or overtime, status, $$, special note). |
| `### accepted_help` / `### posted_help` / `### pickables` | Multi-agent help-board state (hidden if `enable_multi_agent=False`). |
| `### map_snapshot` | Text POI directory: next hops, nearest POIs by shortest-path distance, order endpoints. |
| `### ephemeral_context` | Short-lived hints (pickup-arrival, charging-done) **plus the persistent `[navigation]` block** — the `NAVIGATE` route estimate and the live `next_move:` line, which persists after a successful `NAVIGATE` until another target is chosen (on arrival it stays long enough to say `next_move: you have arrived`). |
| `### available_actions` / `### available_tools` | Per-step hint from `get_valid_actions` / `get_valid_tools`, filtered by stage flags. (`get_valid_tools` now always returns `[]`; `NAVIGATE` is a regular action gated inside the handler.) |

The clean `build_state_observation` deliberately **omits** `past_memory`,
`recent_actions`, `post_action_plan`, and `recent_error` — those belong to
the agent's own memory and are not part of the world state. Errors are
instead exposed via `info["action_error"]`.

In vision mode (`render_mode="vision"` and `enable_map_images=True`), the
per-step images are (each conditional on config, in order): an **FPV cross**
(`enable_fpv` — a single image tiling the four egocentric views around the agent
relative to its facing, FRONT centred and 2× larger; with
`enable_waypoint_marks` every one-hop waypoint is additionally drawn as a
numbered glow marker on the composed canvas — `fpv_marks.overlay_marks`), the **map** (a
Google-Maps-style frame from `tools/render_gmaps.py` with the agent dot + facing
arrow + order pins + the live `NAVIGATE` route, when `use_gmaps_renderer`; else
the legacy `MapExportor`/`MapExportorPil` global/local snapshots). They are
attached as PIL images under `obs["multi_modal_input"]["<image>"]` with one
`<image>` placeholder each at the top of the text. See `PROTOCOL_AGENT.md` §7 for
the authoritative image pipeline (FPV id↔position join, yaw calibration).

---

## 4. `DeliveryBenchEnvConfig`

All fields and their defaults (see `deliverybench_env.py`):

| Field | Default | Notes |
|---|---|---|
| `base_dir` | `<this file>/..` | Root containing `vlm_delivery/` and `maps/`. |
| `map_name` | `"medium-city-22"` | Subdir of `maps/`. |
| `max_steps` | `100` | Max env steps before truncation. |
| `render_mode` | `"text"` | `"text"` or `"vision"`. |
| `image_placeholder` | `"<image>"` | Inserted at the start of obs text in vision mode. |
| `enable_map_images` | `True` | Effective only when `render_mode="vision"`. |
| `map_renderer` | `"qt"` | `"qt"` (pixel-identical to UE viewer; needs headless PyQt5) or `"pil"` (lightweight, default in examples). |
| `prompt_format` | `"free_think"` | `"free_think"` (JSON) or `"wm"` (`<think>`/`<answer>`). |
| `format_reward` | `0.0` | Bonus added when format is correct and engine had no error. |
| `success_reward` | `1.0` | Reserved; not used by current reward. |
| `delivery_reward` | `0.5` | Reserved; the active reward is the earnings delta. |
| `use_example_in_sys_prompt` | `True` | Forwarded to prompt builder. |
| `time_scale` | `1.0` | `VirtualClock` time scale (1.0 = real-time). |
| `time_limit_hours` | `2.0` | Sim-time cap that drives `success`. |
| `map_poi_in_system_prompt` | `False` | Move the static POI directory into the system prompt. |
| `store_catalog_in_system_prompt` | `True` | Move the store catalog into the system prompt. |

**Feature flags** (each removes the matching action/keyword from the
system prompt and per-step obs, and disables the corresponding mechanic):

| Flag | Effect when disabled |
|---|---|
| `enable_battery` | Removes `CHARGE`, `USE_BATTERY_PACK`, `charging_station`, all battery/drag wording. |
| `enable_walking_energy` | Removes `REST`, `USE_ENERGY_DRINK`, `rest_area`, energy-per-meter wording. |
| `enable_food_temperature` | Hides item temperatures, disables `USE_ICE_PACK`/`USE_HEAT_PACK`. |
| `enable_food_smell` | Disables odor mixing in the bag. |
| `enable_food_fragility` | Disables motion damage accumulation. |
| `enable_bag_compartments` | Removes `PLACE_FOOD_IN_BAG`, `VIEW_BAG`. |
| `enable_advanced_transport` | Removes `RENT_CAR`, `RETURN_CAR`, `BOARD_BUS`, `VIEW_BUS_SCHEDULE`, `SWITCH`. |
| `enable_delivery_methods` | Defaults `DROP_OFF` method to `"leave_at_door"` if missing. |
| `enable_special_notes` | (Hook for filtering customer notes.) |
| `enable_multi_agent` | Removes `VIEW_HELP_BOARD`, `POST_HELP`, `ACCEPT_HELP`, `EDIT_HELP`, `PLACE_TEMP_BOX`, `TAKE_FROM_TEMP_BOX`, `REPORT_HELP_FINISHED`, `SAY`. |
| `enable_prep_time` | Orders are ready immediately at the restaurant. |
| `enable_earning_jitter` | Disables the per-order earning/time multiplier RNG. |

**Curriculum knobs:**

| Field | Purpose |
|---|---|
| `enabled_actions` | Whitelist; any action outside this list raises `"Action '<X>' is disabled in current stage config."` |
| `max_orders_in_pool` | Caps `OrderManager.capacity` and trims the pool. |
| `num_restaurants` | Keeps only orders whose pickup road appears in the first N distinct restaurants. Enforced on the initial pool **and** on mid-episode refills (the wrapper wraps `om._spawn_one_order` with a retry filter). |
| `num_customers` | Same idea for dropoffs. Also enforced on refills. |
| `require_single_item` | Trims every order's `items` list to length 1 (initial pool and refills). |
| `fixed_spawn_position` | `[x_m, y_m]` (meters); overrides the random spawn, snapped to the nearest waypoint. |
| `initial_transport_mode` | `"walk"`, `"e-scooter"`, ... or `None` (keep the simulator default, which is e-scooter). Walk-only stages should set `"walk"` so actual speed matches `NAVIGATE(mode="walk")` estimates and the walking-based deadline pricing (`AVG_SPEED_MPS=1.6` in `entities/order.py`). |
| `deadline_multiplier` | Scales every order's `time_limit_s` after spawn (initial pool and refills). Deadlines are priced on pickup→dropoff distance only — the approach leg eats into the budget, so walking stages want ~1.5. |
| `enable_waypoints` | (default `True`) Emit the per-step `### orientation` block (facing-relative forward/left/right/backward directions for `MOVE`). |
| `enable_waypoint_marks` | (default `False`) **Set-of-Marks navigation**: numbered glow markers over every one-hop waypoint in the FPV cross, a per-step `### waypoint_marks` candidate list, and the `MOVE_TO(k)` action alongside `MOVE` (spec bullet, output example, and whitelist entry are all gated on this flag). With `enable_fpv`+vision, reset fails loudly unless the FPV album covers every graph waypoint at all four yaws. Flag off ⇒ byte-identical env. Plan/history: `F1_WAYPOINT_MARKS_PLAN.md`. |
| `enable_feasible_orders` / `enable_infeasible_orders` | Gate the `VIEW_ORDERS` oracle feasibility filter. `(true,false)` = feasible-only pool; `(true,true)` = mixed; `(false,true)` = infeasible-only stress test; `(false,false)` = invalid. See `actions/view_orders.py` + `utils/order_feasibility.py`. |
| `feasible_order_step_budget` | (default 20) Max waypoint+workflow steps an order may need to count as feasible at `VIEW_ORDERS` time (from the agent's current position). |
| `feasible_order_non_move_actions` | (default 5) Fixed non-move workflow actions reserved (accept, pickup, drop-off, …) in the feasibility estimate. |
| `seed` | (currently unused; the per-episode seed comes from `reset(seed)`). |

**Pluggable hazards (default-off, vision-only — see §13):**

| Field | Default | Notes |
|---|---|---|
| `enable_obstacles` | `False` | Load per-map `obstacles.json` (written by the renderer); `MOVE("forward")` into an obstacle edge is blocked (`collisions++`), `BYPASS()` passes at cost. Raises at reset if the sidecar is missing. |
| `enable_traffic_lights` | `False` | Signalised set is built **from the FPV manifest** (`render_kind:"traffic_light"` rows; a `traffic_lights.json` sidecar overrides if present). Crossing a 4-way on red counts a violation (no penalty); `WAIT("traffic_light")` skips to the next minute. Raises at reset if neither manifest rows nor a sidecar exist. See `tools/RENDERING.md`. |
| `passby_cost_scale` | `1.5` | Time/energy multiplier for `BYPASS()`/`PASSBY()` vs a normal `MOVE("forward")`. |

**Locomotion units**: `MOVE` is **direction-based** —
`MOVE(direction="forward"|"left"|"right"|"backward")`. There are no coordinate or
waypoint-id move arguments anymore. All observations report metres; the sim works
in cm internally and snaps the agent to the waypoint graph (`fixed_spawn_position`
is given in metres `[x, y]`).

**Debugging CLI**: `python -m vagen.envs.deliverybench.cli` — interactive
REPL or scripted probes (`--actions 'VIEW_ORDERS(); ACCEPT_ORDER(0)'`),
stage presets (`--stage 1|2|3`), vision mode with image dumps
(`--render vision --fpv --out <dir>`), raw state inspection (`:state`,
`:orders`).

### 4.1 Built-in stage presets

The module exposes three curriculum configs at import time (see also
`PROTOCOL_AGENT.md` §11 / §14 for the full per-stage matrix):

| Preset | Description |
|---|---|
| `STAGE_1_CONFIG` | Minimal: walk-only, 1 restaurant, 1 customer, `max_orders_in_pool=3`, single item, `deadline_multiplier=1.5`, no battery/energy/bag/temperature/smell/fragility/multi-agent/jitter/prep, fixed spawn at `[-217.0, 270.94]`. `enabled_actions = [VIEW_ORDERS, ACCEPT_ORDER, MOVE, PICKUP, DROP_OFF, WAIT, NAVIGATE]` (`NAVIGATE` walk-mode only). |
| `STAGE_2_CONFIG` | Adds battery, walking energy, e-scooter (`CHARGE/REST/SWITCH/BUY/USE_BATTERY_PACK/USE_ENERGY_DRINK`) + `NAVIGATE` walk/e-scooter. Still walk-start. **No bag compartments** (`enable_bag_compartments=False`) — with temperature/fragility/smell off the bag has no payout effect, so it would be busywork; bag mechanics begin in Stage 3. No food temperature, no advanced transport, no multi-agent. |
| `STAGE_3_CONFIG` | All defaults on (full game, all tools, multi-agent). `enabled_actions=None` (no whitelist), so `PASSBY` is technically reachable here even though it is undocumented WIP — see §13. |

These match `scripts/train/earning_reward/train_deliverybench_stage{1,2,3}.yaml`.

---

## 5. Action space

> Authoritative, per-stage availability matrix: **`PROTOCOL_AGENT.md` §5 / §14**.
> This section is the developer-side summary.

Canonical action names (defined in
`vlm_delivery/gameplay/action_space.py::_CANON_RAW`; many aliases map to
the same `DMActionKind`). Each is matched by a `NAME(args)` parser. A single
line `NAME(args)` is required; JSON wrappers (`{"action": "NAME(...)"}`) and
```` ```fences ```` are unwrapped by `sanitize_model_text`. Which actions are
live is gated per-stage by `enabled_actions` (an action outside the list is
filtered from the spec/hint and rejected by the parser with
`"Action '<X>' is disabled in current stage config."`).

**Actions vs. tools.** Callable operations fall into two groups.
*Primitive actions* (`MOVE`, `PICKUP`, `DROP_OFF`, `BOARD_BUS`, …) mutate
simulator state — position, clock, energy, scooter battery, orders.
*Query-only tools* never mutate state; they only add to the observation.
There is exactly **one** tool — `NAVIGATE` — and the single source of truth
for which `DMActionKind` values are tools is `TOOL_ACTION_KINDS` in
`vlm_delivery/base/defs.py` (its only member is `DMActionKind.NAVIGATE`).
`info["is_tool"]` is `True` exactly for `NAVIGATE`. A fully separate Tool API
(distinct call/response format, separate step budget) is not yet implemented —
see §10.

### Locomotion — direction-based `MOVE` (Stage 1+)
- `MOVE(direction="forward"|"left"|"right"|"backward")` (`MOVE("forward")` also
  works) — step one waypoint-graph edge in that direction *relative to the
  agent's persistent compass facing* (`dm.facing_deg`, 0=N/90=E/180=S/270=W),
  then rotate the facing (forward keeps it, right +90, backward +180, left −90).
  The mapping uses `city_map.adjacents` bearings binned to the four quadrants of
  the facing; on the bundled cardinal-grid maps each direction resolves to at
  most one neighbour. A direction with no road fails with a readable error
  listing the legal directions. Charges energy/battery and advances the clock
  exactly as the old `STEP_TO` did. Handler: `actions/move.py`; the shared
  `available_moves(dm)` helper also feeds the `### orientation` observation so
  the text hint and the action always agree.
- `MOVE_TO(k)` / `MOVE_TO("dock_94")` — **only when `enable_waypoint_marks=True`**
  (Set-of-Marks navigation). Steps to ANY one-hop adjacent waypoint by its mark
  number (or waypoint id/name, case-insensitive). Candidates come from
  `actions/move.py::enumerate_candidates` — the full adjacency sorted by
  (absolute bearing, distance, id), indices 1..K, deterministic and stable per
  waypoint; the SAME list is drawn as numbered markers on the FPV cross and
  printed as `### waypoint_marks`. Facing rotates onto the chosen edge's
  bearing. Unlike `MOVE`, non-cardinal edges and same-bin Y-forks are reachable
  (real case: small-city-15 `int_41` has two bearing-90° edges — `MOVE(right)`
  can only ever bind one of them; `MOVE_TO` reaches both). An invalid mark
  fails in place with the valid-marks list; a blocked *chosen* edge fails like
  `MOVE(forward)` into an obstacle (`collisions++`) — for `MOVE_TO` the
  obstacle check covers whichever edge was chosen, not only forward. `MOVE`
  keeps working alongside. Handler: `actions/move.py::handle_move_to`; both
  verbs share `_execute_edge_step` for identical time/energy accounting.
- `PASSBY()` — **work in progress / not in any shipped stage.** Intended to let
  the agent push past a *static obstacle directly ahead* at extra cost. Today its
  handler (`actions/passby.py`) simply steps to the forward neighbour at 1.5×
  time/energy and keeps facing — i.e. it behaves like a costlier `MOVE(forward)`
  and does **not** yet detect or consume any obstacle (the simulator has no
  obstacle model). See §13 / the obstacle-model design before relying on it.

> The id-based `STEP_TO(<waypoint_id>)` and the legacy coordinate `MOVE(x, y)`,
> plus `STEP_FORWARD`/`TURN_AROUND`, are **deleted** — there is no coordinate-
> based locomotion anymore. (The `DMActionKind.MOVE_TO` name once used by the
> deleted coordinate move is reused since F1 by the marked-waypoint action
> above, which takes a mark index or waypoint id, never coordinates.)

### Order lifecycle (Stage 1+)
- `VIEW_ORDERS()` — list the order pool. Optionally filters the pool by
  structural oracle feasibility (`utils/order_feasibility.py`) from the agent's
  current position; see the `enable_feasible_orders` / `enable_infeasible_orders`
  / `feasible_order_step_budget` / `feasible_order_non_move_actions` config knobs.
- `ACCEPT_ORDER(id)` / `ACCEPT_ORDER([id, ...])` / `ACCEPT_ORDER(ids=[...])`.
- `PICKUP(orders=[ids])` — while standing on the order's pickup **dock**
  waypoint, food ready (door-arrival tol `_door_tol_cm(dm)`, default 1000cm).
- `DROP_OFF(oid, method="leave_at_door"|"knock"|"call"|"hand_to_customer")` —
  while standing on the dropoff dock (method auto = `leave_at_door` when
  delivery-methods off).
- `WAIT(minutes=N)` / `WAIT("charge_done")`.

### Resources (Stage 2+)
- `CHARGE(target_pct=100)` at a `charging_station`. Cost $0.05/%; rate 7.5%/min.
- `REST(target_pct=100)` at a `rest_area`. Restores +7.5%/min.
- `BUY(item="energy_drink"|"escooter_battery_pack"|"ice_pack"|"heat_pack", qty=N)`.
- `USE_BATTERY_PACK()` / `USE_ENERGY_DRINK()`.
- `SWITCH(to="walk"|"e-scooter"|"car"|"drag_scooter")`.
- `VIEW_BAG()` / `PLACE_FOOD_IN_BAG(bag_cmd="order 12: 1,2 -> A; 3 -> B")`.

### Stage 3 / full
- `USE_ICE_PACK(comp="A")` / `USE_HEAT_PACK(comp="B")`.
- `RENT_CAR()` / `RETURN_CAR()` (rate $1/min, 12 m/s, 0.008%/m).
- `BOARD_BUS(bus_id, target_stop_id)` / `VIEW_BUS_SCHEDULE()`. $1 flat fare.

### Multi-agent (only when `enable_multi_agent=True`)
- `VIEW_HELP_BOARD()`, `POST_HELP(kind=..., bounty=..., ttl_s=..., payload={...})`,
  `ACCEPT_HELP(req_id)`, `EDIT_HELP(req_id, ...)`.
- `PLACE_TEMP_BOX(req_id, location=(x m, y m), content={"inventory":{...}})`,
  `TAKE_FROM_TEMP_BOX(req_id)`, `REPORT_HELP_FINISHED(req_id)`.
- `SAY("text")` / `SAY(to="agent_id", text="...")`.

### Navigation — the single query-only tool

```
NAVIGATE(target="<address or waypoint>", mode="walk"|"e-scooter"|"bus")
```

`NAVIGATE` (handler `actions/navigate.py`) is **strictly read-only**: it does
not modify position, clock, energy, scooter battery, mode, orders, movement
context, or bus state. It:

- computes the route (Dijkstra on `city_map.waypoint_graph` for walk/e-scooter;
  `navigate_bus.py::compute_bus_routes` for bus);
- **draws it on the city-map image** (vision mode) via `_visual_helpers`, and
  the per-step gmaps frame keeps the route polyline + green source dot + red
  destination pin **under the agent dot** until arrival (`_nav_target_node`) or a
  new `NAVIGATE`;
- emits a `### ephemeral_context [navigation]` block with the **cost estimate**
  (distance / time / personal energy / e-scooter battery / bus wait + fare) and a
  live `next_move:` hint (`move forward` / `move backward` / `turn left` /
  `turn right` / `you have arrived`), recomputed every later step from the
  agent's current waypoint + facing.

It does **not** emit a waypoint-id chain or raw `int_`/`dock_` ids. `mode`
defaults to `"walk"`; `"e-scooter"` requires `enable_battery`, `"bus"` requires
`enable_advanced_transport` (with optional `access_mode`/`egress_mode` =
`"auto"|"walk"|"scooter"`). The system prompt lists only the modes the current
stage enables (`get_tool_spec`). Target resolution is endpoint-first for
accepted/help orders (an order's pickup/dropoff endpoint is resolved before the
global address book), so `NAVIGATE(target="<Pickup/Dropoff address>")` and
`pickup of order #0` / `dropoff of order #0` resolve to the active order.

> **Deleted/merged:** the six former tools (`NAVIGATE_WALK`/`_ESCOOTER`/`_BUS` +
> `VISUAL_*`), the old internal `NAVIGATE`/`NAVIGATE_DIRECTIONS`
> (`navigate_directions.py`), and the internal `VIEW_BUS_OPTIONS` helper are gone.
> The bus planner (`navigate_bus.py::compute_bus_routes` / `_bus_leg_nodes`) is
> the one live copy of the bus-schedule forecast, used by `actions/navigate.py`.

`action_space.action_to_text(action)` produces human-readable action summaries.

---

## 6. Simulator internals (`vlm_delivery/`)

### 6.1 Map & graphs
`map/map.py::Map` builds three graphs from `roads.json`:
- `graph_full` — pedestrian sidewalk graph.
- `graph_skel` — merged road skeleton.
- `graph_drive` — driving lanes with waypoint spacing.

POIs (restaurants, stores, customers/buildings, charging stations,
rest areas, bus stations, car rental) are imported from
`progen_world_enriched.json` as `poi_meta` records. `Map._pick_meta`
returns the nearest matching POI (used by `Order._bind_nodes_initial`).
`Map.shortest_path_nodes(a, b)` / `Map.route_xy_to_xy_mode(...)` are the
routing primitives used by `MOVE`.

Nine prebuilt maps are bundled (`small/medium/large × 3`); each map is
~50–200m on a side in real-world units (positions are stored internally
in **centimeters**).

### 6.2 Addresses & waypoint graph (v2)

On top of the three internal graphs, `Map` also exposes a **coarse,
human-readable layer** of the world built at the end of `import_pois`.
This layer is what powers the direction-based `MOVE` action and the
`### orientation` observation section (the `### waypoints` block it replaced).

#### Street naming
Roads are renamed at import time to look like real US street names. The
producer in `Map.import_roads` consumes a 40-entry pool of US street
basenames (`_STREET_NAME_POOL` in `vlm_delivery/map/map.py`: trees + civic
words such as `Maple`, `Oak`, `Main`, `Church`, ...) and pairs each with
an orientation-aware suffix — **`St` for horizontal-ish roads** and
**`Ave` for vertical-ish roads** — following the Manhattan-grid
convention. The pool is shuffled with `random.Random(crc32(map_name))`
so each scenario produces the same names on every load. With 40 names
and ≤30 roads per bundled scenario, every road gets a unique basename.

A road has **one** user-visible name; the two sidewalks share it and are
distinguished by house-number parity (see below). Internally, sidewalk
edges carry the side-tagged form (`"Maple St (left)"` / `"Maple St
(right)"`) so the skeleton / polyline / projection code still groups
them; `Map._display_road_name` strips the side suffix on every code path
that flows out to the agent. (No legacy `"… (left)"` / `"… (right)"`
address is accepted by `Map.resolve_waypoint`.)

#### Addresses (meters-along-road)
Every POI gets a deterministic street address:

1. Project the POI's `dock_node` onto its road's polyline (built by
   stitching skeleton edges with the matching `road_name`).
2. Round meters-along-road to an integer = house number.
3. Bump the parity to **even on the right** of the polyline direction,
   **odd on the left** (the standard US convention). With the side
   suffix dropped from the name, parity is the only signal of which
   sidewalk a POI is on.
4. Collisions are resolved by `+= 2` so parity is preserved.

Each `poi_meta` entry gains: `address`, `meters_along`, `road_side`,
`kind`. The POI node, dock node, and door node all get an `address`
attribute set to the same string. Sample output on `medium-city-22`
(road 2 = `Walnut St`, road 4 = `Hickory St`, road 5 = `Pine Ave`,
road 6 = `School Ave`):

```
restaurant 1       |  94 Hickory St   |  94.5m along Hickory St   on the right
restaurant 2       |  50 School Ave   |  49.8m along School Ave   on the right
store 1            |  44 Pine Ave     |  42.6m along Pine Ave     on the right
charging_station 3 |  14 Walnut St    |  13.4m along Walnut St    on the right
```

#### Waypoint graph
Two kinds of waypoints, no mid-road sampling:

- **Intersections** — every node in `graph_skel` that is incident to a
  named road. ID `int_<n>`, name = sorted, deduplicated incident road
  names joined with ` & ` (e.g. `"Elm Ave & Poplar St"`). Both sidewalks
  of the same road collapse to the same name, so T-intersections within
  a single road show that road's name once.
- **POI docks** — one per named place, sitting on the road outside the
  building (the existing `dock_node`). ID `dock_<n>`, name = the POI's
  street address.

Adjacency in `Map.waypoint_graph` collapses `graph_full` so only these
nodes survive; an edge `(u, v)` carries `{"dist_cm", "bearing_deg",
"road_name"}`. Two waypoints are adjacent iff there is a path between
them along edges of kind `{road, endcap, crosswalk, aux_perp, aux_ext}`
that passes no other waypoint. On `medium-city-22` this produces 245
waypoints (63 intersections + 182 docks) and 281 edges, one connected
component.

Helpers:

| Method | Returns |
|---|---|
| `Map.address_book[str]` → `Node` | Address / display name / waypoint id → its waypoint node. |
| `Map.resolve_waypoint(token)` | Same as above with case-insensitive fallback; returns `None` on miss. |
| `Map.adjacents(node)` | List of `{id, name, kind, poi_kind, dist_m, bearing_deg, compass, road_name, node}` for `node`'s neighbours in the waypoint graph. |
| `Map.nearest_waypoint(x_cm, y_cm)` | Closest waypoint to an arbitrary position. Used by `MOVE`/`available_moves` to compute the "current" waypoint. |

#### Per-step observation: the `### orientation` section
Enabled by default (`DeliveryBenchEnvConfig.enable_waypoints=True`).
Appears right after `### agent_state` in `vlm_build_state_obs`:

The per-step section is `### orientation` (id-free, facing-relative):

```
### orientation
You are at: 54 Elm Ave   Facing: N
You can MOVE (relative to your facing):
  forward  -> Elm Ave & Poplar St   54m   on Elm Ave
  left     -> (blocked)
  right    -> (blocked)
  backward -> 110 Elm Ave           56m   on Elm Ave  [building]
```

Orders gain `addr:` and `[dock_X]` annotations on their pickup/dropoff
lines (raw coordinates are kept for backward compatibility):

```
[Order #0]
  Pickup : (-522.59m, 221.25m) | road: Park St | addr: 60 Park St [dock_135]
  Dropoff: (-224.20m, -159.01m) | road: Pine Ave | addr: 24 Pine Ave [dock_90]
  ...
```

#### Locomotion: direction-based `MOVE`
`vlm_delivery/actions/move.py::handle_move` advances the agent along exactly one
waypoint-graph edge in the requested direction relative to the persistent compass
facing `dm.facing_deg`:

```
MOVE(direction="forward")    # step to the neighbour ahead; facing unchanged
MOVE(direction="right")      # facing +90, then step
MOVE(direction="backward")   # facing +180, then step
MOVE(direction="left")       # facing −90, then step
```

`available_moves(dm)` bins each neighbour's `bearing_deg` into the four quadrants
of the facing (exact on the cardinal-grid maps) and is shared with the
`### orientation` builder so the hint and the action always agree. On a direction
with no road the handler fails with a readable error listing the legal
directions. On a valid step the agent snaps to the target's position;
`dm.on_move_consumed(dist_cm)` and `dm.clock.advance(dist/speed)` run exactly as
the former `STEP_TO` did, so energy/battery/sim-time accounting is unchanged. The
id-based `STEP_TO` and the coordinate `MOVE(x, y)` are deleted.

(`PASSBY()` reuses `available_moves(...).forward` to step ahead at 1.5×
time/energy cost; it is undocumented WIP and not in any shipped stage — see §13.)

### 6.2 Time
`base/timer.py::VirtualClock` advances simulated time. Each action call
into the sim deterministically advances the clock by the action's
duration (movement distance / speed, charging time, etc.), so the
"2-hour limit" is enforced regardless of wall-clock.

### 6.3 Orders (`entities/order.py`)
- `OrderManager` keeps a fixed-size pool (`capacity`); each order has a
  deterministic per-order RNG seeded from `(env_seed, id)`.
- `Order` binds a random restaurant pickup and random building dropoff,
  anchored on each POI's **dock waypoint** (`dock_node`) so PICKUP/DROP_OFF
  succeed while standing on the order's dock, plans a shortest path, and prices
  by distance:
  ```
  earnings    = PAY_PER_KM (10.0) * km * earning_mult ∈ [0.75, 1.35]
  time_limit  = max(eta_s * 1.25 * time_mult ∈ [0.85, 1.25],
                    longest_prep_time + 60s)
  ```
  `enable_earning_jitter=False` collapses both multipliers to 1.0.
- Food items come from `input/food.json` with thermal/odor/fragility
  metadata; `prep_time_s` controls when an order is `Ready for pickup`.
- `Order.to_text()` renders a single order in `### active_orders`.

### 6.4 Agent (`entities/delivery_man.py`)
~750-line dataclass implementing the courier:
- Movement / energy / battery / earnings.
- Owns `EScooter`, optional `Car`, `InsulatedBag`, `inventory`.
- Holds active/help/completed orders and an action queue.
- Dispatches `DMAction`s through a `_action_handlers` table, one entry per
  `DMActionKind`, each implemented in `vlm_delivery/actions/<name>.py`.
- `build_state_observation()` (RL-friendly, no error/memory) and
  `build_vlm_input()` (legacy, full prompt incl. recent_error and memory).

### 6.5 Bag / temperature / odor
`entities/insulated_bag.py` models 4 compartments (A–D) with air
temperature exchange, odor mixing, and motion-damage tracking. Ice/heat
packs snap a compartment's air to 0°C / 60°C. Wrong-temperature delivery
reduces payout in `gameplay/settlement.py`.

### 6.6 Comms & multi-agent
`gameplay/comms.py` is a process-global hub that handles help requests,
temp boxes, chat, and pickables when `DELIVERYBENCH_MULTI_AGENT=1`. The
text env tears it down with `reset_comms()` on every reset to prevent
state leaks across episodes.

### 6.7 Input JSON configs (in `vlm_delivery/input/`)
| File | Purpose |
|---|---|
| `game_mechanics_config.json` | Speeds, energy decay, scooter battery, rest/charge rates, hospital fee, ambient temp, lifecycle (`duration_hours=2`, `vlm_call_limit=200`). |
| `experiment_config.json` | Defaults for the standalone runner (`map_name=medium-city-22`, `multi_agent=false`). |
| `food.json` | 22 food items with category (HOT/COLD/FROZEN/AMBIENT), odor, motion sensitivity, prep_time, serving_temp_c, heat_capacity. |
| `store_items.json` | 4 buyable items (energy_drink $6, escooter_battery_pack $10, ice_pack/heat_pack $3). |
| `special_notes.json` | Customer notes that constrain `DROP_OFF.method`. |
| `models.json` | Provider/model config used only when the simulator drives its own VLM (not used by VAGEN). |

### 6.8 Bus system (`entities/bus_manager.py`)

`BusManager` owns all `BusRoute` / `BusStop` / `Bus` objects for a scenario.

- `routes[route_id]` — canonical route definition, never mutated after load.
  Each `Bus` gets a `deepcopy` of its route at creation so per-bus state
  (reversed direction, progress) stays isolated from the canonical copy.
- `_birth_sim_time` — sim-time anchor for deterministic schedule forecasts.
  Set once at the top of `init_bus_system()` to `clock.now_sim()` and never
  mutated afterward. `NAVIGATE_BUS` anchors all arrival predictions on this
  value together with canonical route geometry, per-stop dwell times, and the
  full-cycle period `2 × (Σ inter-stop ride times + Σ dwells)`. Live bus
  fields (`bus.x`, `bus.y`, `bus.current_stop_index`, `bus.arrival_time`) are
  not read; the forecast is therefore deterministic and unaffected by sim
  update order.

**Dwell-time tuning.** `bus.waiting_time_s` in
`vlm_delivery/input/game_mechanics_config.json` is the per-stop dwell/wait time
applied to every `BusStop.wait_time_s`, and it enters both the per-stop arrival
offsets and the cycle period above. It is now **`5`** (reduced from `360`).
DeliveryBench is a compressed virtual-world simulation; a 360 s dwell at every
stop made bus routes unrealistically slow (a multi-stop ride accrued minutes of
dwell, and late-index stops were not reached until ~40 min into the cycle),
which produced unintuitive bus-assisted routes — e.g. the planner detouring to a
far early-index stop rather than boarding next to the source. Lowering the dwell
restores compact, sensible bus-assisted routes.

This is a **configuration-level behavior adjustment only** — planner candidate
generation, ranking, visual rendering, and the `NAVIGATE` tool interface are
unchanged. `NAVIGATE(mode="bus")` still returns **bus-assisted routes only**; no
direct-mode gate was added, so the agent is expected to compare
`NAVIGATE(mode="walk")` / `NAVIGATE(mode="e-scooter")` / `NAVIGATE(mode="bus")`
itself to decide whether the bus is worthwhile.

---

## 7. Standalone usage

```bash
# Run the env interactively with a free_think prompt format
python -m vagen.envs.deliverybench.deliverybench_env \
    --render_mode text --max_steps 50

# Run the smoke tests
python -m vagen.envs.deliverybench.test_deliverybench

# Verify error feedback path
python -m vagen.envs.deliverybench.test_error_in_context
```

For the original standalone (non-VAGEN) AgentGym-style HTTP wrapper, see
`vlm_delivery/agentgym_http_server.py` and
`vlm_delivery/agentgym_gateway_server.py`.

---

## 8. VAGEN training entry points

Registered in `vagen/configs/env_registry.yaml`:
```
DeliveryBench: vagen.envs.deliverybench.deliverybench_env.DeliveryBench
```

Example training/eval YAMLs:
- `examples/deliverybench/train_deliverybench.yaml` (n_envs=1000, max_turns=20, vision mode, PIL renderer)
- `examples/deliverybench/val_deliverybench.yaml`
- `scripts/train/earning_reward/train_deliverybench_stage{1,2,3}.yaml`
- `scripts/train/earning_reward/val_deliverybench_stage{1,2,3}.yaml`

A typical env spec inside an `envs:` list:
```yaml
- name: DeliveryBench
  n_envs: 1000
  data_source: deliverybench
  seed: [1, 1000, 1]
  max_turns: 20
  response_length_per_turn: 512
  config:
    map_name: medium-city-22
    render_mode: vision           # or text
    max_steps: 100
    prompt_format: free_think     # or wm
    format_reward: 0.1
    delivery_reward: 0.5
    enable_map_images: true
    map_renderer: pil             # or qt
```

---

## 9. Rewards & success signal — current state

| Component | Source |
|---|---|
| Per-step reward | `dm.earnings_total` delta + optional `format_reward`. |
| Negative reward | Hospital fee (energy=0), charging/buy spend, rental cost, late-delivery penalty (in `settlement.py`). |
| Episode reward | Sum of per-step rewards (≈ final earnings minus initial $100). |
| `info.success` | `True` iff `sim_hours >= time_limit_hours`. |
| `traj_metrics.deliveries_completed` | `len(dm.completed_orders)`. |

If you want a "task success" signal that rewards delivering N orders or
hitting an earnings threshold, edit `DeliveryBench._check_success` and/or
the reward function. The `success_reward` / `delivery_reward` fields are
currently reserved but **not wired** into `_compute_reward`.

---

## 10. Roadmap toward a "formal" environment

> **Movement model v2 status (current)**: locomotion is direction-based
> `MOVE(direction=...)` on the addresses + waypoint graph, with a persistent
> compass facing and FPV/`### orientation` accessibility hints. Navigation is the
> single `NAVIGATE(target, mode)` tool. The earlier coordinate `MOVE(x, y)` and
> id-based `STEP_TO` are deleted. The full game and all stage configs work.

Aspects that would need polish to bring this up to `alfworld`/`webshop`
quality:

1. **Reduce coupling to a copy of the simulator.** `vlm_delivery/` is a
   vendored snapshot; a clean release should pin its version, move it to
   a separate package (e.g. `deliverybench-sim`), and depend on it.
2. **Stable observation schema.** Document the exact `###` sections that
   are guaranteed to exist, and gate optional sections by config flag
   instead of regex stripping.
3. **Wire the unused reward knobs** (`success_reward`, `delivery_reward`)
   or remove them.
4. **Clarify success.** Today `success == True` simply means "ran the
   full 2 sim hours". Consider success = "delivered ≥ K orders" or
   "earned ≥ $X".
5. **Multi-agent.** `gameplay/comms.py` and the `*_help_*` actions are
   ready, but `DeliveryBench` currently only exposes a single
   `DeliveryMan`. Lifting the wrapper to vector environments / multiple
   couriers is the natural next step.
6. **Map procedural generation.** Currently 9 fixed JSON maps live in
   `maps/`. A formal env should ship a procedural generator with a
   documented seed → map function.
7. **Action-space pruning per stage** is already supported via
   `enabled_actions`; expose the three `STAGE_*_CONFIG` presets in
   `env_registry` so users can pick them by name.
8. **Determinism.** Cross-platform determinism of `OrderManager._rng_for`,
   `random.choice`, and `numpy` need a CI test similar to
   `test_deliverybench.py` but covering full-rollout reproducibility.
9. **Image rendering.** Both `MapExportor` (Qt) and `MapExportorPil`
   produce slightly different outputs; pick one as canonical and gate the
   other behind an opt-in flag.
10. **Separate Tool API.** Navigation tools currently use the same
    `parse_action` / `dm_start_action` pipeline as primitive actions and
    return results through the same observation channel. A fully separate
    Tool API — a distinct `"tool"` key in the model response, a dedicated
    `### tool_result` observation block, and optional out-of-step-budget
    invocation — is future work.

---

## 11. Quick reference: most-used hooks

| You want to… | Touch this |
|---|---|
| Change reward shaping | `DeliveryBench._compute_reward` |
| Change what counts as success | `DeliveryBench._check_success` |
| Add a new action | `vlm_delivery/actions/*.py` + register in `DeliveryMan._action_handlers` + add a parser branch in `gameplay/action_space.py::parse_action` + add to `_CANON_RAW`, `_FULL_ACTION_API_SPEC`, `_FULL_OUTPUT_EXAMPLES`. (The direction-based `MOVE` was wired this way — see `actions/move.py`.) |
| Change addressing scheme or waypoint graph | `vlm_delivery/map/map.py::_build_addresses`, `::_build_waypoint_graph`. |
| Disable the `### orientation` section | Set `enable_waypoints: false` on the env config. |
| Add a curriculum stage | Add a new `STAGE_X_CONFIG = DeliveryBenchEnvConfig(...)` constant. |
| Change the world | Drop a new map dir under `maps/` and pass `map_name="..."`. |
| Change game balance | Edit `vlm_delivery/input/game_mechanics_config.json` (and update the prompt template if user-facing numbers change). |
| Add a new food item | Append to `vlm_delivery/input/food.json`. |
| Add a new store item | Append to `vlm_delivery/input/store_items.json` and handle it in `BUY` / a new `USE_*` action. |
| Change observation sections | `vlm_delivery/utils/vlm_prompt.py::vlm_build_state_obs`. |
| Change the system prompt | `vlm_delivery/gameplay/prompt.py::_SYSTEM_PROMPT_TEMPLATE`. |

---

## 12. `tools/` directory

Most scripts under `tools/` are **offline** (read scenario JSON, write artifacts
to disk). The one exception is **`render_gmaps.py`**, which is now **also on the
runtime hot path**: in vision mode with `use_gmaps_renderer`, the per-step map
frame (agent dot + facing arrow + order pins + live `NAVIGATE` route) is produced
by `render_gmaps.py` and its `View` transform — a single shared renderer, so the
offline PNG and the per-step observation are pixel-aligned (see
`PROTOCOL_AGENT.md` §7/§9).

Inventory:

| Tool | Runtime? | Purpose |
|---|---|---|
| `render_gmaps.py` | **yes** (vision) + offline | Google-Maps-style PIL renderer; per-step frame **and** offline scenario PNGs (§12.1). |
| `render_trajectory_html.py` | offline | Renders a rollout `log.jsonl` into a self-contained HTML trajectory (images embedded base64); imports `benchmark.CHECKS`. |
| `analyze_order_feasibility.py` | offline (CLI) | Oracle estimator that enumerates raw restaurant/building order candidates with `OrderManager`'s binding logic and reports structurally-feasible counts. Shares its logic with the runtime `VIEW_ORDERS` filter (`utils/order_feasibility.py`). |
| `visualize_waypoints.py` | offline | Waypoint-graph debug viewer (§12.2). |

`benchmark.py` (package root, not under `tools/`) reads a rollout run dir and
reports per-trajectory metrics (`called_navigation`, `arrived_pickup`,
`delivered`); used by `rollout_qwen.py` and `render_trajectory_html.py`.

### 12.1 `tools/render_gmaps.py` — Google-Maps-style scenario PNG

A pure-Pillow renderer that turns a scenario folder
(`roads.json` + `progen_world_enriched.json`) into a single PNG styled like
Google Maps day mode. Used for paper figures, scenario inspection, and batch
pre-rendering — **and**, via its cached background + `View` transform, as the
canonical per-step map frame in vision mode when `use_gmaps_renderer` is set
(see §7). The CLI usage below is the offline path; the same module is imported by
the env wrapper for the runtime frame.

**What it draws**
- Warm-beige outer background with a slightly greener "city interior" fill
  inside the road bounding box (so gaps between buildings read as land).
- Road casing + inner fill, with synthesized street names
  (`1st St`, `Main Ave`, `2nd Ave`, …) numbered by perpendicular offset and
  rotated along centerlines. `is_highway=true` segments get a yellow fill
  with darker casing and a dashed white centerline (no shipped scenario
  uses highways yet, but the styling is in place).
- Building polygons with a subtle 3D extrusion offset and a soft drop
  shadow. POI buildings (restaurant / store / hospital / rest_area /
  car_rental) get a category-tinted fill plus a teardrop pin with a vector
  icon inside the head:
  - fork + knife (restaurant)
  - shopping bag (store)
  - white cross (hospital)
  - tree (rest area)
  - car silhouette (car rental)
- Charging stations as green pill markers with a white lightning bolt,
  labeled `EV 1`, `EV 2`, …
- Bus stations as orange pill markers with a bus silhouette, labeled
  `Bus 1`, `Bus 2`, …, plus the bus route polyline.
- Optional agent dot (blue, with translucent halo) at the supplied
  `--agent X Y` position (centimeters).
- Compass + scale bar.
- Anti-collision label placement (POI labels avoid road names and other
  pins).

**Usage**

```bash
# Render one scenario.
python -m vagen.envs.deliverybench.tools.render_gmaps \
    vagen/envs/deliverybench/maps/medium-city-22 \
    -o /tmp/medium-city-22.png \
    --agent 5000 5000

# Batch-render every scenario.
mkdir -p ./rendered_maps
for d in vagen/envs/deliverybench/maps/*/; do
    name=$(basename "$d")
    python -m vagen.envs.deliverybench.tools.render_gmaps \
        "$d" -o "./rendered_maps/${name}.png"
done
```

**Inputs / outputs**

| Argument | Meaning |
|---|---|
| `scenario_dir` (positional) | Path to a `maps/<name>/` folder. Must contain `roads.json` and `progen_world_enriched.json`. |
| `-o`, `--out` | Output PNG path. Defaults to `./<scenario_name>_v2.png` in the current working directory. |
| `--agent X Y` | Optional. Centimetres. If supplied, draws the blue agent dot at that world position. |

The output's long side is fixed at ~2400 px; the short side is sized from
the data's aspect ratio so large `is_highway=true`-less cities and small
single-block scenarios both fit without empty cream borders.

**Implementation notes**
- Coordinate convention follows the simulator: `roads.json` is in **meters**
  (×100 → cm at load time), the enriched-world `properties.location` and
  building bboxes are already in **cm**.
- All styling tokens live in the `STYLE` dict at the top of
  `render_gmaps.py`. Road / building / POI colors, pin radius, drop-shadow
  alpha, and street-label font size are tunable from there.
- Adding a new POI category: extend `STYLE["building_poi"]`, add an entry
  to `STYLE["poi_full"]`, and add a `_draw_poi_glyph` branch for the
  inside-the-pin icon.
- Adding a new scenario: drop a folder under `maps/`; no renderer change
  needed. Street numbering is derived from the road segments at render
  time.

### 12.2 `tools/visualize_waypoints.py`

Existing debug viewer for the waypoint graph produced by
`Map._build_waypoint_graph` (§6.2). Useful when investigating
direction-based `MOVE` movement or address resolution.

---

## 13. Pluggable hazards: obstacles, traffic lights, `BYPASS`, `WAIT("traffic_light")`

Two **default-off, vision-only** safety / social-navigation mechanics. Full spec:
`OBSTACLE_TRAFFIC_DESIGN.md`; capture spec: `deliverybench_fpv/FPV_CAPTURE_PLAN.md`.

**Status:** runtime mechanics are implemented and tested; the FPV perception
images (obstacle sprite, red/green crosswalk) are **not baked yet**, so the
features are kept off in all shipped stages. Enabling a flag without its per-map
sidecar (`obstacles.json` / `traffic_lights.json`) **raises at reset**.

- **Obstacles** (`enable_obstacles`): a static block on a directed waypoint edge
  `src→dst` (sidecar-defined, seeded). `MOVE("forward")` onto it is rejected
  ("blocked, unable to proceed"), the agent stays put, and
  `traj_metrics.collisions` increments. `BYPASS()` (alias of `PASSBY`) steps past
  it at `passby_cost_scale`× (default 1.5) time/energy. Forward-edge only.
- **Traffic lights** (`enable_traffic_lights`): 4-way intersections alternate by
  wall-minute parity (odd → N–S red / E–W green; even flipped). Crossing on red
  still moves but increments `traj_metrics.traffic_violations`.
  `WAIT("traffic_light")` advances the clock to the next minute boundary so the
  light flips. Light state is chosen by the travel axis (N–S vs E–W).
- **Vision-only:** hazards are *not* named in `### orientation`; the agent must
  read the obstacle / red light from the FPV. Counters are surfaced in
  `info["metrics"]["traj_metrics"]` but **not wired into reward** (a penalty term
  can consume them later).
- **Wiring:** `vlm_delivery/utils/hazards.py` (`ObstacleField` / `TrafficController`),
  `DeliveryBench._load_hazards`, and the `move.py` / `passby.py` / `wait.py`
  handlers. `BYPASS` and `WAIT("traffic_light")` are normal gated actions — add
  them to a stage's `enabled_actions` only once the FPV variants exist.

`PASSBY`/`BYPASS` is therefore no longer redundant with `MOVE("forward")`: it is
the only way across an obstacle edge.
