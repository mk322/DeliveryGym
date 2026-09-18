# DeliveryBench — Environment Protocol (coding-agent reference)

Authoritative contract for the VAGEN DeliveryBench environment as wired on
branch `dev/env_v2`. Audience: engineers/coding agents integrating, training
against, or modifying the env. Everything below was verified against the code
(`vagen/envs/deliverybench/`), not from memory.

---

## 1. Layout & entry points

```
deliverybench_env.py          # VAGEN wrapper: DeliveryBench(GymImageEnv) + DeliveryBenchEnvConfig
vlm_delivery/                  # the underlying text simulator
  entities/delivery_man.py     # DeliveryMan ("dm"): state, action handler table, observation builders
  gameplay/action_space.py     # parse_action(), action spec/examples, canonical name map
  gameplay/prompt.py           # get_system_prompt(cfg)
  gameplay/valid_actions.py    # per-step available_actions / available_tools hint
  actions/*.py                 # one handler module per action kind
  map/map.py                   # CityMap: road graph + waypoint graph + resolve/adjacents
  utils/order_feasibility.py   # oracle shortest-path step estimate for order feasibility filters
  utils/vlm_prompt.py          # build_state_observation() body (vlm_build_state_obs)
  base/defs.py                 # DMActionKind enum, TOOL_ACTION_KINDS, TransportMode
tools/render_gmaps.py          # Google-Maps-style PIL renderer (shared by per-step frame + NAVIGATE overlay)
tools/analyze_order_feasibility.py  # CLI oracle estimator for raw map order candidates
cli.py                         # debug REPL / scripted / vision probes
rollout_qwen.py                # reference rollout loop (OpenRouter / vLLM)
maps/<map_name>/               # world files; deliverybench_fpv/<map_name>/ = FPV dataset
```

Run tests / probes with the system Python and `PYTHONPATH=.`:
`PYTHONPATH=. python3 -m vagen.envs.deliverybench.test_deliverybench`.

---

## 2. Gym interface (`DeliveryBench`, async)

`DeliveryBench` subclasses `GymImageEnv`. All methods are `async`.

| Method | Signature | Returns |
|---|---|---|
| `system_prompt()` | `() -> {"obs_str": str}` | Static system prompt (action space + rules + vision-input description + optional store catalog). Call **once** per episode; cache it. |
| `reset(seed)` | `(seed:int) -> (obs, info)` | Fresh episode. |
| `step(action_str)` | `(action_str:str) -> (obs, reward, done, info)` | One decision step. `action_str` is the raw model output. |
| `close()` | `() -> None` | Release the underlying sim. |

### Observation dict (`obs`)
- `obs_str: str` — the textual state (see §6). In vision mode it is prefixed with one `<image>` placeholder per image.
- `multi_modal_input: {"<image>": [PIL.Image, ...]}` — present only in `render_mode="vision"`. Order: **[FPV?, global map?, local map?, ephemeral tool image?]** (each conditional on config; see §7).

### `info` dict from `step`
- `parsed`: `{"action": str|None, "reasoning": str, "format_correct": bool, ...}` from `parse_response`.
- `is_tool: bool` — true iff the executed action kind ∈ `TOOL_ACTION_KINDS` (query-only nav tools).
- `action_error: str|None` — **the only place handler errors surface** (see §8). Relay it to the model yourself.
- `metrics`:
  - `turn_metrics.action_is_valid` = `format_correct and not action_error`
  - `turn_metrics.action_is_effective` = `not action_error`
  - `traj_metrics`: `success`, `deliveries_completed`, `sim_hours`, `time_limit_reached`
- `success: bool`, `raw_info: dict`.

### `done` is set when **any** of:
- simulated time ≥ `time_limit_hours` (default 2.0 h) — the primary terminator; the underlying sim never ends on its own.
- the **same action FAILS** `_repeat_limit` (=3) times in a row (anti-stuck guard). The counter resets on any *effective* action, so a long straight walk — many identical `MOVE("forward")`s — never triggers it.
- the underlying step returns terminated/truncated.

`success` = at least one completed delivery (`deliveries_completed ≥ 1`).

---

## 3. Reward

`_compute_reward` (per step):
```
reward = (dm.earnings_total - prev_earnings)          # Δ money this step
       + format_reward (default 0.0) if format_correct and no error
```
Earnings start at $100.00. Net profit = `earnings_total - 100`. The reference
rollout reports **hourly profit** = `net_profit / sim_hours`. There is no
shaping beyond Δearnings; deadline misses / penalties show up as smaller or
negative earnings deltas.

---

## 4. Units & coordinates

- The simulator works in **centimetres** internally; **all observations report metres**.
- Locomotion is **direction-based** (`MOVE(direction=...)`); there are no
  coordinate or waypoint-id move arguments. NAVIGATE targets are POI
  names/addresses. Facing is a compass angle (0=N/90=E/180=S/270=W).
- Position is snapped to the waypoint graph on spawn (`fixed_spawn_position` is given in metres `[x, y]`); initial facing is North.

---

## 5. Action space (curriculum-gated)

Actions are parsed by `parse_action` (`action_space.py`). A single line
`NAME(args)` is required; JSON wrappers (`{"action": "NAME(...)"}`) and
``` ```fences``` ``` are unwrapped by `sanitize_model_text`. Names are
canonicalised (case-insensitive) via `_CANON_RAW`.

**Which actions are live is controlled per-stage by `enabled_actions`.** An
action not in `enabled_actions` is (a) filtered out of the system-prompt spec,
(b) filtered out of the per-step `available_actions` hint, and (c) rejected by
the parser (`"Action '<X>' is disabled in current stage config."`).

### Locomotion — **direction-based `MOVE`** (waypoint granularity, FPV-driven)
```
MOVE(direction="forward"|"left"|"right"|"backward")   # MOVE("forward") also works
```
- The agent has a **persistent compass facing** (`dm.facing_deg`, 0=N/90=E/180=S/270=W,
  init 0). `MOVE` steps one edge to the adjacent waypoint in that direction
  *relative to the facing*, then **rotates the facing**: forward keeps it, right +90,
  backward +180, left −90 (so after a turn you face the way you walked).
- The mapping uses `city_map.adjacents` bearings binned to the four quadrants of
  the facing. On the bundled **cardinal-grid maps** every edge is exactly N/E/S/W,
  so each direction resolves to at most one neighbour (no ambiguity).
- **Accessibility:** a direction with no road fails with
  `MOVE: no reachable waypoint to your <dir> (facing <C>). Directions you can move: <list>.`
- Charges energy/battery and advances the sim clock (distance from edge metadata,
  time = dist / (speed × pace)). A planned NAVIGATE route stays on the map until
  the agent reaches its destination (see §9), so a single move does not clear it.
- The handler lives in `actions/move.py`; `available_moves(dm)` is the shared
  helper that the observation builder also uses, so the text hint and the action
  always agree.

> The old `STEP_TO(<waypoint_id>)` and the legacy coordinate `MOVE(x, y)` are
> **deleted** — there is no id-based or coordinate-based locomotion anymore.

> **`BYPASS()` / `PASSBY()` — obstacle bypass (pluggable, default-off).** Steps one
> waypoint forward at `passby_cost_scale`× (default 1.5) time/energy. When
> `enable_obstacles` is on it is the *only* way across an obstacle edge (a normal
> `MOVE("forward")` into an obstacle is rejected with "blocked, unable to proceed"
> and counts a collision). It is allowed on a clear edge too (just costlier). Not
> in any shipped stage's `enabled_actions` yet (FPV obstacle images aren't baked).
> See `OBSTACLE_TRAFFIC_DESIGN.md` / `DELIVERYBENCH_DOC.md` §13.

> **`WAIT("traffic_light")` — pluggable, default-off.** With `enable_traffic_lights`
> on, 4-way intersections alternate by wall-minute parity; crossing on red still
> moves but counts a `traffic_violations`. `WAIT("traffic_light")` advances the
> clock to the next minute boundary so the light flips. Both hazard mechanics are
> **vision-only** (read the obstacle / red light from the FPV — not named in text);
> counters surface in `traj_metrics` (`collisions`, `traffic_violations`) and are
> not wired into reward.

### Order lifecycle
```
VIEW_ORDERS()
ACCEPT_ORDER(0)  | ACCEPT_ORDER([0, 3])
PICKUP(orders=[0])          # while standing on the order's pickup dock waypoint, food ready
DROP_OFF(oid=0)             # while standing on the dropoff dock (method auto = leave_at_door when delivery_methods off)
WAIT(minutes=3) | WAIT("charge_done")
```

`VIEW_ORDERS()` can optionally filter the visible order pool by structural
oracle feasibility. The filter lives in `actions/view_orders.py` and uses
`utils/order_feasibility.py` at the moment the agent calls `VIEW_ORDERS`, so the
start position is the agent's current `(dm.x, dm.y)`, not necessarily the spawn.
The estimate is:
`non_move_actions + shortest_path(current -> pickup) + shortest_path(pickup -> dropoff)`,
where shortest paths are waypoint-graph edge counts. Configure it through
`DeliveryBenchEnvConfig` / rollout YAML:

```
enable_feasible_orders: true
enable_infeasible_orders: true
feasible_order_step_budget: 20
feasible_order_non_move_actions: 5
```

Semantics: `enable_feasible_orders=true, enable_infeasible_orders=false` gives a
feasible-only pool; `true,true` is the natural mixed pool; `false,true` is an
infeasible-only stress test; `false,false` is invalid. Runtime
`feasible_order_non_move_actions` defaults to 5 because `VIEW_ORDERS` itself has
already been spent, leaving accept, navigate-to-pickup, pickup,
navigate-to-dropoff, and drop-off. The CLI estimator defaults to 6 when
estimating the whole episode from spawn.

### Resource / transport / packing / multi-agent (gated by feature flags)
`CHARGE, REST, BUY, USE_BATTERY_PACK, USE_ENERGY_DRINK, SWITCH, RENT_CAR,
RETURN_CAR, BOARD_BUS, VIEW_BUS_SCHEDULE, PLACE_FOOD_IN_BAG, VIEW_BAG,
USE_ICE_PACK, USE_HEAT_PACK, DROP_OFF(method=...), VIEW_HELP_BOARD, POST_HELP,
ACCEPT_HELP, EDIT_HELP, PLACE_TEMP_BOX, TAKE_FROM_TEMP_BOX,
REPORT_HELP_FINISHED, SAY`. Each is gated by the relevant `enable_*` flag; see
`get_action_spec` / `get_valid_actions`.

### Tool — query-only (do **not** mutate position/clock/energy/orders/...)
A **single** navigation tool; the transport mode is a parameter:
```
NAVIGATE(target="<address or waypoint>", mode="walk"|"e-scooter"|"bus")
```
- Draws the route to `target` on the city-map image (vision mode) **and reports
  cost estimates**: travel time, personal energy, e-scooter battery, and (bus)
  wait + fare. Estimates are the point — feasibility vs. deadlines/energy.
- It **does NOT emit a waypoint-id chain**. It does emit one live
  `next_move:` hint (`move forward`, `move backward`, `turn left`, `turn right`,
  or `you have arrived`) that is recomputed from the agent's current waypoint and
  facing on every later observation while the navigation target is active.
  `_nav_helpers._MOVE_PHRASE[180]` intentionally says **`move backward`**, not
  "turn around", so the text maps directly to `MOVE(direction="backward")`.
- `mode` defaults to `"walk"`; `"e-scooter"` requires `enable_battery`, `"bus"`
  requires `enable_advanced_transport` (else the call errors). For `"bus"`,
  optional `access_mode`/`egress_mode` = `"auto"|"walk"|"scooter"`.
- Target resolution is task-aware: for accepted/help orders,
  `NAVIGATE(target="<Pickup/Dropoff address>")` first resolves against that
  order's actual pickup/dropoff endpoint before falling back to the global map
  address book. Explicit strings like `pickup of order #0` / `dropoff of order #0`
  also resolve to the active order endpoint. Ground and bus navigation both use
  this resolver.
- `is_tool=True`; never advances the clock or moves the agent. The system prompt
  lists only the modes enabled by the current stage (see `get_tool_spec`).

### Removed (do not reference)
- **Locomotion:** `STEP_TO(<id>)` and coordinate `MOVE(x, y)` (kind `MOVE_TO`),
  plus `STEP_FORWARD`, `TURN_AROUND` — all deleted. `MOVE` now means the
  direction action above (`DMActionKind.MOVE`).
- **Nav:** the six former tools (`NAVIGATE_WALK`/`_ESCOOTER`/`_BUS` + `VISUAL_*`),
  the old internal `NAVIGATE`/`NAVIGATE_DIRECTIONS`, and `navigate_directions.py`
  were deleted/merged into the single `NAVIGATE`. The bus planner
  (`navigate_bus.py::compute_bus_routes`/`_bus_leg_nodes`) is still used by
  `actions/navigate.py`.

---

## 6. Observation text (`build_state_observation` → `vlm_build_state_obs`)

The wrapper uses the **clean state observation** (no agent memory, no
`### recent_error` — errors are relayed via `info`). Sections, in order:

1. `### agent_state` — id, time, transport mode, position (m), speed, pace,
   energy% (if walking-energy on), earnings, **`Active orders: #<id>...`**,
   **`Carrying: #<id>...`**, inventory, scooter/charging status.
   *(IDs are `#`-prefixed so order `#0` is not misread as a count of zero.)*
2. `### orientation` (when `enable_waypoints`, default on) — "You are at:
   `<addr>` … Facing: `<N/E/S/W>`" plus **"You can MOVE (relative to your
   facing): forward → … / left → … / right → … / backward → …"** (each shows the
   destination name + distance + road, or `(blocked)`). Facing-relative,
   id-free; this is the live accessibility hint that pairs with `MOVE`.
3. `### store_catalog` (when a store manager exists; in the reference config it
   is hoisted into the system prompt instead via `store_catalog_in_system_prompt`).
4. `### active_orders` — full block per accepted order (pickup/dropoff
   coords+addr+dock id, time left, payout, status) or "You currently have no
   accepted orders."
5. `### accepted_help` / `### posted_help` / `### pickables` (multi-agent only).
6. `### map_snapshot` — text POI directory: next hops, next intersections,
   all POIs by shortest-path distance, order endpoints.
7. `### ephemeral_context` — short-lived hints plus persistent navigation state.
   `[navigation]` contains the route estimate and the live `next_move:` line; it
   persists after a successful `NAVIGATE` until another destination is chosen.
   On arrival it remains long enough to say `next_move: you have arrived`.
8. `### available_actions` / `### available_tools` — the per-step hint from
   `get_valid_actions` / `get_valid_tools`, filtered by stage flags.

---

## 7. Vision inputs

When `render_mode="vision"`, per-step images (in this order, each conditional):
- **FPV cross** (`enable_fpv`): a **single** image tiling the four egocentric
  views around the agent **relative to its facing** — BACK on top, LEFT/RIGHT on
  the sides, FRONT centred and 2× larger (front ½×½, others ¼×¼), each panel
  text-labelled. FRONT = where `MOVE("forward")` goes, RIGHT = `MOVE("right")`,
  etc. Built by `_build_fpv_cross(pos_key, facing_deg)` from the four cardinal
  yaws in `deliverybench_fpv/<map>/manifest.jsonl` (all waypoints have all 4 yaws).
  **Keyed by position, not waypoint_id**: the capture's `dock_*` ids were
  renumbered vs. the current graph (~1/3 mismatch), so the lookup joins on the
  waypoint's `(x_cm, y_cm)` (`_load_fpv_lookup` / `_fpv_images_at`) — positions
  match 1:1. (An id-join was the cause of "FRONT sometimes faces a wall.")
  **Yaw calibration**: the UE yaw is a left-handed **reflection** of the sim
  compass, so a direction D maps to stored yaw `(fpv_yaw_offset_deg − D)`
  (default K=**90**), NOT `D + offset`. Using a rotation made turns mirror-reversed
  (RIGHT showed the left-hand scene). With the reflection, FRONT = heading and
  LEFT/RIGHT are correct. Adjust K by ±90 if a future dataset differs.
- **Map** (`enable_map_images`): if `use_gmaps_renderer` → one Google-Maps-style
  frame from `tools/render_gmaps.py` (cached background + agent dot **with a white
  facing-arrow** + order pins + planned nav route). Else the legacy exporter's
  global (+ local unless `map_global_only`) snapshots.
- **Route overlay**: `NAVIGATE` no longer relies on a separate one-shot tool
  image. The per-step map redraws the live route under the agent dot until
  arrival or a new `NAVIGATE`.

The system prompt's `### vision_inputs` block tells the model how many images
to expect and in what order; keep it in sync if you change the image pipeline.

---

## 8. Error contract (important)

Handler-level failures (PICKUP not ready, DROP_OFF too far, MOVE direction
blocked, …) are written to `dm.vlm_errors`, **not** into the observation. The
wrapper copies them to `info["action_error"]`, then calls
`dm.vlm_clear_errors()`. `build_state_observation` deliberately omits a
`### recent_error` section. **Your harness must surface `info["action_error"]`
to the model** (the reference rollout prepends "⚠ Your previous action
FAILED: …" to the next user turn). The stale `test_error_in_context.py` asserts
the old in-observation behaviour and fails by design.

---

## 9. Navigation tool ↔ movement wiring (the "Google-Maps" loop)

1. `NAVIGATE(target=..., mode=...)` runs query-only (`actions/navigate.py`),
   computes the route (Dijkstra on `city_map.waypoint_graph` for walk/e-scooter;
   `compute_bus_routes` for bus), **renders it as a map-image overlay** via
   `_visual_helpers.render_route_overlay`/`render_legs_overlay`, and emits a
   `### ephemeral_context [navigation]` block with the cost estimate
   (distance/time/energy/battery/fare). It does **not** expose waypoint ids or a
   full route chain.
2. Later observations keep `[navigation]` and append a live `next_move:` line
   from `vlm_prompt._live_next_move_line()`. The hint is the single next
   direction to execute in the current facing frame. Prompt v2.1 tells the model
   to follow this hint exactly unless the move is blocked or energy/battery
   makes it impossible.
3. It sets `dm._nav_route_path` / `_nav_route_color` / `_nav_target_node`. The
   per-step gmaps frame draws that polyline **plus a green source dot and red
   destination pin, as a layer UNDER the agent dot**, and it **persists on every
   later frame** until another NAVIGATE overwrites it or the agent reaches
   `_nav_target_node`. On arrival, `MOVE` clears the drawn route but keeps the
   navigation target/block so the next observation can explicitly say
   `next_move: you have arrived`; a later NAVIGATE overwrites it.
4. The per-step frame **reuses `tools/render_gmaps.py` +
   its `View` transform** — pixel-aligned, single renderer.
5. Target parsing is endpoint-first for active orders. This avoids address-book
   ambiguity where an order's displayed pickup/dropoff address has nearby street
   waypoints with similar labels.

---

## 10. Waypoint graph (accessibility model)

Built in `map.py` `_build_waypoint_graph`:
- **Nodes**: intersections (`int_<n>`) + POI docks (`dock_<n>`).
- **Edges**: two waypoints are adjacent iff reachable along the road graph
  without passing another waypoint; edge metadata carries `dist_cm`,
  `bearing_deg`, `road_name`.
- **API**: `city_map.adjacents(node) -> [{id,name,kind,dist_m,bearing_deg,compass,road_name,node}]`,
  `city_map.nearest_waypoint(x_cm,y_cm)`, `city_map.resolve_waypoint(token)`
  (id / address / display name / intersection name).

`MOVE` consults this graph via `actions/move.py::available_moves`, which bins
each neighbour's `bearing_deg` into forward/left/right/backward relative to
`dm.facing_deg`; the `### orientation` section exposes those four directions
every step. (On the cardinal-grid maps the binning is exact.)

---

## 11. Curriculum configs (`deliverybench_env.py`)

`DeliveryBenchEnvConfig` feature flags gate constraints, spec lines, prompt
text, POI directory, and the action/tool lists. Presets:

- **STAGE_1_CONFIG** — walk-only, single restaurant + single customer,
  `max_orders_in_pool=3`, single item, no jitter, no battery/energy/temperature/
  smell/fragility/bag/advanced-transport/delivery-methods/notes/multi-agent/
  prep-time. `enabled_actions = [VIEW_ORDERS, ACCEPT_ORDER, MOVE, PICKUP,
  DROP_OFF, WAIT, NAVIGATE]`. `deadline_multiplier=1.5`
  (deadlines are priced on the pickup→dropoff leg only; the approach leg is
  unbudgeted, so deadlines are loosened). `NAVIGATE` here offers `mode="walk"` only.
- **STAGE_2_CONFIG** — adds battery, walking energy, e-scooter (charge/rest/buy/
  switch/packs): `… MOVE, CHARGE, REST, SWITCH, BUY, USE_BATTERY_PACK,
  USE_ENERGY_DRINK, NAVIGATE]`. `NAVIGATE` now offers `mode="walk"|"e-scooter"`.
  Still walk-start.
  **No bag compartments** (`enable_bag_compartments=False`): while temperature/
  fragility/smell are off, bag placement has no payout effect and is never
  required, so it would be busywork — bag mechanics begin in Stage 3.
- **STAGE_3_CONFIG** — defaults (everything on, full map, multi-agent, all tools).
  `MOVE` here means the same direction-based action (the only locomotion).

Key config notes:
- `initial_transport_mode`: the sim spawns the agent on an e-scooter; walk
  stages must set `"walk"` so speed matches `NAVIGATE` walk estimates and the
  walking-based deadline pricing.
- Curriculum order filters (`num_restaurants`/`num_customers`/`require_single_item`/
  `deadline_multiplier`) are applied at reset **and** re-applied to mid-episode
  pool refills via a wrapped `_spawn_one_order`.
- `time_scale=0.0` → deterministic: the sim clock advances only via
  `clock.advance()` inside MOVE/WAIT/CHARGE/REST; API latency has zero effect.

---

## 12. Integrating a rollout loop

1. `sys = await env.system_prompt()` → system message (cache it; it's static).
2. `obs, _ = await env.reset(seed)`.
3. Each step: append `obs` (images + `obs_str`, with `<image>` placeholders
   stripped/replaced for your provider) as the user turn; if the previous step
   returned `info["action_error"]`, prepend it; call the model; `await
   env.step(model_text)`; log; stop on `done`.
4. Manage context: strip images from older turns (the text state is refreshed
   each step), compress/summarise history past a token threshold.
5. Report `success`, `deliveries_completed`, `sim_hours`, hourly profit.

- **`rollout_qwen.py`** — model rollout (default model `qwen/qwen3-8b`, override
  with `OPENROUTER_MODEL_ID` / `VLLM_MODEL_NAME`).
- **`rollout_scripted.py`** — deterministic, no-API autopilot that drives
  `MOVE(direction)` by translating shortest-path hops into directions; completes a
  delivery and writes `outputs/scripted_<ts>/` (log.jsonl + per-step images).
  Useful as an end-to-end smoke test and a reference trajectory.

Because locomotion is per-waypoint, a delivery costs several `MOVE` calls per
leg (one model call each); budget `max_steps` accordingly (the reference uses
200). Sim-time feasibility is unchanged (same distance/speed); only the step
count grows.

---

## 13. Mechanics coherence audit & known gaps (2026-06-13)

Audited the full feature set for "does it make sense / is it wired."

**Coherent — no action needed:**
- Energy / battery / hospitalization / towing; CHARGE, REST, BUY, SWITCH,
  RENT_CAR/RETURN_CAR, BOARD_BUS all mutate the right state and gate on
  POI/vehicle proximity; economic constants are balanced. Gating is consistent
  across parser / spec / valid_actions / prompt / observation.
- Food temperature / fragility / odor are real and feed the settlement star
  ratings → payout (`gameplay/settlement.py`). `pace="accel"` damages fragile
  items; ice/heat packs have real effects.
- Multi-agent help is correctly hidden when `enable_multi_agent=False`; no
  observation leakage in single-agent runs (comms exists but nothing posts).

**Resolved (2026-06-15 direction-MOVE refactor):**
- Locomotion is now `MOVE(direction)` in every stage; the old coordinate `MOVE`
  and `STEP_TO` are gone, so the "Stage 3 still allows legacy MOVE" gap is moot.
- **Pickup/drop-off reachability fixed**: orders now anchor on the POI's **dock
  waypoint** (`entities/order.py` binds `dock_node`), and PICKUP/DROP_OFF succeed
  while standing on it — so the "waypoint not within 6 m of the door" bug can no
  longer make an order uncompletable.
- **`hand_to_customer` now works at the dock** (`actions/drop_off.py`): the first
  call at the dropoff dock "knocks" and starts a customer come-out timer
  (`_CUSTOMER_COME_OUT_S`, default 30 s); the agent `WAIT`s, then a second call
  hands the food over. (Previously it required an off-graph handoff point and a
  customer-emergence delay that was **never implemented** — now it is, minimally.)

**Known gaps / follow-ups (still deferred):**
1. `PLACE_FOOD_IN_BAG` is never a hard gate (`_force_place_food_now` only drives a
   hint; DROP_OFF unloads from `dm.carrying` regardless). Fine as-is; bag
   arrangement only matters when temperature/fragility/smell are on (hence
   dropped from Stage 2).
2. **Non-grid maps.** The forward/left/right/back binning is exact only because
   the bundled maps are cardinal grids (verified: 0° max bearing offset). A map
   with diagonal roads could put two neighbours in one bin; add a tie-break /
   guard before using such maps.

---

## 14. Constraint & action matrix (per stage)

Legend: ✓ = active/available, · = off/unavailable. S1 = Stage 1, S2 = Stage 2,
S3 = Stage 3 (full).

### 14a. Constraints — what each tests and how to satisfy it

| Constraint (config) | S1 | S2 | S3 | Capacity tested | How the agent satisfies it |
|---|:--:|:--:|:--:|---|---|
| 2-hour shift (`time_limit_hours`) | ✓ | ✓ | ✓ | Global time budgeting | Maximise deliveries per sim-hour; don't spend steps on infeasible orders. |
| Per-order deadline (`Time Left`) | ✓ | ✓ | ✓ | Time estimation & prioritisation | Accept only orders you can pick up **and** drop off before the deadline (`NAVIGATE` reports the time estimate); deliver before it hits 0. |
| Direction movement + facing (waypoint graph) | ✓ | ✓ | ✓ | Spatial grounding / egocentric orientation | `MOVE(forward/left/right/backward)` relative to facing; read the 4-way FPV + map arrow + `### orientation` to pick the way toward the `NAVIGATE` route. |
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
| Delivery methods (`enable_delivery_methods`) | · | · | ✓ | Instruction following | `DROP_OFF` with the required method (`leave_at_door`/`knock`/`call`/`hand_to_customer`). |
| Special notes (`enable_special_notes`) | · | · | ✓ | Constraint satisfaction | Read the order note and obey its delivery-method requirement. |
| Advanced transport — car/bus (`enable_advanced_transport`) | · | · | ✓ | Mode selection under cost | `RENT_CAR` for speed ($/min) or `BOARD_BUS` ($1) when faster/cheaper than walking/scooter for a leg. |
| Multi-agent help (`enable_multi_agent`) | · | · | ✓ | Coordination / comms | Only meaningful with ≥2 agents: `VIEW_HELP_BOARD`/`ACCEPT_HELP` for bounties, `POST_HELP`+`PLACE_TEMP_BOX` when you need help, `SAY` to coordinate. |

Always-on baseline (every stage): walk-start, the 2-hour clock, per-order
deadlines, the waypoint graph, and the reward = Δearnings / success = ≥1
delivery contract.

### 14b. Action space (state-changing) — availability by stage

Actions mutate the world (move, pick up, charge, …). Tools are listed separately
in §14c.

| Action | S1 | S2 | S3 | Purpose |
|---|:--:|:--:|:--:|---|
| `VIEW_ORDERS` / `ACCEPT_ORDER` | ✓ | ✓ | ✓ | See the pool / accept (one or a list). |
| `MOVE(direction=...)` | ✓ | ✓ | ✓ | Step to the adjacent waypoint forward/left/right/backward of your facing; turns you. The only locomotion. |
| `PICKUP` / `DROP_OFF` | ✓ | ✓ | ✓ | Collect ready food / deliver — while standing on the order's pickup / dropoff dock. |
| `WAIT` | ✓ | ✓ | ✓ | Pass time (prep/charge) — last resort. |
| `CHARGE` / `REST` | · | ✓ | ✓ | Recharge scooter / restore energy at a POI. |
| `BUY` / `USE_ENERGY_DRINK` / `USE_BATTERY_PACK` | · | ✓ | ✓ | Buy at a store; consume for energy/battery. |
| `SWITCH` | · | ✓ | ✓ | Change transport mode (walk ↔ e-scooter[/car]). |
| `PLACE_FOOD_IN_BAG` / `VIEW_BAG` | · | · | ✓ | Arrange/inspect the insulated bag. |
| `USE_ICE_PACK` / `USE_HEAT_PACK` | · | · | ✓ | Hold a compartment's temperature. |
| `RENT_CAR` / `RETURN_CAR` / `BOARD_BUS` / `VIEW_BUS_SCHEDULE` | · | · | ✓ | Car & bus transport. |
| `VIEW_HELP_BOARD` / `POST_HELP` / `ACCEPT_HELP` / `EDIT_HELP` / `PLACE_TEMP_BOX` / `TAKE_FROM_TEMP_BOX` / `REPORT_HELP_FINISHED` / `SAY` | · | · | ✓ | Multi-agent help & coordination. |
| `MOVE` | · | · | ✓¹ | Legacy free-coordinate routing. ¹Present in S3 only because it inherits the all-actions default — pending removal (§13). |

### 14c. Tools (query-only) — availability by stage

Tools never change the world (no move/clock/energy/order mutation); they only
add to the observation. There is exactly **one** tool — `NAVIGATE` — whose
transport mode is a parameter, gated by stage flags.

| Tool | S1 | S2 | S3 | `mode=` options by stage | Purpose |
|---|:--:|:--:|:--:|---|---|
| `NAVIGATE(target, mode)` | ✓ | ✓ | ✓ | S1: `walk` · S2: `walk`,`e-scooter` · S3: `walk`,`e-scooter`,`bus` | Draw the route on the city-map image, report time/energy/battery/(bus) fare estimate, and keep a live `next_move` hint. No waypoint-id chain. For `mode="bus"`: optional `access_mode`/`egress_mode`. |

`info["is_tool"]` is `True` exactly for `NAVIGATE` (the only member of
`TOOL_ACTION_KINDS`).

---

## 15. Reference rollout: backends & local Qwen3-VL serving (`rollout_qwen.py`, 2026-06-15)

The client only needs the **VAGEN env** (`vagen` conda env: it imports
DeliveryBench and renders frames via `PyQt5`/`pyqtgraph` — both required for
vision mode). The model is served **separately** over an OpenAI-compatible HTTP
API; the client uses no GPU. `rollout_qwen.py` now supports **three backends**
via `BACKEND`:

| `BACKEND` | base-url env | model env | structured output |
|---|---|---|---|
| `openrouter` (default) | — | `_OPENROUTER_MODEL_ID` | `response_format` json_schema + `reasoning=none` |
| `vllm` | `VLLM_BASE_URL` | `VLLM_MODEL_NAME` | `extra_body.guided_json` |
| `sglang` | `SGLANG_BASE_URL` (`http://localhost:30000/v1`) | `SGLANG_MODEL_NAME` (`qwen3-vl-8b`) | `response_format` json_schema, **gated by `SGLANG_GRAMMAR` (default off)** |

### Context budget (must stay under the server `context_len`)
With vision on, each turn carries an FPV photo + a 2400×1572 gmaps frame; real
Qwen3-VL image tokens far exceed the old flat estimate, which let the prompt
overrun a 32768-token window before compression fired. Retuned defaults:
- `CONTEXT_TOKEN_THRESHOLD = 14_000` (was 150_000)
- `IMAGE_TURNS_KEPT = 1` (was 2)
- `_TOKENS_PER_IMAGE = 3000` (was 765)

A `400 … context length` error means the estimate undershot — lower the threshold
or `IMAGE_TURNS_KEPT`, or shrink images (`gmaps_out_scale`). With a larger server
`context_len` these can be relaxed.

### Serving Qwen3-VL-8B on an old-driver node (workarounds, not VAGEN requirements)
Observed on this box: GPU driver **550.144 (CUDA 12.4)** but the serving env
(`qwen_env`, sglang `0.5.10.post1`) is a **cu128** build (torch cu128, NCCL
2.29.7, flashinfer 0.6.7). Three "CUDA runtime newer than driver" crashes, each
with a fix needed to serve at all:
- flashinfer `rmsnorm_cute` (cutlass-DSL) → `cudaErrorInsufficientDriver`.
  → launch with **`FLASHINFER_USE_CUDA_NORM=1`** (forces the CUDA-JIT norm kernel).
- sglang runtime JIT kernels (`clamp_position`, …) → *"set CUDA_HOME"*.
  → launch with **`CUDA_HOME=<qwen_env prefix>`** (its `bin/nvcc` is cuda-toolkit 12.8).
- grammar-constrained sampling (`response_format` → xgrammar) calls
  `sampler._sync_token_ids_across_tp`, an **NCCL all-reduce** that crashes on this
  driver. → **`SGLANG_GRAMMAR` defaults off**; the prompt already mandates pure
  JSON and DeliveryBench parses it from raw text, so unconstrained decoding works.
  Set `SGLANG_GRAMMAR=1` only where the driver matches the CUDA runtime.

Launch (single GPU; see `start_sglang.sh`):
```
CUDA_VISIBLE_DEVICES=2 FLASHINFER_USE_CUDA_NORM=1 CUDA_HOME=<qwen_env> \
  python -m sglang.launch_server --served-model-name qwen3-vl-8b --port 30000 \
  --context-length 32768 --mem-fraction-static 0.82 --trust-remote-code \
  --enable-multimodal --disable-cuda-graph --disable-overlap-schedule
# then, in the vagen env:
BACKEND=sglang python -m vagen.envs.deliverybench.rollout_qwen
```
A node whose driver is new enough for the cu128 stack needs none of these three
workarounds and can set `SGLANG_GRAMMAR=1`.

### Current local server snapshot (2026-06-17)
The active local deployment for the latest rollout work is Qwen3.5-VL-9B served
from the `qwen_env` environment through SGLang on GPU 2:

```
tmux session: qwen35_sglang_gpu2
base URL:     http://127.0.0.1:30002/v1
model id:     qwen3.5-9b
max_model_len: 32768
launcher:     scripts/start_sglang_qwen35_9b_gpu2.sh
```

The rollout client reads explicit YAML configs:
```
python -m vagen.envs.deliverybench.rollout_qwen <config.yaml>
```
Use `ROLLOUT_OUT_DIR` and `RUN_LABEL` to put runs under
`experiments/rollout_qwen35_sglang_visual/runs/`.

---

## 16. Curriculum rollout: Qwen3-VL-8B + SGLang visual difficulty search (2026-06-15)

**Goal.** Run many visual DeliveryBench rollouts to find env-parameter configs
that correspond to distinct empirical difficulty bands for Qwen3-VL-8B, as the
basis for a future curriculum. Model/backend settings are held **fixed** after
calibration; difficulty must come from env knobs, not from weakening the model.
Working dir for this effort: `experiments/rollout_qwen_sglang_visual/`
(`EXPERIMENT_PLAN.md`, `FINDINGS.md`, probes, `configs/`, `runs/`, `logs/`).

### 16.1 Visual path — VERIFIED
Images **do** reach the model. `experiments/.../visual_smoke_test.py` builds the
Stage-1 vision obs, encodes it with the exact `rollout_qwen._obs_to_user_message`
path, and sends one request to SGLang. Evidence (`logs/visual_smoke_result.json`):
obs has 2 images (FPV 640×480 + gmaps 1200×786) → request carries 2
`image_url` data-URL parts → SGLang `prompt_tokens` 1336 **with** images vs 88
text-only (**+1248 image tokens**) → model describes per-image content. When you
change the image pipeline, re-run this probe; do not assume vision works.

### 16.2 `rollout_qwen.py` harness changes (eval harness, NOT env code)
All env-var driven; defaults are the frozen "strong setting".
- **`DB_CONFIG_JSON`** — path or inline JSON of `DeliveryBenchEnvConfig`
  overrides, layered on the Stage-1 vision base. Unknown fields raise. Use for
  difficulty bands. `max_steps` override syncs the loop cap.
- **`SEED_LIST="42,43,…"`** — distinct seed per rollout (overrides
  `NUM_ROLLOUTS`/`SEED`); needed for difficulty/seed variance. Plain `SEED` keeps
  all rollouts identical (variance = model only).
- **Frozen model settings (env-overridable):** `MAX_TOKENS=2048`, `TOP_P=0.9`,
  `TEMPERATURE=0.2`, `REQUEST_TIMEOUT=120`, `CONCURRENCY=4` (bounded semaphore).
- **Output dirs:** `ROLLOUT_OUT_DIR` + `RUN_LABEL` → `<root>/<label>_<ts>/`
  (`log.jsonl` + `images/`). A `run_meta` record at the top of every `log.jsonl`
  captures backend, model, frozen settings, full env config, and seeds — logs are
  self-describing; read it before interpreting a run.
- **Action normalizer** `_normalize_action_text` (Fix A) — *transparent parser
  tolerance*, applied to the model response **before** `env.step` and before it is
  appended to history. **Logged per step:** `action_raw` (exactly what the model
  emitted, via `_extract_action_field`), `action` (normalized + env-parsed), and
  `action_normalized` (# rewrites). Two tolerant rewrites that never change
  *which* waypoint/flag the model chose:
  1. bareword waypoint ids → quoted: `STEP_TO(int_10)` → `STEP_TO('int_10')`
     (the AST parser rejects bare `Name` nodes with "Unsupported expression").
  2. JS booleans → Python: `NAVIGATE(target="x", text=true)` → `text=True`.
     **This one matters a lot:** the parser rejects bare `true`, so every
     `NAVIGATE(text=true)` silently failed and starved the agent of its
     turn-by-turn waypoint chain → it wandered. (Note: the docs/examples elsewhere
     write `text=true`; the *parser* needs `text=True`.)
- **Prompt version is logged — see §16.7/§16.8.** The historical B0 scan used
  `Scaffolded Task-Navigation Prompt v1`, which first completed B0. Current
  Qwen3.5-VL-9B visual runs use `Scaffolded Task-Navigation Prompt v2.1`, with
  stronger "use NAVIGATE once the Pickup/Dropoff address is known" discipline.
  Treat prompt version as an experiment axis; do not silently mutate it inside a
  difficulty comparison.

### 16.3 Env change made (Option A — APPROVED, door arrival tolerance)
`vlm_delivery/gameplay/action_space.py`: added `_DOOR_TOL_CM_DEFAULT = 1000.0`
and `_door_tol_cm(dm)` = `get_tol(dm.cfg, "door", 1000.0)`; PICKUP and DROP_OFF
now use it (previously a hardcoded `tol_cm=600.0` for PICKUP; DROP_OFF fell back
to the 5 m `nearby` tol). Override per-config via `cfg["tolerance_cm"]["door"]`.
Rationale below. This is the only DeliveryBench env-code change in this effort.

### 16.4 B0 (sanity/trivial band) debugging history — READ BEFORE JUDGING RESULTS
B0 config: `configs/B0_sanity.json` (Stage-1, small-city-11, deadline ×3,
1 rest/1 cust, pool 1, e-scooter), seed 42.

| Run dir (under `runs/`) | State | Deliveries | Why |
|---|---|---|---|
| `B0_sanity_20260615_135009` | raw baseline | **0** | wandered, never reached pickup; 27% parse errors (barewords) |
| `B0_sanity_fixAB_20260615_140820` | Fix A+B | 0 | reached pickup (step 37); `text=true` still parse-failed → PICKUP too far |
| `B0_fixAB2_20260615_143113` | + bool normalize | 0 | NAVIGATE worked, reached `dock_78`, but PICKUP impossible (see below) |
| `B0_optA_20260615_145107` | + Option A | **≥1 ✅** | **first delivery: `DROP_OFF(oid=0)` at step 36, `deliv=1`, $100→$107.97** |

**Root cause found between fixAB2 and optA (env feasibility, not model):**
order pickup/dropoff targets are *building doors* (`order.py:137`
`pickup_node = pu_meta["door_node"]`), offset from the street waypoint graph.
`feasibility_sweep.py` over 40 seeds (`logs/feasibility_sweep.json`): door→
nearest-waypoint distances up to **8.16 m**, and **~71% of doors exceeded the
old 6 m tolerance** — with `MOVE` disabled in Stage-1, those orders were
**impossible to PICKUP/DROP_OFF via `STEP_TO` by any agent**. 10 m covers 100%
of measured doors with margin → Option A.

### 16.5 Current status & paths
- **Successful run:** `experiments/rollout_qwen_sglang_visual/runs/B0_optA_20260615_145107/`
  (`log.jsonl`, `images/`); console `logs/B0_optA_console.log`.
- As of this writing the B0_optA process is **still running** (pid 436407, a live
  Monitor watches for completion): 1 delivery so far; after delivering it accepted
  the next order and is now looping/issuing premature `DROP_OFF`s (genuine nav
  inefficiency, ~9% action-error rate). History compression fired at step 125 as
  designed. **The final B0 summary has not been saved yet.**
- Probes: `visual_smoke_test.py`, `feasibility_probe.py`, `feasibility_sweep.py`
  (all read-only; run in the `deliverybench` conda env, `python -m
  experiments.rollout_qwen_sglang_visual.<name>`).
- Known cosmetic issue: the Qt/gmaps renderer core-dumps at **process teardown**
  (after logs are written) → nonzero exit; harmless to data, but batch
  orchestration must not treat the exit code as run failure.

### 16.6 Warnings for the next agent
- **Do NOT cite pre-Option-A B0 (0 deliveries) as a Qwen3-VL capability result.**
  Those runs were blocked by parse format, broken `NAVIGATE(text=true)`, and an
  env infeasibility (door tolerance). They measure harness/env bugs, not the model.
- **Distinguish two baselines:** the *raw* baseline (no harness fixes) vs the
  *scaffolded* baseline (Fix A normalizer + Fix B prompt + Option A tolerance).
  Curriculum difficulty numbers must state which baseline they used; the intended
  reference is the scaffolded one.
- **Do NOT launch the full difficulty sweep until the final B0 summary is saved**
  and B0 success is confirmed across a few seeds. Reachability is fixed, but
  navigation competence (looping, premature `DROP_OFF`) is still the dominant
  difficulty — verify B0 is reliably solvable before reading harder bands.
- Re-run `feasibility_sweep.py` for any new `map_name`/order-shaping config before
  trusting its difficulty numbers (door-vs-graph offset is map-specific; FPV
  vision is currently only available for `small-city-11`).

### 16.7 Historical prompt policy — `Scaffolded Task-Navigation Prompt v1`
**Rule: within a difficulty comparison, keep the prompt fixed and log its
version.** Difficulty comparisons are only valid under one prompt; a moving
prompt confounds the difficulty signal. The notes below are the historical B0
prompt evidence; current Qwen3.5-VL-9B prompt changes are in §16.8.

The prompt = the env `system_prompt()` (authoritative action space, syntax, rules,
observation + vision description) plus the rollout prompt policy/version logged
in each run's `run_meta.prompt_version`.

**This is the operational prompt for the curriculum rollout scan** — the scaffolded
suffix restored from source history (the version that actually completes B0). The
minimal/protocol-aware prompts below are recorded as **ablations that failed B0**.

**Why scaffolded (the decisive evidence).**
- **Scaffolded (this prompt) — completes B0. ✅** `B0_optA_20260615_145107` (seed
  42): first end-to-end delivery — `DROP_OFF(oid=0)` at step 36, $100→$107.97,
  `deliv=1`. This is the only prompt that has produced a B0 delivery, so it is the
  fixed baseline for the scan.
- **ABLATION — Minimal Interface (interface-only) — too sparse, 0 deliveries.**
  `B0_v1_20260615_160707` (seed 42): the agent PICKED UP (step 21) but **never
  attempted DROP_OFF** — acted as if pickup were the goal — then hit the
  repeat-guard at step 81.
- **ABLATION — Minimal Task-Interface v1.1 (interface + delivery semantics) — 0
  deliveries.** `B0_v1_1_20260615_163813` (seed 42): after pickup (step 15) the
  agent **did** head for the customer but never reached the drop-off; repeat-guard
  killed it at step 27 (`int_12`×3).
- **ABLATION — Protocol-Aware v1 (interface + semantics + env protocol facts, no
  workflow scaffold) — 0 deliveries.** `B0_protoaware_20260615_170257` (seed 42):
  survived to step ~188 but circled by compass dead-reckoning, abandoned NAVIGATE
  after pickup, closest approach ~21 m (never within the 10 m door tolerance),
  never delivered. Trajectory diagnosis: navigation/localization failure, not
  task-state misunderstanding.

> Takeaway: removing the per-order workflow scaffold (view→accept→pickup→dropoff +
> "use NAVIGATE to route, walk it with adjacent STEP_TO") reliably broke B0. The
> scaffolded prompt restores that scaffold and is therefore the operational prompt.

What the suffix **contains** (objective + format + light task-navigation scaffold):
- task objective (courier on foot, earn money within the 2-h window);
- response format (the 3-field JSON: `reasoning_and_reflection` / `action` /
  `future_plan`, nothing outside it; exactly one action/step);
- per-order **workflow**: VIEW_ORDERS → ACCEPT_ORDER → walk to pickup → PICKUP →
  walk to dropoff → DROP_OFF;
- historical **STEP_TO**/turn-by-turn routing guidance from the old B0 prompt
  (current code uses direction-based `MOVE` and live `next_move`, not STEP_TO);
- **NAVIGATE(target=…)** for routing — current code draws the route on the map
  and refreshes one live `next_move` hint, without exposing waypoint ids;
- visual-obs format (FPV + top-down map each step);
- read the error message on failure and fix the cause instead of repeating.

Note this is light scaffolding (it names the workflow and how to use the nav tool),
**not** an oracle: it gives no shortest/optimal route, no distance-must-decrease
rule, and no PICKUP/DROP_OFF distance thresholds — the agent still has to navigate.

**Action normalization is allowed** as transparent parser tolerance (Fix A) — it
is not strategy — but every step logs `action_raw`, `action`, and
`action_normalized` so the format-tax stays measurable and auditable.

If you ever need to compare prompt variants, bump the version and treat it as a
separate experiment axis — never silently mutate the prompt mid-sweep.

### 16.8 2026-06-17 navigation/prompt update — Qwen3.5-VL-9B

This section supersedes the older v1-specific rollout notes above for the current
Qwen3.5-VL-9B experiments.

**Prompt version.** Current rollout logs should show
`PROMPT_VERSION = "Scaffolded Task-Navigation Prompt v2.1"`. The v2 family makes
navigation discipline explicit:
- after accepting an order, call `NAVIGATE` with the actual Pickup address string
  shown in `active_orders` before moving;
- after `PICKUP`, call `NAVIGATE` with the actual Dropoff address string before
  moving toward the customer;
- do not copy placeholder text such as `<exact Pickup address>`;
- while `[navigation]` is present, follow `next_move` directly; when it says
  `you have arrived`, stop moving and call `PICKUP` or `DROP_OFF`.

**Navigation runtime changes.**
- `NAVIGATE` target resolution now checks accepted/help order endpoints before
  the global map resolver. This fixes ambiguous address cases such as an order
  display address that is near ordinary street waypoints with similar labels.
- `next_move` now uses action-shaped phrases (`move backward` instead of
  `turn around`) and can report `next_move: you have arrived`.
- `MOVE` no longer drops the navigation block immediately on arrival; it clears
  the route overlay but preserves enough state for the next observation to say
  arrived.

**Current text-nav rollout defaults (formerly B2; `B2 = text_nav`).**
Current configs under `experiments/rollout_qwen35_sglang_visual/configs/` use:
- `max_steps: 25` in both `rollout` and `env`;
- `context_token_threshold: 26000`;
- `image_turns_kept: 1`, `tokens_per_image: 3000`;
- `concurrency: 2`;
- simplified text-nav order/action shape: one restaurant, one customer, single item,
  prep/jitter/temperature/bag/multi-agent/advanced transport off, actions
  `[VIEW_ORDERS, ACCEPT_ORDER, PICKUP, DROP_OFF, WAIT, MOVE, NAVIGATE]`.
Older notes may still contain `B2`/`b2`; this is the old name for the same
text-navigation baseline where `NAVIGATE` exposes textual `next_move` guidance.
Current config and run names use the shorter `textnav_*` convention.

**Multi-order and feasibility tooling.**
- The underlying multi-order mechanism already existed (`max_orders_in_pool`,
  `VIEW_ORDERS`, `ACCEPT_ORDER([ids])`, and refill-on-accept). The recent
  multi-order configs make it meaningful by using a larger pool, unrestricted
  restaurants/customers, and payout jitter.
- `tools/analyze_order_feasibility.py` enumerates the raw
  restaurant/building candidate space with the same binding logic as
  `OrderManager`. For `small-city-11`, spawn `[-17.0, 256.58]`, and a 25-step
  whole-episode budget, it reports `88 / 352` raw candidate orders as
  structurally feasible (`308` unique endpoint pairs after binding).
- The live order-pool filter uses the same oracle logic but starts from the
  agent's position at `VIEW_ORDERS` time and defaults to a 20-step post-view
  budget.

**Latest 8-seed results.**

| Mode | Config | Run dir | NAVIGATE | Pickup | Delivered |
|---|---|---|---:|---:|---:|
| walk | `textnav_walk_multi_unfiltered_s80-87_25.yaml` | `textnav_walk_multi_unfiltered_s80-87_25_pv2` | 8/8 | 5/8 | 3/8 |
| e-scooter | `textnav_scooter_multi_unfiltered_s90-97_25.yaml` | `textnav_scooter_multi_unfiltered_s90-97_25_pv2` | 8/8 | 6/8 | 3/8 |

Interpretation: prompt v2.x fixed the main "agent does not call NAVIGATE"
failure mode. Remaining failures are mostly graph-step budget: `MOVE` still
advances one waypoint edge per action, so e-scooter reduces sim time/battery cost
but does **not** reduce the number of model turns needed to traverse a long route.
Examples: seeds that target `200 Cherry St` often arrive at or near pickup around
step 25 with no budget left for `PICKUP`.

### 16.9 Visual route-following mode — Qwen3.5-VL-9B training target

The current visual training target is no longer "make the agent call
`NAVIGATE`." In visual route-following mode the agent already calls `NAVIGATE`
reliably. The remaining target is VLM reasoning: read the top-down route image
and convert the highlighted route plus current facing into the correct primitive
`MOVE(direction=...)`.

**Current evaluation configuration.**

Use the visual single-order walk config as the controlled probe:

- config: `visnav_walk_single_feasible_s100-107_25.yaml`
- task mode: `visual_route_following`
- model: `qwen3.5-9b`
- backend: local SGLang, grammar off
- transport: walk
- order pool: feasible-only, single-order
- seeds: `[100, 107]`
- max steps: `25`
- actions: `VIEW_ORDERS`, `ACCEPT_ORDER`, `NAVIGATE`, `MOVE`, `PICKUP`,
  `DROP_OFF`, `WAIT`

This mode preserves the complete delivery workflow. It is not a vacuum
route-image-to-action task.

**Policy observation contract.**

- `NAVIGATE` remains available to the policy.
- Textual `next_move`, `oracle_next_move`, and full oracle route must not appear
  in policy-visible system prompts or observations.
- Oracle fields remain available in `info` for debugging, evaluation, SFT label
  extraction, and reward shaping.
- Visual observations include FPV plus a top-down map with current location,
  pickup/dropoff markers, active blue route, route arrows when they fit, and the
  orange facing arrow.
- Stale route-estimate fields are exposed as `planned_route_*`, e.g.
  `planned_route_distance_m` and `planned_route_estimated_time`. The prompt says
  these are estimates from the last `NAVIGATE` route plan and must not be used to
  judge arrival.
- Arrival and task transitions are gated by the visual route plus
  `[pickup_hint]` / `[dropoff_hint]`.

**Recent prompt/render cleanup.**

- Fixed the output JSON example from double braces to a normal single JSON
  object.
- Shortened `reasoning_and_reflection` and `future_plan` instructions so the
  model is not encouraged to write long self-reflection.
- Kept the long-horizon courier objective and order-selection framing.
- Added a visual-mode short `NAVIGATE` action-space line; detailed visual route
  semantics live in the workflow and `vision_inputs` blocks.
- Clarified that the blue highlighted line is the route; blue-line arrows may
  appear on longer segments but may be absent near the final step; the orange
  arrow attached to the agent marker is facing only, not a route arrow.
- Added/kept renderer improvements for route readability: active-order labels,
  route waypoint labels in visual mode, compact legend text, and a larger
  navigation crop margin so legend/text is less likely to be truncated.
- Repeated same-target `NAVIGATE` is a no-op guard that returns route-already
  active semantics instead of changing state.

**Focused verification.**

Run:

```bash
python -m vagen.envs.deliverybench.test_visual_route_following
python -m py_compile vagen/envs/deliverybench/deliverybench_env.py \
  vagen/envs/deliverybench/vlm_delivery/gameplay/action_space.py \
  vagen/envs/deliverybench/vlm_delivery/gameplay/prompt.py
```

The focused visual test checks that:

- policy text does not leak `next_move` / oracle fields;
- oracle next action remains available in `info`;
- visual route rendering still draws waypoint labels;
- visual observations keep `[navigation]` while hiding textual next-step hints.

**Latest controlled visual rollout.**

Run:

`visnav_walk_single_feasible_s100-107_25_promptfix_plannedroute_20260622_130308`

Metrics:

- called `NAVIGATE`: 8/8 = 100%
- arrived pickup: 1/8 = 12.5%
- delivered: 0/8 = 0%
- oracle MOVE turns: 169
- correct MOVE: 81/169 = 47.9%
- wrong but valid MOVE: 76
- failed MOVE: 12
- parser no-valid: 0
- repeated `NAVIGATE` when oracle expected MOVE: 3
- failed pickup: 2

Compared with prior same-seed visual runs, the config and order-generation
settings are essentially the same; the differences are prompt/renderer/runtime
state. The cleanup removed parser/no-valid failures but did not materially solve
visual route following. The consistent bottleneck is still selecting the correct
`MOVE(direction=...)` from the visual route image.

**Training implication.**

It is reasonable to start SFT/RL design now. Use realistic DeliveryBench
trajectories, not an isolated route-following-only environment. For SFT, label
oracle MOVE turns with `oracle_next_action_before_action`. For RL, reward correct
visual MOVE direction, route progress, successful pickup/dropoff, and penalize
failed MOVE plus premature `PICKUP`/`DROP_OFF`. Keep evaluation in the complete
workflow so gains reflect delivery competence rather than a toy route-image
classifier.

### 16.10 Minimal visual SFT export pipeline

The first implemented training component is an oracle SFT-data exporter:

- exporter: `vagen/envs/deliverybench/tools/generate_visual_sft_data.py`
- smoke test: `vagen/envs/deliverybench/test_visual_sft_export.py`
- output: parquet with `messages`, `images`, `seed`, `turn_index`, `stage`,
  and `action`
- image convention: each row stores pre-action visual observations as absolute
  image paths; `<image>` placeholders in the user message must match the image
  count

The exporter preserves the real DeliveryBench workflow:

`VIEW_ORDERS -> ACCEPT_ORDER -> NAVIGATE(pickup) -> MOVE... -> PICKUP -> NAVIGATE(dropoff) -> MOVE... -> DROP_OFF`

It should be used before attempting RL. It can generate full-workflow examples
or a `--move-only` subset emphasizing visual route-following decisions while
still sampling from realistic delivery trajectories. `--move-repeat` may be used
to upsample MOVE examples without changing the environment.

Leakage contract:

- policy messages must not contain textual `next_move`, `oracle_next_move`, or
  `oracle_next_action`;
- oracle labels may come from env `info`, especially
  `oracle_next_action_before_action`;
- oracle fields remain valid for SFT, reward shaping, debugging, and
  evaluation, but not for policy observation.

Run focused checks after changing the exporter or visual observation contract:

```bash
python -m vagen.envs.deliverybench.test_visual_sft_export
python -m vagen.envs.deliverybench.test_visual_route_following
python -m py_compile vagen/envs/deliverybench/tools/generate_visual_sft_data.py
```

Current training caveats:

- the `vagen` Python environment has not yet been confirmed as Qwen3.5-VL SFT
  ready because its Transformers stack does not correctly load `qwen3_5`;
- `qwen_env` can load Qwen3.5-VL classes but still needs the training stack
  aligned;
- Qwen3.5-VL-9B LoRA SFT likely requires freeing suitable GPU memory first;
- multimodal multi-turn GRPO should wait until after SFT export and a LoRA SFT
  smoke test succeed.
