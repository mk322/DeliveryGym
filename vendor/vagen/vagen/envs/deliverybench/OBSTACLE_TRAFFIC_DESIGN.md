# DeliveryBench — Obstacles & Traffic Lights (design + status)

> Status: **runtime mechanics IMPLEMENTED & tested (2026-06-19); FPV image
> variants PENDING the next capture run.** The config flags, sidecar loaders,
> `MOVE`/`BYPASS`/`WAIT("traffic_light")` logic, and counters are wired and
> default-off; the simplest rollout (Stage 1 / current configs) is unaffected.
> The perception images (obstacle / red-green crosswalk) do not exist yet — until
> they are baked, enabling the flags raises at reset (sidecar missing) and, if a
> sidecar is present but variants are not, the FPV falls back to plain photos.
>
> Goal: make the FPV channel *load-bearing* and add a safety / social-navigation
> axis — the agent must look at the first-person view, notice a static obstacle or
> a red light, and choose the right action.
>
> Companions: `deliverybench_fpv/FPV_CAPTURE_PLAN.md` (what images to bake),
> `DELIVERYBENCH_DOC.md` §13 (developer summary).

## 0. Where it lives in code (implemented)

> **Unified 2026-06.** There is now a **single runtime traffic-light path**
> (`TrafficController`) and one shared data schema (the FPV manifest). The earlier
> duplicate world-JSON `edge_signal_check` runtime path was removed and the env
> now reads the schema that was actually rendered. See `TRAFFIC_LIGHT_AUDIT.md`.

| Piece | Location |
|---|---|
| Config flags `enable_obstacles` / `enable_traffic_lights` / `passby_cost_scale` | `deliverybench_env.py::DeliveryBenchEnvConfig` |
| Hazard load + counters + loud-fail | `deliverybench_env.py::_load_hazards` (called from `reset`); traffic lights built **from the FPV manifest** (sidecar overrides), obstacles from `obstacles.json` |
| `ObstacleField` / `TrafficController` (single runtime SSOT) | `vlm_delivery/utils/hazards.py` |
| Obstacle hard-block (collision++) + red-light violation (**count only, no penalty**) | `vlm_delivery/actions/move.py`, `actions/passby.py` |
| `WAIT("traffic_light")` | `vlm_delivery/actions/wait.py` + `gameplay/action_space.py` |
| Flag → action vocabulary (`enable_obstacles→PASSBY`, `enable_traffic_lights→WAIT`) | `gameplay/action_space.py::_MECHANIC_ACTIONS` |
| Counters in `info` | `deliverybench_env.py::step` → `traj_metrics.collisions` / `.traffic_violations` |
| FPV variant selection | `deliverybench_env.py::_select_fpv_image` / `_build_fpv_cross` (reads manifest `render_kind` rows: `traffic_light`→green/red by `(minute parity, axis)`, `obstacle`→blocked; falls back to plain) |
| Renderers | `tools/render_fpv_dataset_ue.py` (**unified**: plain + light + obstacle, canonical naming, one manifest). Standalone: `render_waypoint_pedestrian_light_ue.py` (lights), `render_dock_obstacle_ue.py` (dock→next-dock road blocks) |
| Data-gen helpers (not runtime) | `utils/pedestrian_lights.py`, `vlm_delivery/utils/traffic_lights.py` |

> **Obstacle scope:** for `MOVE`, the block is applied to the **forward** edge
> only — the head-on case the FPV shows and the one `BYPASS()` (forward-only)
> can clear. `MOVE(left/right/backward)` is not obstacle-checked; obstacles are
> sampled to be met head-on along normal `NAVIGATE` routes. Revisit if non-grid
> maps or sideways obstacle encounters become common.
>
> **F1 update (`enable_waypoint_marks`):** `MOVE_TO(k)` lets the agent choose an
> arbitrary edge, and the marked FPV shows a barrier in whichever panel faces
> it — so `handle_move_to` obstacle-checks the **chosen** edge and fails
> identically (in place, `collision_count++`, "blocked, unable to proceed").
> `BYPASS()` stays forward-bound; a `BYPASS_TO(k)` generalisation is a
> documented non-goal for F1.

---

## 1. Feature A — road obstacles

**Model (simplified).** A static obstacle sits on a **directed waypoint edge**
`src → dst`, ~2 m in front of `src` (purely a rendering placement — the agent
stays waypoint-snapped). It is **one-sided**: only `src → dst` is blocked; the
reverse edge is clear. Types: `slow_pedestrian`, `road_block`, *(others HOLD)*.
The set of obstacles is **pre-sampled and prebaked** per map in
`deliverybench_fpv/<map>/obstacles.json` (seeded; same file the FPV capture uses).

**Perception.** When the agent stands at `src` and faces `dst`, the FRONT FPV
panel uses the obstacle-variant image for that (position, yaw). So the obstacle is
detectable purely from the front view (the point of the feature). Optionally the
`### orientation` text annotates the direction `... [obstacle ahead]` — see Open
decisions (we may *withhold* the text to force visual grounding).

**Mechanics.**
- `MOVE("forward")` into an obstacle edge → **invalid**: the agent does **not**
  move (stays at `src`), `collision_count += 1`, and `info["action_error"] =
  "blocked, unable to proceed"`.
- `BYPASS()` (alias of the existing `PASSBY`) → traverses the obstacle edge to
  `dst` at `passby_cost_scale` (default 1.5×) time/energy; facing unchanged.
- On a **clear** forward edge, `MOVE("forward")` behaves normally and `BYPASS()`
  is unnecessary (see Open decisions for whether it's an error or just allowed at
  extra cost).

This finally makes `BYPASS`/`PASSBY` non-redundant with `MOVE("forward")` — it is
the *only* way across an obstacle edge, and the wrong choice is counted.

**Runtime wiring.**
- `Map` (or a thin `ObstacleField`) loads `obstacles.json` into a directed lookup
  keyed by `(round(src_x,1), round(src_y,1), bearing_to_dst)`.
- `actions/move.py`: before stepping `forward`, if the chosen edge is obstacle-
  blocked → reject + `collision_count`.
- `actions/passby.py`: rework so it requires/consumes an obstacle edge (today it's
  just a costlier forward step). Keep the 1.5× cost.
- FPV builder (`_build_fpv_cross`): when the FRONT yaw at the current waypoint
  matches an obstacle edge, swap in `obstacle_<type>_yaw_<NNN>.png`.

---

## 2. Feature B — traffic lights

**Model (simplified).** At **signalised intersections** (incident to both a N–S
and an E–W road; listed in `traffic_lights.json`) the light alternates by
wall-minute parity:
- **odd minute** → N–S **red**, E–W **green**; **even minute** → flipped.

No yellow, no per-intersection phase offset — a single global parity clock. The
relevant light is chosen by the **travel axis** of the agent's MOVE (N/S vs E/W),
exact on the cardinal-grid maps.

**Perception.** At a signalised intersection, the FPV (facing the travel
direction) shows the **red** or **green** variant for that direction at the
current minute.

**New interface: `WAIT("traffic_light")`.** Advances the clock to the **start of
the next wall-minute**, flipping the light, so the agent can cross on green. (If
the agent's axis is already green, it's a near-no-op; see Open decisions.)

**Mechanics.**
- `MOVE` across a signalised intersection whose **travel-axis light is red** →
  **valid**: the agent *does* move and pays normal cost, but
  `traffic_violation_count += 1`.
- `MOVE` on green → normal.
- `WAIT("traffic_light")` → jump to next minute boundary; observation re-renders
  with the flipped light.

**Runtime wiring.**
- A `TrafficController` reads `traffic_lights.json`; `light_for(axis, sim_time)`
  returns red/green from `floor(now_s/60) % 2` and `odd_minute_red_axis`.
- `actions/move.py`: if `src` is signalised, classify the travel bearing into
  axis, check the light; if red, `traffic_violation_count`.
- `actions/wait.py`: handle the `"traffic_light"` argument (advance to next minute
  boundary). `WAIT(minutes=N)` / `WAIT("charge_done")` unchanged.
- FPV builder: at a signalised intersection, pick `light_{green,red}_yaw_<NNN>.png`
  from `(parity, axis)`.

---

## 3. Counters, reward, pluggability

- New counters live on `DeliveryMan` and surface in `info` / `traj_metrics`:
  `collision_count`, `traffic_violation_count`. **Not wired into reward yet** —
  they are observation/metrics only, so a later penalty term (or a safety-aware
  curriculum) can consume them without touching the core loop.
- Config flags on `DeliveryBenchEnvConfig`, **default `False`**:
  `enable_obstacles`, `enable_traffic_lights`, plus `passby_cost_scale`
  (default 1.5). When off, no sidecars are loaded, FPV uses only plain images,
  and `BYPASS` / `WAIT("traffic_light")` are not added to any stage's
  `enabled_actions`.
- Stage 1/2 and the rollout config are untouched. A future "safety" stage turns
  the flags on and adds `BYPASS` (and keeps `WAIT`) to `enabled_actions`.
- Both features require the FPV dataset variants (§ capture plan). With flags on
  but variants missing, the env should **fail loudly at reset** (don't silently
  fall back to plain images, or the obstacle/light would be invisible yet active).

---

## 4. Does the simplification hold together? (review)

Yes, with the clarifications above. Specifically:
- **Waypoint-snapped movement is preserved.** "2 m from current waypoint" is only
  where the sprite is drawn; the agent never stops mid-edge. ✓
- **One-sided obstacles** need a *directed* edge lookup (not symmetric). ✓
- **Obstacle = hard block (no move), red light = soft block (moves + violation).**
  Two deliberately different semantics — physical collision vs rule-breaking —
  which is exactly the safety vs social-rules split you want to test. ✓
- **Determinism:** obstacle set is seeded/prebaked; the light is a pure function
  of `sim_time` minute parity and travel axis. With `time_scale=0` the clock only
  advances inside MOVE/WAIT/etc., so light state is reproducible. ✓
- **Feasibility is preserved:** obstacles only raise cost (BYPASS always crosses),
  they never disconnect the graph, so orders stay deliverable; the `VIEW_ORDERS`
  feasibility oracle may slightly **undercount** cost on obstacle/red-light paths
  — acceptable for v1, note it.

---

## 5. Decisions (settled 2026-06-19)

1. **`BYPASS` on a clear edge:** **allowed**, at `passby_cost_scale` (1.5×). It is
   never an error; obstacles merely *require* it and the extra cost discourages
   spamming.
2. **Text leakage:** **vision-only** for now. Obstacles and red lights are **not**
   annotated in `### orientation`; the agent must read them from the FPV. (The
   accessibility hint still shows the direction as a normal reachable edge.)
3. **`slow_pedestrian` vs `road_block`:** **treated identically** for v1 — both are
   always-present static blocks that `BYPASS` clears at cost. `type` only selects
   the FPV sprite (and a future penalty weight); no walks-away dynamics.
4. **`WAIT("traffic_light")`:** **always advances to the next wall-minute
   boundary**, regardless of the current light (so it's deterministic and simple).
5. **Signalised set:** **4-way (both-axis) intersections only.** Single-axis
   `int_*` nodes have no light and are crossed freely.
6. **Penalty weights:** **unweighted** — `collision_count` and
   `traffic_violation_count` are surfaced in `info`/`traj_metrics` only; reward is
   untouched. A penalty term can consume them later.
