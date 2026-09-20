# Traffic-light & obstacle wiring audit + unification (2026-06)

Context: the traffic-light + road-block work was done in parallel by two people,
so the runtime ended up with **three** overlapping code paths and a schema that
did **not** match the FPV images that were actually rendered
(`deliverybench_fpv/small-city-11-new`). This note records what was found and how
it was unified. Companion docs: `OBSTACLE_TRAFFIC_DESIGN.md`,
`deliverybench_fpv/FPV_CAPTURE_PLAN.md`.

## What was found

### Three traffic-light code paths

| # | File | Built from | Behaviour on red | Status |
|---|------|-----------|------------------|--------|
| 1 | `vlm_delivery/utils/hazards.py` `TrafficController` | `traffic_lights.json` sidecar, position-keyed, minute-parity | **count only** | kept (now the single runtime) |
| 2 | `vlm_delivery/utils/traffic_lights.py` `edge_signal_check` | world-JSON pedestrian-light faces, geometric face selection | **time + energy penalty** | runtime path removed |
| 3 | `utils/pedestrian_lights.py` | builds the world-JSON light *nodes* + renders | 120 s phase model (`phase_for_time`) | kept (data-gen only) |

### The double-wiring bug

`actions/move.py::handle_move` invoked **both** #1 and #2 on every `MOVE`:

- `apply_traffic_light_check(...)` (path #2) → applied `red_light_penalty_s`
  seconds + an energy multiplier, appended to `dm.traffic_light_violations`, and
  emitted ephemeral messages.
- `dm._traffic.is_signalised / light_for_bearing` (path #1) → incremented
  `dm.traffic_violation_count`.

So one red crossing was **counted twice** (two different counters) **and
penalised** — which contradicts the "for now, just count" intent.

### Schema mismatch vs the rendered images

The `small-city-11-new` manifest rows are:
`render_kind:"traffic_light"`, `signal_state:"green"|"red"`,
`signal_axis:"south-north"|"east-west"`, files `images/<int>/yaw_<NNN>_{green,red}.png`,
keyed by `(x_cm, y_cm, yaw)`. 44 faces × 2 states = 88 images; 24 intersections
have one rendered yaw-face, 10 have two (the two-axis "front vs right" case).

But the env's loader looked for a *different*, never-produced schema
(`variant:"light"` + `light_state` + `light_<state>_yaw_<NNN>.png`) and required
a `traffic_lights.json` sidecar that does not exist. Net effect: **none of the 88
light images loaded**, the deleted plain `yaw_<NNN>.png` left blank panels, and
`enable_traffic_lights=True` raised at reset.

## What was changed (unification)

**Single runtime source of truth = `TrafficController` (`utils/hazards.py`),
built from the FPV manifest** so the signalised set and per-face axes match the
rendered images.

- `move.py` / `passby.py`: only the count path remains. A red crossing moves the
  agent at normal cost and does `traffic_violation_count += 1` — no penalty,
  vision-only. The axis is taken from the **travel bearing**, so the forward and
  perpendicular directions are scored against opposite signals (front red ⇒
  counted, right green ⇒ free).
- `apply_traffic_light_check` + the `traffic_lights` import were removed from
  `move.py`; `last_traffic_light_check` / `traffic_light_violations` removed from
  `DeliveryMan` (now `_traffic`, `_obstacle_field`, `traffic_violation_count`,
  `collision_count`).
- `map/map.py`: dropped `load_traffic_lights` + `edge_signal_check`. The map only
  keeps the **geometry** flag `crosses_vehicle_road` (no red/green state, no
  text leakage). `self.traffic_lights` is now an unused empty list (back-compat).
- `vlm_delivery/utils/traffic_lights.py`: trimmed to data-gen helpers
  (`is_traffic_light_node`, `load_traffic_lights`, `signal_state_for_axis` — the
  last still used by `tools/render_waypoint_crossings_2d.py`). Its
  `signal_state_for_axis` parity matches `TrafficController` (`odd minute → NS
  red`).
- `deliverybench_env.py`: `_load_hazards` builds the controller from the manifest
  (sidecar overrides if present); `_load_fpv_lookup` reads the real
  `render_kind`/`signal_state` schema and uses the manifest's own image
  basenames; `_select_fpv_image` picks green/red per panel from
  `(minute parity, axis_of(panel direction))`.
- `gameplay/action_space.py`: coupled the hazard flags to the action vocabulary
  (`enable_obstacles → PASSBY/BYPASS`, `enable_traffic_lights → WAIT`).

## Unified data schema (one representation)

`deliverybench_fpv/<map>/manifest.jsonl`, one row per rendered view, joined to
the runtime by `(round(x_cm,1), round(y_cm,1), yaw)`:

| `render_kind` | extra fields | image basename | runtime use |
|---|---|---|---|
| *(absent)* / `plain` | — | `yaw_<NNN>.png` | plain FPV |
| `traffic_light` | `signal_state`, `signal_axis` | `yaw_<NNN>_<green\|red>.png` | light variant; signalised set + axis |
| `obstacle` | `obstacle_type`, `obstacle_state`, `edge_dst_*` | `yaw_<NNN>_blocked.png` | obstacle variant + `obstacles.json` edge |

Sidecars (optional overrides / obstacle source of truth):
- `obstacles.json` — directed `src→dst` edges (`ObstacleField`).
- `traffic_lights.json` — signalised positions + `odd_minute_red_axis` /
  `minute_period_s` (`TrafficController`), overrides the manifest-derived set.

## Unified renderer + canonical naming

`tools/render_fpv_dataset_ue.py` is the single dataset generator: it renders the
agent's camera at every waypoint × 4 yaws (plain), green+red at every intersection
crossing face, and a `RoadCone_C` blocked view for a seeded 5 % sample of docks on
the **dock→next-dock** edge. Canonical names under `images/<waypoint_id>/`:

```
yaw_<NNN>.png            # plain (render_kind: plain)
yaw_<NNN>_green.png      # render_kind: traffic_light, signal_state: green
yaw_<NNN>_red.png        # render_kind: traffic_light, signal_state: red
yaw_<NNN>_blocked.png    # render_kind: obstacle (clear twin = the plain view)
```

`<NNN>` is the stored FPV yaw (`yaw = (90 - compass_dir) % 360`). All kinds share
the agent's waypoint camera, so the variants at one `(waypoint, yaw)` differ only
by what is spawned. On `small-city-11` the plan is 544 plain + 116 light (58
crossing faces, 24 intersections with both perpendicular faces) + 5 obstacle.
Standalone tools remain for targeted reruns:
`render_waypoint_pedestrian_light_ue.py`, `render_dock_obstacle_ue.py`.
