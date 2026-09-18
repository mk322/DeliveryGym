# Rendering the DeliveryBench FPV images — runbook

This is the start-to-finish guide for (re)generating the first-person-view (FPV)
image dataset a map needs: the plain per-waypoint views **plus** the traffic-light
(green/red) and road-block (obstacle) variants. One command does all of it.

> TL;DR — with a SimWorld Studio server running (city/roads loaded in the level)
> on MCP port 55565:
> ```bash
> python -m vagen.envs.deliverybench.tools.render_fpv_dataset_ue \
>     vagen/envs/deliverybench/maps/<map> --mcp-port 55565
> ```
> Output lands in `vagen/envs/deliverybench/deliverybench_fpv/<map>/`
> (`images/`, `manifest.jsonl`, `obstacles.json`). Then point the env at it and
> flip the hazard flags (see "Use it in the env" below).
>
> The driver **delegates rendering to the proven per-kind UE scripts**
> (`render_waypoint_pedestrian_light_ue` for plain+light, `render_dock_obstacle_ue`
> for blocks); it only plans jobs and assigns canonical filenames. Its defaults
> **reproduce `deliverybench_fpv/<map>/ue_render_setup_record.md`** (the config
> hosted on the MCP/editor server):
> - traffic lights → **face-aligned** camera 1600 cm in front of the signal face,
>   z 170, 1280×720, light render-location `center + (json−center)×10`;
> - plain → z 160, 640×480;
> - world assets **are spawned** from progen JSON (the .md spawned them). Pass
>   `--no-spawn-map-assets` only if the loaded level already holds the full city.

---

## 0. What gets rendered (canonical naming)

All images live under `deliverybench_fpv/<map>/images/<waypoint_id>/` and are
joined to the runtime by **`(round(x_cm,1), round(y_cm,1), yaw)`** (never by
`waypoint_id` — those drift). The renderer also writes one `manifest.jsonl`
(one row per image) and one `obstacles.json` sidecar.

| file | `render_kind` | what / when |
|---|---|---|
| `yaw_<NNN>.png` | `plain` | the agent's view — **every** waypoint × 4 yaws |
| `yaw_<NNN>_green.png` | `traffic_light` | intersection crossing face, light green |
| `yaw_<NNN>_red.png` | `traffic_light` | intersection crossing face, light red |
| `yaw_<NNN>_blocked.png` | `obstacle` | sampled dock, `RoadCone_C` ahead (clear twin = the plain view) |

`<NNN>` is the stored FPV yaw, `yaw = (90 − compass_dir) % 360` → 000=E, 090=N,
180=W, 270=S. Every kind is shot from the **agent's camera at the waypoint**, so
the plain / green / red / blocked images at one `(waypoint, yaw)` differ only by
what is spawned in front — that is what lets the env swap them per panel.

Traffic-light rule (decided at runtime, not baked): odd wall-minute → North-South
red / East-West green; even minute flips. The renderer just needs **both** colours
to exist for each crossing face.

---

## 1. Prerequisites (one-time)

You need a running **SimWorld Studio** server with the map loaded and an MCP
(TCP) port the renderer can talk to. Example, matching the original capture:

```bash
cd /path/to/SimWorld-Studio
UE_ROOT=/path/to/UE_5.3.2 \
UE_PROJECT_PATH=/path/to/SimWorld/SimWorld.uproject \
UCV_PORT=9001 \
.venv/bin/simworld-studio start \
  --data-dir "$PWD/simworld_studio_workspace" \
  --gpu 0 --port 3008 \
  --mcp-port 55565 \
  --cirrus-http-port 8695 --cirrus-ws-port 8696 --cirrus-sfu-port 8994 \
  --map /Game/Main --skip-auth-check
```

Also have the asset catalogue used to spawn the generated world:
`simworld/data/ue_assets.json` in the DeliveryBench checkout, or `$UE_ASSETS_JSON` (pass a different path
with `--ue-assets-json` if yours differs). The road-block uses
`/Game/CityDatabase/blueprints/RoadCone.RoadCone_C` (override with `--cone-asset`).

> The exact engine/exposure/sky settings are **not** captured in the dataset
> metadata — see `deliverybench_fpv/<map>/ue_render_setup_record.md` for what is
> and isn't recorded.

---

## 2. Dry-run first (no UE needed)

Validate the plan, naming, sample, and sidecar before spending render time:

```bash
python -m vagen.envs.deliverybench.tools.render_fpv_dataset_ue \
    vagen/envs/deliverybench/maps/small-city-11 --dry-run
```

It prints the job counts (e.g. `plain=544 traffic_light=116 obstacle=5`) and
writes `manifest.jsonl` + `obstacles.json` without contacting UE. Good for code
review and for checking the obstacle sample.

---

## 3. Render everything

```bash
python -m vagen.envs.deliverybench.tools.render_fpv_dataset_ue \
    vagen/envs/deliverybench/maps/<map> --mcp-port 55565
#   add --no-spawn-map-assets if the loaded level already holds the full city
```

To render from a JSON-only city scene, start the same server but pass
`--clear-existing-scene`. The first render command destroys existing visible/map
actors from the loaded UE level, keeps lighting/sky/fog/postprocess/camera
infrastructure, then spawns only the non-pedestrian-light assets listed in the
map's `progen_world_enriched.json`:

```bash
python -m vagen.envs.deliverybench.tools.render_fpv_dataset_ue \
    vagen/envs/deliverybench/maps/<map> \
    --out-dir vagen/envs/deliverybench/deliverybench_fpv/<map>-json-only \
    --mcp-port 55565 \
    --clear-existing-scene
```

Useful flags (defaults reproduce `ue_render_setup_record.md`):

| flag | default | meaning |
|---|---|---|
| `--out-dir` | `deliverybench_fpv/<map>` | where images/manifest go |
| `--sample-frac` / `--seed` | `0.05` / `5` | fraction of docks blocked + RNG seed (reproducible) |
| `--obstacle-dist-cm` | `300` | how far ahead (3 m) the cone sits |
| `--cone-scale` | `5.0` | actor scale of the road cone (bigger = clearer) |
| `--light-camera-mode` | `face` | `face` (.md) = camera 1600 cm in front of the signal face; `waypoint` = agent forward view |
| `--camera-distance` | `1600` | face-mode distance in front of the signal |
| `--light-z` / `--normal-z` | `170` / `160` | camera height for light vs plain renders |
| `--light-render-xy-scale` | `10` | render-location fix: `center + (json−center)×scale` |
| `--light-asset-scale` | `2.0` | actor scale of the spawned signal (.md whole-dataset value) |
| `--light-image-width/height` | `1280`/`720` | traffic-light resolution |
| `--normal-image-width/height` | `640`/`480` | plain resolution |
| `--camera-backoff-cm` | `900` | waypoint-mode (plain) backoff |
| `--camera-fov` | `90` | viewport FOV |
| `--spawn-map-assets` / `--no-spawn-map-assets` | **on** | spawn progen world (the .md did); turn off if the level already has the city |
| `--clear-existing-scene` | off | before the first render, remove existing visible/map actors from the loaded UE level, then spawn the JSON-defined map assets |
| `--only {plain,traffic_light,obstacle}` | all | render just one kind |
| `--max-waypoints N` | all | smoke test on the first N waypoints |

Re-running is safe: it overwrites `images/` + `manifest.jsonl` + `obstacles.json`
for the map. If a render still looks off, first reproduce a single light with the
proven tool (`render_waypoint_pedestrian_light_ue.py <map_dir> --max-waypoints 1
--camera-mode face --camera-backoff-cm -1600 --camera-z 170`) to confirm the
server/level is good, then compare its output to this driver's.

---

## 4. Use it in the env

Point the config at the rendered dataset and enable the mechanics:

```python
DeliveryBenchEnvConfig(
    map_name="small-city-11",
    fpv_dir=".../deliverybench_fpv/small-city-11",  # default if omitted
    render_mode="vision", enable_fpv=True,           # hazards are vision-only
    enable_traffic_lights=True,                      # signalised set built from the manifest
    enable_obstacles=True,                           # reads obstacles.json
)
```

The env builds the traffic-light controller **from the manifest** (no separate
sidecar needed) and reads `obstacles.json` for the blocked edges. Red-light
crossings are counted (`info["metrics"]["traj_metrics"]["traffic_violations"]`),
obstacles are a hard block cleared by `BYPASS()`
(`...["collisions"]`). Enabling `enable_obstacles`/`enable_traffic_lights`
auto-adds `PASSBY`/`WAIT` to the action vocabulary. If a flag is on but the
manifest/sidecar has no matching rows, `reset()` fails loudly.

---

## 5. Targeted reruns (optional)

The unified driver supersedes these, but they remain for re-shooting one kind:

```bash
# lights only (per signalised crossing face):
python -m vagen.envs.deliverybench.tools.render_waypoint_pedestrian_light_ue <map_dir> --mcp-port 55565
# road blocks only (5% docks, dock->next-dock edge):
python -m vagen.envs.deliverybench.tools.render_dock_obstacle_ue <map_dir> --mcp-port 55565 --spawn-map-assets
```

---

## 6. Troubleshooting

- **All panels grey / no FPV** — the env joins by position+yaw; confirm
  `manifest.jsonl` rows have the right `x_cm/y_cm/yaw` and the `images/<wp>/…`
  files exist. `--dry-run` shows the planned paths.
- **`reset()` raises "no signalised intersections"** — the manifest has no
  `render_kind:"traffic_light"` rows; render lights (or disable the flag).
- **Signal renders off to the side / wrong size** — tune `--light-xy-scale`,
  `--camera-backoff-cm`, `--camera-z` (the close-up look in the old
  `small-city-11-new` used a different, light-aligned camera; this driver uses
  the agent's waypoint camera so the agent sees what it would actually see).
- **Cone not found** — pass the correct `--cone-asset` path for your project.

See also: `../OBSTACLE_TRAFFIC_DESIGN.md`, `../deliverybench_fpv/FPV_CAPTURE_PLAN.md`,
`../TRAFFIC_LIGHT_AUDIT.md`.
