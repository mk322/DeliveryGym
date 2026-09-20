# Any-point navigation (pixel-goal): what it is and how to evaluate it

The waypoint benchmark moves the courier along a street graph: `walk_to("Rue de
Grenelle", "east")` is one decision per junction, executed over pre-rendered
photographs. The **pixel-goal** track removes the graph. The policy looks at the
photograph in front of it (and, in the front/rear variant, the one behind it),
names a normalised point in that image — `walk_to_pixel(view="front", u=0.62,
v=0.81)` — and a pawn in a **live Unreal Engine** Paris walks there over the
engine's NavMesh. Every metre is real locomotion; the observation is a fresh
capture from wherever the pawn actually stopped.

It is the same delivery task (an order, a pickup door, a drop-off door, a fee,
a deadline, `collect()` and `hand_over()`), on the same phone map, scored by the
same `CourierEnv` summary. Only the way of moving changes.

```
policy ──── walk_to_pixel(view,u,v) ──▶ CourierSession ──▶ EmbodiedCourierEnv
                                                            │  SpearTrackBClient
                                                            ▼
                                            SimWorldEditor (UE 5.8 + SPEAR RPC)
                                            SpPixelGoalSubsystem: raycast → NavMesh
                                            projection → MoveTo → capture pair
```

The wire contract is the dataclasses in
`embodiedbench/runtime/live/protocol.py`, one per JSON shape; the certified
order pool is `configs/pixel_goal/paris_trusted_pedestrian_pool_v3.json`,
which carries its own audit, hashes and engine certification block, and the
loader in `tools/pixel_goal_order_pool.py` refuses a pool that fails any of
them.

## What a run measures

One run = one delivery scenario from the certified pool, one model, one UE
boot. `delivery_report.json` records the transcript, every engine event, the
frames the model saw, and the `CourierEnv` summary — `delivered`, `earnings`,
`late`, `walked_m`, `rejected_actions`, `sim_seconds`. The summary is what the
waypoint benchmark reports too, so the two tracks read on one scale
(dollars per shift, deliveries).

The pool's comparison protocol (30–80 m, at least one turn, through a marked
crossing) has **16 eligible directed scenarios**: four orders — 14 → 11 and
11 → 14 Rue Oberkampf (37.1 m), 11 → 16 and 16 → 11 (57.7 m) — each from
the four certified spawns. Random mode with seeds
`0 1 2 3 5 6 7 9 12 14 15 16 17 18 23 31` covers each of them once. Seeds 6,
7 and 9 carry the audited order (11 → 16 Rue Oberkampf, 57.7 m, three turns,
crosswalk 94) from three different spawns; fixed mode with that pickup and
drop-off and no `--spawn-id` is seed 6's scenario, `scenario-4ca424faacfb8dbc`
on pool v3 (`scenario-1034149f06cf7117` on v2). A scenario id hashes the
pool file's bytes, so it changes whenever the pool is re-pinned; the seed →
order mapping did not change from v2 to v3 (the sixteen scenarios come out
in the same order, so the split below holds unchanged).

**The split.** The sixteen scenarios are divided once and for all in
`tools/pixel_goal_order_pool.py` (`PROTOCOL_SPLITS`, checked by
`tests/test_pixel_goal_protocol.py`):

| split | seeds | what it is for |
|---|---|---|
| `dev` | 0 1 2 6 7 9 | the six scenarios looked at while the harness and the prompt were built; every 4B run, relay run and oracle walk before the split used one of them (seed 9 is the one the harness was debugged on) |
| `heldout` | 3 5 12 14 15 16 17 18 23 31 | the ten scenarios never run, read or tuned against before the split; **a reported number is a number on these** |

Run and tune on `dev`; report on `heldout`; never move a seed across. The
limit is stated with it: four doors carry all sixteen scenarios, so the
split holds out scenarios, not doors — three of the four orders occur in
`dev` (only 16 → 11 does not), and every held-out scenario shares its doors
with a dev one. A door- or street-level held-out set needs a second
certified region, which is a live audit of another block
(`tools/audit_trusted_pedestrian_graph_live.py`,
`tools/build_trusted_pedestrian_order_pool.py`) and the reason the pool is
versioned. Nothing trains on this track today; if something does, it
trains on `dev` scenarios and the rule above is what keeps the number
honest.

## Requirements (this is the expensive part)

| what | where it is read | how to provide it |
|---|---|---|
| **A SimWorld UE editor build carrying `SpPixelGoalSubsystem`** | `.simworld-ue/` at the repo root (symlink is fine) | **Ask the authors for the bundle**: the Unreal project it is built from is private, so it cannot be built outside the team. It is the Linux `SimWorldEditor` bundle built from the SimWorld_SPEAR UE project with the engine-side code in `ue/` applied (`ue/README.md`: nine source files, of which three are new and six carry the any-point changes the patch beside them records; the build the numbers were made on is branch `pixel-goal-1b-poc`, commit `7a1131a`, of the team's fork). The identity helper pins the seven bundle files by SHA-256 (`tools/pixel_goal_launcher_identity.py`); a differently built binary is refused, so the build has to be the one those hashes name — or the hashes updated deliberately. Copy `Saved/DerivedDataCache` with it (3 GB) or the first Paris load takes far longer than the 30-minute readiness cap. |
| CityCore Paris content | `CITYCORE_PARIS_CONTENT` (required) | The Epic marketplace asset, **read-only** to the launching user (the launcher refuses a writable copy). Symlinked into the build's `Content/` by the launcher. |
| SPEAR python client | `SIMWORLD_SPEAR_PYTHON`, `SIMWORLD_SPEAR_EXT_PYTHON` | The `spear` package from the SPEAR fork plus its built `python_ext` (`spear_ext.abi3.so`, cp312). Both go on `PYTHONPATH`; the launcher does that itself. |
| Interpreter | `PIXEL_GOAL_PYTHON` (default: `python3` on PATH) | Python 3.12 with `cairosvg` (+libcairo), Pillow, pydantic 2, pyyaml, and SPEAR's deps (`yacs`, numpy, psutil, scipy, opencv-python, msgpack-rpc-python, mcp). The launcher checks cairosvg before touching the engine. |
| Model endpoint | `QWEN_ENDPOINT` (`…/v1/chat/completions`), `QWEN_MODEL_NAME` | An OpenAI-compatible server listing the model id under `/v1/models`. The runner accepts only the model ids in `AUDITED_MODEL_LABELS` (`qwen3-vl-4b`, `qwen3-vl-8b`, `gpt-5.6-codex-interactive`, `claude-fable-5.1-interactive`, `human-interactive`) so a report always names what produced it; add a line there for a new model. A person or an interactive model answers through `tools/relay_policy_server.py`, which is an OpenAI-compatible endpoint whose replies come from a file, under exactly the rules a served model gets. Serve at least 3 images per prompt. The launcher never starts or stops this server. |
| Two GPUs | `PIXEL_GOAL_GPU` (UE, `-graphicsadapter` index), `QWEN_GPU` | Must differ. The UE card must have < 4 GiB in use (`PIXEL_GOAL_ALLOW_SHARED_UE_GPU=1` raises that to 8 GiB). |
| A free RPC port | `SIMWORLD_RPC_PORT` (default 30155) | Loopback only. |
| binutils | — | `readelf`, `nm` for the identity checks. |

Everything else — the certified pool and its Recast surface audit
(`configs/pixel_goal/`), the vendored Paris map — ships in the repository.

## Running it

```bash
# 1. serve the model (example: vLLM 0.11, one 24 GB card; the 4B fits with room)
CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct --served-model-name qwen3-vl-4b \
  --port 30001 --max-model-len 12288 --limit-mm-per-prompt '{"image": 16}'

# 2. one delivery, the audited scenario, fresh UE boot, report under artifacts/
PIXEL_GOAL_PYTHON=/path/to/py312 PIXEL_GOAL_GPU=7 QWEN_GPU=5 \
QWEN_ENDPOINT=http://127.0.0.1:30001/v1/chat/completions QWEN_MODEL_NAME=qwen3-vl-4b \
bash tools/run_pixel_goal_front_rear_delivery.sh \
  --route-profile validated_pool --order-mode fixed \
  --spawn-id spawn-near-citycore-building-0529 \
  --pickup-id stop-citycore-building-0529 --dropoff-id stop-citycore-building-0535 \
  --min-delivery-m 30 --max-delivery-m 80 --min-route-turns 1 --require-marked-crossing

# 3. the protocol, one split at a time: the same launcher with
#    --order-mode random --seed N for each seed of the split (one UE boot
#    each), then one results file with every seed's row
PIXEL_GOAL_PROTOCOL_OUT=results/anypoint/heldout PIXEL_GOAL_PROTOCOL_LABEL=four_view_heldout \
bash tools/run_pixel_goal_protocol.sh heldout      # or dev, protocol, or "3 5 12"
```

The harness's own oracle -- a policy-shaped script that always points at
the certified route, reading the run's engine events for the capture
poses -- shows what a scenario's harness can carry and what it pays:

```bash
# 4. the harness oracle on one scenario: the relay is the model endpoint,
#    the launcher runs as for any model, the oracle answers the relay's turns
python tools/relay_policy_server.py --dir /tmp/relay --port 8601 --model harness-oracle &
QWEN_ENDPOINT=http://127.0.0.1:8601/v1/chat/completions QWEN_MODEL_NAME=harness-oracle \
PIXEL_GOAL_OUTPUT=results/oracle/seed-9 bash tools/run_pixel_goal_front_rear_delivery.sh \
  --route-profile validated_pool --order-mode random --seed 9 \
  --min-delivery-m 30 --max-delivery-m 80 --min-route-turns 1 --require-marked-crossing &
python tools/pixel_goal_oracle.py --relay-dir /tmp/relay --events results/oracle/seed-9/events.jsonl --seed 9
```

Its report goes through the same validator as a model's and is grouped
apart in the results file (`harness_oracle_*`): the number it gives is
the ceiling the harness allows, never a benchmark row.

`run_pixel_goal_protocol.sh` takes the launcher's environment, adds the
protocol's own flags, and ends by running `tools/pixel_goal_results.py`,
which refuses to write a held-out results file unless every held-out seed
has a report and no other seed does. Its `results.json` is the layout of
the results file it writes: one group per harness version
or model, every row naming the scenario it resolved and the harness rules
the report declares.

`--list-order-stops` prints the certified spawn and stop ids without touching
the engine. The runner writes `delivery_report.json` (validated against the
pool before it is written), `events.jsonl` (every RPC), `album/` and `frames/`
(what the model saw), and `ue.log`.

## The harness since 2026-09-12: four views, and a walk along pedestrian ways

The first release of this track showed two photographs (front, rear) and
let the engine walk the straight line to the pixel or refuse it. Three
relay runs by a frontier model showed what that could not do (the limits
at the end of this document): after any walk across the pavement both
views looked away from it, a crossing could only be entered from a pose
that happened to line up with its stripes, and a walk could end on a
patch of floor with no legal way off. The runner's default (`--views
quad`) now does the following; `--views pair` is the original behaviour,
kept for comparison.

- **Four photographs a turn**: `front`, `left`, `right`, `rear`, the
  whole horizon. The engine's capture is a front/rear pair at the pawn's
  facing; the harness turns the pawn a quarter turn to the right through
  SPEAR (`K2_SetActorRotation`), takes a second pair (its front is the
  courier's right, its rear the left), and turns the pawn back. The
  captures are about a second apart in a static scene. The engine resolves
  a pixel against the snapshot it was taken from, whichever pair that was,
  as long as the pawn stands as it did for that snapshot (checked live: a
  first-pair pixel resolved after the second pair had been taken; a
  second-pair pixel was "stale" until the pawn was turned to that pair's
  yaw again), so the harness turns the pawn to the selected pair's yaw
  before resolving and back afterwards.
- **A pixel names a destination; the harness walks there.** The pixel is
  resolved without walking: the same `ResolveAndMove` call with an
  acceptance radius wider than the map, which the engine completes on the
  spot after the ray hit, the NavMesh projection and its verdict on the
  straight path (measured: zero distance, zero seconds). The hit is then
  read against the certified pedestrian graph:
  - on a pavement, a marked crossing, a paved island or a floor patch, the
    nearest certified node within **1.5 m** is the destination;
  - on an object or the base of a wall no higher than **60 cm** above the
    ground (a kerb stone, a planter's foot), the same: the paving beside
    it; higher up a wall, refused;
  - on the carriageway, only a node within **60 cm** (the gutter); otherwise
    refused, with the hint that the far pavement is a destination too;
  - the sky, and points with no certified node in range, refused.
  The route to the destination is planned over the pool's **certified
  legs** (the section below): the cheapest sequence of straight moves in
  metres, ties broken towards fewer moves, each move riding a leg the
  engine accepted from the very node it starts on. A move is walked by
  turning the pawn to face the leg's end, taking a capture, and giving the
  engine the pixel that end projects to (`project_world_point_to_pixel`;
  camera location and yaw from the capture, fixed intrinsics); a move that
  stops at a node inside the leg lets the acceptance radius stop the pawn
  there, which is how a node nearer than the picture's bottom edge (about
  2.9 m) is reached -- by aiming past it along a certified leg. The engine
  keeps its verdict on every leg; a leg it refuses is logged and the walk
  is planned again from the node the pawn stands on without that leg (at
  most four times an action). Before every move the pawn is set back onto
  its node when the controller left it a step to 75 cm off it (the legs
  were certified from the nodes; `world_nudge` in the report); further off
  than that the harness has lost the pawn and says so (`pedestrian_off_node`)
  rather than walk an uncertified line. The harness always knows the node
  the pawn stands on: each accepted move's stop node, never a map-matcher's
  guess. Measured on seed 9's approach (57 m, with the certified crossing):
  every leg accepted, landing 5-30 cm from its node, the pickup door
  reached within 15 cm of its anchor.
- **One action, one turn, the route's length at the declared walking
  speed** on the clock (the engine's own seconds are recorded beside the
  charge; see the pawn-speed note below). The pawn ends every action on a
  certified node, facing along its last leg, and the next observation is
  taken there. A refused pixel still costs a turn, as before. A door counts
  as reached within 3 m (the picture's bottom edge is 2.85 m away).
- **Checked end to end** before any model saw it: a harness oracle behind
  the relay endpoint (model id `harness-oracle`, a script that points at
  the certified route's next point using the run's own capture poses; not
  a policy, never a benchmark number) walked seed 9 under the real runner:
  pickup at turn 14, hand-over at turn 26, 140 m, 35 certified legs of
  which 34 accepted, every action ending on a certified node, the report
  validated.
- **The prompt** describes the four photographs and says what the walk does
  (`QUAD_MOVEMENT_CONTRACT` in `embodiedbench/agent/courier/prompts.py`).
  The tool is still `walk_to_pixel(view, u, v)`, with four labels.

What this changes about the task: the policy no longer has to plan the
entry into a crossing or keep the pawn facing along the pavement; it has
to look at four pictures, find the pavement that goes the route's way, and
point at it. Reading the pictures is still the whole task. Under the
four-view harness the zero-shot 4B walks far further than it did under the
two-view one and collects the order on some development scenarios, but the
drop-off leg defeats it: it points at the carriageway or at a wall and
repeats that pixel until the stuck rule ends the run. The harness's own
oracle, which points at the certified route instead of reading the
pictures, completes the held-out scenarios on time and without a refused
action, so what stands between a model and a delivery here is choosing the
pixel -- which is what this track measures. Run the protocol yourself for
the numbers, or read them in the paper; groups are labelled by harness
version and pool version, and a group is only ever a whole split.

One limit of the routing is the pool itself: a pavement pixel outside
the certified region has no node within 1.5 m and is refused as "not on a
pedestrian way you can reach" (seed 2 met it thirteen times). The
certified region covers the protocol's routes with a margin; a policy
that wanders off them is told so rather than walked onto uncertified
ground.

## The pool, certified by the engine (v3): directed legs

The v2 pool's edges were certified against the Recast surface under them.
The engine judges a walk by something else -- whether the controller path
from where the pawn stands to the point it was given crosses a road
polygon outside a marked crossing, and what the ray it was aimed by hit --
and the held-out oracle pass of 2026-09-13 found the two disagreeing at
the traffic islands between crossings 92 and 94 (three of its ten
scenarios lost there, "Limits" below).

**The engine's verdict is on a leg, not an edge.** A first attempt at v3
(2026-09-14, 01:40 UTC, since withdrawn) judged every straight run of
edges from its start node and dropped the 45 edges of the runs it refused.
That was the wrong unit: a verdict is on one directed walk from one pose
to one aim -- the same paving is accepted one way and refused the other
(`recast-grid--186--23 -> --186--19` accepted, the reverse refused), and a
long run refused at its far end says nothing about the short walks along
it (15 of the 45 dropped edges had been walked by 139 accepted legs in the
runs of the two days before). The held-out oracle on that pool got stuck
on its first scenario, 4.4 m from the pickup door. The same night showed a
second fault in the harness itself: after a walk the engine cut short, the
pawn's node was taken from the waypoint track's map-matcher, which put it
on a node 1.4 m away, and every later leg -- planned from a node the pawn
was not on -- was refused. Both are gone: the pool is certified by the
unit the engine judges, and the harness tracks the pawn's node exactly.

Pool v3 is v2 with a table of the **directed legs the engine accepted**,
and the harness walks nothing else:

- A **leg** is one straight walk between two pool nodes, 3 to 10 m long
  (the picture's bottom edge is 2.9 m away, so nothing nearer can be aimed
  at; longer walks are chained), whose chain of pool nodes stays within
  40 cm of the line (the shortest edge path between the ends inside that
  band; `candidate_legs`, `leg_chain` in `tools/pixel_goal_order_pool.py`).
  The rule is geometric, so a pair of nodes is the same leg however a
  route reached it. Pool v2 offers 11,154 of them.
- The certifier (`tools/certify_pixel_goal_pool_live.py`) sets the pawn on
  each leg's start node (SPEAR `K2_TeleportTo`, the placement checked
  against the status RPC to 15 cm), turns it to face the end node,
  captures, projects the end node into the picture and resolves that
  pixel with an acceptance radius wider than the map, so the engine
  resolves and judges without walking -- exactly the question the harness
  asks before every move. Every verdict goes to `verdicts.jsonl` as it is
  made, with the controller path's length, the projected point and the
  actor the ray hit; a stale-snapshot refusal (the pawn still settling) is
  judged again, never counted.
- `tools/prune_pixel_goal_pool.py` keeps the accepted legs whose controller
  path is no longer than the chord by more than 150 cm (a path that bends
  round something is not the straight walk that was judged), only the
  nodes those legs join **both ways** (the largest strongly connected
  component, to a fixed point: a leg over a node that goes, goes too), and
  only the edges some kept leg walks over. A stop or spawn outside the
  component stops the tool. The pool records the verdict file, the counts
  and what was removed under `engine_certification`, and the loader
  refuses a table that is not one component -- a pawn on a v3 pool can
  leave every node it can reach.
- A pawn **aims only at a leg's end** and may stop at any node on the
  leg's chain (`certified_moves`): a node nearer than 3 m is reached by
  aiming past it along a certified leg and stopping early with the
  acceptance radius. There is no other kind of move.

```bash
# 1. ask the engine about every leg (one engine boot; ~3.4 s a leg, so
#    ~10 h in one shard -- or two engines with --shard 1/2 and --shard 2/2)
PIXEL_GOAL_RUNNER_PY=tools/certify_pixel_goal_pool_live.py PIXEL_GOAL_OUTPUT=results/certify \
  bash tools/run_pixel_goal_front_rear_delivery.sh --route-profile validated_pool \
  --order-mode random --seed 0 --min-delivery-m 30 --max-delivery-m 80 \
  --min-route-turns 1 --require-marked-crossing \
  --pool configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json
# 2. the verdicts into the next pool version; the loader is the judge of the result
python tools/prune_pixel_goal_pool.py --pool configs/pixel_goal/paris_trusted_pedestrian_pool_v2.json \
  --verdicts results/certify/verdicts.jsonl --out configs/pixel_goal/paris_trusted_pedestrian_pool_v3.json
```

**What the engine said about v2's legs (2026-09-14, two engines in
parallel, 11,154 legs, 3.4 s a leg):** 10,887 accepted, 267 refused --
210 `controller_path_enters_unmarked_road`, 33 `hit_not_walkable_ground`
(the ray to the end node lands on a shop front or a kerb), 18
`controller_path_detour_exceeded`, 5 `controller_path_surface_unverified`,
1 `navmesh_projection_failed` -- and one accepted leg whose controller
path bent 156 cm past its chord, counted as refused. Refusals sit where
the earlier runs met them: the traffic islands between crossings 92 and
94, the pavement beside crossing 94, and two nodes the engine accepts no
leg out of at all -- `recast-grid--186--19` (where the held-out oracle
was trapped 4.4 m from the pickup door on the withdrawn v3) and
`recast-grid--188--15`. **Pool v3** keeps 10,884 legs (the accepted ones
that join the component both ways), 538 of the 540 nodes (those two
gone), one node as pass-through (`crosswalk:PR_Crossswalk_94:150`, the
middle of crossing 94: the engine crosses end to end but accepts no leg
aimed at the middle, so the pawn walks over it and never stops on it),
and 1,234 of the 1,443 edges (the 209 gone are sidewalk stubs no
straight walk of 3 m covers). The same sixteen
protocol scenarios come out of the same seeds; every route is 0.6 m
longer with one more turn (the stub beside 12 Rue Oberkampf's door is
gone), and scenario ids change with the pool's bytes.

The runner, the oracle and the results tool read v3; v2 stays in the
repository as the pool the legs were judged on and as what the numbers
before 2026-09-14 were run on.

## Results so far (Qwen3-VL-4B zero-shot; older 8B and GPT-5.6 runs for scale)

See the paper for the numbers and the exact harness
revision they were produced under. Three things to know when reading them:

- The system prompt (`FRONT_REAR_SYSTEM_TEMPLATE`) carries a *choosing a good
  pixel* section since 2026-09-12: find the pavement band along the
  buildings, aim a few metres ahead in the lower third of the frame, cross
  only on the stripes of a marked crosswalk, treat the rear view as a real
  option, and what each refusal means for the next pixel. Runs before that
  date had only the `(u, v)` convention. The pixel the model names and the
  point the engine walks to were checked to be the same point (every action
  of a run overlaid on the frame the model saw, against the engine's
  verdict).
- The engine pawn walks at about **68 cm/s** of engine time (measured over
  140 m of the oracle walk: 205 engine seconds), while the deadlines are
  priced at the courier's declared 140 cm/s (`WALK_SPEED_CM_S`). Under the
  two-view harness the engine's seconds were charged as they came, which
  makes every delivery late by construction; the four-view harness charges
  a routed walk at the declared speed (`route_cm / 140`) and records the
  engine's own seconds beside it (`engine_seconds` in the movement record).
  The pixel-goal subsystem exposes no speed setter.
- Before this release, "where you are" named the target door from up to 8 m
  away while `collect()`/`hand_over()` required 1 m, and the refusal carried no
  distance; four identical refusals ended a run as `stuck`. The harness now
  names the door only where the action would succeed. Numbers produced under
  the earlier harness (the Aug-28 runs in the findings document) are not
  comparable with numbers produced now.

## Limits, stated

- Hazards are off in embodied mode (no obstacle dressing, no signal charging):
  locomotion realism is the thing under test.
- The UE-side source is in this repository under `ue/`, MIT-licensed by
  SimWorld_SPEAR: the `SpPixelGoalSubsystem` and its 25 automation tests
  (new files), the humanoid pawn, the camera capture pool and the NavMesh
  helper it drives (carrying the any-point changes, which the patch beside
  them records hunk by hunk), and nothing else from that module -- the
  agent bases, the cluster beacon and the rest of SimWorld stay there.
  `ue/README.md` says how to drop them in, what they still compile
  against and how to re-pin the launcher's bundle hashes. They build
  inside the SimWorld_SPEAR Unreal project, which stays private (owner's
  decision, 2026-09-18), and the Paris scene is Epic marketplace content.
  So the editor bundle cannot be built outside the team and is given on
  request; the launcher pins its seven files by SHA-256, which is how a
  bundle received that way is checked before a run starts. Without such a
  bundle nothing on this track runs and the waypoint track is the one that
  runs anywhere, so a reader of the paper should take the any-point numbers
  from the paper and read the rules they were produced under here, in the
  Python harness and in `ue/`.
- One delivery per engine boot, by design of the launcher: a boot is minutes,
  and the identity checks assume they own the process they started.
- The pool certifies two marked crossings (`PR_Crossswalk_92` and
  `PR_Crossswalk_94`). The Paris scene has more. A pixel on the stripes of
  any other one is refused, and since 2026-09-12 the refusal says why ("the
  only crossing that counts here is the one the blue route on your phone
  uses") and the prompt says the same before the first move. In the earlier
  re-runs the 4B, told only to aim at stripes, aimed at `PR_Crossswalk_77`
  four times in a row on seed 1 and was ended by the `stuck` rule for it.
  Certifying the other crossings would need a new live audit and is still
  open.
- A door's task anchor is the certified pavement point about half a metre
  in front of the entrance, and the pawn cannot stand closer than roughly
  1.5 m to a facade (its capsule plus the NavMesh setback). With the
  original 1 m tolerance the action succeeded only on the one NavMesh edge
  point straight in front of the anchor: a relay policy reached the door
  three times on 2026-09-12 and stopped 100, 106 and 114 cm from it. The
  tolerance became 175 cm then, and **300 cm** with the four-view harness
  (`VALIDATED_POOL_PICKUP_TOLERANCE_CM`, `CORRIDOR_DROPOFF_TOLERANCE_CM`):
  the picture's bottom edge is about 2.85 m from the camera, so a courier
  standing nearer the door than that can no longer point at it, and the
  harness's own oracle walk of seed 9 stopped 2.7 m from the door node with
  nothing left to aim at. A door whose foot has gone under the picture's
  edge is a door the courier has reached. Every reported number before
  each change is labelled. The 4B runs are unaffected by it: on seeds 0, 1, 2
  and 9 the pawn never came within 3.8 m of a pickup, and on seed 7 it
  spawned 9 cm from the pickup, was told so, collected on its first turn,
  and then never moved: every later action aimed at building fronts.
- **Handled by the four-view harness (the section above); listed for the
  two-view runs in the table.** Under `--views pair` the pawn faces the way
  its last walk went and the capture takes no yaw, so a walk ending at a
  kerb leaves the pavement in the blind sectors of both views (the second
  relay run ended `stuck` that way at its second walk); the engine's
  "outside a marked crosswalk" check reads the straight NavMesh path, so a
  crossing can only be entered from a pose that lines up with its stripes
  (the third relay run lost five actions to it and never got onto the
  drop-off route's crossing); and the engine accepts a move onto the base
  floor beside a crossing (`Template_Map_Floor`) and then refuses every
  path off it. The four-view harness turns the pawn through SPEAR, walks
  the certified graph leg by leg so that no leg's straight line ever
  crosses a carriageway outside the stripes, and ends every action on a
  certified node. The engine-side checks themselves are unchanged and
  still apply to each leg.
- **The pool's certification and the engine's road check disagree at one
  junction (found by the held-out oracle pass, 2026-09-13).** Pool v2's
  `sidewalk` edges were certified against the Recast surface under them;
  the engine judges a leg by whether its controller path crosses a road
  polygon outside a marked crosswalk. Around the traffic islands between
  crossings 92 and 94 (`PR_SidewalkIsland_42/43/46/47`, x -21300 to
  -20200, y 200 to 1700) it refuses certified legs between islands
  (`controller_path_enters_unmarked_road`, at times
  `controller_path_detour_exceeded`): 49 directed certified legs were
  refused and never accepted over the four-view runs. The approach from
  the 10 Rue Oberkampf spawn (seeds 2, 6, 14, 31) crosses three of them,
  and from island 46 (`recast-grid--202-6`) every leg is refused: the
  harness detours round one refused leg but cannot leave a node the
  engine refuses every leg from, which is where its own oracle was
  trapped on held-out seed 14. The verdict also moves with where the pawn
  stops: once, 14 cm from the end node of crossing 94, every leg was
  refused on a delivery route the oracle completed from two other spawns
  (held-out seed 16). The route is the harness's choice, not the
  policy's: a destination whose shortest certified path runs through the
  islands meets this, and the routes from the other three spawns do not,
  apart from the one refused leg every route shares
  (`recast-grid--199-22 -> --200-21`), which the detour handled. **Closed
  by pool v3** (the section above): every leg the harness can plan was
  judged by the engine from its start node, the harness walks only the
  legs it accepted, and every node of the pool can be left as well as
  reached. Scenario ids changed with the pool (the protocol seeds were
  re-derived for the same spawn, pickup and drop-off triples; the split
  is defined on those). What remains is the engine's verdict itself, which
  the harness now takes as given rather than discovers on the way.
- Each certified spawn stands in front of its building's door and was
  certified facing it. Three of the four look past the door along the
  pavement; the fourth (`spawn-near-citycore-building-0044`, seed 2 of the
  protocol) looked straight into a shopfront 0.8 m away, so its front view
  was glass and its rear view the street. Since 2026-09-12 a spawn certified
  within 1.2 m of its own door and within 35° of it is turned round at
  scenario resolution (`face_away_from_own_door`): the door goes into the
  rear view, the pavement into the front. The pool file is unchanged; the
  report records the yaw actually used. Numbers before the change are
  labelled.
