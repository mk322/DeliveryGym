# Running the courier environment

Three things you might want to do: run an episode with a reference policy,
run one with a model, or evaluate a model. All three go through the same
objects. Install first (`docs/INSTALL.md`); set `ALBUMS_DIR`.

## The shape of it

```
CourierEnv              the world: streets, doors, orders, lights, barriers, a clock
CourierSession          the harness: turns the world into a prompt, a reply into a call
CourierGymEnv           the adapter the trainer and the evaluator drive (prompt + image budget)
ObservationOnlyCourier  a policy that reads only what the world says (the floor)
ShortestPathCourier     reads the graph, plans in metres   (the ceiling; never scored as a policy)
FewestMovesCourier      reads the graph, plans in turns    (the ceiling the 60-turn benchmark is read against)
```

`CourierEnv` is the only thing that knows the truth. `CourierSession`
decides nothing — it renders, dispatches and charges. That split is
deliberate: a harness that computed routes would be measuring itself.

## Run an episode

```python
import os
from pathlib import Path
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv

MAPS = Path("vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris")
ALBUMS = Path(os.environ["ALBUMS_DIR"])
net = build_road_network(MAPS, map_name="citycore-paris")

env = CourierEnv(
    net,
    seed=0,
    difficulty="endless",       # solo | pair | triple | shift | endless
    stride="block",             # block | waypoint
    embodiment="human_on_foot",
    album_root=ALBUMS / "paris_streets_v2/citycore-paris",              # carriageway views
    pavement_album_root=ALBUMS / "paris_streets_pavement/citycore-paris",  # the walker's eyes
    signal_album_root=ALBUMS / "paris_lamps_real/citycore-paris",         # pedestrian lamps
    obstacle_album_root=ALBUMS / "paris_obstacles/citycore-paris",        # barriers
    pavement_obstacle_album_root=ALBUMS / "paris_obstacles_pavement/citycore-paris",
)
env.reset()
```

The album arguments are what switch the mechanics on. Pass a signal album
and crossing on red is charged; pass an obstacle album and barriers exist.
Omit one and the corresponding mechanic is *off*, because charging for
something the photographs cannot show is the defect the whole visibility
gate exists to prevent. (The trainer and the evaluator refuse to start with
an album missing, for the same reason from the other side.)

**`difficulty`** sets how many orders and how deep the queue: `solo` 1
order, `pair` 2, `triple` 3 with 2 in hand, `shift` 10 with 3 in hand,
`endless` an unbounded stream against a one-hour clock. **The benchmark is
`endless`**: the dispatcher refills after every delivery and the score is
the money earned before the clock runs out.

**`stride`** sets how far one `walk_to` carries: `waypoint` is one waypoint
(~18 m), `block` one whole block. Same metres, same seconds, same
photographs — only the number of decisions changes.

**`embodiment`** sets what is doing the delivering:

| | speed | stamina | drain | stopping | viewpoint |
|---|---|---|---|---|---|
| `human_on_foot` (default, the benchmark) | 1.4 m/s | 100 | 0.02/m → 5 km | free | pavement |
| `human_on_scooter` | 4.2 m/s | 100 | 0.004/m → 25 km | 20 s | carriageway |
| `human_in_car` | 7.0 m/s | 100 | none | 75 s | carriageway |
| `robot_dog`, `humanoid_robot` | — | — | — | — | declared, **refuse to run** |

Stamina goes on *distance*, and a spent courier slows to 60 % rather than
stopping; `rest()` buys the tank back for a minute. A walking courier must
get the pavement albums or `env.summary()["viewpoint_matches_embodiment"]`
reports `False`. Note that the shift clock and the deadlines are priced at
walking pace whatever the embodiment; the vehicle tiers are declared for
completeness and are not part of the reported benchmark.

**Optional constraints**, all keyword flags on the constructor, all default
off except the stamina drain which has always been on:

| flag | what it adds |
|---|---|
| `enable_earning_jitter` | fees vary per job; the slip says what this one pays |
| `enable_food_temperature` | hot food goes cold 8 min after collection and pays 70 % |
| `enable_food_categories` | the slip names hot meal / ice cream (melts in 5 min, 60 %) / groceries (never spoil) |
| `enable_special_notes` | "leave at the door" is quick, "ring first" is slow |
| `enable_walking_energy` (default on) | the stamina tank above |
| `enable_phone_battery` | `navigate()` costs 4 %, every hop with the map lit 0.5 %; at zero the map is gone for the shift |
| `enable_phone_recharge` | `charge_phone()`: 90 s standing still for 40 % back |
| `deadline_slack` | scales every deadline |

Every flag has its paragraph in the system prompt and its compliance
metric in the trainer's `info` (`tests/test_constraint_flags.py` pins all of
it). A config that doesn't mention a flag runs the benchmark exactly as
published.

## Run a model against it

```python
from embodiedbench.agent.courier.session import CourierSession

session = CourierSession(env, city="Paris")
print(session.system_prompt())          # built from the tools this env enables

while not session.finished:
    observation = session.observe()
    reply = your_model(
        system=session.system_prompt(),
        text=observation.text,
        images=[f.path for f in observation.frames if f.kind == "photograph"],
        drawings=[f.svg for f in observation.frames if f.kind == "map"],
    )
    session.step(reply)

print(session.report())
```

The reply must be a short `THOUGHT:` line followed by exactly one fenced
call. Streets are named, with the bearing the junction lists them under:

    THOUGHT: the map's line runs east and photograph 1 shows the street is clear.
    ```
    walk_to("Rue de Grenelle", "east")
    ```

Three malformed replies in a row end the episode: a policy that cannot
emit an action is not being measured on navigation. `observe()` is
memoised within a turn (two calls give the same frames); `session.refresh()`
discards it if you change the world outside `step()`.

For the model side of a real evaluation use `embodiedbench-eval` (below)
or `CourierGymEnv` directly: they add the image budget (`max_images`), the
downscale (`image_max_side`) and the lamp-visibility gating that the trainer
applies, so the pictures a model is scored on are the pictures it is
trained on.

**Images versus drawings.** `observation.frames` carries two `kind`s that
must not be merged. A `photograph` came out of the world through the
courier's eyes and is the only place a traffic light, a barrier or a
shopfront ever appears. A `map` is drawn from the survey and can see
*nothing* — it is SVG, so rasterise it (`cairosvg`) if your model wants
pixels. The map is on the phone from the first turn, centred on the job in
hand, re-drawn every turn with the courier's position live; `navigate()`
re-routes it (after the courier has moved, or towards some other address).

**Frame paths are deliberately meaningless.** The albums name their files
after what is in them (`..._road_block.png`); a policy handed the raw path
could read the answer off the string. `CourierSession` hands out
content-addressed names (`a3f1c9….png`) that link to the same image.

## The verbs

| call | kind | costs | what only it can do |
|---|---|---|---|
| `walk_to(street, bearing)` | act | the walk | moves the courier one stride |
| `follow_street(street, bearing, n)` | act | the walk | several waypoints of one street in one turn (waypoint stride only) |
| `collect()` | act | 30 s | takes the parcel, at the pickup door |
| `hand_over()` | act | 30 s | gives it, at the drop-off door |
| `wait()` | act | to the end of the light's phase (1–60 s); 15 s elsewhere | sees a red light phase out |
| `rest()` | act | 60 s | refills stamina; offered only to a body that tires |
| `charge_phone()` | act | 90 s | +40 % battery (with `enable_phone_recharge`) |
| `look(street, bearing)` | look | 2 s | the door numbers down a street you are *not* on |
| `check_order()` | consult | 1 s + 1 s per live job | both ends of every job, the fee, the note |
| `check_map(address)` | consult | 5 s | where any named address is, and how far on foot |
| `navigate(address?)` | consult | 15 s (+4 % battery) | re-centres the phone's route, or points it at another address |
| `note(text)` | consult | free | writes to the courier's own notebook, shown every turn |

Every call except `note` costs simulated time. A refused action costs 5 s
and a turn; walking into a barrier costs 45 s. **There is no way to tell the
phone anything.** It routes on a survey that never learns, so it will name
a street a barrier is standing in every time; going round is worked out from
the photographs. That is the point of the environment.

## Scoring

`env.summary()` returns both currencies and the things that explain them:

```
earnings                      the money -- what ENDLESS is scored on
delivered / orders_issued     what got there; on_time, late, expired
turns, sim_seconds            what it cost, in decisions and on the clock
walked_m vs optimal_walk_m    route quality (chained from the spawn)
blocked_attempts              walked into a barrier
red_crossings, waits_at_red   crossed against a light you could see
rejected_actions              asked the world for something impossible
```

A fee is `3.00 + 0.01 × metres`, paid in full on time and in part late;
undelivered pays nothing. Report `earnings` **and** `delivered` **and**
`sim_seconds`: a policy can be cheap in turns and slow on the clock, or the
reverse.

## Evaluate a model

```bash
python -m embodiedbench.eval run --model qwen3-vl-4b \
    --base-url http://127.0.0.1:8000/v1 --split val --out results/
python -m embodiedbench.eval compare results/a.json results/b.json
```

`--split val` is the benchmark: the 64 frozen shifts (seeds 1000–1063),
**60 turns** each (the training horizon, since 2026-09-13), greedy. The
task — tier, image budget, downscale, constraint flags, the turn cap — is
read from `embodiedbench/training/vagen/val_courier.yaml`, the same file
the trainer validates on. The summary carries `earnings_mean` with
a bootstrap 95 % interval, `deliveries_mean`, `zero_delivery_rate` and the
per-seed scores; `compare` pairs two result files seed by seed. Episodes
ended by the infrastructure (a dead endpoint) are excluded from every mean
and make the exit status non-zero: a run's number is either clean or not
reported. `examples/eval_api_model.sh` serves a model with vLLM and runs
this end to end.

Validation is noisy: three runs of the *same* untrained model gave delivery
rates of 23.4 %, 14.1 % and 17.2 %. Compare paired, never two means.

### Reproducing the published numbers with Qwen3-VL-4B

Both tracks were run with `Qwen/Qwen3-VL-4B-Instruct`, greedy, served by
vLLM 0.11 under an OpenAI-compatible endpoint. This is the whole recipe;
the paper's numbers were produced by exactly these commands.

**Waypoint, the benchmark number (64 shifts, seeds 1000-1063, 60 turns).**

```bash
# 1. serve (one 24 GB card; the 4B fits with room to spare)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct --served-model-name qwen3-vl-4b \
  --port 8000 --max-model-len 16384 --limit-mm-per-prompt '{"image": 16}'
# 2. the 64 shifts; ALBUMS_DIR is where the photograph albums were unpacked
ALBUMS_DIR=/path/to/albums python -m embodiedbench.eval run \
  --model qwen3-vl-4b --base-url http://127.0.0.1:8000/v1 \
  --split val --workers 4 --transcripts --out results --tag qwen3-vl-4b
```

`results/qwen3-vl-4b.json` carries `earnings_mean` (the number) with its
bootstrap interval and, under `earnings_at_mean`, the running earnings at
turns 20, 40 and 60 averaged over the same shifts. The horizon is the
training one, 60 turns (`train_courier.yaml`), so the shift the courier is
scored on is the shift it is trained for; the training runs' own
validation logs the same checkpoints
(`val-aux/courier_paris_val/earnings_at_60/mean@1` and friends). The runs
before 2026-09-13 used a 100-turn readout with checkpoints up to 100;
greedy decoding walks the same first sixty turns either way, so their
`earnings_at_60` is the 60-turn number up to vLLM batching noise (53 of
64 shifts identical between the two runs in the audit). `--max-turns`
overrides the cap for a labelled experiment. The task -- endless tier,
one order at a time, block stride, hazards on, three 320 px photographs
and the phone's map per turn -- is read from
`embodiedbench/training/vagen/val_courier.yaml`, the file the trainer
validates on; the seeds are the immutable 1000-1063. Greedy decoding
makes a shift replayable: the same seed against the same served model
reproduces the same earnings to the cent, as the re-runs in the audit
show; a different vLLM version or GPU kernel can change a token and with
it a shift, which is why the number is quoted with its interval.

**Any-point (pixel-goal), the 16-scenario protocol.**

The engine, the CityCore content and the SPEAR client are the expensive
part; `docs/PIXEL_GOAL.md` lists each one, where it is read from, and how to
provide it. With them in place:

```bash
# the same served model; the launcher checks /v1/models for the id
PIXEL_GOAL_PYTHON=/path/to/py312 PIXEL_GOAL_GPU=1 QWEN_GPU=0 \
CITYCORE_PARIS_CONTENT=/path/to/CityCore_Paris/Content/CityCore_Paris \
SIMWORLD_SPEAR_PYTHON=/path/to/spear/python \
SIMWORLD_SPEAR_EXT_PYTHON=/path/to/spear/python_ext/python \
QWEN_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions QWEN_MODEL_NAME=qwen3-vl-4b \
PIXEL_GOAL_PROTOCOL_OUT=results/anypoint/heldout PIXEL_GOAL_PROTOCOL_LABEL=four_view_heldout \
bash tools/run_pixel_goal_protocol.sh heldout
# one seed by hand: the launcher itself, with the protocol's flags
PIXEL_GOAL_OUTPUT=results/anypoint/seed-3 \
bash tools/run_pixel_goal_front_rear_delivery.sh \
  --route-profile validated_pool --order-mode random --seed 3 \
  --min-delivery-m 30 --max-delivery-m 80 --min-route-turns 1 --require-marked-crossing
```

One engine boot and one delivery per seed; `delivery_report.json` in each
seed's directory carries the transcript, every engine call and the
`CourierEnv` summary (`delivered`, `earnings`, `walked_m`,
`rejected_actions`), and `results.json` at the top holds every seed's row
and the split's summary (`tools/pixel_goal_results.py`). The runner's default is the
four-view harness with pedestrian routing (`--views quad`); `--views pair`
after the split name reproduces the original two-view runs. The number is
the `heldout` split's; `dev` is where a prompt or a harness change is
tried first (`docs/PIXEL_GOAL.md`, "The split").

## Train / eval separation

Both tracks keep what a policy is tuned on and what it is scored on apart,
and the separation is data in the repository, not a convention:

| track | tuned / trained on | scored on | where it is fixed |
|---|---|---|---|
| waypoint | seeds 0–799 (`embodiedbench/training/vagen/train_courier.yaml`) | seeds 1000–1063, the `val` split (`val_courier.yaml`, `SPLITS` in `embodiedbench/eval/run.py`) | `tests/test_eval_cli.py` asserts the two ranges; the yaml states the 200-seed gap and why |
| any-point | the six `dev` scenarios (seeds 0 1 2 6 7 9) | the ten `heldout` scenarios (seeds 3 5 12 14 15 16 17 18 23 31) | `PROTOCOL_SPLITS` in `tools/pixel_goal_order_pool.py`; `tests/test_pixel_goal_protocol.py` pins the partition to the shipped pool |

On the waypoint track a seed fixes the shift — the orders, their timing,
the hazards — on the one Paris map, so the held-out set is unseen shifts
on a seen city; the `trainprobe` split (seeds 0–31) exists to be compared
against `val`, never reported as it. The training runs' own held-out
column in the paper was seeds 1001–1064 (63 of the 64 in common), which
the audit notes. On the any-point track the held-out set is unseen
scenarios on the four certified doors; holding out a door needs a second
certified region, as `docs/PIXEL_GOAL.md` says. Nothing in this repository
trains on the any-point track.

## Reference policies

```python
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier
from embodiedbench.tasks.courier_router import run_shortest_path_courier

env.reset()
ObservationOnlyCourier(env, max_steps=9000).run(seed)                  # floor
env.reset()
ObservationOnlyCourier(env, max_steps=9000, sighted=True).run(seed)    # + perfect sight
run_shortest_path_courier(env, seed)                                   # ceiling (resets itself)
env.reset()
ObservationOnlyCourier(env, stop_at_turns=60).run(seed)                # the floor at the benchmark's turn budget
```

The floor reads only what the world says, so if it delivers, the words are
sufficient. `sighted=True` gives it the two things the words never say —
what is standing in a street, what colour its light is — with recognition
assumed perfect. The ceiling reads the road network directly and spends no
turn on the phone; **it is not a policy and is never scored as one**. It
is the benchmark's upper bound, and it is reported as that, labelled.
There are two of it: `ShortestPathCourier` plans in metres, and
`FewestMovesCourier` plans in turns -- the thing a capped shift actually
rations -- over the world's own block rule, keeping each job's deadline.
`tools/waypoint_reference.py` runs both on the held-out shifts through the
evaluator's own adapter, at the benchmark's horizon, keeps the better per
shift, and writes the evaluator's results shape, so `python -m
embodiedbench.eval compare results/model.json results/ceiling.json` pairs
a model with the bound seed by seed. `--objective floor` runs the
text-only courier the same way -- to the same **turn** budget, where a
phone check costs it a turn as it costs a model one (its own step count
is not the model's scale: a floor allowed 60 steps was spending 110-130
turns) -- and names it `text-floor`. A number quoted between the floor and
the ceiling is a claim about a policy; a number outside them is a bug in
the measurement.

## Conditions

`condition="full"` is the only validated rung. `no_phone` and `visual` are
declared and **not** validated: the reference courier scores zero on both,
which measures the absence of information rather than the absence of
perception. Do not report scores on them until a policy clears one.

## Running a person, or an interactive model, as the policy

Both evaluators talk to an OpenAI-compatible chat endpoint and nothing else,
so anything that can answer one can be the courier. `tools/relay_policy_server.py`
is that endpoint with no model behind it: every request is written to a
directory as the text the policy would read and the images it would see
(`prompt.txt`, `img-K.png`, one `composite.png` of the turn's images), and the
reply is whatever appears in `answer.txt` there. The harness cannot tell the
difference — same prompt, same image budget, same turn limit, same refusals.

```bash
python tools/relay_policy_server.py --dir /tmp/relay --port 8600 --model human-interactive
# another shell: the waypoint benchmark, one episode at a time
python -m embodiedbench.eval run --model human-interactive --base-url http://127.0.0.1:8600/v1 \
    --seed-list 1000,1001,1002 --workers 1 --transcripts --out results/
# or the any-point runner
QWEN_ENDPOINT=http://127.0.0.1:8600/v1/chat/completions QWEN_MODEL_NAME=human-interactive \
    bash tools/run_pixel_goal_front_rear_delivery.sh ...
```

Then, per turn: read `/tmp/relay/pending` for the turn directory, look at its
`prompt.txt` and `composite.png`, and write the reply into `answer.txt` in the
exact shape the system prompt asks for. Use `--workers 1`; with more, several
turns wait at once. The model client retries a request that takes longer than
ten minutes, so answer within that.

This is how the human and interactive-model rows in the results were produced.
It is also the honest way to play the benchmark yourself before reading a
model's transcript: the rules are easier to judge from the inside.
