# Installing on a fresh machine

Everything in this repository is text. The cities the courier walks are not:
they are rendered photographs that cannot go in git. Three things to put in
place — the **albums**, a **Python environment** for evaluation, and (only
for training) a **second environment** with the pinned trainer stack.

## 1. The albums

One archive, `albums_v3.tar` (ten cities, 7.8 GB packed, ~9 GB unpacked),
published as the Hugging Face dataset
[`mk322/DeliveryGym-albums`](https://huggingface.co/datasets/mk322/DeliveryGym-albums)
in four parts with a `SHA256SUMS` file. Download, join, verify, unpack
anywhere with room, and point `ALBUMS_DIR` at the directory that holds the
album folders:

```bash
pip install -U "huggingface_hub[cli]"
hf download mk322/DeliveryGym-albums --repo-type dataset --local-dir albums
cd albums
cat albums_v3.tar.part0{0,1,2,3} > albums_v3.tar
sha256sum -c SHA256SUMS                       # the four parts and the joined archive
mkdir -p /data/albums && tar -xf albums_v3.tar -C /data/albums && cd ..
export ALBUMS_DIR=/data/albums
```

The parts can be deleted once the archive verifies. To publish the dataset
again from a machine that holds the parts (maintainers only):

```bash
hf auth login                                 # a token with write access to the dataset
hf upload mk322/DeliveryGym-albums <directory with the four parts and SHA256SUMS> \
    --repo-type dataset
```

You should end up with:

```
$ALBUMS_DIR/
  paris_streets_v2/citycore-paris/           the view down each Paris street (carriageway)
  paris_streets_pavement/<city>/             the same views from the pavement -- the walker's eyes
  paris_lamps_real/<city>/                   pedestrian lamps, one frame per phase (+ signal_visibility.json)
  paris_obstacles/<city>/                    barriers and crowded pavements (+ obstacle_visibility.json)
  paris_obstacles_pavement/<city>/           the same, from the pavement
  city_streets_v1/<city>/                    carriageway views of the nine procedural cities
```

where `<city>` is `citycore-paris` and, for the multi-city experiments,
`small-city-11/13/15`, `medium-city-18/20/22`, `large-city-26/28/30`.
The `*_visibility.json` sidecars matter: the environment charges for
crossing on red, and dresses barriers, **only** on the approaches the album
can actually show. An album without its sidecar charges nothing — the
intended failure mode, but a silent one — so check they arrived:

```bash
ls $ALBUMS_DIR/paris_lamps_real/citycore-paris/signal_visibility.json
ls $ALBUMS_DIR/paris_obstacles/citycore-paris/obstacle_visibility.json
```

Every yaml in the repository names albums under `/data/albums/...`.
`ALBUMS_DIR` re-roots those paths at run time
(the last two components, `<album>/<city>`, identify an album; everything
before them is where a machine keeps it), so the yamls need no editing. A
missing album fails loudly at the first `reset()`, naming the path it
looked for.

The compiled maps (nodes, streets, addresses) are small and tracked, under
`vendor/vagen/vagen/envs/deliverybench/maps/`.

## 2. Evaluation and tests — any Python ≥ 3.11, no GPU

```bash
conda create -n courier python=3.12 -y && conda activate courier
pip install -e .            # pydantic, pillow, pyyaml, cairosvg, numpy
pip install pytest          # for the test suite
```

`cairosvg` needs the cairo library (`apt install libcairo2` / `conda install
cairo`). It is easy to forget and used to fail silently: the phone's map is
rendered from SVG, and without it the map was simply never sent. The
evaluator now refuses to start without it.

Verify — no GPU, no server, no model:

```bash
python -c "
from pathlib import Path
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.tasks.courier_router import run_shortest_path_courier
net = build_road_network(Path('vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris'), map_name='citycore-paris')
env = CourierEnv(net, seed=0, difficulty='solo', stride='block')
run_shortest_path_courier(env, 0)
print('reference courier delivered', env.summary()['delivered'], 'of', env.summary()['orders_issued'])
"
# -> reference courier delivered 1 of 1

pytest tests -q            # ~10 min; a few tests skip without albums or without a live UE
```

Then score a model. Any OpenAI-compatible endpoint works — a local vLLM,
or a hosted API:

```bash
bash examples/eval_api_model.sh                       # serves Qwen3-VL-4B with vLLM, scores it
SKIP_SERVE=1 BASE_URL=https://api.example/v1 MODEL=some-vlm API_KEY=... \
  bash examples/eval_api_model.sh                     # a hosted model; CPU only
```

## 3. Training — Python 3.12 and the pinned stack

Training needs vLLM and a CUDA torch that match what the vendored trainer
expects. Keep it in its own environment:

```bash
conda create -n courier-rl python=3.12 -y && conda activate courier-rl
pip install torch==2.8.0
pip install vllm==0.11.0 ray==2.53.0 transformers==4.57.1 accelerate==1.12.0 ninja
pip install -e vendor/vagen/verl -e vendor/vagen -e .
```

`vendor/` is **tracked** in this repository — VAGEN and verl at pinned
commits with our local modifications (`vendor/VENDOR_PINS.md`). Do not
replace it with an upstream checkout.

The model:

```bash
export HF_HOME=/somewhere/with/40GB
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct
export MODEL_PATH=$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/<hash>
```

Then:

```bash
MODEL_PATH=... ALBUMS_DIR=... bash examples/train_single_node.sh        # 4 x 80 GB by default
CARD_GB=24 WANT_GPUS=4 MODEL_PATH=... ALBUMS_DIR=... bash examples/train_single_node.sh
```

The launcher picks the freest cards at launch time, opens each one before
starting (a card released seconds ago is not yet openable), sizes the
rollout engine from the tightest card, and writes `$EXPERIMENT_DIR.launch.log`.
`examples/README.md` lists the knobs; `experiments/run_arm.sh` launches the
paper's arms by name.

## 4. What breaks first

Ordered by how often it has actually happened.

- **`Free memory on device (…) is less than desired`** — someone else took
  the card between your config and vLLM starting. Relaunch; the launcher
  recomputes.
- **`device >= 0 && device < num_gpus`** — a GPU released seconds earlier
  is not yet openable. The launcher waits; if you launch by hand, wait
  fifteen seconds.
- **NCCL collective timeout** — `NCCL_P2P_DISABLE=1` is set by default
  (a workaround for consumer cards); on an NVLink box you may unset it.
- **The map never appearing** — `cairosvg` missing in the training
  environment while present in the evaluation one. Check both.
- **`FileNotFoundError: … album …`** at the first reset — `ALBUMS_DIR` is
  unset or the archive is incomplete. The message names the path.
- **A delivery rate that looks impossibly low** — read `infra_error_seeds`
  in the evaluator's summary first. Those episodes are excluded from the
  mean, but a run with any of them is not a clean number.

## 5. The any-point (pixel-goal) track

Needs a live Unreal Engine build and a second GPU; it is documented on its
own in `docs/PIXEL_GOAL.md`.

The build is the part you cannot install from here. The engine-side code is
in `ue/`, but the Unreal project it builds inside is private, so **ask the
authors for the editor bundle**; the launcher verifies whatever you are given
against seven SHA-256 hashes pinned in this repository. Nothing above in this
guide depends on it, and neither does the waypoint benchmark.
