<h1 align="center">DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.19801"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.19801-b31b1b.svg"></a>
  <a href="https://huggingface.co/datasets/mk322/DeliveryGym-albums"><img alt="albums" src="https://img.shields.io/badge/albums-7.8%20GB-ffd21e.svg?logo=huggingface&logoColor=black"></a>
  <a href="https://github.com/mk322/DeliveryGym/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/mk322/DeliveryGym/actions/workflows/tests.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-3776ab.svg">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-2ea44f.svg"></a>
</p>

<img src="docs/assets/figure1_overview.png" width="100%" alt="DeliveryGym overview: (a) the task, maximize shift income under time, location, energy, money, food and safety constraints; (b) a courier shift through food orders A to D in a rendered Paris, with a charging stop; (c) the event trace of pickups, deliveries and charging; (d) the verifier that scores outcomes from simulator events" />

**DeliveryGym** is a 3D environment for evaluating and training vision-language agents on continuous **courier shifts**. The model is the courier: it reads the street, the phone and the clock, then walks, waits at lights, detours around barriers, charges, and collects and hands over food orders in rendered cities. Every delivery uses time, energy and money that later orders need, so the score rewards planning across the whole shift, not just finishing the order in hand.

- **Long-horizon, coupled tasks**: endless shifts where the dispatcher refills after every delivery, under shared time, location, energy, money, food and safety constraints
- **Multimodal tool use**: street views, pedestrian signals, a phone map and an order slip each turn, answered with one thought and one tool call
- **Verified rewards from simulator events**: income is computed from the event trace of what actually happened in the world, so the whole trajectory becomes RL feedback
- **Fixed benchmark, adaptive training**: 64 held-out shifts with floor/ceiling reference policies, plus a GRPO trainer whose curriculum adapts to the policy's weaknesses

[**[Paper]**](https://arxiv.org/abs/2609.19801) &ensp; [**[Dataset]**](https://huggingface.co/datasets/mk322/DeliveryGym-albums) &ensp; [**[Environment API]**](docs/RUNNING.md) &ensp; [**[Task Design]**](docs/TASK_DESIGN.md) &ensp; [**[Any-point Runbook]**](docs/PIXEL_GOAL.md)
<!-- TODO(link): add [Home page] first once the project site is live -->

-------
## Updates
* [9/12/2026] **v0.1.0**: Initial release with the waypoint benchmark and RL environment, ten city albums, reference policies, the GRPO trainer, and the engine-side any-point code.
<!-- TODO: add entries (e.g. paper acceptance, leaderboard, new cities) -->

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [The Any-point Track](#the-any-point-track)
- [Held-out Splits](#held-out-splits)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Citation](#citation)

-------
<!-- ## Overview

The environment is one courier shift in a rendered city. A vision-language model plays the courier: it collects and hands over a queue of food orders before the clock and the turn cap run out. The score is the **money earned** on that shift, not whether the order in hand succeeded.

Each turn the courier sees the streets leaving the junction, the phone with the route drawn on it, the order slip, the clock and its own notes. It answers with one thought and one tool call:

```
THOUGHT: the route runs east and photograph 1 shows Rue de Grenelle is clear; the light is green.
walk_to("Rue de Grenelle", "east")
```

A delivery pays `3.00 + 0.01 × metres`: in full on time, in part if late, nothing if it never arrives. The dispatcher refills after every delivery, so a shift is a queue of orders against one clock: what the courier spends on the order in hand is gone from the next.

Two reference policies ship here and both run in about a minute on a CPU: a **floor** that reads only the text, and a **ceiling** that reads the road graph directly. A number has a floor and a ceiling beside it. The paper reports the measurements; this repository is how you reproduce them.

------- -->
## Installation
Evaluation runs on CPU. Training needs four 80 GB GPUs (or four 24 GB GPUs with `CARD_GB=24`).

1. Set up a conda environment:

   ```sh
   conda create -c conda-forge -n deliverygym python=3.11
   conda activate deliverygym
   ```
2. Clone and install this repo:

   ```sh
   git clone https://github.com/mk322/DeliveryGym.git
   cd DeliveryGym
   pip install -e .        # cairosvg needs libcairo2 (apt) or cairo (conda)
   ```
3. The photographs are a dataset, not part of the repository: ten cities, **7.8 GB** in four parts. Download, join, verify, unpack, and point `ALBUMS_DIR` at the result:

   ```sh
   pip install -U "huggingface_hub[cli]"
   hf download mk322/DeliveryGym-albums --repo-type dataset --local-dir albums
   cd albums && cat albums_v3.tar.part0{0,1,2,3} > albums_v3.tar && sha256sum -c SHA256SUMS
   mkdir -p /data/albums && tar -xf albums_v3.tar -C /data/albums && cd ..
   export ALBUMS_DIR=/data/albums
   ```
   [`docs/INSTALL.md`](docs/INSTALL.md) lists the folders you should end up with.

4. *(Training only)* Set up the pinned trainer stack in a **separate** environment:

   ```sh
   conda create -n courier-rl python=3.12 -y && conda activate courier-rl
   pip install torch==2.8.0
   pip install vllm==0.11.0 ray==2.53.0 transformers==4.57.1 accelerate==1.12.0 ninja
   pip install -e vendor/vagen/verl -e vendor/vagen -e .
   ```

-------
## Basic Usage

### Evaluate a model
The benchmark is 64 held-out Paris shifts (seeds 1000–1063), 60 turns each, with greedy decoding, scored through the same adapter the trainer uses. Any OpenAI-compatible endpoint works, and the evaluator itself needs no GPU.

```sh
# serve Qwen3-VL-4B on one 24 GB card
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct --served-model-name qwen3-vl-4b \
  --port 8000 --max-model-len 16384 --limit-mm-per-prompt '{"image": 16}'

# score it
python -m embodiedbench.eval run --model qwen3-vl-4b --base-url http://127.0.0.1:8000/v1 \
  --split val --workers 4 --out results --tag qwen3-vl-4b
```
`results/qwen3-vl-4b.json` reports `earnings_mean` (with a bootstrap 95% CI), `deliveries_mean`, `zero_delivery_rate`, running earnings at turns 20/40/60, and per-seed scores. [`examples/eval_api_model.sh`](examples/eval_api_model.sh) wraps both steps.

> [!NOTE]
> Episodes ended by infrastructure failures are excluded from every mean, and the evaluator exits non-zero when any occur.

### Compare against the floor and ceiling
```sh
python tools/waypoint_reference.py --out results/ceiling.json                    # reads the road graph
python tools/waypoint_reference.py --objective floor --out results/floor.json    # reads only the text
python -m embodiedbench.eval compare results/qwen3-vl-4b.json results/ceiling.json
```

### Train with GRPO
Training uses the vendored, pinned VAGEN + verl trainer. It runs on seeds 0–799 at the 60-turn horizon and validates on the 64 held-out shifts.

```sh
conda activate courier-rl
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/Qwen3-VL-4B-Instruct
MODEL_PATH=models/Qwen3-VL-4B-Instruct ALBUMS_DIR=/data/albums bash examples/train_single_node.sh
```
The validation metric is logged as `val-aux/courier_paris_val/earnings_at_60/mean@1`. See [`examples/README.md`](examples/README.md) for all options.

### Train with the adaptive curriculum
The evolving dispatcher samples training cities from a learnability profile rebuilt from the episode log, and is judged on a city no run trains on:
```sh
MODEL_PATH=models/Qwen3-VL-4B-Instruct ALBUMS_DIR=/data/albums bash examples/train_adaptive_city.sh
# ADAPTIVE=0 for the uniform-sampling control
```

### Reproduce a paper experiment
Every arm of the paper (reward, curriculum, constraints, map scaling, horizon) is a named config in [`experiments/`](experiments):
```sh
bash experiments/run_arm.sh <arm>
```

-------
## The Any-point Track

Each run is one delivery in a live Unreal Engine Paris, on a certified pool of doors, crossings and pedestrian ways. The engine resolves the pixel the policy points at. [`docs/PIXEL_GOAL.md`](docs/PIXEL_GOAL.md) is the runbook.

The protocol's sixteen scenarios split into six development and ten held-out ones (scenarios, not doors: the pool has four doors, and every held-out scenario uses doors a development scenario also uses).

```sh
PIXEL_GOAL_PROTOCOL_OUT=results/anypoint/heldout bash tools/run_pixel_goal_protocol.sh heldout
```

The engine-side code is in [`ue/`](ue): the `SpPixelGoalSubsystem`, its 25 automation tests, and the three SimWorld classes it drives (see [`ue/README.md`](ue/README.md)).

> [!IMPORTANT]
> The Unreal project these files build inside is private, so **ask the authors for the editor bundle**. The launcher pins that bundle's seven files by SHA-256 and checks them before a run starts. The waypoint track does not need it.

-------
## Held-out Splits

| Track | Train / development | Held out |
|---|---|---|
| Waypoint | seeds 0–799 | seeds 1000–1063 |
| Any-point | scenarios 0 1 2 6 7 9 | scenarios 3 5 12 14 15 16 17 18 23 31 |

Both partitions are stored as data in the repository and asserted by tests. **Report results on the held-out sets only.**

-------
## Repository Structure
```
embodiedbench/
  runtime/city/       CourierEnv: the world and every mechanic
  agent/courier/      prompts, tools, the session, the model client
  training/           CourierGymEnv (trainer/evaluator adapter), the GRPO launcher
  eval/               embodiedbench.eval: score an endpoint, compare two runs
  runtime/live/       live-UE backends; runtime/pixel_goal*: any-point navigation
  compiler/           city maps -> road network, albums, signal legibility, obstacles
ue/                   engine-side any-point code (SpPixelGoalSubsystem and friends)
examples/             eval and training on an ordinary server
experiments/          every arm of the paper, by name
configs/pixel_goal/   the certified pedestrian order pool and its engine audit
tools/                any-point protocol, relay endpoint, reference policies, album bakers
vendor/               VAGEN + verl at pinned commits, and the compiled city maps
docs/                 INSTALL · RUNNING · TASK_DESIGN · PIXEL_GOAL
tests/                ~2000 tests; no GPU, albums or engine needed
```

-------
## License
Code: [MIT License](LICENSE)

City albums: released as a [dataset on Hugging Face](https://huggingface.co/datasets/mk322/DeliveryGym-albums) <!-- TODO: name the dataset license -->

Engine-side code in `ue/`: SimWorld_SPEAR's MIT License

-------
## Citation

```bibtex
@article{deliverygym2026,
  title   = {DeliveryGym: An RL Environment for Long-Horizon Embodied Agent
             Planning with Adaptive Curriculum},
  author  = {Kang, Haoqiang and Zhang, Yiming and Guo, Yiyang and Li, Chuying
             and Shen, Jianzhi and Xu, Tianruo Rose and Ye, Xiaokang and
             Qin, Lianhui},
  journal = {arXiv preprint arXiv:2609.19801},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.19801}
}
```
