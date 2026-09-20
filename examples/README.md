# Examples — the benchmark on an ordinary GPU server

No cloud dependency: one machine (or N, for the multi-node script), local
GPUs, a HuggingFace snapshot, this repository, the albums. Install per
`docs/INSTALL.md` first.

```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct     # a local HF snapshot, no trailing slash
export ALBUMS_DIR=/data/albums                      # the unpacked album archive
```

## The runs

| Script | What it does |
|---|---|
| `eval_api_model.sh` | Serve a model with vLLM and score it on the 64 frozen held-out shifts (`SKIP_SERVE=1 BASE_URL=... MODEL=...` scores a hosted API instead). CPU is enough for the eval itself. |
| `train_single_node.sh` | The base recipe: GRPO on Qwen3-VL-4B, earnings reward, Paris. Every experiment arm is this script plus a yaml — `bash experiments/run_arm.sh <arm>` sets them. |
| `train_adaptive_city.sh` | The evolving dispatcher (RQ2): 8-city pool, city-level curriculum, judged on unseen large-city-30. `ADAPTIVE_MODE=city_ladder` is the schedule control, `ADAPTIVE=0` the static control. |
| `train_constraints.sh` | RQ5: fee jitter + food temperature + notes + phone battery, with per-constraint compliance metrics on the validation columns. |
| `train_multi_node.sh` | The same trainer over N machines (a plain TCP Ray cluster). |

Common knobs, all environment variables and all optional:

| variable | default | meaning |
|---|---|---|
| `CARD_GB` | `80` | sizing row: `80` (H100/A100) or `24` (a 24 GB card you have to yourself); sets `ROLLOUT_MEM`, `SP`, `OFFLOAD`, `RESP_LEN` unless you set them |
| `WANT_GPUS` | `4` | how many cards the launcher takes (it picks the freest); the 48-episode batch must divide evenly: 2, 3, 4, 6, 8 |
| `TRAIN_BS`, `GROUP` | `6`, `8` | prompts per step, samples per prompt (48 episodes a step) |
| `TOTAL_STEPS`, `TEST_FREQ`, `SAVE_FREQ` | `1000`, `10`, `20` | length, validation cadence, checkpoint cadence |
| `EXPERIMENT_DIR` | `exps/local_<timestamp>` | checkpoints, dumps, logs. **Reuse it to resume** — the trainer restarts from the last checkpoint in that directory |
| `EXPERIMENT_NAME` | `grpo_courier_qwen3vl4b` | log name and wandb run id |
| `DATASET_TRAIN`, `DATASET_VAL` | `embodiedbench/training/vagen/{train,val}_courier.yaml` | the task; see `experiments/` |
| `WANDB_MODE` | `offline` | `online` streams to wandb after `wandb login` (the logger follows this) |
| `ADAPTIVE_MODE` | `city` | which curriculum the sidecar builds (`experiments/README.md`) |

Watch a run with `tail -f $EXPERIMENT_DIR.launch.log`; the held-out curve is
the `val-core/courier_paris_val/earnings_at_100` lines.

## Reproducibility contract

- Validation seeds 1000–1063 are frozen, named exactly in every validation
  block, and no training pool overlaps them.
- Every optional mechanic is a config flag defaulting **off**; a config that
  doesn't mention a flag runs the benchmark exactly as published. The
  curriculum is a config decision too (`adaptive: true` in the training
  yaml) — nothing in the process environment can switch it on.
- Adaptive runs are deterministic in the seed stream: the same seed replays
  the same shift, city choice included; the curriculum's decisions are
  logged in `$EXPERIMENT_DIR/adaptive_profile.{json,log}`.
- The evaluator drives the same environment adapter the trainer does
  (`CourierGymEnv`): same prompt, same image budget and downscale, same
  lamp-visibility gating, all read from the validation yaml.
