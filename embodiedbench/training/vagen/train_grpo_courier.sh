#!/bin/bash
# GRPO on the courier benchmark, using VAGEN's verl stack.
#
# Adapted from vendor/vagen/examples/deliverybench/train_grpo_qwen25vl3b.sh.
# Two deliberate differences from that script:
#
#   * the environment is registered from the command line
#     (+env_registry.Courier=...), so nothing inside the gitignored vendor/
#     checkout has to be edited for this to run;
#   * no API key is baked in. The original has a live WANDB_API_KEY committed
#     into it; export your own or leave the logger on console.
#
# Prerequisites that are NOT satisfied by the default env -- see WORK_SUMMARY.md:
#   ray, and a rollout backend (sglang, or vllm on a torch build that works).
set -x

PROJECT_NAME="${PROJECT_NAME:-verl_vagen}"
# Named by the arm (experiments/run_arm.sh exports it), so each experiment's
# log file and wandb run carry its own name -- monitoring five parallel arms
# that all called themselves grpo_courier_qwen3vl4b was the alternative.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_courier_qwen3vl4b}"

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
VAGEN=${REPO}/vendor/vagen
EXPERIMENT_DIR=${EXPERIMENT_DIR:-~/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}}
DATASET_TRAIN=${DATASET_TRAIN:-${REPO}/embodiedbench/training/vagen/train_courier.yaml}
DATASET_VAL=${DATASET_VAL:-${REPO}/embodiedbench/training/vagen/val_courier.yaml}
# Repo-relative values resolve against this checkout, so one --job_env value
# is correct on every node whatever its root is.
case "$DATASET_TRAIN" in /*) ;; *) DATASET_TRAIN="${REPO}/${DATASET_TRAIN}" ;; esac
case "$DATASET_VAL"   in /*) ;; *) DATASET_VAL="${REPO}/${DATASET_VAL}" ;; esac
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a local checkpoint}
# Stripped here, where every caller passes through, because verl asserts on a
# trailing slash from inside copy_to_local and the traceback names neither the
# model nor whichever launcher built the path with `ls -d .../snapshots/*/`.
MODEL_PATH="${MODEL_PATH%/}"

export HYDRA_FULL_ERROR=1
# Peer-to-peer between these cards hangs. A multi-GPU run dies with
#   Watchdog caught collective operation timeout:
#   WorkNCCL(SeqNum=3, OpType=BROADCAST, ...) ran for 600000ms
# ten minutes into what looks like a healthy startup -- no error, no OOM, the
# log simply stops. The same hang appeared when serving an 8B with
# tensor-parallel 2 on this machine and the same switch cleared it. It costs
# interconnect bandwidth, not correctness.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
# NOTE: do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# It looks like the right answer to the backward pass OOMing on a 1.92 GiB
# allocation with 889 MiB free, but vLLM refuses to start with it:
#   AssertionError: Expandable segments are not compatible with memory pool.
# The failure surfaces as "Engine core initialization failed" during
# load_model, which reads as not enough memory for the weights and sends you
# tuning gpu_memory_utilization instead. Shorten the sequence instead.
#
# And "the sequence" means the rollout's own context, not the data lengths.
# Without max_model_len the engine takes the model's default -- 40960 for this
# one -- and reserves KV cache for it, so it refuses to start whatever
# gpu_memory_utilization is set to. Seven launches were spent moving that
# fraction between 0.30 and 0.91 before the number in the error message was
# read: it says 40960, not the 12288 the data lengths imply.
export WANDB_MODE=${WANDB_MODE:-offline}
# The logger list follows WANDB_MODE: `online` streams to wandb (after
# `wandb login`), anything else keeps the console only. TRAIN_LOGGER still
# overrides both.
if [ -z "${TRAIN_LOGGER:-}" ]; then
    if [ "$WANDB_MODE" = "online" ]; then TRAIN_LOGGER="['console','wandb']"; else TRAIN_LOGGER="['console']"; fi
fi
# One experiment, one line. Every process restart used to wandb.init a fresh
# run id, so a crash-and-resume experiment drew as four disconnected
# segments under five identical names. A deterministic id per experiment
# plus resume=allow makes every restart append to the same run -- offline
# too: `wandb sync --append` stitches the segments on the server because
# they share the id. Steps come from the trainer's global step, which the
# checkpoint restores, so the appended history lines up on the x-axis.
export WANDB_RUN_ID=${WANDB_RUN_ID:-${EXPERIMENT_NAME}}
export WANDB_RESUME=${WANDB_RESUME:-allow}
# The adapter lives in this repo, not in VAGEN, so both must be importable.
export PYTHONPATH=${REPO}:${VAGEN}:${PYTHONPATH}
mkdir -p "${EXPERIMENT_DIR}"

# MICRO_BS and OFFLOAD default to the 24 GB settings -- one sequence at a time,
# parameters and optimizer state living on the CPU -- because that is what a
# 24 GB card needs and it is where this has run. On 80 GB both are pure cost:
# the offloads move 8 GB over PCIe per micro-batch to save memory there is no
# shortage of. See the H100 launcher.
#
# PREFLIGHT: torch must actually see the GPUs before ray is told there are
# N of them.
#
# A GPU released moments earlier still lists free memory in nvidia-smi while
# CUDA cannot open it yet, and torch then reports a smaller device_count than
# CUDA_VISIBLE_DEVICES names. verl asks for the rank it was configured for and
# dies with "device >= 0 && device < num_gpus ... device=2, num_gpus=2", which
# reads like a framework bug about device indexing. It is not: it is launching
# too soon after killing whatever held the card. Four separate GPU counts were
# blamed for this before the cards were simply checked.
# Counting is not enough: device_count has reported four while a later import
# in the same run saw two. Every card is opened and written to here, which is
# the only claim that matters.
SEEN=$("${PYTHON:-python3}" - <<'PYEOF'
import torch
ok = 0
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(8, device=f"cuda:{i}")
        ok += 1
    except Exception as error:
        print(f"cuda:{i} unusable: {error}", flush=True)
print(ok)
PYEOF
)
SEEN=$(echo "${SEEN}" | tail -1)
WANT=${N_GPUS:-1}
if [ "${SEEN}" != "${WANT}" ]; then
  echo "PREFLIGHT FAILED: torch sees ${SEEN} GPU(s), N_GPUS=${WANT}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "the cards named are not all open yet -- wait for whatever held them to exit, then retry"
  exit 1
fi
echo "PREFLIGHT: torch sees ${SEEN} GPU(s), as configured"

cd "${VAGEN}" || exit 1

# trainer.validation_data_dir below dumps per-episode validation results, not
# just the mean. Two validations are 64 greedy episodes each on the same 64
# seeds; comparing their means compares two proportions with a standard error
# of about 5 points, so a real 5-point gain and pure noise look identical. The
# episodes are paired by construction, and a paired test looks only at the
# seeds that changed -- several times more sensitive, at no extra compute.
#
# Note for anyone editing the command below: it is one command continued with
# backslashes, and a comment line inside it ENDS THE COMMAND THERE. Putting
# this note in the middle silently dropped the last three trainer.* arguments
# and one run started with no checkpoint directory and no data dumps at all.
# Skip recomputing old_log_probs: the engine already produced them while
# sampling, and recomputing cost 207 s of a 1424 s step. This is verl's own
# rollout-correction bypass mode, documented at
# https://verl.readthedocs.io/en/latest/algo/rollout_corr.html -- with
# old_log_prob = rollout_log_prob the PPO ratio becomes pi_theta/pi_rollout,
# which is the right correction when the two policies are the same checkpoint,
# as they are here.
#
# Two things are needed and each fails differently on its own. The helper
# does `policy_loss_config["rollout_correction"] = rollout_corr_config`, so:
#   * the key must EXIST in the composed config, or omegaconf's struct mode
#     refuses the assignment -- hence the null override below. The config is
#     built from VAGEN's YAML, which knows only the keys written in it, so
#     adding the dataclass field alone does not put it there.
#   * PolicyLossConfig must ACCEPT the field, or instantiate raises
#     "unexpected keyword argument 'rollout_correction'" -- hence the patch
#     in embodiedbench/training/vagen/patches/.
#
# Both are in place and both are harmless with BYPASS_OLDLP off, which is
# where it stays: with them filled, the path fails a third time on
# KeyError: 'temperature', missing from meta_info inside verl's own actor.
# The feature is backported in pieces here. Leave it off unless verl is
# upgraded.
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="${VAGEN}/vagen/configs" \
    --config-name='vagen_multiturn' \
    +env_registry.Courier=embodiedbench.training.vagen_courier_env.CourierGymEnv \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files="${DATASET_VAL}" \
    data.train_batch_size=${TRAIN_BS:-2} \
    data.max_prompt_length=${PROMPT_LEN:-4096} \
    data.max_response_length=${RESP_LEN:-8192} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-5e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS:-2} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS:-1} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL:-True} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF:-0.005} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEF:-0.0} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP:-1} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${OFFLOAD:-True} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-1} \
    actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND:-vllm} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${GROUP:-4} \
    filter.enable=${FILTER:-False} \
    filter.name=${FILTER_NAME:-reward_variance_top_p} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM:-0.4} \
    actor_rollout_ref.rollout.enforce_eager=${EAGER:-True} \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=${CUDAGRAPH_SIZES:-null} \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_CTX:-12288} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_SEQS:-32} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_TOKENS:-32768} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=${BYPASS_OLDLP:-False} \
    ++algorithm.rollout_correction.bypass_mode=${BYPASS_OLDLP:-False} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction=null \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-1} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${VAGEN}/vagen/configs/agent.yaml" \
    critic.strategy=fsdp2 \
    critic.model.path="${MODEL_PATH}" \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD:-True} \
    reward_model.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAIN_LOGGER" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS:-1} \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.total_epochs=${TOTAL_EPOCHS:-100000} \
    trainer.total_training_steps="${TOTAL_STEPS:-1000}" \
    `# total_epochs is a ceiling the step count must never hit: at 10 epochs a` \
    `# 60-seed pool stopped at step 100 whatever TOTAL_STEPS said.` \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir="${EXPERIMENT_DIR}/verl_checkpoints" \
    trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
    trainer.validation_data_dir="${EXPERIMENT_DIR}/validation_data" \
    2>&1 | tee "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log"

# The exit status of a pipeline is its LAST command's, and tee always succeeds.
# So a trainer that died on an assertion in its first minute exited 0 through
# this shell, the supervisor logged "Training process finished", and the run
# ended with no error anywhere -- which is how a crash spent a day looking
# like a mysteriously short training run. PIPESTATUS[0] is python's own.
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo ""
    echo "=================================================================="
    echo "  TRAINER EXITED $rc"
    echo "  log: ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log"
    echo "------------------------------------------------------------------"
    # The traceback, where whoever is reading the supervisor's output can see
    # it without going and finding the file.
    grep -nE "Error|Traceback|assert|Exception|CUDA out of memory" \
        "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log" 2>/dev/null | tail -n 30
    echo "=================================================================="
fi
exit "$rc"
