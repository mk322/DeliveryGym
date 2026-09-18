#!/usr/bin/env bash
# Launch a GRPO run on whatever GPUs are free right now.
#
# Everything machine-specific is an environment variable with a default, so
# this is the file another machine runs unchanged.
#
# Required:
#   MODEL_PATH        the model directory (a HF snapshot path)
# Usually set:
#   HF_HOME           where the weights live
#   EXPERIMENT_DIR    where checkpoints, rollouts and validation dumps go
# See docs/INSTALL.md for the rest.
set -u

# Make CUDA number the cards the way nvidia-smi does. The default order is
# FASTEST_FIRST, which on eight identical cards is arbitrary -- so an index
# chosen by looking at nvidia-smi can select a different physical card, and
# "the free one" becomes the one a vLLM has 98% full. The soak run's OOM
# ("GPU 0 ... Process <other user> has 2.42 GiB") showed another user's job on
# what nvidia-smi called an empty card: same trap, seen from the other side.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
export PYTHONPATH="$REPO:$REPO/vendor/vagen:$REPO/vendor/vagen/verl:${PYTHONPATH:-}"

: "${MODEL_PATH:?set MODEL_PATH to the model directory}"

# A writable HOME, and the caches that hang off it.
#
# On a workstation these are already fine and this block changes nothing. In a
# container HOME can be /, owned by root and not writable, so every library
# that falls back to $HOME/.cache dies the first time it compiles something --
# flashinfer's JIT does it inside vLLM's engine init and takes the engine with
# it, reporting `PermissionError: [Errno 13] Permission denied: '/.cache'`,
# which reads as an engine fault and is a HOME fault. Cheap to set here too,
# so a run launched by any route gets it.
if [ ! -w "${HOME:-/}" ]; then
    export HOME="/tmp/home"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$XDG_CACHE_HOME/flashinfer}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_ext}"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_DIR" \
         "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" 2>/dev/null || true

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO/exps/grpo_courier}"
mkdir -p "$EXPERIMENT_DIR"

# flashinfer JIT-compiles against CUDA_HOME. Where the system nvcc is older
# than the CUDA torch was built for, compilation fails inside math.h with an
# error that names neither. Pointing CUDA_HOME at the conda environment that
# holds the matching toolkit is the fix; harmless when they already agree.
# CONDA_PREFIX is only set by `conda activate`, and the launchers put the env's
# bin on PATH instead -- so this block was skipped entirely and the fix below
# never ran. Derive the prefix from whichever python is on PATH.
_PREFIX="${CONDA_PREFIX:-$(dirname "$(dirname "$(command -v python 2>/dev/null || echo /nonexistent/bin/python)")")}"
if [ -x "${_PREFIX}/bin/nvcc" ]; then
  export CUDA_HOME="${_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  # nvcc hands ld a -L for $CUDA_HOME/lib64; conda puts the runtime in lib/,
  # so the link step of the JIT build ends with
  #     /usr/bin/ld: cannot find -lcudart
  # and the rollout engine dies before it serves a request. Machines that
  # already have a compiled flashinfer in ~/.cache never reach the link and
  # never see this, which is why it surfaced only on a fresh host.
  export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# A C compiler for Triton's JIT.
#
# Triton compiles kernels during the rollout engine's init and looks for one by
# the plain names cc and gcc, falling over with
#     RuntimeError: Failed to find C compiler. Please specify via CC
# which surfaces as "Engine core initialization failed" and looks like an
# engine problem. A CUDA *runtime* container has no toolchain at all.
#
# The packed conda environment does, under a target-prefixed name that Triton's
# search will never try -- x86_64-conda-linux-gnu-gcc. Naming it explicitly is
# better than installing a system gcc as well: it is the toolchain the rest of
# the environment was built against, so the ABI matches, and it travels in the
# bundle rather than depending on the image. Triton reads CC directly
# (runtime/build.py), so this is its own supported route.
if [ -z "${CC:-}" ] && ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  for _gcc in "${_PREFIX}"/bin/*-linux-gnu-gcc "${_PREFIX}"/bin/gcc; do
    if [ -x "$_gcc" ]; then
      export CC="$_gcc"
      [ -x "${_gcc%gcc}g++" ] && export CXX="${_gcc%gcc}g++"
      break
    fi
  done
fi
# Printed whichever way it went. The last time this failed, the question that
# cost the most was simply "did the fix reach this run", and one line answers
# it forever.
echo "COMPILER cc=$(command -v cc 2>/dev/null || echo none)" \
     "gcc=$(command -v gcc 2>/dev/null || echo none)" \
     "CC=${CC:-<unset, Triton will search PATH>}"
if [ -z "${CC:-}" ] && ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  echo "COMPILER WARNING: none found. Triton JIT-compiles at engine start and" >&2
  echo "  will fail with 'Failed to find C compiler', which reads as an engine" >&2
  echo "  fault. Install build-essential in the image, or put a compiler in the" >&2
  echo "  packed environment." >&2
fi

# Chosen fresh, from what is free at this second. A snapshot taken when the
# config was edited is already wrong by the time vLLM starts on a shared
# machine: fifteen launches died on "Free memory on device ... less than
# desired" before this existed.
read -r _DEV _MEM _N _SP <<< "$(bash "$HERE/pick_gpus.sh")"
if [ "$_DEV" = "NONE" ]; then
  echo "No GPU has ${MIN_FREE_MB:-18000} MB free right now." >&2
  nvidia-smi --query-gpu=index,memory.free --format=csv >&2
  exit 1
fi
# Honoured if the caller named the cards, for the same reason ROLLOUT_MEM is:
# the picker ranks by free memory at this instant, and on a shared host the
# instant is not the run. It chose two cards with another user's 3 GB still on
# them over four that were completely empty.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$_DEV}
# Computed from live free memory, unless the caller has already said what it
# wants. It used to overwrite an explicit setting without a word, which on a
# shared host is exactly when someone is overriding it: the computed fraction
# asked for more than another user had left free, the rollout engine refused
# to start, and the trainer stayed alive and produced no steps -- a run that
# reads as slow rather than as broken.
export ROLLOUT_MEM=${ROLLOUT_MEM:-$_MEM}
# Honoured, like CUDA_VISIBLE_DEVICES, ROLLOUT_MEM and SP before it: naming
# five cards and being given four is the same silent override that once ran
# five hours on one card.
export N_GPUS=${N_GPUS:-$_N}
# Honoured for the same reason as the two above, and it was not: this line
# overwrote an explicit SP without a word, so a run launched with SP=2 trained
# with SP=1 and the OOM it was meant to fix was credited to it anyway.
export SP=${SP:-$_SP}
# What the run will actually use, not what the picker suggested. Printing the
# picker's numbers next to settings the caller had already overridden is how
# five hours went by on one card under a line that read gpus=1 as if that were
# the request.
echo "PICKED devices=$CUDA_VISIBLE_DEVICES gpus=$N_GPUS" \
     "vllm_mem=$ROLLOUT_MEM sp=$SP (picker offered $_DEV/$_N/$_MEM/$_SP)"

# Sizes that have actually run. See the table in docs/INSTALL.md
# before changing them; the learning rate and the reward scale multiply.
export TRAIN_BS="${TRAIN_BS:-6}"
# The prompt segment holds the system prompt and the FIRST observation. At
# 4096 the agent loop logged
#     In env:Courier, prompt_ids length 4345 exceeds prompt_length 4096
# on nearly every episode and truncated the first frame -- a WARNING, so the
# run looked healthy while the model answered about an image it had only
# partly seen.
#
# Measured over 26 overflowing episodes: median 4223, p90 4443, max 4611. Note
# what that sample is -- only the episodes that already exceeded 4096 -- so the
# real distribution reaches further right than it shows and the max is a floor,
# not a ceiling. 4864 leaves 253 tokens over the widest seen. 4608 does not:
# it was the first choice here and one observation cleared it by three.
export PROMPT_LEN="${PROMPT_LEN:-4864}"
# Lowered by exactly what the prompt gained. Their SUM is what reserves the KV
# cache -- vllm_async_server.py overwrites rollout.max_model_len with it -- so
# holding the sum at 40960 keeps every memory figure already solved for these
# cards valid.
export RESP_LEN="${RESP_LEN:-36096}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export KL_COEF="${KL_COEF:-0.005}"
export TEST_FREQ="${TEST_FREQ:-10}"

# The fixes VAGEN and verl need live in embodiedbench/training/vagen/patches/
# rather than as edits to vendor/, so that they survive a vendor bump and stay
# readable as a list. They are re-applied here on every launch. Each is
# idempotent and refuses rather than guesses if upstream has moved, so a run
# stops with a readable message instead of training on a checkout that is only
# partly patched.
for patch in agent_loop_qwen3vl_rope \
             agent_loop_image_safe_truncation \
             agent_loop_generation_budget \
             policy_loss_rollout_correction_field \
             multiturn_image_safe_truncation; do
    python -m "embodiedbench.training.vagen.patches.${patch}" || exit 1
done

exec bash "$REPO/embodiedbench/training/vagen/train_grpo_courier.sh"
