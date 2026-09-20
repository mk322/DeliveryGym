#!/usr/bin/env bash
# GRPO on the courier benchmark, one machine, no cloud anything.
#
#   MODEL_PATH=/path/to/Qwen3-VL-4B-snapshot bash examples/train_single_node.sh
#
# Everything else has a working default; the knobs are in examples/README.md
# (the sizing table: 24 GB cards vs 80 GB cards). The launcher picks the free
# GPUs itself and prints a PICKED line with what it chose -- read it.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MODEL_PATH:?set MODEL_PATH to a local HF snapshot directory (no trailing slash)}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PWD/exps/local_$(date +%m%d_%H%M)}"
export ALBUMS_DIR="${ALBUMS_DIR:?set ALBUMS_DIR to the directory holding the albums}"

# Sizing follows the cards: CARD_GB=80 (default; H100/A100) or CARD_GB=24
# (RTX 4090 / A5000 class, cards to yourself). examples/README.md explains
# every number. Anything already set in the environment wins.
CARD_GB="${CARD_GB:-80}"
if [ "$CARD_GB" = "24" ]; then
    export ROLLOUT_MEM="${ROLLOUT_MEM:-0.88}" SP="${SP:-2}" OFFLOAD="${OFFLOAD:-True}"
    export PROMPT_LEN="${PROMPT_LEN:-4864}" RESP_LEN="${RESP_LEN:-36096}"
else
    export ROLLOUT_MEM="${ROLLOUT_MEM:-0.70}" SP="${SP:-1}" OFFLOAD="${OFFLOAD:-False}"
    export PROMPT_LEN="${PROMPT_LEN:-4864}" RESP_LEN="${RESP_LEN:-57344}"
fi
export TRAIN_BS="${TRAIN_BS:-6}" GROUP="${GROUP:-8}"

# The redirect below lands next to EXPERIMENT_DIR; on a fresh checkout that
# parent does not exist yet, and a redirect that fails on a backgrounded
# command does not trip set -e -- the script reported "launched" with no log
# and no trainer.
mkdir -p "$EXPERIMENT_DIR"
setsid nohup bash scripts/train_grpo.sh > "$EXPERIMENT_DIR.launch.log" 2>&1 < /dev/null &
disown
echo "launched; watch with:"
echo "  tail -f $EXPERIMENT_DIR.launch.log"
echo "  grep -E 'PICKED|PREFLIGHT|val-core|step:' $EXPERIMENT_DIR.launch.log"
