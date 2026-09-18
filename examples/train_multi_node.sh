#!/usr/bin/env bash
# GRPO across N machines: one Ray cluster over plain TCP, no scheduler.
#
# On EVERY node (rank 0 first):
#   MASTER_ADDR=<rank0-host> NNODES=2 NODE_RANK=<this node's rank> \
#   MODEL_PATH=... ALBUMS_DIR=... bash examples/train_multi_node.sh
#
# Rank 0 runs the trainer; other ranks contribute GPUs and hold. All nodes
# need the same checkout, the same environment, and the same assets at the
# same paths (NFS or identical local copies).
#
# HONESTY: the single-node path is exercised daily; this one shares its
# machinery but has seen fewer miles. The supervisor prints a RUN FAILED
# block on any failure -- file that block, not a screenshot of the tail.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MASTER_ADDR:?rank 0's reachable hostname/IP}"
: "${NNODES:?total number of nodes}"
: "${NODE_RANK:?this node's rank, 0-based}"
: "${MODEL_PATH:?local HF snapshot directory}"
: "${ALBUMS_DIR:?directory holding the albums}"
RAY_PORT="${RAY_PORT:-6379}"
N_GPUS="${N_GPUS:-$(nvidia-smi -L | wc -l)}"

# N_GPUS reaches the trainer too: Ray registers every card, and without this
# the picker's default of four cards silently trained a 2x8 cluster on eight.
export NNODES MODEL_PATH ALBUMS_DIR N_GPUS
export WANT_GPUS="$N_GPUS"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PWD/exps/multinode_$(date +%m%d_%H%M)}"

if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head --port="$RAY_PORT" --num-gpus="$N_GPUS" --disable-usage-stats
    export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
    want=$(( NNODES * N_GPUS ))
    echo "waiting for the cluster to reach $want GPUs..."
    for _ in $(seq 1 60); do
        have=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
        [ "${have:-0}" -ge "$want" ] && break
        echo "  $have/$want"; sleep 10
    done
    exec bash scripts/train_grpo.sh
else
    until ray start --address="$MASTER_ADDR:$RAY_PORT" --num-gpus="$N_GPUS" --disable-usage-stats; do
        echo "head not up; retrying"; sleep 5
    done
    echo "rank $NODE_RANK holding as a Ray worker (ctrl-c to leave the cluster)"
    exec sleep infinity
fi
