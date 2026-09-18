#!/usr/bin/env bash
# RQ2 flagship: the evolving dispatcher (city-level curriculum).
#
# Trains on an 8-city pool and is judged on large-city-30, which no arm
# ever trains on. A sidecar rebuilds a city-learnability profile from the
# episode log every 10 minutes (pure statistics -- no extra model calls);
# the dispatcher samples cities from it on 30% of episodes, the rest stay
# on the uniform base rotation. Deterministic per seed, replayable.
#
#   MODEL_PATH=... ALBUMS_DIR=... bash examples/train_adaptive_city.sh
#
#   ADAPTIVE_MODE=city         (default) learnability curriculum
#   ADAPTIVE_MODE=city_ladder  fixed small-to-large schedule (control arm)
#   ADAPTIVE=0                 uniform rotation (static control arm)
#
# Everything else (GPU picking, memory sizing) is inherited from
# train_single_node.sh. There is no watchdog on this path: if the trainer
# dies, relaunch with the same EXPERIMENT_DIR and it resumes from its last
# checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-courier-adaptive-city}"
# The curriculum is a config decision: the `_cur` yaml carries
# `adaptive: true`; the plain one is the static control. The environment
# variable ADAPTIVE only decides whether the sidecar runs.
export ADAPTIVE="${ADAPTIVE:-1}"
if [ "$ADAPTIVE" = "1" ]; then
  export DATASET_TRAIN="${DATASET_TRAIN:-experiments/train_m8_maps_cur.yaml}"
else
  export DATASET_TRAIN="${DATASET_TRAIN:-experiments/train_m8_maps.yaml}"
fi
export DATASET_VAL="${DATASET_VAL:-experiments/val_courier_rq3v2.yaml}"
export ENTROPY_COEF="${ENTROPY_COEF:-0}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PWD/exps/${EXPERIMENT_NAME}_$(date +%m%d_%H%M)}"
mkdir -p "$EXPERIMENT_DIR"

if [ "$ADAPTIVE" = "1" ]; then
  export ADAPTIVE_EPISODE_LOG="$EXPERIMENT_DIR/episodes.jsonl"
  export ADAPTIVE_PROFILE="$EXPERIMENT_DIR/adaptive_profile.json"
  python3 tools/adaptive_profile.py \
      --log "$ADAPTIVE_EPISODE_LOG" --out "$ADAPTIVE_PROFILE" \
      --mode "${ADAPTIVE_MODE:-city}" --watch 600 \
      > "$EXPERIMENT_DIR/adaptive_profile.log" 2>&1 &
  echo "curriculum sidecar pid $! -> $EXPERIMENT_DIR/adaptive_profile.log"
fi

bash examples/train_single_node.sh
