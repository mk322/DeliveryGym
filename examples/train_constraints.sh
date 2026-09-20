#!/usr/bin/env bash
# RQ5 showcase: the constraint world -- fee jitter, food that cools,
# customer notes, a phone battery. Validation reports both the standard
# 64 shifts (comparable with every other run) and the same shifts with
# the constraints on; watch warm_delivery_rate and phone_alive_rate learn.
#
#   MODEL_PATH=... ALBUMS_DIR=... bash examples/train_constraints.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-courier-constraints}"
export DATASET_TRAIN="${DATASET_TRAIN:-experiments/train_c2_battery.yaml}"
export DATASET_VAL="${DATASET_VAL:-experiments/val_courier_constraints.yaml}"
export ENTROPY_COEF="${ENTROPY_COEF:-0}"

bash examples/train_single_node.sh
