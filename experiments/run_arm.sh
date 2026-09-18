#!/usr/bin/env bash
# Launch one experiment arm on this machine, by its short name.
#
#   MODEL_PATH=... ALBUMS_DIR=... bash experiments/run_arm.sh base2
#   MODEL_PATH=... ALBUMS_DIR=... bash experiments/run_arm.sh a-frontier
#   bash experiments/run_arm.sh --list
#
# Every arm is the base recipe (examples/train_single_node.sh) plus the
# per-arm settings in the table below -- a training yaml, sometimes a
# validation yaml, sometimes the curriculum sidecar. Nothing else differs
# between arms, which is what makes them comparable. The table is the same
# one the paper's runs were launched from; experiments/README.md says what
# each arm asks.
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A ARMS=(
  # RQ1 -- reward ablations against base2.
  [base2]="courier-base2       DATASET_TRAIN=experiments/train_e0.yaml"
  [r1-red]="courier-r1-red     DATASET_TRAIN=experiments/train_r1_red.yaml"
  [r1-all]="courier-r1-all     DATASET_TRAIN=experiments/train_r1_all.yaml"
  [r1-shape]="courier-r1-shape DATASET_TRAIN=experiments/train_r1_shape.yaml"
  # RQ4 -- task scaling by repetition rate (60/200/600-seed pools).
  [t60]="courier-t60           DATASET_TRAIN=experiments/train_t60.yaml"
  [t200]="courier-t200         DATASET_TRAIN=experiments/train_t200.yaml"
  [t600]="courier-t600         DATASET_TRAIN=experiments/train_t600.yaml"
  # RQ3 -- map scaling: 1/2/4/8 training cities, judged on an unseen city.
  [m1v2]="courier-m1v2        DATASET_TRAIN=experiments/train_m1_maps.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16"
  [m2v2]="courier-m2v2        DATASET_TRAIN=experiments/train_m2_maps.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16"
  [m4v2]="courier-m4v2        DATASET_TRAIN=experiments/train_m4_maps.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16"
  [m8u]="courier-m8-uniform   DATASET_TRAIN=experiments/train_m8_maps.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16"
  # RQ2 -- the evolving dispatcher, single city: four curriculum signals.
  [a-fail]="courier-a-fail       DATASET_TRAIN=experiments/train_e4b_gentle.yaml ADAPTIVE=1 ADAPTIVE_MODE=fail"
  [a-frontier]="courier-a-frontier DATASET_TRAIN=experiments/train_e4b_gentle.yaml ADAPTIVE=1 ADAPTIVE_MODE=frontier"
  [a-hard]="courier-a-hard       DATASET_TRAIN=experiments/train_a_hard.yaml ADAPTIVE=1 ADAPTIVE_MODE=hard"
  [a-ladder]="courier-a-ladder     DATASET_TRAIN=experiments/train_e4b_gentle.yaml ADAPTIVE=1 ADAPTIVE_MODE=ladder"
  # RQ2 -- the evolving dispatcher, eight cities: the dispatcher chooses WHERE.
  [m8c]="courier-m8-citycur   DATASET_TRAIN=experiments/train_m8_maps_cur.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16 ADAPTIVE=1 ADAPTIVE_MODE=city"
  [m8l]="courier-m8-cityladder DATASET_TRAIN=experiments/train_m8_maps_cur.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16 ADAPTIVE=1 ADAPTIVE_MODE=city_ladder"
  [m8n]="courier-m8-cityneed  DATASET_TRAIN=experiments/train_m8_maps_cur.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16 ADAPTIVE=1 ADAPTIVE_MODE=city_need"
  [m8x]="courier-m8-mix60     DATASET_TRAIN=experiments/train_m8_maps_mix60.yaml DATASET_VAL=experiments/val_courier_rq3v2.yaml ROLLOUT_SEQS=16 ADAPTIVE=1 ADAPTIVE_MODE=city"
  # RQ5 -- optional constraints, and what reward does to behaviour.
  [c1]="courier-c1-econ       DATASET_TRAIN=experiments/train_c1_econ.yaml DATASET_VAL=experiments/val_courier_constraints.yaml"
  [c2]="courier-c2-battery    DATASET_TRAIN=experiments/train_c2_battery.yaml DATASET_VAL=experiments/val_courier_constraints.yaml"
  [c3]="courier-c3-recharge   DATASET_TRAIN=experiments/train_c3_recharge.yaml DATASET_VAL=experiments/val_courier_recharge.yaml ROLLOUT_SEQS=24"
  [c3f]="courier-c3-skillcur  DATASET_TRAIN=experiments/train_c3_recharge_cur.yaml DATASET_VAL=experiments/val_courier_recharge.yaml ROLLOUT_SEQS=24 ADAPTIVE=1 ADAPTIVE_MODE=skill_battery"
  [q2]="courier-q2           DATASET_TRAIN=experiments/train_q2.yaml"
  [f-cat]="courier-f-categories DATASET_TRAIN=experiments/train_f_categories.yaml DATASET_VAL=experiments/val_courier_categories.yaml"
  [d-tight]="courier-d-tight     DATASET_TRAIN=experiments/train_d_tight.yaml"
  # RQ-V -- what the pictures are worth: route / everything narrated in text.
  [v-route]="courier-v-route     DATASET_TRAIN=experiments/train_v_route.yaml DATASET_VAL=experiments/val_courier_v_route.yaml"
  [v-all]="courier-v-all       DATASET_TRAIN=experiments/train_v_all.yaml DATASET_VAL=experiments/val_courier_v_all.yaml"
)

if [ "${1:-}" = "--list" ] || [ $# -eq 0 ]; then
  for key in $(printf '%s\n' "${!ARMS[@]}" | sort); do
    printf '  %-11s %s\n' "$key" "${ARMS[$key]}"
  done
  exit 0
fi

key="$1"; shift
spec="${ARMS[$key]:?unknown arm '$key' (try --list)}"
read -r name envs <<<"$spec"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$name}"
# Entropy bonus off on every arm: the runs were made that way, and the
# entropy telemetry is not understood well enough to tune it (TASK_DESIGN 5).
export ENTROPY_COEF="${ENTROPY_COEF:-0}"
for kv in $envs; do export "$kv"; done
# Arms without their own validation yaml are judged on the standard 64 Paris
# shifts -- stated here so the adaptive launcher's multi-city default never
# applies to a single-city arm.
export DATASET_VAL="${DATASET_VAL:-embodiedbench/training/vagen/val_courier.yaml}"

echo "== $key -> $EXPERIMENT_NAME  ($envs)"
if [ "${ADAPTIVE:-0}" = "1" ]; then
  # The sidecar rebuilds the sampling profile from the episode log; the
  # training yaml carries `adaptive: true` so the env reads it.
  exec bash examples/train_adaptive_city.sh
fi
exec bash examples/train_single_node.sh
