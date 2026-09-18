#!/usr/bin/env bash
# The any-point comparison protocol, end to end: one certified delivery per
# seed of a split, one UE boot each, then one results file.
#
#   tools/run_pixel_goal_protocol.sh heldout            # the ten held-out scenarios
#   tools/run_pixel_goal_protocol.sh dev                # the six development scenarios
#   tools/run_pixel_goal_protocol.sh protocol           # all sixteen
#   tools/run_pixel_goal_protocol.sh "3 5 12"           # an explicit seed list (no split check)
#
# Extra arguments go to tools/run_pixel_goal_front_rear_delivery.sh after the
# protocol's own flags (for instance ``--views pair`` for the two-view harness).
# The launcher's environment is this script's environment: PIXEL_GOAL_PYTHON,
# CITYCORE_PARIS_CONTENT, SIMWORLD_SPEAR_PYTHON, SIMWORLD_SPEAR_EXT_PYTHON,
# QWEN_ENDPOINT, QWEN_MODEL_NAME, QWEN_GPU, PIXEL_GOAL_GPU and the rest
# (docs/PIXEL_GOAL.md). This script adds:
#
#   PIXEL_GOAL_PROTOCOL_OUT       where seed-N/ runs and results.json go
#                                 (default artifacts/pixel_goal_protocol/<stamp>)
#   PIXEL_GOAL_PROTOCOL_LABEL     the results group's label (default: QWEN_MODEL_NAME)
#   PIXEL_GOAL_PROTOCOL_SEED_TIMEOUT_S   wall-clock cap per seed (default 5400)
#
# A seed whose run leaves no delivery_report.json is reported and the loop goes
# on; the results step then fails, because a split's number needs every seed.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
split="${1:?usage: run_pixel_goal_protocol.sh <heldout|dev|protocol|\"seed list\"> [launcher args...]}"
shift
python_bin="${PIXEL_GOAL_PYTHON:-$(command -v python3 || true)}"
out="${PIXEL_GOAL_PROTOCOL_OUT:-$repo_root/artifacts/pixel_goal_protocol/$(date -u +%Y%m%dT%H%M%SZ)}"
label="${PIXEL_GOAL_PROTOCOL_LABEL:-${QWEN_MODEL_NAME:-qwen3-vl-8b}}"
seed_timeout="${PIXEL_GOAL_PROTOCOL_SEED_TIMEOUT_S:-5400}"

case "$split" in
  heldout|dev|protocol)
    seeds="$(cd "$repo_root" && PYTHONPATH="$repo_root" "$python_bin" -c \
      "from tools.pixel_goal_order_pool import protocol_seeds; print(' '.join(map(str, protocol_seeds('$split'))))")" || exit 1
    split_check=(--split "$split")
    ;;
  *)
    seeds="$split"
    split_check=()
    ;;
esac

mkdir -p "$out"
echo "protocol split=$split seeds=[$seeds] label=$label out=$out"
for seed in $seeds; do
  echo "=== seed $seed  $(date -u +%H:%M:%SZ) ==="
  PIXEL_GOAL_OUTPUT="$out/seed-$seed" timeout "$seed_timeout" \
    bash "$repo_root/tools/run_pixel_goal_front_rear_delivery.sh" \
      --route-profile validated_pool --order-mode random --seed "$seed" \
      --min-delivery-m 30 --max-delivery-m 80 --min-route-turns 1 --require-marked-crossing \
      "$@" > "$out/seed-$seed.log" 2>&1
  rc=$?
  report="$out/seed-$seed/delivery_report.json"
  if [ -f "$report" ]; then
    "$python_bin" - "$report" "$rc" <<'PY'
import json, sys
report = json.load(open(sys.argv[1])); s = report["summary"]
print(f"rc={sys.argv[2]} delivered={s['delivered']} earnings={s['earnings']} late={s.get('late')} "
      f"walked_m={s.get('walked_m')} turns={report.get('turns')} rejected={s.get('rejected_actions')} "
      f"termination={report.get('termination')}")
PY
  else
    echo "rc=$rc no report; last log line: $(tail -n 1 "$out/seed-$seed.log" | cut -c1-160)"
  fi
done

cd "$repo_root" && PYTHONPATH="$repo_root" "$python_bin" tools/pixel_goal_results.py \
  --out "$out/results.json" --group "$label=$out" "${split_check[@]}"
