#!/usr/bin/env bash
# Choose GPUs and a memory fraction at launch time, from what is free now.
#
# Every previous failure of this run was the same shape: pick cards from a
# snapshot, set gpu_memory_utilization from that snapshot, and by the time vLLM
# actually starts two or three minutes later somebody else has taken memory and
# it refuses with "Free memory on device (5.75/23.55 GiB) is less than desired".
# Fifteen attempts died that way. A shared machine needs the choice made at the
# moment of use, not by me editing a constant between retries.
#
# Two rules, both learned the hard way:
#
#   * Prefer emptier cards over more cards. The rollout is data parallel, so
#     every GPU holds a whole vLLM replica and the KV cache each one gets comes
#     from its own budget. gpu_memory_utilization is global, so the tightest
#     card sets it for all of them -- adding a card someone else is using cuts
#     the budget on the empty ones too. Measured: three empty cards at 0.88 gave
#     6.11 GiB of KV; six cards including shared ones at 0.72 gave 2.34.
#   * Leave headroom. The fraction is computed from the tightest chosen card
#     minus MARGIN_MB, so a neighbour can grow by that much without killing the
#     run.
#
# Prints "CUDA_VISIBLE_DEVICES ROLLOUT_MEM N_GPUS SP" for the caller to eval.

WANT=${WANT_GPUS:-4}
# Make CUDA number the cards the way nvidia-smi does. The default order is
# FASTEST_FIRST, which on eight identical cards is arbitrary -- so an index
# chosen by looking at nvidia-smi can select a different physical card, and
# "the free one" becomes the one a vLLM has 98% full. The soak run's OOM
# ("GPU 0 ... Process <other user> has 2.42 GiB") showed another user's job on
# what nvidia-smi called an empty card: same trap, seen from the other side.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
          # batch 24 divides 2, 3, 4 and 6
MARGIN_MB=${MARGIN_MB:-1800}  # room for a neighbour to grow
MIN_FREE_MB=${MIN_FREE_MB:-18000}

read -r -a free_list <<< "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# Sort indices by free memory, descending, and keep the ones worth having.
order=$(for i in "${!free_list[@]}"; do echo "${free_list[$i]} $i"; done | sort -rn |
        awk -v m="$MIN_FREE_MB" '$1 >= m {print $2}')
chosen=()
for i in $order; do
  [ "${#chosen[@]}" -ge "$WANT" ] && break
  chosen+=("$i")
done

# The batch must divide evenly across ranks -- and the batch is whatever the
# caller set, not the 24 that was welded in here. With TRAIN_BS 10 and
# rollout.n 8 the batch is 80, which five cards divide exactly; the hardcoded
# 24 does not, so the picker silently dropped a card and reported gpus=4 for
# a run that had named five.
_BATCH=$(( ${TRAIN_BS:-6} * ${GROUP:-4} ))
while [ "${#chosen[@]}" -gt 1 ] && [ $((_BATCH % ${#chosen[@]})) -ne 0 ]; do
  unset 'chosen[-1]'
  chosen=("${chosen[@]}")
done

if [ "${#chosen[@]}" -lt 1 ]; then
  echo "NONE 0 0 1"
  exit 1
fi

tightest=999999
for i in "${chosen[@]}"; do
  [ "${free_list[$i]}" -lt "$tightest" ] && tightest=${free_list[$i]}
done
usable=$(( tightest - MARGIN_MB ))
mem=$(awk -v u="$usable" -v t="$total_mb" 'BEGIN{printf "%.2f", (u/t)}')

# Sequence parallelism needs an even world size and must divide the 32 heads.
sp=1
[ $(( ${#chosen[@]} % 2 )) -eq 0 ] && sp=2

echo "$(IFS=,; echo "${chosen[*]}") $mem ${#chosen[@]} $sp"
