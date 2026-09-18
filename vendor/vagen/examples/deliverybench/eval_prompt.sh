#!/bin/bash
# Prompt tuning eval script — SGLang server + direct env interaction.
#
# WORKFLOW
# --------
# Step 1 (ONCE): Launch the SGLang server in a separate terminal and keep it running.
#   You only need to do this once per session; the model stays loaded between eval runs.
#
#   python -m sglang.launch_server \
#       --model-path ~/models/Qwen2.5-VL-3B-Instruct \
#       --port 30000 \
#       --chat-template qwen2-vl
#
# Step 2 (EACH ITERATION): Edit your prompt, then re-run this script.
#   Prompt locations:
#     - vagen/envs/deliverybench/utils/prompt.py   (system_prompt, format_prompt, templates)
#     - examples/deliverybench/val_deliverybench.yaml  (prompt_format: free_think | wm)
#
# Step 3: Compare metrics (reward, success_rate, mean_turns) across runs.

set -e

MODEL_PATH=/root/VAGEN/models/Qwen2.5-VL-7B-Instruct
SERVER_URL="http://localhost:30000"
VAL_YAML="scripts/train/earning_reward/val_deliverybench.yaml"

OUTPUT_DIR=/root/VAGEN/exps

python -m vagen.eval.prompt_tuning \
    --server-url "${SERVER_URL}" \
    --model-path "${MODEL_PATH}" \
    --val-yaml "${VAL_YAML}" \
    --n-per-env 1 \
    --max-concurrent 32 \
    --temperature 0 \
    --max-tokens 512 \
    --output-dir "${OUTPUT_DIR}"
