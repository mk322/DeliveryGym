#!/usr/bin/env bash
# Evaluate a model on the benchmark, end to end, on one machine with one GPU.
#
#   bash examples/eval_api_model.sh                       # serve + eval Qwen3-VL-4B
#   MODEL=Qwen/Qwen3-VL-8B-Instruct bash examples/eval_api_model.sh
#   SKIP_SERVE=1 BASE_URL=http://host:8000/v1 MODEL=name bash examples/eval_api_model.sh
#
# The eval itself needs no GPU -- point BASE_URL at any OpenAI-compatible
# endpoint (a hosted API included) and set SKIP_SERVE=1.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT/v1}"
TAG="${TAG:-$(basename "$MODEL" | tr 'A-Z' 'a-z')}"

if [ "${SKIP_SERVE:-0}" != "1" ]; then
    echo "==> serving $MODEL on :$PORT (vLLM)"
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" \
        --limit-mm-per-prompt '{"image": 16}' \
        --max-model-len 32768 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null' EXIT
    # --limit-mm-per-prompt matters: a turn offers up to ~10 pictures, and a
    # server capped below that returns HTTP 400, which once got counted as
    # the MODEL's format-error rate. See model_io.py's docstring. The value
    # is JSON: vLLM 0.10+ rejects the older image=16 spelling.
    until curl -sf "$BASE_URL/models" >/dev/null; do
        kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }
        sleep 3
    done
fi

echo "==> smoke test: 2 episodes"
python -m embodiedbench.eval run --model "$MODEL" --base-url "$BASE_URL" \
    --split val --limit 2 --out results --tag "${TAG}-smoke"

echo "==> the benchmark: all 64 validation seeds"
python -m embodiedbench.eval run --model "$MODEL" --base-url "$BASE_URL" \
    --split val --out results --tag "$TAG"

echo "==> done: results/$TAG.json (the number is earnings_mean with its CI)"
