python -m sglang.launch_server \
    --model-path /root/VAGEN/models/Qwen2.5-VL-7B-Instruct \
    --port 30000 \
    --chat-template qwen2-vl \
    --enable-cache-report \
    --dp-size 4