#!/bin/bash
set -x

MNT_PATH="${MNT_PATH:-/path/to/your/data}"
MODEL_PATH="$MNT_PATH/models/YOUR_VERIFIER_MODEL_PATH"

tmux new-session -s verifier_server "CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "verifier" \
    --host "0.0.0.0" \
    --port "8000" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager \
    --dtype bfloat16 \
    --max-num-seqs 32"
