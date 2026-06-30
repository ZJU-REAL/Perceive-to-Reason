#!/bin/bash
# Batch convert FSDP checkpoints to HuggingFace format.
# Usage: bash scripts/convert_fsdp_to_hf.sh <ckpt_root_dir>

set -e

CKPT_ROOT=${1:?"Usage: $0 <ckpt_root_dir>"}

if [ ! -d "$CKPT_ROOT" ]; then
    echo "[ERROR] Directory not found: $CKPT_ROOT"
    exit 1
fi

for step_dir in $(find "$CKPT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -t '_' -k3 -n); do
    actor_dir="$step_dir/actor"
    target_dir="$actor_dir/huggingface"
    step_name=$(basename "$step_dir")

    if [ ! -d "$actor_dir" ]; then
        echo "[SKIP] $step_name: actor dir not found"
        continue
    fi

    # Skip if huggingface dir already has model shards
    if [ -d "$target_dir" ] && ls "$target_dir"/model-*.safetensors 1>/dev/null 2>&1; then
        echo "[SKIP] $step_name: already converted"
        continue
    fi

    echo "[CONVERT] $step_name"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target_dir"

    echo "[DONE] $step_name"
done

echo "All conversions finished!"
