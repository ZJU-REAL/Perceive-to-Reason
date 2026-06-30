export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=ERROR

MNT_PATH="${MNT_PATH:-/path/to/your/data}"
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


MODEL_NAMES=(
    "YOUR_MODEL_NAME"
)

# default, thinking, p2r
EVAL_MODE="p2r"

# Available tasks: "V-Star" "HR-Bench" "MME-RealWorld-lite" "MME-RealWorld"
TASKS=("V-Star")

declare -A MODEL_PATH_MAP=(
    ["YOUR_MODEL_NAME"]="${MNT_PATH}/models/YOUR_MODEL_PATH"
)


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_PATH="${MODEL_PATH_MAP[$MODEL_NAME]}"

    if [[ -z "$MODEL_PATH" ]]; then
        echo "Error: unsupported model ${MODEL_NAME}"
        exit 1
    fi

    for TASK in "${TASKS[@]}"; do
        echo "Evaluating ${MODEL_NAME} on ${TASK} (Mode: ${EVAL_MODE})..."

        python -m main_eval \
            --model_path "$MODEL_PATH" \
            --data_dir "${MNT_PATH}/datasets" \
            --task "$TASK" \
            --eval_mode "$EVAL_MODE" \
            --temperature 0 \
            --top_p 1 \
            --nframes 32 \
            --max_pixels $((4096*32*32)) \
            --min_pixels $((16*32*32)) \
            --log_dir "${MNT_PATH}/logs/${MODEL_NAME}" \
            --batch_size 256 \
            --debug_size 256
    done
done