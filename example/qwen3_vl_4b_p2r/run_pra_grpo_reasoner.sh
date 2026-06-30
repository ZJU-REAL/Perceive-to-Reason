#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=65536

MNT_PATH="${MNT_PATH:-/path/to/your/data}"
ENGINE=${1:-vllm}

# ─── Paths ──────────────────────────────────────────────────────────────────────
train_files="$MNT_PATH/data/YOUR_TRAIN_DATA.parquet"
val_files="$MNT_PATH/data/YOUR_VAL_DATA.parquet"
model_path="$MNT_PATH/ckpts/qwen3_vl_4b_p2r/run_pra_grpo_perceiver/YOUR_PERCEIVER_CKPT"
project_name="qwen3_vl_4b_p2r"
exp_name="run_pra_grpo_reasoner"
save_path="$MNT_PATH/ckpts/$project_name/$exp_name"

mkdir -p "$save_path"

export SWANLAB_LOG_DIR="$save_path/swanlab"
export SWANLAB_MODE="offline"

# ─── Frozen perceiver service (called during dataset construction) ──────────────
PERCEIVER_HOST="${PERCEIVER_HOST:-YOUR_PERCEIVER_HOST}"
PERCEIVER_PORT=${PERCEIVER_PORT:-8000}
PERCEIVER_MODEL="perceiver"

# ─── Verifier service (for free_form answer evaluation) ────────────────────────
VERIFIER_HOST="${VERIFIER_HOST:-YOUR_VERIFIER_HOST}"
VERIFIER_PORT=${VERIFIER_PORT:-8000}
VERIFIER_MODEL="verifier"

MAX_PIXELS=2097152

# ─── Training ───────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=8704 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    +data.image_prefix="$MNT_PATH/datasets/P2R-10k/images" \
    +data.role=reasoner \
    +data.max_pixels=$MAX_PIXELS \
    data.perceiver_host="$PERCEIVER_HOST" \
    data.perceiver_port=$PERCEIVER_PORT \
    data.perceiver_model="$PERCEIVER_MODEL" \
    data.custom_cls.path=pkg://verl.utils.dataset.p2r_dataset \
    data.custom_cls.name=P2RDataset \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    reward.custom_reward_function.path=pkg://verl.utils.reward_score.p2r_reward \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.role=reasoner \
    +reward.custom_reward_function.reward_kwargs.verifier_host="$VERIFIER_HOST" \
    +reward.custom_reward_function.reward_kwargs.verifier_port=$VERIFIER_PORT \
    +reward.custom_reward_function.reward_kwargs.verifier_model="$VERIFIER_MODEL" \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$exp_name" \
    trainer.default_local_dir="$save_path" \
    trainer.rollout_data_dir="$save_path/rollout_logs" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 "$@"
