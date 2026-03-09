#!/bin/bash

# Null expert experiments: retrain B and C with proper model_tags
# Uses torchrun with 3 GPUs (DDP) per run, runs sequentially
# usage: bash sweep_null_experts.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
mkdir -p $NANOCHAT_BASE_DIR

# Common args matching original runs (from wandb configs)
COMMON="--depth=12 --model_dim=768 --num_heads=6 --max_seq_len=1024 \
    --device_batch_size=10 --total_batch_size=552960 --num_iterations=4521 \
    --eval_every=250 --core_metric_every=-1 --sample_every=2000 --save_every=1000 \
    --num_active_experts=8 \
    --use_bias_balancing=False \
    --load_balance_loss_weight=0.2 \
    --router_z_loss_weight=0.001 \
    --compute_loss_weight=0.0"

# (B) 48 real uniform + 16 null experts
echo "===== Run B: 48 real + 16 null (3 GPUs, torchrun) ====="
uv run torchrun --nproc_per_node=3 -m scripts.base_train \
    --run=null-B-48real-16null-v2 \
    --model_tag=null_expert_sweep/null-B \
    --expert_sizes="[(48, 256)]" \
    --num_null_experts=16 \
    $COMMON

# (C) 24 small + 24 large real + 16 null experts
# widths 128 + 384 keeps total expert width = 48*256 = 12288
echo "===== Run C: 24 small + 24 large + 16 null (3 GPUs, torchrun) ====="
uv run torchrun --nproc_per_node=3 -m scripts.base_train \
    --run=null-C-24s24l-16null-v2 \
    --model_tag=null_expert_sweep/null-C \
    --expert_sizes="[(24, 128), (24, 384)]" \
    --num_null_experts=16 \
    $COMMON

# (C-resumed) Resume Run C from step 3000
echo "===== Run C RESUME: 24 small + 24 large + 16 null (3 GPUs, torchrun) ====="
uv run torchrun --nproc_per_node=3 -m scripts.base_train \
    --run=null-C-24s24l-16null-v2-resumed \
    --model_tag=null_expert_sweep/null-C \
    --resume_from=/media/henry/MoreFiles/base_checkpoints/null_expert_sweep/null-C \
    --resume_step=3000 \
    --expert_sizes="[(24, 128), (24, 384)]" \
    --num_null_experts=16 \
    $COMMON

echo "Null expert sweep complete. Check wandb for results."
