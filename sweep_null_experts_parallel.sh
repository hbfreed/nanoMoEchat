#!/bin/bash

# Null expert experiments: baseline vs null-expert MoE configurations
# 3 runs in PARALLEL on GPUs 0, 1, 2
# (A) 64 uniform, (B) 48 real + 16 null, (C) 24 small + 24 large + 16 null
# All top-8, global LBL, no compute loss, no bias balancing
# usage: bash sweep_null_experts_parallel.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
mkdir -p $NANOCHAT_BASE_DIR logs

# venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env" 2>/dev/null || true

# Build rustbpe tokenizer if not already built
if ! python -c "import rustbpe" &> /dev/null; then
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# download shards
python -m nanochat.dataset -n 50

# train tokenizer if not already present
python -m scripts.tok_train --max_chars=500000000

# Common args matching pretrain_smol.sh model config
COMMON="--depth=12 --model_dim=768 --num_heads=6 --max_seq_len=1024 \
    --device_batch_size=10 --total_batch_size=552960 --num_iterations=4521 \
    --eval_every=250 --core_metric_every=-1 --sample_every=2000 --save_every=1000 \
    --num_active_experts=8 \
    --use_bias_balancing=False \
    --load_balance_loss_weight=0.08 \
    --router_z_loss_weight=0.001 \
    --compute_loss_weight=0.0"

# (A) Baseline: 64 uniform experts, no null experts — GPU 0
echo "===== Launching Run A: baseline 64 uniform (GPU 0) ====="
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.base_train \
    --run=null-A-baseline-64uniform \
    --expert_sizes="[(64, 256)]" \
    --num_null_experts=0 \
    $COMMON > logs/null-A.log 2>&1 &
PID_A=$!

# (B) 48 real uniform + 16 null experts — GPU 1
echo "===== Launching Run B: 48 real + 16 null (GPU 1) ====="
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.base_train \
    --run=null-B-48real-16null \
    --expert_sizes="[(48, 256)]" \
    --num_null_experts=16 \
    $COMMON > logs/null-B.log 2>&1 &
PID_B=$!

# (C) 24 small + 24 large real + 16 null experts — GPU 2
# widths 128 + 384 keeps total expert width = 48*256 = 12288
echo "===== Launching Run C: 24 small + 24 large + 16 null (GPU 2) ====="
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.base_train \
    --run=null-C-24s24l-16null \
    --expert_sizes="[(24, 128), (24, 384)]" \
    --num_null_experts=16 \
    $COMMON > logs/null-C.log 2>&1 &
PID_C=$!

echo "Launched: A=$PID_A (GPU 0), B=$PID_B (GPU 1), C=$PID_C (GPU 2)"
echo "Logs: logs/null-{A,B,C}.log"

wait $PID_A $PID_B $PID_C
echo "All 3 runs complete. Check wandb for results."
