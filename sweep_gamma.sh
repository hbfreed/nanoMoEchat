#!/bin/bash

# Gamma (bias_update_speed) sweep for bias balancing with variable experts
# Expert setup: 4x2560 + 4x512, top-2 active
# Runs 3 gamma values in parallel, one per GPU (single-process, no DDP)
# usage: bash sweep_gamma.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
mkdir -p $NANOCHAT_BASE_DIR

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

# Common args: variable experts (4x2560 + 4x512), top-2
# total_batch_size=552960 / (10 * 1024 * 1 GPU) = 54 grad_accum_steps
# eval_every=50 for better resolution over 500 steps (11 eval points)
COMMON_ARGS="--depth=12 --model_dim=768 --num_heads=6 --max_seq_len=1024 \
    --expert_sizes=[(4,2560),(4,512)] --num_active_experts=2 \
    --device_batch_size=10 --total_batch_size=552960 --num_iterations=500 \
    --eval_every=50 --core_metric_every=-1 --sample_every=2000 --save_every=-1 \
    --use_bias_balancing=True --load_balance_loss_weight=0.0 --router_z_loss_weight=0.0 \
    --compute_loss_weight=0.0"

# Launch 3 gamma values in parallel, one per GPU
echo "Launching 3 gamma sweeps in parallel (one per GPU)..."

CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.base_train \
    --run=sweep-varexp-gamma-0.001 \
    --bias_update_speed=0.001 \
    $COMMON_ARGS > sweep_gamma_0.001.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.base_train \
    --run=sweep-varexp-gamma-0.01 \
    --bias_update_speed=0.01 \
    $COMMON_ARGS > sweep_gamma_0.01.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.base_train \
    --run=sweep-varexp-gamma-0.1 \
    --bias_update_speed=0.1 \
    $COMMON_ARGS > sweep_gamma_0.1.log 2>&1 &
PID2=$!

echo "PIDs: gamma=0.001 ($PID0), gamma=0.01 ($PID1), gamma=0.1 ($PID2)"
echo "Logs: sweep_gamma_0.001.log, sweep_gamma_0.01.log, sweep_gamma_0.1.log"
echo "Waiting for all runs to finish..."

FAILED=0
wait $PID0 || { echo "gamma=0.001 failed (see sweep_gamma_0.001.log)"; FAILED=1; }
wait $PID1 || { echo "gamma=0.01 failed (see sweep_gamma_0.01.log)"; FAILED=1; }
wait $PID2 || { echo "gamma=0.1 failed (see sweep_gamma_0.1.log)"; FAILED=1; }

if [ $FAILED -eq 0 ]; then
    echo "All gamma sweeps completed successfully. Check wandb for results."
else
    echo "Some runs failed. Check logs above."
fi
