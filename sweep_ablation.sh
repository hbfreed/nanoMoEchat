#!/bin/bash

# Full ablation runs for MoE bias balancing vs load-balance losses
# 5 runs x 4521 iterations each (2.5B tokens)
# usage: bash sweep_ablation.sh
#
# IMPORTANT: Set GAMMA_STAR below to the best gamma from sweep_gamma.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
mkdir -p $NANOCHAT_BASE_DIR

GAMMA_STAR=0.001  # <-- Replace with best gamma from sweep

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
    --eval_every=250 --core_metric_every=-1 --sample_every=2000 --save_every=1000"

# Run 1: baseline (standard load-balance + z-loss + compute-loss)
echo "===== Run 1: baseline ====="
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=ablation-baseline \
    --use_bias_balancing=False \
    --load_balance_loss_weight=0.08 \
    --router_z_loss_weight=0.001 \
    --compute_loss_weight=0.004 \
    --bias_update_speed=0.001 \
    $COMMON

# Run 2: lbl-high (higher load-balance weight, no compute loss)
echo "===== Run 2: lbl-high ====="
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=ablation-lbl-high \
    --use_bias_balancing=False \
    --load_balance_loss_weight=0.12 \
    --router_z_loss_weight=0.001 \
    --compute_loss_weight=0.0 \
    --bias_update_speed=0.001 \
    $COMMON

# Run 3a: bias-zloss (bias balancing + z-loss only)
echo "===== Run 3a: bias-zloss ====="
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=ablation-bias-zloss \
    --use_bias_balancing=True \
    --load_balance_loss_weight=0.0 \
    --router_z_loss_weight=0.001 \
    --compute_loss_weight=0.0 \
    --bias_update_speed=${GAMMA_STAR} \
    $COMMON

# Run 3b: bias-only (bias balancing, no auxiliary losses)
echo "===== Run 3b: bias-only ====="
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=ablation-bias-only \
    --use_bias_balancing=True \
    --load_balance_loss_weight=0.0 \
    --router_z_loss_weight=0.0 \
    --compute_loss_weight=0.0 \
    --bias_update_speed=${GAMMA_STAR} \
    $COMMON

# Run 4: bias-compute (bias balancing + compute loss)
echo "===== Run 4: bias-compute ====="
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=ablation-bias-compute \
    --use_bias_balancing=True \
    --load_balance_loss_weight=0.0 \
    --router_z_loss_weight=0.0 \
    --compute_loss_weight=0.004 \
    --bias_update_speed=${GAMMA_STAR} \
    $COMMON

echo "Ablation sweep complete. Check wandb for results."
