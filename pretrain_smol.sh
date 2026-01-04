#!/bin/bash

# smol pretrain script for SYNTH dataset experiments
# usage: bash pretrain_smol.sh

set -e  # exit on error
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
mkdir -p $NANOCHAT_BASE_DIR

# venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Install Rust and build rustbpe tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# download shards (~55M tokens each, need ~1818 for 100B)
python -m nanochat.dataset -n 1820

# train tokenizer if not already present
python -m scripts.tok_train --max_chars=500000000

# Model: depth=12, dim=768, 12 heads (~167M active params with MoE)
# Training: 100B tokens on 3 GPUs
# batch: 8 * 1024 * 3 * 21 (grad_accum) = 516,096 tokens/step
# iterations: 100B / 516,096 = 193,765
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=synth-moe-12L-768d-100B \
    --depth=12 \
    --model_dim=768 \
    --num_heads=12 \
    --max_seq_len=1024 \
    --device_batch_size=8 \
    --total_batch_size=516096 \
    --num_iterations=193765 \
    --eval_every=250 \
    --core_metric_every=-1 \
    --sample_every=2000 \
    --save_every=1000
