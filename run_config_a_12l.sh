#!/bin/bash

# Config A: 12-layer MoE, ~10M active / ~306M total (trunk)
# H=256 (2 heads x 128), E=192 experts x D=256, top-4 active
# Per-layer: 4*256^2 attn + 256*192 router + 2*4*256*256 MLP-active
#          = 262K + 49K + 524K = 835K active/layer x 12 = ~10.0M
# Total MLP: 2*192*256*256 = 25.2M * 12 = 302M + 3.15M attn + 0.59M router = ~306M
# Aiming for ~320M total was tempting but E=192 (2^6*3) is cleanest near here;
# Ling EL paper (2507.17702) confirms off-optimal configs still work fine.
# usage: bash run_config_a_12l.sh

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

# Download shards and train tokenizer (skip if already present)
python -m nanochat.dataset -n 50
python -m scripts.tok_train --max_chars=500000000

# Training: 2.5B tokens on 3 GPUs (matches run_baseline_bias.sh horizon)
# batch: 60 * 1024 * 3 = 184,320 tokens/micro-batch; grad_accum = 552960/184320 = 3
# iterations: 2.5B / 552,960 = 4,521
# --window-pattern=L : SM86 GPUs have no FA3, avoid SDPA sliding-window perf cliff
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=config-a-12L-256d-10Mact \
    --model-tag=config-a-12L-256d-10Mact \
    --depth=12 \
    --model-dim=256 --num-heads=2 --num-kv-heads=2 \
    --max-seq-len=1024 \
    --window-pattern=L \
    --expert-sizes='[[192,256]]' --num-active-experts=4 \
    --load-balance-loss-weight=0.001 \
    --router-z-loss-weight=0.001 \
    --compute-loss-weight=0.0 \
    --device-batch-size=60 --total-batch-size=552960 --num-iterations=4521 \
    --warmup-ratio=0.02 --warmdown-ratio=0.2 --final-lr-frac=0.0 \
    --eval-every=250 --core-metric-every=-1 --sample-every=2000 --save-every=1000
