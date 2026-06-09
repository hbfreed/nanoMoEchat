#!/bin/bash

# Config B: 12-layer MoE, ~32M active / ~1B total
# H=512 (4 heads x 128), E=192 experts x D=384, top-4 active
# Per-layer: 4*512^2 attn + 512*192 router + 2*4*512*384 MLP-active
#          = 1.05M + 98K + 1.57M = 2.72M active/layer x 12 = ~32.6M
# Total MLP: 2*192*512*384 = 75.5M * 12 = 906M + 12.6M attn + 1.2M router = ~920M trunk
# With embeddings (wte + lm_head ~67M): total ~987M ≈ 1B
# Activation ratio A = 4/192 = 2.1%, Granularity G = 2*512/384 ≈ 2.67
# usage: bash run_config_b_12l_32M_1B.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
# expandable_segments: reduces fragmentation
# garbage_collection_threshold: allocator self-GCs at 85% to avoid runaway high-water mark
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85
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
if ! python3 -c "import rustbpe" &> /dev/null; then
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# Download enough ClimbMix shards for 100B tokens (~1660 at ~60M tok/shard)
python3 -m nanochat.dataset -n 1660 -w $(nproc)

# Tokenizer (skip if already exists)
python3 -m scripts.tok_train --max_chars=500000000

# Training: 100B tokens on 3 GPUs
# batch: 30 * 1024 * 3 = 92,160 tokens/micro-batch; grad_accum = 552960/92160 = 6
# iterations: 100B / 552,960 = 180,845
# device-batch=30 with GC threshold pushes ~22-23GB (close to redline, but the
# allocator self-manages at 85% via PYTORCH_CUDA_ALLOC_CONF). If the first few
# steps push past 23GB, drop to 20 (grad_accum=9) or 18 (grad_accum=10).
# Clean divisors of 180 (=552960/(1024*3)): 18, 20, 30, 36.
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=config-b-12L-512d-32Mact-100B \
    --model-tag=config-b-12L-512d-32Mact \
    --depth=12 \
    --model-dim=512 --num-heads=4 --num-kv-heads=4 \
    --max-seq-len=1024 \
    --window-pattern=L \
    --expert-sizes='[[192,384]]' --num-active-experts=4 \
    --load-balance-loss-weight=0.001 \
    --router-z-loss-weight=0.001 \
    --compute-loss-weight=0.0 \
    --device-batch-size=30 --total-batch-size=552960 --num-iterations=180845 \
    --warmup-ratio=0.02 --warmdown-ratio=0.2 --final-lr-frac=0.0 \
    --eval-every=250 --core-metric-every=-1 --sample-every=2000 --save-every=10000
