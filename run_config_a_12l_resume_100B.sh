#!/bin/bash

# Resume config-a-12L-256d-10Mact from pre-decay checkpoint (+100M tokens).
# WSD trick: the stable-phase checkpoint still has peak LR momentum,
# so continuation adds real training signal rather than re-heating a cooled model.
#
# Original run: num_iter=4521, warmdown_ratio=0.2 → decay started at step 3617.
# Save points were 1000/2000/3000/4000/4521 — step 3000 is the last pre-decay save.
#
# Extension math:
#   100M tokens / 552,960 per step = 181 new steps
#   num_iter = 4521 + 181 = 4702
#   warmdown_iters = 0.2 * 4702 = 940, warmdown_start = 3762
#   From step 3000: 762 stable + 940 decay = 1702 steps total
#   Model sees 3000·552960 + 1702·552960 = 2.6B tokens (net +100M)
# usage: bash run_config_a_12l_resume_100M.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"

source .venv/bin/activate
source "$HOME/.cargo/env" 2>/dev/null || true

torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=config-a-12L-256d-10Mact-resume100B \
    --model-tag=config-a-12L-256d-10Mact \
    --resume-from-step=3000 \
    --depth=12 \
    --model-dim=256 --num-heads=2 --num-kv-heads=2 \
    --max-seq-len=1024 \
    --window-pattern=L \
    --expert-sizes='[[192,256]]' --num-active-experts=4 \
    --load-balance-loss-weight=0.001 \
    --router-z-loss-weight=0.001 \
    --compute-loss-weight=0.0 \
    --device-batch-size=60 --total-batch-size=552960 --num-iterations=180845 \
    --warmup-ratio=0.02 --warmdown-ratio=0.2 --final-lr-frac=0.0 \
    --eval-every=250 --core-metric-every=-1 --sample-every=2000 --save-every=500
