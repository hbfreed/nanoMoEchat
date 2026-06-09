#!/bin/bash

# Resume Config B (12L, ~32M active / ~1B total) from step 62500.
# History:
#   Run 1 cold-started, crashed at step ~27600 (system reboot, May 8).
#   Run 2 resumed from 20000, crashed at step ~62718 (system reboot, May 13).
# Latest checkpoint on disk: step 62500. The resume step is auto-protected
# from keep_last_n=5 pruning. Total iterations and LR schedule unchanged:
# warmdown still begins at step 144676 (protected snapshot).
# usage: bash run_config_b_12l_32M_1B_resume.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate
source "$HOME/.cargo/env" 2>/dev/null || true

torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=config-b-12L-512d-32Mact-100B-resume20k \
    --wandb-run-id=8bivozl7 \
    --model-tag=config-b-12L-512d-32Mact \
    --resume-from-step=62500 \
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
    --eval-every=250 --core-metric-every=-1 --sample-every=2000 \
    --save-every=500 --keep-last-n=5
