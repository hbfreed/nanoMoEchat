#!/bin/bash
# Post-training pipeline v2: SFT (now with open-book QA) -> DPO -> OPSD -> evals.
# Changes vs v1: CookingOpenBook in the SFT mixture, clean_chunk on OPSD
# teacher passages. Everything else mirrors the v1 runs (sft-cook-2 /
# dpo-cook-2 / opsd-cook-1 settings).
set -e
cd "$(dirname "$0")"
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
export OMP_NUM_THREADS=1
PY=.venv/bin/python

# v1 checkpoints would be overwritten on step-number collision; keep them.
BK=$NANOCHAT_BASE_DIR/v1_checkpoints_backup
mkdir -p $BK
cp -n $NANOCHAT_BASE_DIR/chatsft_checkpoints/d20/* $BK/ 2>/dev/null || true
cp -n $NANOCHAT_BASE_DIR/chatdpo_checkpoints/d20/* $BK/ 2>/dev/null || true
cp -n $NANOCHAT_BASE_DIR/chatrl_checkpoints/d20/* $BK/ 2>/dev/null || true

echo "=== [1/5] SFT (sft-cook-3, + open-book) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.chat_sft --source=mid --model-tag=d20 --run=sft-cook-3

echo "=== [2/5] DPO (dpo-cook-3) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.chat_dpo --source=sft --model-tag=d20 --run=dpo-cook-3 --init-lr-frac=0.01 --num-epochs=3

echo "=== [3/5] OPSD (opsd-cook-2) ==="
torchrun --standalone --nproc_per_node=2 -m scripts.chat_opsd -- --source=dpo --model-tag=d20 --run=opsd-cook-2

echo "=== [4/5] Dashboard on final model ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.sft_dashboard --source=rl --model-tag=d20 --max-tokens=640

echo "=== [5/5] In-corpus closed-book quiz: dpo vs rl ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.incorpus_quiz --source=dpo --model-tag=d20
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.incorpus_quiz --source=rl --model-tag=d20

echo "=== pipeline v2 done ==="
