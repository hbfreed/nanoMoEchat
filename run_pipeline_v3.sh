#!/bin/bash
# Pipeline v3 ("push"): heavier open-book SFT (3x CookingOpenBook), then
# long OPSD (10 epochs vs v2's 1) to actually test knowledge internalization.
set -e
cd "$(dirname "$0")"
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
export OMP_NUM_THREADS=1
PY=.venv/bin/python

BK=$NANOCHAT_BASE_DIR/v2_checkpoints_backup
mkdir -p $BK
cp -n $NANOCHAT_BASE_DIR/chatsft_checkpoints/d20/* $BK/ 2>/dev/null || true
cp -n $NANOCHAT_BASE_DIR/chatdpo_checkpoints/d20/* $BK/ 2>/dev/null || true
cp -n $NANOCHAT_BASE_DIR/chatrl_checkpoints/d20/* $BK/ 2>/dev/null || true

echo "=== [1/5] SFT (sft-cook-4, 3x open-book) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.chat_sft --source=mid --model-tag=d20 --run=sft-cook-4

echo "=== [2/5] DPO (dpo-cook-4) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.chat_dpo --source=sft --model-tag=d20 --run=dpo-cook-4 --init-lr-frac=0.01 --num-epochs=3

echo "=== [3/5] OPSD (opsd-cook-3, 10 epochs) ==="
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.chat_opsd -- --source=dpo --model-tag=d20 --run=opsd-cook-3 --num-epochs=10

echo "=== [4/5] Dashboard on final model ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.sft_dashboard --source=rl --model-tag=d20 --max-tokens=640

echo "=== [5/5] In-corpus closed-book quiz: dpo vs rl ==="
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.incorpus_quiz --source=dpo --model-tag=d20
CUDA_VISIBLE_DEVICES=0 $PY -m scripts.incorpus_quiz --source=rl --model-tag=d20

echo "=== pipeline v3 done ==="
