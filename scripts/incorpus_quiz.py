"""In-corpus closed-book quiz: measures OPSD knowledge internalization.

The holdout quiz (sft_dashboard part 2) uses never-trained books, so it
measures generalization. This one asks questions whose source chunks WERE
the OPSD training distribution, closed-book — if OPSD internalized
anything, it shows up here first. Prints the source passage for eyeball
scoring.

Run:
    python -m scripts.incorpus_quiz --source=rl --model-tag=d20
"""
import argparse
import json
import os
import random
from contextlib import nullcontext

import torch
from huggingface_hub import hf_hub_download

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from tasks.cooking import clean_chunk

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="rl")
parser.add_argument("--model-tag", type=str, default="d20")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=400)
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.step
)
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda" else nullcontext()
)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

_local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "posttrain", "out", "opsd_pairs.jsonl")
if os.path.exists(_local):
    pairs_path = _local
else:
    pairs_path = hf_hub_download("hbfreed/cook-posttrain", "opsd_pairs.jsonl", repo_type="dataset")
rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
sample = random.Random(3).sample(rows, args.n)

for row in sample:
    ids = [bos, user_start] + tokenizer.encode(row["question"]) + [user_end, assistant_start]
    toks = []
    with autocast_ctx:
        for col, _ in engine.generate(
            ids, num_samples=1, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        ):
            toks.append(col[0])
    stopped = len(toks) > 0 and toks[-1] == assistant_end
    body = toks[:-1] if stopped else toks
    ans = tokenizer.decode(body)
    print(f"\n--- [{row['book']}] ({row['chunk']})")
    print(f"Q: {row['question']}")
    print(f"MODEL{'' if stopped else ' (no stop)'}: {ans}")
    print(f"SOURCE (first 600 chars): {clean_chunk(row['passage'])[:600]}")
