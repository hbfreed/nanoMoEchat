"""Flavor-pairing eval: score the model's culinary *ideas*, not its recall.

Hand-seeded pairing matrix (~30 ingredients, classic affinities; tier 2 =
canonical pairings a chef would name first). For each probe ingredient the
model proposes pairings; we scan the answer for accepted pairings and score
tier * rarity (inverse frequency across the matrix, so naming butter/lemon
everywhere earns little). Admittedly ham-fisted — the matrix is "correct
pairings by fiat" — but it's verifiable, which is the point (RLVR candidate).

Run:
    python -m scripts.flavor_eval --source=dpo --model-tag=d20
"""
import argparse
import json
import math
import re
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

from tasks.flavor_matrix import ALIASES, MATRIX  # noqa: F401


def find(term, text):
    for a in ALIASES.get(term, [term]):
        if a in text:
            return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="rl")
parser.add_argument("--model-tag", type=str, default="d20")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--max-tokens", type=int, default=120)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, r, lr, ws, device = compute_init(device_type)
model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.step
)
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda" else nullcontext()
)
engine = Engine(model, tokenizer)
bos = tokenizer.get_bos_token_id()
us, ue = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
astart, aend = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# rarity: how many matrix entries list this pairing
df = {}
for pairings in MATRIX.values():
    for p in pairings:
        df[p] = df.get(p, 0) + 1
N = len(MATRIX)
idf = {p: math.log(N / c) for p, c in df.items()}

total, per_probe = 0.0, []
for probe, pairings in MATRIX.items():
    q = (f"Name five ingredients that pair well with {probe}. "
         "Answer with a short comma-separated list only.")
    ids = [bos, us] + tokenizer.encode(q) + [ue, astart]
    toks = []
    with autocast_ctx:
        for col, _ in engine.generate(ids, num_samples=1, max_tokens=args.max_tokens,
                                      temperature=args.temperature, top_k=50):
            toks.append(col[0])
    text = tokenizer.decode(toks[:-1] if toks and toks[-1] == aend else toks).lower()
    hits = [(p, t) for p, t in pairings.items() if find(p, text) and p != probe]
    score = sum(t * idf[p] for p, t in hits)
    best = sum(sorted((t * idf[p] for p, t in pairings.items()), reverse=True)[:5])
    per_probe.append((probe, score / best if best else 0.0, [p for p, _ in hits]))
    total += score / best if best else 0.0
    if args.verbose:
        print(f"[{probe}] {text[:160]}")
        print(f"   hits: {[p for p, _ in hits]}  score {score/best:.2f}")

per_probe.sort(key=lambda x: -x[1])
print(f"\n=== flavor sense ({args.source}, step {meta.get('step','?')}) ===")
for probe, s, hits in per_probe:
    print(f"{probe:>12}: {s:.2f}  {', '.join(hits) if hits else '-'}")
print(f"\nMEAN normalized score: {total/len(MATRIX):.3f}  (1.0 = named the top-5 canonical pairings)")
