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

# ingredient -> {pairing: tier}. Tier 2 = canonical.
MATRIX = {
    "rhubarb": {"strawberry": 2, "ginger": 2, "vanilla": 1, "orange": 1, "cream": 1, "cinnamon": 1, "almond": 1},
    "beet": {"goat cheese": 2, "orange": 1, "walnut": 1, "dill": 1, "horseradish": 1, "apple": 1, "yogurt": 1},
    "chocolate": {"hazelnut": 2, "orange": 2, "coffee": 2, "raspberry": 1, "caramel": 1, "chili": 1, "mint": 1, "peanut": 1},
    "lamb": {"rosemary": 2, "garlic": 2, "mint": 2, "yogurt": 1, "cumin": 1, "apricot": 1, "eggplant": 1},
    "pork": {"apple": 2, "sage": 2, "fennel": 1, "mustard": 1, "cabbage": 1, "maple": 1, "prune": 1},
    "duck": {"orange": 2, "cherry": 2, "five-spice": 1, "honey": 1, "fig": 1, "turnip": 1},
    "salmon": {"dill": 2, "lemon": 2, "cucumber": 1, "soy": 1, "miso": 1, "horseradish": 1, "caper": 1},
    "scallop": {"brown butter": 2, "bacon": 1, "cauliflower": 1, "lemon": 1, "pea": 1, "corn": 1},
    "shrimp": {"garlic": 2, "chili": 1, "lime": 1, "cilantro": 1, "butter": 1, "feta": 1, "tomato": 1},
    "tomato": {"basil": 2, "mozzarella": 2, "olive oil": 1, "garlic": 1, "balsamic": 1, "anchovy": 1, "watermelon": 1},
    "mushroom": {"thyme": 2, "garlic": 2, "cream": 1, "sherry": 1, "parmesan": 1, "egg": 1, "soy": 1},
    "potato": {"butter": 2, "leek": 2, "rosemary": 1, "cream": 1, "cheese": 1, "bacon": 1, "garlic": 1},
    "cauliflower": {"curry": 2, "brown butter": 1, "caper": 1, "raisin": 1, "parmesan": 1, "anchovy": 1},
    "carrot": {"cumin": 2, "orange": 1, "ginger": 1, "honey": 1, "coriander": 1, "yogurt": 1, "dill": 1},
    "corn": {"butter": 2, "lime": 1, "chili": 1, "cheese": 1, "basil": 1, "bacon": 1},
    "peach": {"raspberry": 2, "basil": 1, "bourbon": 1, "cream": 1, "almond": 1, "prosciutto": 1, "honey": 1},
    "apple": {"cinnamon": 2, "caramel": 2, "pork": 1, "cheddar": 1, "walnut": 1, "sage": 1},
    "pear": {"blue cheese": 2, "walnut": 2, "honey": 1, "vanilla": 1, "ginger": 1, "chocolate": 1},
    "banana": {"chocolate": 2, "peanut": 2, "caramel": 1, "rum": 1, "coconut": 1},
    "strawberry": {"cream": 2, "balsamic": 2, "basil": 1, "rhubarb": 1, "chocolate": 1, "black pepper": 1},
    "fig": {"prosciutto": 2, "honey": 2, "goat cheese": 1, "walnut": 1, "balsamic": 1},
    "melon": {"prosciutto": 2, "mint": 1, "lime": 1, "feta": 1, "chili": 1},
    "cucumber": {"dill": 2, "yogurt": 2, "mint": 1, "vinegar": 1, "feta": 1, "gin": 1},
    "avocado": {"lime": 2, "cilantro": 2, "chili": 1, "tomato": 1, "corn": 1, "sesame": 1},
    "egg": {"bacon": 2, "chive": 2, "truffle": 1, "cheese": 1, "asparagus": 1, "hot sauce": 1},
    "blue cheese": {"walnut": 2, "honey": 2, "pear": 1, "celery": 1, "beef": 1, "port": 1},
    "goat cheese": {"beet": 2, "honey": 2, "thyme": 1, "fig": 1, "arugula": 1},
    "ginger": {"garlic": 2, "soy": 2, "lime": 1, "honey": 1, "scallion": 1, "carrot": 1},
    "coffee": {"chocolate": 2, "cream": 2, "cardamom": 1, "caramel": 1, "hazelnut": 1},
}

ALIASES = {
    "cream": ["cream", "creme", "custard"],
    "soy": ["soy"],
    "chili": ["chili", "chile", "chilli", "pepper flake", "cayenne", "jalape"],
    "cheese": ["cheese"],
    "olive oil": ["olive oil"],
    "black pepper": ["black pepper", "cracked pepper"],
    "brown butter": ["brown butter", "browned butter", "beurre noisette"],
    "hot sauce": ["hot sauce", "tabasco", "sriracha"],
    "five-spice": ["five spice", "five-spice"],
    "prune": ["prune"],
    "caper": ["caper"],
    "walnut": ["walnut"],
    "peanut": ["peanut"],
    "bourbon": ["bourbon", "whiskey"],
}


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
