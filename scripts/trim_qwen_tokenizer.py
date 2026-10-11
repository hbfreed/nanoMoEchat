"""
Build a trimmed Qwen tokenizer: the Qwen3.5/3.6 BPE restricted to the tokens our
data actually uses, so the output layer costs what our own 65k vocab did while
Qwen models stay usable as same-vocab logit-distillation teachers.

Recipe:
1. count Qwen token frequencies per source (web pretraining text, cookbooks,
   cooking SFT conversations) and score each token by a weighted mix of its
   per-source frequencies, so the small cooking sources are not drowned out
2. add tokens in score order, each with every token on its BPE merge path (a
   kept token must stay buildable), until base vocab + special tokens is exactly
   --vocab-size; all 256 byte tokens are always kept, so any text still encodes
3. write a tokenizer.json with the kept tokens renumbered 0..N-1 (original order),
   only the merges among kept tokens, and the special tokens after them, plus
   teacher_ids.json: student id -> Qwen id

Words whose Qwen tokens were dropped come out as a few smaller kept tokens.
Every student token is a Qwen token, so a teacher reads the student's sequence
directly and its distribution is restricted to teacher_ids and renormalized.

Held-out coverage per source is reported: web val shard, the held-out cookbooks
(never counted), and 10% of the SFT conversations.

Run with:
python -m scripts.trim_qwen_tokenizer --vocab-size=49152
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, pre_tokenizers

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files
from scripts.build_cookbook_mix import HOLDOUT_BOOKS, load_books

parser = argparse.ArgumentParser(description="Trim the Qwen tokenizer to our data")
parser.add_argument("--vocab-size", type=int, default=49152, help="final size including special tokens")
parser.add_argument("--source", type=str, default="Qwen/Qwen3.5-0.8B",
                    help="HF repo whose tokenizer.json to trim (instruct: has the chat/think tokens)")
parser.add_argument("--web-bytes", type=float, default=3e8, help="web text bytes to count, over 32 shards")
parser.add_argument("--weights", type=str, default="web=0.7,cookbooks=0.15,sft=0.15")
parser.add_argument("--out", type=str, default="", help="default: <base_dir>/tokenizer_qwen<vocab-size>")
args = parser.parse_args()
weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
out_dir = args.out or os.path.join(get_base_dir(), f"tokenizer_qwen{args.vocab_size}")

src_path = hf_hub_download(args.source, "tokenizer.json")
tj = json.load(open(src_path, encoding="utf-8"))
full = Tokenizer.from_file(src_path)
vocab = tj["model"]["vocab"]  # base (non-special) tokens: str -> id
merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in tj["model"]["merges"]]
specials = tj["added_tokens"]
V = len(vocab) + len(specials)
assert all(s["id"] >= len(vocab) for s in specials)
print(f"source {args.source}: {len(vocab):,} base tokens, {len(merges):,} merges, {len(specials)} special")

# every merge that produces a token; a token may be producible by more than one pair
producers = {}
for a, b in merges:
    producers.setdefault(vocab[a + b], []).append((vocab[a], vocab[b]))


def encode(texts):
    return [e.ids for e in full.encode_batch(texts, add_special_tokens=False)]


def counts_of(texts):
    c = np.zeros(V, dtype=np.int64)
    for ids in encode(texts):
        c += np.bincount(ids, minlength=V)
    return c


def read_web(paths, budget):
    texts, n = [], 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                texts.append(t)
                n += len(t.encode("utf-8"))
                if n >= budget:
                    return texts
    return texts


def render_chat(messages):
    """Qwen chat format, so formatting tokens around the turns are counted too."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


# ---- sources: (count texts, held-out texts) ----
t0 = time.time()
parquets = list_parquet_files()
train_paths = parquets[:-1]
web_count = []
for path in train_paths[:: max(1, len(train_paths) // 32)][:32]:
    web_count += read_web([path], args.web_bytes / 32)
web_val = read_web(parquets[-1:], 2e7)

books = load_books()
book_count = [t for b, t in books.items() if b not in HOLDOUT_BOOKS]
book_val = [t for b, t in books.items() if b in HOLDOUT_BOOKS]

from tasks.cooking import CookingSFT

sft = [render_chat(c) for c in CookingSFT().conversations]
random.Random(0).shuffle(sft)
n_val = len(sft) // 10
sft_val, sft_count = sft[:n_val], sft[n_val:]

sources = {"web": (web_count, web_val), "cookbooks": (book_count, book_val), "sft": (sft_count, sft_val)}
assert set(weights) == set(sources), f"weights must name {sorted(sources)}"
for name, (cnt, val) in sources.items():
    print(f"  {name:9s}: count on {sum(map(len, cnt))/1e6:6.1f}M chars ({len(cnt):,} docs), "
          f"held out {sum(map(len, val))/1e6:5.1f}M chars ({len(val):,} docs)")

score = np.zeros(V)
val_counts = {}
for name, (cnt, val) in sources.items():
    c = counts_of(cnt)
    score += weights[name] * c / c.sum()
    val_counts[name] = counts_of(val)
print(f"counted in {time.time()-t0:.0f}s")

# ---- choose the kept set ----
byte_ids = [vocab[ch] for ch in pre_tokenizers.ByteLevel.alphabet()]
assert len(byte_ids) == 256
budget = args.vocab_size - len(specials)
keep = set(byte_ids)
order = np.argsort(-score[: len(vocab)], kind="stable")


def add_with_closure(t):
    """Add t and every token on any merge path building it; False if over budget."""
    new, stack = {t}, [t]
    while stack:
        for a, b in producers.get(stack.pop(), ()):
            for p in (a, b):
                if p not in keep and p not in new:
                    new.add(p)
                    stack.append(p)
    new -= keep
    if len(keep) + len(new) > budget:
        return False
    keep.update(new)
    return True


for t in order:
    if len(keep) >= budget or score[t] == 0:
        break
    if t not in keep:
        add_with_closure(int(t))
# fill any remaining slots with single tokens that need no new parents
for t in order:
    if len(keep) >= budget:
        break
    if t not in keep and all(a in keep and b in keep for a, b in producers.get(int(t), [])):
        keep.add(int(t))
assert len(keep) == budget, (len(keep), budget)

# ---- write the trimmed tokenizer ----
kept_base = sorted(keep)
new_id = {old: i for i, old in enumerate(kept_base)}
id_to_str = {i: s for s, i in vocab.items()}
tj["model"]["vocab"] = {id_to_str[old]: new_id[old] for old in kept_base}
tj["model"]["merges"] = [[a, b] for a, b in merges
                         if vocab[a] in keep and vocab[b] in keep and vocab[a + b] in keep]
teacher_ids = list(kept_base)
for i, s in enumerate(specials):
    teacher_ids.append(s["id"])
    s["id"] = len(kept_base) + i
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
    json.dump(tj, f, ensure_ascii=False)
with open(os.path.join(out_dir, "teacher_ids.json"), "w") as f:
    json.dump({"source": args.source, "teacher_ids": teacher_ids}, f)
for name in ("tokenizer_config.json", "chat_template.jinja"):
    try:
        cfg_path = hf_hub_download(args.source, name)
    except Exception:
        continue
    with open(cfg_path, encoding="utf-8") as f:
        content = f.read()
    if name == "tokenizer_config.json":  # drop the id-keyed special token table; tokenizer.json has it
        cfg = json.loads(content)
        cfg.pop("added_tokens_decoder", None)
        # tokens named only here would be appended past --vocab-size; keep those tokenizer.json has
        named = {s["content"] for s in specials}
        cfg["extra_special_tokens"] = {k: v for k, v in cfg.get("extra_special_tokens", {}).items() if v in named}
        cfg["additional_special_tokens"] = [v for v in cfg.get("additional_special_tokens", []) if v in named]
        content = json.dumps(cfg, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write(content)

# ---- verify on held-out text ----
trimmed = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
assert trimmed.get_vocab_size() == args.vocab_size, trimmed.get_vocab_size()
from transformers import AutoTokenizer
assert len(AutoTokenizer.from_pretrained(out_dir)) == args.vocab_size, "tokenizer_config adds tokens"
t_ids = np.array(teacher_ids)
stats = {"vocab_size": args.vocab_size, "base_tokens": len(kept_base),
         "merges": len(tj["model"]["merges"]), "weights": weights, "sources": {}}
print(f"\nwrote {out_dir}: {len(kept_base):,} base + {len(specials)} special tokens, "
      f"{len(tj['model']['merges']):,} merges")
print("held-out text, trimmed vs full Qwen tokenizer:")
for name, (_, val) in sources.items():
    vc = val_counts[name]
    out_frac = 1 - vc[t_ids].sum() / vc.sum()
    full_ids = encode(val)
    trim_ids = [e.ids for e in trimmed.encode_batch(val, add_special_tokens=False)]
    n_full, n_trim = sum(map(len, full_ids)), sum(map(len, trim_ids))
    roundtrip = all(trimmed.decode(ids, skip_special_tokens=False) == full.decode(f, skip_special_tokens=False)
                    for ids, f in zip(trim_ids[:500], full_ids[:500]))
    # documents whose full tokenization is all-kept must tokenize identically
    kept_mask = np.zeros(V, dtype=bool)
    kept_mask[t_ids] = True
    same = [list(t_ids[np.array(t, dtype=np.int64)]) == f
            for t, f in zip(trim_ids, full_ids) if kept_mask[f].all()]
    s = {"tokens_outside_kept": out_frac, "token_inflation": n_trim / n_full - 1,
         "identical_when_all_kept": float(np.mean(same)) if same else None, "roundtrip_ok": roundtrip}
    stats["sources"][name] = s
    print(f"  {name:9s}: {100*out_frac:.3f}% of Qwen tokens dropped -> "
          f"{100*s['token_inflation']:+.3f}% tokens | identical where all kept: "
          f"{'n/a' if not same else f'{100*np.mean(same):.2f}%'} ({len(same)} docs) | decode roundtrip {'ok' if roundtrip else 'FAIL'}")
with open(os.path.join(out_dir, "trim_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
