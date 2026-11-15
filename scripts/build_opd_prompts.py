"""Build a clean on-policy distillation prompt set from allenai/Dolci-Instruct-RL.

Dolci's general and instruction-following prompts include sexual role-play and similar
material that we do not want in a cooking model. Two filters, both must pass:
  1. a keyword blocklist (cheap, catches the obvious cases), then
  2. a Qwen judge served by vLLM (the same server the distillation teacher uses), asked for
     SAFE/UNSAFE in non-thinking mode.
Only single-turn prompts that fit --max-prompt-tokens are kept. Writes JSONL rows
{"source": category, "prompt": text, "dataset_source": ...} for chat_opsd --prompts-file.

Run (teacher server up on :8011):
    python -m scripts.build_opd_prompts --out posttrain/out/opd_prompts_clean.jsonl \\
        --counts general:1300,ifeval:850,math:150
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from nanochat.prompt_filter import BLOCK, judge as _judge
from nanochat.tokenizer import get_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--counts", default="general:1300,ifeval:850,math:150",
                help="prompts to keep per category (general, ifeval, math, code)")
ap.add_argument("--max-prompt-tokens", type=int, default=512)
ap.add_argument("--judge", default="Qwen/Qwen3.5-4B")
ap.add_argument("--judge-url", default="http://localhost:8011")
ap.add_argument("--tokenizer", default="tokenizer_qwen49152")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

CATS = {"general": {"general-quality_ref", "general-quality"}, "ifeval": {"ifeval"},
        "math": {"math"}, "code": {"code"}}
counts = {k: int(v) for k, v in (kv.split(":") for kv in args.counts.split(","))}
assert set(counts) <= set(CATS)

judge = lambda p: _judge(p, args.judge_url, args.judge)
tok = get_tokenizer(args.tokenizer)
dummy = {"role": "assistant", "content": ""}
pool = {k: [] for k in counts}
for i in range(3):
    path = hf_hub_download("allenai/Dolci-Instruct-RL", f"data/train-0000{i}-of-00003.parquet", repo_type="dataset")
    for r in pq.read_table(path, columns=["dataset", "prompt", "dataset_source"]).to_pylist():
        d = r["dataset"][0] if isinstance(r["dataset"], list) else r["dataset"]
        text = r["prompt"]
        if not text.startswith("user: ") or "\nassistant:" in text:
            continue  # single-turn only
        for k in counts:
            if d in CATS[k]:
                pool[k].append({"source": k, "prompt": text[len("user: "):], "dataset_source": r["dataset_source"]})

rng = random.Random(args.seed)
kept, stats = [], {}
with ThreadPoolExecutor(32) as ex:
    for k, want in counts.items():
        rng.shuffle(pool[k])
        cand = []
        for row in pool[k]:
            if len(cand) >= int(want * 1.6) + 50:
                break
            if BLOCK.search(row["prompt"]):
                continue
            sp = tok.render_for_completion({"messages": [{"role": "user", "content": row["prompt"]}, dummy]})
            if len(sp) <= args.max_prompt_tokens:
                cand.append(row)
        verdicts = list(ex.map(judge, [c["prompt"] for c in cand]))
        safe = [c for c, ok in zip(cand, verdicts) if ok]
        stats[k] = (len(cand), len(safe))
        kept += safe[:want]
        print(f"{k}: {len(cand)} candidates after keywords/length, {len(safe)} judged safe, kept {len(safe[:want])}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(kept)} prompts to {args.out}")
