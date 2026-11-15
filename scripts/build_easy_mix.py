"""Easy distillation prompts beyond cooking: GSM8K train questions and short SmolTalk chat.

- gsm8k: the train split's word problems (the shape of the GSM8K chat eval). No teacher
  reference: Qwen3.5-4B already gets ~93% (non-thinking), so it grades a student's path well.
- chat: first user turns from HuggingFaceTB/smol-smoltalk's short/simple sources (everyday
  chat, simple constraints, rewrites), passed through nanochat.prompt_filter (keywords + judge).
Prompts longer than --max-prompt-tokens are dropped. Writes JSONL {source, prompt}.

Run (teacher server up on :8011, for the judge):
    python -m scripts.build_easy_mix --out posttrain/out/opd_easy_mix.jsonl --gsm8k 900 --chat 900
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

from nanochat.prompt_filter import BLOCK, judge
from nanochat.tokenizer import get_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--gsm8k", type=int, default=900)
ap.add_argument("--chat", type=int, default=900)
ap.add_argument("--max-prompt-tokens", type=int, default=256)
ap.add_argument("--tokenizer", default="tokenizer_qwen49152")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

CHAT_SOURCES = {"smol-magpie-ultra-short", "everyday-conversations", "smol-contraints", "smollm-rewrite-30k"}
tok = get_tokenizer(args.tokenizer)
dummy = {"role": "assistant", "content": ""}
fits = lambda p: len(tok.render_for_completion({"messages": [{"role": "user", "content": p}, dummy]})) <= args.max_prompt_tokens
rng = random.Random(args.seed)

gsm = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
rng.shuffle(gsm)
out = [{"source": "gsm8k", "prompt": q} for q in gsm if fits(q)][: args.gsm8k]

chat = []
for r in load_dataset("HuggingFaceTB/smol-smoltalk", split="train"):
    if r["source"] in CHAT_SOURCES and r["messages"] and r["messages"][0]["role"] == "user":
        chat.append(r["messages"][0]["content"].strip())
rng.shuffle(chat)
cand = [p for p in chat[: args.chat * 3] if not BLOCK.search(p) and fits(p)][: int(args.chat * 1.4)]
with ThreadPoolExecutor(32) as ex:
    safe = [p for p, ok in zip(cand, ex.map(judge, cand)) if ok]
out += [{"source": "chat", "prompt": p} for p in safe[: args.chat]]
print(f"gsm8k {sum(r['source'] == 'gsm8k' for r in out)}, chat {len(safe[:args.chat])} "
      f"({len(cand) - len(safe)} of {len(cand)} chat candidates judged unsafe)")

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} to {args.out}")
