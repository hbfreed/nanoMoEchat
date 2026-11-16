"""Generate off-policy distillation data: teacher-written answers with top-k teacher log-probs.

The teacher is the vocab-trimmed Qwen (scripts/trim_qwen_teacher.py) served by vLLM, so prompt
and output token ids are ours and the returned log-probs are the full model's distribution
restricted to our vocabulary. Log-probs are vLLM's raw ones (before temperature), so they are
soft labels from the teacher's real distribution even though answers are sampled at --temperature.

Prompt sources (JSONL rows {source, prompt[, teacher_context]}):
  - cooking: easy home-cook questions; the teacher also sees the book excerpt and a reference
    answer (teacher_context), the student only the question
  - gsm8k: train questions; --gsm8k-format-frac of them carry a format instruction (#### / boxed /
    Answer: / number only). Kept only if the final answer is correct and, when instructed, the
    format is followed
  - chat, ifeval: as is
The teacher also gets a hidden system prompt (concise; never mention reference material). Answers
are dropped if they hit max tokens (the student should learn to stop), are mostly non-ASCII,
repeat a line 3+ times, or (cooking) mention the excerpt the student never sees; each prompt gets
--tries samples to pass.

Writes a torch file: list of dicts {source, prompt, fmt, ids: int32[T] (answer + <|im_end|>),
topk_ids: int32[T,K], topk_lp: float16[T,K]}.

Run (trimmed teacher up on :8011):
    python -m scripts.gen_kd_data --out /media/henry/MoreFiles/kd/kd_easy_v1.pt
"""

import argparse
import collections
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import _norm, extract_answer, extract_final_answer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="posttrain/out/opd_easy_cooking.jsonl,posttrain/out/opd_easy_mix.jsonl,"
                "posttrain/out/opd_prompts_clean.jsonl")
ap.add_argument("--sources", default="cooking,gsm8k,chat,ifeval")
ap.add_argument("--max-prompt-tokens", type=int, default=512)
ap.add_argument("--max-tokens", type=int, default=768)
ap.add_argument("--tries", type=int, default=2, help="samples per prompt until one passes the filters")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.8)
ap.add_argument("--top-k-lp", type=int, default=20)
ap.add_argument("--gsm8k-format-frac", type=float, default=0.35)
ap.add_argument("--url", default="http://localhost:8011")
ap.add_argument("--tokenizer", default="tokenizer_qwen49152")
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--limit", type=int, default=0, help="debug: only this many prompts")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

FORMATS = {  # same instructions and strict checks as chat_opsd
    "hash": ("End with a final line in the form '#### <number>'.",
             lambda t: re.search(r"(^|\n)\s*####\s*-?\$?\d[\d,]*(\.\d+)?\s*$", t.strip()) is not None),
    "boxed": ("Put your final answer in \\boxed{}.",
              lambda t: re.search(r"\\boxed\{\s*-?\$?\d[\d,]*(\.\d+)?\s*\}", t) is not None),
    "answer": ("Finish with a line of the form 'Answer: <number>'.",
               lambda t: re.search(r"(^|\n)\s*Answer:\s*-?\$?\d[\d,]*(\.\d+)?\s*\.?\s*$", t.strip()) is not None),
    "only": ("Reply with only the final number, nothing else.",
             lambda t: re.fullmatch(r"\s*-?\$?\d[\d,]*(\.\d+)?\s*", t) is not None),
}
TEACHER_SYSTEM = ("Answer helpfully, directly and concisely; prefer short answers unless the question needs "
                  "detail. Any reference material in the user's message is private background: never mention "
                  "it, an excerpt, a book or an author; answer as if from your own knowledge.")
LEAK = re.compile(r"\b(excerpt|the passage|the reference|the text above|the book|according to the)\b", re.I)

tok = get_tokenizer(args.tokenizer)
model = requests.get(args.url + "/v1/models").json()["data"][0]["id"]
think_header = tok.encode("<think>\n\n</think>\n\n")
im_end = tok.encode_special("<|im_end|>")
dummy = {"role": "assistant", "content": ""}
render = lambda content, sys=None: tok.render_for_completion({"messages": ([{"role": "system", "content": sys}] if sys else [])
                                                                + [{"role": "user", "content": content}, dummy]})
gold_by_q = None
rng = random.Random(args.seed)
want = set(args.sources.split(","))

rows = []
for path in args.prompts.split(","):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["source"] not in want:
            continue
        if r["source"] == "gsm8k":
            if gold_by_q is None:
                from datasets import load_dataset
                gold_by_q = {x["question"]: _norm(extract_answer(x["answer"]))
                             for x in load_dataset("openai/gsm8k", "main", split="train")}
            r["gold"] = gold_by_q[r["prompt"]]
            r["fmt"] = rng.choice(sorted(FORMATS)) if rng.random() < args.gsm8k_format_frac else None
            if r["fmt"]:
                r["prompt"] = r["prompt"] + "\n\n" + FORMATS[r["fmt"]][0]
        sp = render(r["prompt"])
        if len(sp) > args.max_prompt_tokens:
            continue
        ctx = r.get("teacher_context")
        tp = render(r["prompt"] + "\n\n" + ctx if ctx else r["prompt"], TEACHER_SYSTEM)
        r["teacher_prompt"] = tp[1:] + think_header  # Qwen chat has no leading <|endoftext|>
        rows.append(r)
rng.shuffle(rows)
if args.limit:
    rows = rows[: args.limit]
print("prompts: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(r["source"] for r in rows).items()), flush=True)


def parse_id(s):
    assert s.startswith("token_id:"), s
    return int(s[len("token_id:"):])


def gen(r):
    reason = None
    for _ in range(args.tries):
        res, reason = gen1(r)
        if res:
            return res, reason
    return None, reason


def gen1(r):
    body = {"model": model, "prompt": r["teacher_prompt"], "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "logprobs": args.top_k_lp,
            "return_tokens_as_token_ids": True, "skip_special_tokens": False}
    for attempt in range(6):
        try:
            resp = requests.post(args.url + "/v1/completions", json=body, timeout=600)
            resp.raise_for_status()
            c = resp.json()["choices"][0]
            break
        except requests.RequestException as e:
            if attempt == 5:
                return None, f"error {e}"
            time.sleep(5 * 2 ** attempt)
    if c["finish_reason"] != "stop":
        return None, "truncated"
    lp = c["logprobs"]
    ids = [parse_id(t) for t in lp["tokens"]]
    if not ids or ids[-1] != im_end:
        return None, "no im_end"
    text = tok.decode(ids[:-1]).strip()
    if sum(not ch.isascii() for ch in text) > 0.05 * max(1, len(text)):
        return None, "non-english"
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    if lines and max(collections.Counter(lines).values()) >= 3:
        return None, "repetitive"
    if r["source"] == "cooking" and LEAK.search(text):
        return None, "mentions excerpt"
    if r["source"] == "gsm8k":
        if _norm(extract_final_answer(text) or "") != r["gold"]:
            return None, "gsm8k wrong"
        if r["fmt"] and not FORMATS[r["fmt"]][1](text):
            return None, "format not followed"
    K = args.top_k_lp
    topk_ids = torch.zeros(len(ids), K, dtype=torch.int32)
    topk_lp = torch.full((len(ids), K), -1e4, dtype=torch.float16)
    for t, d in enumerate(lp["top_logprobs"]):
        items = sorted(((parse_id(k), v) for k, v in d.items()), key=lambda kv: -kv[1])[:K]
        for j, (i, v) in enumerate(items):
            topk_ids[t, j], topk_lp[t, j] = i, max(v, -1e4)
    return {"source": r["source"], "prompt": r["prompt"], "fmt": r.get("fmt"),
            "ids": torch.tensor(ids, dtype=torch.int32), "topk_ids": topk_ids, "topk_lp": topk_lp}, "ok"


t0 = time.time()
out, why = [], collections.Counter()
with ThreadPoolExecutor(args.workers) as ex:
    for n, (r, (res, reason)) in enumerate(zip(rows, ex.map(gen, rows)), 1):
        why[(r["source"], reason)] += 1
        if res:
            out.append(res)
        if n % 250 == 0:
            print(f"{n}/{len(rows)} kept {len(out)} ({time.time() - t0:.0f}s)", flush=True)

ntok = sum(len(o["ids"]) for o in out)
print(f"kept {len(out)} of {len(rows)}, {ntok} answer tokens, {time.time() - t0:.0f}s", flush=True)
for (src, reason), v in sorted(why.items()):
    print(f"  {src:8s} {reason:20s} {v}")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(out, args.out)
print(f"wrote {args.out}")
