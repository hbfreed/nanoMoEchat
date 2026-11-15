"""Easy home-cooking questions from the cookbook corpus, for on-policy distillation.

The OPSD questions (posttrain/out/opsd_pairs.jsonl) are book-specific chef recipes that a small
student cannot answer even roughly. This asks the Qwen teacher (vLLM) to read each of the same
OPSD-partition chunks (so nothing overlaps the SFT/DPO chunks; holdout books are already
excluded) and write one simple, everyday question a home cook might ask that the excerpt helps
answer, plus a short correct answer grounded in it. In distillation the student sees only the
question; the teacher also sees the excerpt and the reference answer.

Writes JSONL rows {source: "cooking", prompt, teacher_context, book, chunk}.

Run (teacher server up on :8011):
    python -m scripts.build_easy_cooking --out posttrain/out/opd_easy_cooking.jsonl
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from tasks.cooking import clean_chunk

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="posttrain/out/opsd_pairs.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--max-passage-chars", type=int, default=4000)
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--url", default="http://localhost:8011")
args = ap.parse_args()

PROMPT = """Here is an excerpt from the cookbook "{book}":

<<<
{passage}
>>>

Write ONE simple, everyday question that a home cook might ask and that this excerpt helps answer.
Good questions are short and general: a basic technique, why something happens, a substitution,
timing or temperature, storage, a common mistake. Do not ask about a named restaurant dish, the
book, or the author, and do not mention "the excerpt".

Then write a short, correct, practical answer (3 to 6 sentences, or a short numbered list), using
the excerpt for facts and quantities. Plain text, no markdown headings.

Reply with JSON only: {{"question": "...", "answer": "..."}}"""


def make(row):
    passage = clean_chunk(row["passage"])[: args.max_passage_chars]
    for attempt in range(3):
        try:
            r = requests.post(f"{args.url}/v1/chat/completions", timeout=300, json={
                "model": args.model, "max_tokens": 600, "temperature": 0.7,
                "messages": [{"role": "user", "content": PROMPT.format(book=row["book"], passage=passage)}],
                "chat_template_kwargs": {"enable_thinking": False}})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            q, a = obj["question"].strip(), obj["answer"].strip()
            if 10 < len(q) < 300 and 40 < len(a) < 2500:
                ctx = (f"Reference excerpt (from \"{row['book']}\"):\n{passage}\n\n"
                       f"A good answer:\n{a}")
                return {"source": "cooking", "prompt": q, "teacher_context": ctx,
                        "book": row["book"], "chunk": row.get("chunk")}
        except (requests.RequestException, AttributeError, KeyError, ValueError, TypeError):
            pass
    return None


rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
with ThreadPoolExecutor(32) as ex:
    made = [m for m in ex.map(make, rows) if m]
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for m in made:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(f"wrote {len(made)} of {len(rows)} to {args.out}")
