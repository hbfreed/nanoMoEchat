"""Round-2 generation: holdout eval quiz + multi-turn SFT extensions.

Emits a second Mistral batch input file with two request kinds:

- eval:  questions + reference answers from the HOLDOUT books' chunks.
  Eval-only artifact -- these books are never trained on in any phase; the
  quiz measures closed-book generalization to never-seen sources.
- multiturn: takes a round-1 SFT pair (question, answer) plus its source
  passage and generates one realistic follow-up exchange (substitution
  requests, failure diagnosis, technique clarification).

Usage:
    python -m posttrain.generate_round2 --out posttrain/out/batch_input_r2.jsonl
    python -m posttrain.run_generation_batch submit   # after pointing it at r2
"""

import argparse
import hashlib
import json
import os
import random

from posttrain.generate_dpo_requests import CORPUS_DIR, HOLDOUT_BOOKS, _IMG, _XML, usable

MODEL = "zai-glm-5-2"
HERE = os.path.dirname(os.path.abspath(__file__))

EVAL_INSTRUCTIONS = """\
You are writing an evaluation quiz for a small cooking chat model. Given
SOURCE (a passage from a cookbook the model has NEVER seen), produce JSON:

1. "question": a closed-book cooking question a curious home cook might ask,
   answerable by someone who genuinely understands cooking -- NOT a trivia
   question about this specific book, recipe name, or author. Test transferable
   knowledge the passage demonstrates (technique, ratios, why-it-works).
2. "reference": a correct, concise answer grounded in SOURCE.
3. "rubric": 2-4 bullet points a grader should check in a model's answer.

Return strict JSON: {"question": ..., "reference": ..., "rubric": [...]}
"""

MULTITURN_INSTRUCTIONS = """\
You are extending a cooking Q&A into a realistic two-turn conversation.
Given SOURCE, plus FIRST_QUESTION and FIRST_ANSWER, produce JSON:

1. "followup": a natural user follow-up (substitution request, failure
   diagnosis like "mine came out gummy", scaling, or technique clarification).
2. "followup_answer": a professional-register answer, grounded in SOURCE
   where possible, honest about uncertainty where not.

Return strict JSON: {"followup": ..., "followup_answer": ...}
"""


def holdout_chunks():
    for book in sorted(HOLDOUT_BOOKS):
        book_dir = os.path.join(CORPUS_DIR, book)
        for f in sorted(os.listdir(book_dir)):
            if f.endswith(".md"):
                with open(os.path.join(book_dir, f), encoding="utf-8") as fh:
                    text = _IMG.sub("", _XML.sub("", fh.read())).strip()
                if usable(text):
                    yield book, f, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out", "batch_input_r2.jsonl"))
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-multiturn", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    hc = list(holdout_chunks())
    rng.shuffle(hc)
    for book, fname, text in hc[: args.n_eval]:
        rid = "eval-" + hashlib.md5(f"{book}/{fname}".encode()).hexdigest()[:10]
        rows.append({
            "custom_id": rid,
            "body": {"model": MODEL, "max_tokens": 1200, "temperature": 0.7,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "user", "content":
                                   EVAL_INSTRUCTIONS + f"\n\nSOURCE:\n{text}"}]},
        })

    sft = [json.loads(l) for l in open(os.path.join(HERE, "out", "sft_pairs.jsonl"), encoding="utf-8")]
    meta = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(HERE, "generation_requests.jsonl"), encoding="utf-8")}
    passages = {(m["book"], m["chunk"]): m["passage"] for m in meta.values()}
    rng.shuffle(sft)
    for row in sft[: args.n_multiturn]:
        passage = passages.get((row["book"], row["chunk"]), "")
        rid = "mt-" + hashlib.md5((row["book"] + row["question"]).encode()).hexdigest()[:10]
        content = (MULTITURN_INSTRUCTIONS
                   + f"\n\nSOURCE:\n{passage[:4000]}"
                   + f"\n\nFIRST_QUESTION: {row['question']}"
                   + f"\n\nFIRST_ANSWER: {row['answer']}")
        rows.append({
            "custom_id": rid,
            "body": {"model": MODEL, "max_tokens": 1200, "temperature": 0.8,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "user", "content": content}]},
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_eval = sum(1 for r in rows if r["custom_id"].startswith("eval-"))
    print(f"wrote {len(rows)} requests ({n_eval} eval, {len(rows)-n_eval} multiturn) -> {args.out}")


if __name__ == "__main__":
    main()
