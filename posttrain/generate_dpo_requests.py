"""Emit generation requests for the DPO/OPSD shared dataset.

Walks Cooking-Corpus chunks (holdouts excluded), samples usable ones, and
writes a JSONL of *generation requests*: each row carries the source passage
plus the exact instructions an LLM needs to produce (question, chosen,
rejected). Backend-agnostic on purpose -- pipe the rows through whatever model
Henry wants to spend tokens on, then join the outputs back on `id`.

The same rows serve OPSD for free: (question, passage) is the privileged-
teacher pair; only DPO needs chosen/rejected filled in.

Usage:
    python -m posttrain.generate_dpo_requests --n 2000 --out posttrain/dpo_requests.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re

CORPUS_DIR = "/home/henry/Documents/PythonProjects/Cooking-Corpus/data"
HOLDOUT_BOOKS = {"Amrikan", "The Noma Guide to Fermentation", "Guerrilla Tacos"}

_XML = re.compile(r"^<\?xml[^>]*\?>\s*$", re.MULTILINE)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")

GENERATION_INSTRUCTIONS = """\
You are building a preference dataset for a small cooking chat model.
Given SOURCE (a passage from a professional cookbook), produce JSON with:

1. "question": a natural user request answerable from SOURCE. Vary types:
   recipe requests, technique questions, why-does-this-work questions.
2. "chosen": the answer in professional register. Ingredients with real
   quantities first when it's a recipe; numbered method; concise; grounded in
   SOURCE (paraphrase -- do not copy long spans verbatim); no autobiography.
3. "rejected": the SAME dish/topic with the SAME core ingredients, but in
   food-blog register: 300+ words of personal story before any food, vague
   quantities ("a cup or so of butter"), engagement bait ("my picky eater
   DEVOURED these!!"), buried or incomplete method, missing salt quantities.
   Do NOT change the dish or swap main ingredients -- register and precision
   are the only differences.

Return strict JSON: {"question": ..., "chosen": ..., "rejected": ...}
"""


def iter_chunks():
    for book in sorted(os.listdir(CORPUS_DIR)):
        if book in HOLDOUT_BOOKS or not os.path.isdir(os.path.join(CORPUS_DIR, book)):
            continue
        for f in sorted(os.listdir(os.path.join(CORPUS_DIR, book))):
            if f.endswith(".md"):
                with open(os.path.join(CORPUS_DIR, book, f), encoding="utf-8") as fh:
                    text = _IMG.sub("", _XML.sub("", fh.read())).strip()
                yield book, f, text


def usable(text):
    # Long enough to ground an answer; skip front matter and index-like chunks.
    words = len(text.split())
    return 150 <= words and not text.lower().startswith(("copyright", "index", "contents"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="posttrain/dpo_requests.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    chunks = [(b, f, t) for b, f, t in iter_chunks() if usable(t)]
    random.Random(args.seed).shuffle(chunks)
    chunks = chunks[: args.n]

    with open(args.out, "w", encoding="utf-8") as out:
        for book, fname, text in chunks:
            rid = hashlib.md5(f"{book}/{fname}".encode()).hexdigest()[:12]
            out.write(json.dumps({
                "id": rid,
                "book": book,
                "chunk": fname,
                "passage": text,
                "instructions": GENERATION_INSTRUCTIONS,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} generation requests to {args.out}")
    by_book = {}
    for b, _, _ in chunks:
        by_book[b] = by_book.get(b, 0) + 1
    print(f"coverage: {len(by_book)} books; top: "
          + ", ".join(f"{b} ({n})" for b, n in sorted(by_book.items(), key=lambda kv: -kv[1])[:5]))


if __name__ == "__main__":
    main()
