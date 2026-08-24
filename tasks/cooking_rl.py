"""RLVR task for the cook: verifiable gates x (flavor coherence + pairwise judge).

Reward design (see surya.website/rling-qwen-to-paint-with-code for the shape):
  gates (multiplicative): stopped cleanly, non-degenerate (distinct-4 ratio),
    and for recipe-shaped prompts a parseable quantity list + numbered method.
  flavor (30%): idf-weighted co-occurrence of mentioned ingredients against
    the hand-seeded pairing matrix (train half only; held-out half is eval).
  judge (70%): group RANKING vs the curated DPO `chosen` answer for the SAME
    prompt (one call ranks the whole GRPO group + unmarked reference),
    order shuffled. Comparative, not absolute, on purpose.

Prompts come from posttrain/out/dpo_pairs.jsonl so every prompt has a
reference. Judge endpoint via COOK_JUDGE_URL (default localhost:8300).
"""

import json
import os
import random
import re

import requests

from tasks.common import Task

_JUDGE_URL = os.environ.get("COOK_JUDGE_URL", "http://127.0.0.1:8300/v1/chat/completions")
_JUDGE_MODEL = os.environ.get("COOK_JUDGE_MODEL", "judge")
_PAIRS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "posttrain", "out", "dpo_pairs.jsonl")

_QTY = re.compile(r"\b\d+[\d/.\s]*(cups?|cup|tablespoons?|tbsp|teaspoons?|tsp|ounces?|oz|pounds?|lbs?|grams?|g\b|ml\b|liters?|cloves?|sticks?)", re.I)
_STEP = re.compile(r"^\s*(\d+[.)]|step \d+)", re.I | re.M)
_RECIPEISH = re.compile(r"\brecipe\b|\bhow do i make\b|\bhow to make\b", re.I)

# train half of the flavor matrix (held-out half stays in scripts/flavor_eval.py)
from tasks.flavor_matrix import ALIASES, MATRIX  # noqa: E402

_TRAIN_KEYS = sorted(MATRIX)[::2]  # deterministic split; evens train, odds held out
_TRAIN = {k: MATRIX[k] for k in _TRAIN_KEYS}
_DF = {}
for _pairings in _TRAIN.values():
    for _p in _pairings:
        _DF[_p] = _DF.get(_p, 0) + 1
import math  # noqa: E402

_IDF = {p: math.log(len(_TRAIN) / c) for p, c in _DF.items()}


def _mentions(term, text):
    for a in ALIASES.get(term, [term]):
        if a in text:
            return True
    return False


def distinct4(text):
    toks = text.split()
    if len(toks) < 8:
        return 1.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return len(set(grams)) / len(grams)


def flavor_score(text):
    """For each train-matrix ingredient mentioned, credit its accepted
    pairings also mentioned (tier x idf, capped); mean over mentioned keys."""
    low = text.lower()
    present = [k for k in _TRAIN if _mentions(k, low)]
    if not present:
        return 0.0
    total = 0.0
    for k in present:
        pairings = _TRAIN[k]
        score = sum(
            t * _IDF.get(p, 1.0)
            for p, t in pairings.items()
            if p != k and _mentions(p, low)
        )
        best = sum(sorted((t * _IDF.get(p, 1.0) for p, t in pairings.items()), reverse=True)[:3])
        total += min(score / best, 1.0) if best else 0.0
    return total / len(present)


_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def rank_group(question, candidates, reference, rng, timeout=120):
    """One judge call ranks all candidates plus the (unmarked) reference.

    Returns (scores, ref_score): normalized rank scores in [0, 1] for each
    candidate (best = 1), plus the reference's score — when rollouts start
    out-ranking the curated reference, that's the interesting signal.
    On judge failure every score is 0.5.
    """
    items = list(candidates) + [reference]
    order = list(range(len(items)))
    rng.shuffle(order)
    labeled = "\n\n".join(
        f"=== ANSWER {_LABELS[i]} ===\n{items[j][:1800]}" for i, j in enumerate(order)
    )
    prompt = (
        "You are judging answers to a cooking question. Rank them best to "
        "worst. Prefer real quantities, a coherent method, plausible flavor "
        "pairings, and no rambling or repetition.\n\n"
        f"QUESTION: {question}\n\n{labeled}\n\n"
        f"Reply with ONLY the ranking, best first, like: "
        f"{' > '.join(_LABELS[:len(items)][:4])}{' > ...' if len(items) > 4 else ''}"
    )
    n = len(items)
    try:
        r = requests.post(_JUDGE_URL, timeout=timeout, json={
            "model": _JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8 * n, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        out = r.json()["choices"][0]["message"]["content"].strip().upper()
        seen = [c for c in out if c in _LABELS[:n]]
        ranking = list(dict.fromkeys(seen))  # first occurrence order, deduped
        if len(ranking) != n:
            raise ValueError(f"bad ranking: {out!r}")
    except Exception:
        return [0.5] * len(candidates), 0.5
    # position in ranking -> score; label i corresponds to items[order[i]]
    score_by_item = {}
    for pos, label in enumerate(ranking):
        score_by_item[order[_LABELS.index(label)]] = 1.0 - pos / (n - 1) if n > 1 else 1.0
    return [score_by_item[i] for i in range(len(candidates))], score_by_item[len(items) - 1]


def _pair_one(question, cand, reference, flip, timeout):
    a, b = (cand, reference) if not flip else (reference, cand)
    prompt = (
        "You are judging two answers to a cooking question. Prefer the "
        "answer with real quantities, a coherent method, plausible flavor "
        "pairings, and no rambling or repetition.\n\n"
        f"QUESTION: {question}\n\n=== ANSWER A ===\n{a[:1800]}\n\n"
        f"=== ANSWER B ===\n{b[:1800]}\n\n"
        "Reply with exactly one letter: A or B."
    )
    try:
        r = requests.post(_JUDGE_URL, timeout=timeout, json={
            "model": _JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        out = r.json()["choices"][0]["message"]["content"].strip().upper()
        if out[:1] not in ("A", "B"):
            raise ValueError(out)
        return 1.0 if (out[:1] == "A") != flip else 0.0
    except Exception:
        return 0.5


def pair_wins(question, candidates, reference, rng, timeout=120):
    """Pairwise arm: each candidate vs the reference, calls run concurrently
    against the judge server's parallel slots.

    Returns (scores, ref_score) with the same shape as rank_group; scores
    are 0/1 wins (0.5 on judge failure), ref_score = 1 - mean(win rate)."""
    from concurrent.futures import ThreadPoolExecutor

    flips = [rng.random() < 0.5 for _ in candidates]
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(
            lambda cf: _pair_one(question, cf[0], reference, cf[1], timeout),
            zip(candidates, flips),
        ))
    valid = [s for s in scores if s != 0.5]
    ref_score = 1.0 - (sum(valid) / len(valid)) if valid else 0.5
    return scores, ref_score


class CookingRL(Task):
    """Prompts from the DPO split; reward = gates x (0.3 flavor + 0.7 judge).

    judge_mode: "rank" (one call ranks the whole group + unmarked reference)
    or "pair" (one call per rollout vs the reference)."""

    def __init__(self, seed=0, judge_mode="rank", **kwargs):
        super().__init__(**kwargs)
        assert judge_mode in ("rank", "pair")
        self.judge_mode = judge_mode
        self.rng = random.Random(seed)
        self.rows = []
        with open(_PAIRS, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("prompt") and row.get("chosen"):
                    self.rows.append(row)

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": ""},  # completion slot
            ],
            "_reference": row["chosen"],
        }

    def _gate(self, question, text):
        gate = 1.0
        if distinct4(text) < 0.55:
            gate *= 0.1  # degenerate loops
        if _RECIPEISH.search(question):
            if len(_QTY.findall(text)) < 3:
                gate *= 0.3
            if not _STEP.search(text):
                gate *= 0.5
        return gate

    def reward_group(self, conversation, texts):
        """Score a whole GRPO group with ONE judge ranking call.

        Returns (rewards, ref_score); ref_score is where the curated DPO
        reference landed among the rollouts (0.5 = judge failure)."""
        question = conversation["messages"][0]["content"]
        texts = [t.strip() for t in texts]
        judge = rank_group if self.judge_mode == "rank" else pair_wins
        ranks, ref_score = judge(
            question, texts, conversation["_reference"], self.rng
        )
        rewards = [
            (self._gate(question, t) * (0.3 * flavor_score(t) + 0.7 * r)) if t else 0.0
            for t, r in zip(texts, ranks)
        ]
        return rewards, ref_score

    def reward(self, conversation, generated_text):
        rewards, _ = self.reward_group(conversation, [generated_text])
        return rewards[0]
