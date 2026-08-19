"""Cooking SFT task: grounded QA pairs from Henry's cookbook corpus.

Loads the private HF dataset hbfreed/cook-posttrain (sft_pairs.jsonl +
multiturn.jsonl). Single-turn rows become (user, assistant) conversations;
rows with a matching multi-turn extension get the follow-up exchange
appended. Chunk provenance is disjoint from the DPO and OPSD splits by
construction (see nanoMoEchat posttrain/PLAN.md).
"""

import hashlib
import json
import os
import re

from huggingface_hub import hf_hub_download
from tasks.common import Task

_REPO = "hbfreed/cook-posttrain"
_CORPUS_DIR = os.environ.get(
    "COOKING_CORPUS_DIR",
    os.path.expanduser("~/Documents/PythonProjects/Cooking-Corpus/data"),
)

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_AUTOLINK = re.compile(r"<((?:https?|ftp)://[^\s<>]+)>")  # <http://...> bare-URL link
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>|<[!?][^>]*>")
_HEADING_ANCHOR = re.compile(r"\s*\{#[^}]*\}")
_TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|?\s*$")  # | --- | --- |
_BULLETED_PIPE = re.compile(r"^[*+-]\s*\|")  # table row nested under a list bullet
_EMPTY_HEADING = re.compile(r"^#{1,6}\s*$")  # e.g. bare "##" ebook conversion junk
_EMPTY_PARENS = re.compile(r"\(\s*\)")  # e.g. "(![](icon.jpg))" -> "()" once stripped


def clean_chunk(text):
    """Strip ebook-conversion artifacts (tables, inline links, html) so a
    passage reads as prose when placed inside a chat turn. Raw chunks are
    fine for pretraining but flip the model into document-continuation mode
    when used as privileged context (see the OPSD teacher postmortem)."""
    raw_lines = text.splitlines()
    lines = []
    for i, line in enumerate(raw_lines):
        stripped = line.lstrip()
        if _EMPTY_HEADING.match(stripped):  # bare "##" left by conversion, no text
            continue
        if stripped.startswith("|") or _BULLETED_PIPE.match(stripped):
            # Some ebook conversions wrap a plain sub-recipe/section title in a
            # single-cell "| Title |" line with no header separator and no
            # neighboring table rows - that's not a real table, just a mangled
            # heading, so don't drop it outright. Only treat a pipe line as a
            # table row (and discard it) when it looks like part of an actual
            # table: a separator row, a multi-column row, a row nested under a
            # list bullet, or a row adjacent to another pipe-prefixed line.
            is_separator = bool(_TABLE_SEPARATOR.match(stripped))
            is_multi_col = stripped.count("|") > 2
            prev_pipe = i > 0 and raw_lines[i - 1].lstrip().startswith("|")
            next_pipe = (
                i + 1 < len(raw_lines) and raw_lines[i + 1].lstrip().startswith("|")
            )
            if (
                _BULLETED_PIPE.match(stripped)
                or is_separator
                or is_multi_col
                or prev_pipe
                or next_pipe
            ):
                continue
            line = stripped.strip("|").strip()
        line = _MD_IMAGE.sub("", line)
        line = _MD_LINK.sub(r"\1", line)
        line = _AUTOLINK.sub(r"\1", line)
        line = _HTML_TAG.sub("", line)
        line = _HEADING_ANCHOR.sub("", line)
        line = _EMPTY_PARENS.sub("", line)
        if _EMPTY_HEADING.match(line.strip()):
            # heading whose only content was an image/link/tag we just stripped
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def load_chunk_text(book, chunk):
    path = os.path.join(_CORPUS_DIR, book, chunk)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CookingSFT(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sft_path = hf_hub_download(_REPO, "sft_pairs.jsonl", repo_type="dataset")
        mt_path = hf_hub_download(_REPO, "multiturn.jsonl", repo_type="dataset")
        followups = {}
        with open(mt_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                followups[row["id"]] = row
        self.conversations = []
        with open(sft_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q, a = row["question"].strip(), row["answer"].strip()
                if not q or not a:
                    continue
                messages = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
                # multiturn ids were derived from md5(book + question)
                mt_id = "mt-" + hashlib.md5((row["book"] + row["question"]).encode()).hexdigest()[:10]
                fu = followups.get(mt_id)
                if fu and fu.get("followup") and fu.get("followup_answer"):
                    messages.append({"role": "user", "content": fu["followup"].strip()})
                    messages.append({"role": "assistant", "content": fu["followup_answer"].strip()})
                self.conversations.append(messages)

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return {"messages": self.conversations[index]}


class CookingOpenBook(Task):
    """Open-book variant of CookingSFT: the user turn carries the (cleaned)
    source passage in the same framing chat_opsd uses for its teacher prompt,
    so by OPSD time "answer the question using this reference" is in-
    distribution instead of collapsing into document continuation. Same
    sft-split chunks as CookingSFT, so no bleed into the dpo/opsd splits."""

    def __init__(self, max_passage_chars=4000, **kwargs):
        super().__init__(**kwargs)
        # Prebuilt rows on HF (posttrain/build_openbook_sft.py); fall back to
        # assembling from the local Cooking-Corpus checkout if not uploaded.
        try:
            ob_path = hf_hub_download(_REPO, "openbook_sft.jsonl", repo_type="dataset")
            rows = [json.loads(l) for l in open(ob_path, encoding="utf-8")]
        except Exception:
            rows = []
            sft_path = hf_hub_download(_REPO, "sft_pairs.jsonl", repo_type="dataset")
            with open(sft_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    try:
                        row["passage"] = clean_chunk(load_chunk_text(row["book"], row["chunk"]))
                    except OSError:
                        continue
                    rows.append(row)
        self.conversations = []
        for row in rows:
            q, a = row["question"].strip(), row["answer"].strip()
            passage = row.get("passage", "").strip()
            if not q or not a or not passage:
                continue
            user = (
                q
                + f"\n\nReference (from \"{row['book']}\"):\n"
                + passage[:max_passage_chars]
            )
            self.conversations.append([
                {"role": "user", "content": user},
                {"role": "assistant", "content": a},
            ])

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return {"messages": self.conversations[index]}
