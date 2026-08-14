"""Cooking SFT task: grounded QA pairs from Henry's cookbook corpus.

Loads the private HF dataset hbfreed/cook-posttrain (sft_pairs.jsonl +
multiturn.jsonl). Single-turn rows become (user, assistant) conversations;
rows with a matching multi-turn extension get the follow-up exchange
appended. Chunk provenance is disjoint from the DPO and OPSD splits by
construction (see nanoMoEchat posttrain/PLAN.md).
"""

import hashlib
import json

from huggingface_hub import hf_hub_download
from tasks.common import Task

_REPO = "hbfreed/cook-posttrain"


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
