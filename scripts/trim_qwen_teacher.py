"""Trim a Qwen3.5 checkpoint to our tokenizer's vocabulary, for off-policy distillation.

Our tokenizer (tokenizer_qwen49152) is Qwen's BPE restricted to 49152 tokens and renumbered;
teacher_ids.json maps our id -> Qwen id. Slicing the (tied) embedding / output rows to those
ids, in our order, gives a teacher that reads and writes our ids directly. Its softmax runs over
our vocabulary only, which is exactly the full model's distribution restricted to our tokens and
renormalized. vLLM then samples only tokens the student can represent and returns log-probs
indexed by our ids, with no remapping.

Everything else (vision tower, MTP head, preprocessor configs) is copied unchanged; the config
gets vocab_size 49152 and the vision special-token ids in our numbering. The tokenizer files are
ours.

Run:
    python -m scripts.trim_qwen_teacher --model Qwen/Qwen3.5-4B \\
        --out /media/henry/MoreFiles/qwen35_4b_trim49152
"""

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--tokenizer-dir", default="/media/henry/MoreFiles/tokenizer_qwen49152")
ap.add_argument("--out", required=True)
args = ap.parse_args()

src = snapshot_download(args.model)
with open(os.path.join(args.tokenizer_dir, "teacher_ids.json")) as f:
    teacher_ids = json.load(f)["teacher_ids"]
V = len(teacher_ids)
idx = torch.tensor(teacher_ids)
os.makedirs(args.out, exist_ok=True)

with open(os.path.join(src, "model.safetensors.index.json")) as f:
    index = json.load(f)
sliced = []
for shard in sorted(set(index["weight_map"].values())):
    sd = load_file(os.path.join(src, shard))
    for k in list(sd):
        if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"):
            assert sd[k].shape[0] == 248320, (k, sd[k].shape)
            sd[k] = sd[k][idx].contiguous()
            sliced.append(k)
    save_file(sd, os.path.join(args.out, shard), metadata={"format": "pt"})
    print(f"wrote {shard}", flush=True)
print("sliced:", sliced)
assert sliced, "no embedding found"
shutil.copy(os.path.join(src, "model.safetensors.index.json"), args.out)

# Config: vocab size and the special-token ids, renumbered into our vocabulary.
with open(os.path.join(args.tokenizer_dir, "tokenizer.json")) as f:
    ours = {a["content"]: a["id"] for a in json.load(f)["added_tokens"]}
q2o = {q: o for o, q in enumerate(teacher_ids)}
with open(os.path.join(src, "config.json")) as f:
    cfg = json.load(f)
for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
    cfg[k] = q2o[cfg[k]]
cfg["text_config"]["vocab_size"] = V
cfg["text_config"]["eos_token_id"] = q2o[cfg["text_config"]["eos_token_id"]]  # <|endoftext|>
for name, qid in (("<|endoftext|>", 248044), ("<|im_end|>", 248046)):
    assert ours[name] == q2o[qid], name
with open(os.path.join(args.out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

for name in ("preprocessor_config.json", "video_preprocessor_config.json", "LICENSE"):
    if os.path.exists(os.path.join(src, name)):
        shutil.copy(os.path.join(src, name), args.out)
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(os.path.join(args.tokenizer_dir, name), args.out)
with open(os.path.join(args.out, "generation_config.json"), "w") as f:
    json.dump({"eos_token_id": [ours["<|im_end|>"], ours["<|endoftext|>"]], "pad_token_id": ours["<|endoftext|>"]}, f)
print(f"done: {args.out} (vocab {V})")
