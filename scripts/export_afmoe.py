"""
Export an arch="afmoe" base checkpoint to a Hugging Face AfmoeForCausalLM directory,
loadable by transformers and servable by vLLM (`vllm serve <out-dir>`).

The tokenizer is not converted: pass --tokenizer-dir to copy HF tokenizer files
(e.g. the teacher's, if the model was trained with its tokenizer) into the export.

Run:
    python -m scripts.export_afmoe --model-tag=<tag> --out=<dir> [--step=N] [--tokenizer-dir=<hf dir>]
"""

import argparse
import json
import os
import shutil

import torch

from nanochat.afmoe_export import afmoe_hf_config, to_hf
from nanochat.checkpoint_manager import find_last_step, load_checkpoint
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig

parser = argparse.ArgumentParser(description="Export an afmoe checkpoint to Hugging Face format")
parser.add_argument("--model-tag", type=str, required=True, help="base_checkpoints/<model-tag>")
parser.add_argument("--step", type=int, default=-1, help="checkpoint step (-1: latest)")
parser.add_argument("--out", type=str, required=True, help="output directory")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
parser.add_argument("--tokenizer-dir", type=str, default="", help="HF tokenizer files to copy into the export")
args = parser.parse_args()

checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", args.model_tag)
step = args.step if args.step >= 0 else find_last_step(checkpoint_dir)
model_data, _, meta = load_checkpoint(checkpoint_dir, step, "cpu")
config = GPTConfig(**meta["model_config"])
assert config.arch == "afmoe", f"checkpoint arch is {config.arch!r}; only 'afmoe' exports"

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cpu")
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in model_data.items()},
                      strict=True, assign=True)

hf = to_hf(model, getattr(torch, args.dtype))
hf.save_pretrained(args.out)
# save_pretrained writes transformers' AfmoeConfig, which lacks the routing keys
# vLLM reads; write the full config (as in Arcee's released checkpoints) over it
with open(os.path.join(args.out, "config.json")) as f:
    saved = json.load(f)
saved.update(afmoe_hf_config(config, model.window_sizes))
saved["dtype"] = args.dtype
with open(os.path.join(args.out, "config.json"), "w") as f:
    json.dump(saved, f, indent=2)

if args.tokenizer_dir:
    for name in os.listdir(args.tokenizer_dir):
        if name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "chat_template")):
            shutil.copy(os.path.join(args.tokenizer_dir, name), args.out)
print(f"Exported step {step} of {args.model_tag} to {args.out}")
