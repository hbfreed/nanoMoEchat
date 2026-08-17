"""Post-SFT sufficiency dashboard, single GPU.

Three checks against an SFT checkpoint:
  1. Stop rate: does the model emit <|assistant_end|> before max_tokens,
     over a spread of cooking/general/identity prompts?
  2. Holdout quiz sample: answers to eval_quiz questions (holdout books,
     never trained on in any phase), printed with reference + rubric for
     eyeball scoring.
  3. Forgetting guard: ClimbMix val bpb vs the pretrain baseline (pure-CE
     0.8278 for base-u96x256-swiglu-r10).

Run:
    python -m scripts.sft_dashboard -- --source=sft --model-tag=d20
"""
import argparse
import json
import random
from contextlib import nullcontext

import torch
from huggingface_hub import hf_hub_download

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.engine import Engine
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="sft")
parser.add_argument("--model-tag", type=str, default="d20")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--quiz-n", type=int, default=10)
parser.add_argument("--bpb-steps", type=int, default=20)
parser.add_argument("--skip-bpb", action="store_true")
args = parser.parse_args()

COOK_PROMPTS = [
    "How long should I rest a brisket?",
    "How do I make a roux?",
    "My hollandaise broke. What went wrong?",
    "What's the difference between braising and stewing?",
    "How much salt goes in a basic brine?",
    "What temperature should I roast a chicken at?",
    "How do I keep my pie crust from shrinking?",
    "What's mise en place?",
    "How do I know when my steak is medium rare?",
    "Why do you salt pasta water?",
    "What's the point of resting meat?",
    "How do I fix a sauce that's too salty?",
    "What's the best way to caramelize onions?",
    "How do I sharpen a knife?",
    "What's a beurre blanc?",
    "How do I make stock from chicken bones?",
    "Why did my rice come out mushy?",
    "What's the smoke point of olive oil?",
    "How do I temper chocolate?",
    "What does deglazing mean?",
    "How do I get crispy skin on roast duck?",
    "What's the ratio for vinaigrette?",
    "How long can I keep fish in the fridge?",
    "What's the difference between baking soda and baking powder?",
    "How do I proof yeast?",
    "Why is my bread dense?",
    "What cut of beef is best for stew?",
    "How do I clean a cast iron pan?",
    "What's sous vide?",
    "How do I make mayonnaise by hand?",
]
GENERAL_PROMPTS = [
    "What's the capital of France?",
    "What's 12 times 8?",
    "Name three planets in the solar system.",
    "What language is spoken in Brazil?",
    "What's the opposite of hot?",
    "How many days are in a week?",
    "What color do you get mixing blue and yellow?",
    "What's the chemical symbol for gold?",
    "Who wrote Romeo and Juliet?",
    "What season comes after winter?",
]
IDENTITY_PROMPTS = [
    "Who are you?",
    "What can you help me with?",
    "What's your favorite thing to cook?",
    "Are you a chef?",
    "Tell me about yourself.",
    "Do you know anything about barbecue?",
    "What should I make for dinner tonight?",
    "Hi!",
    "Can you help me plan a menu?",
    "What's your favorite cookbook?",
]

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.step
)
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda" else nullcontext()
)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")


def ask(question, max_tokens=None):
    ids = [bos, user_start] + tokenizer.encode(question) + [user_end, assistant_start]
    toks = []
    with autocast_ctx:
        for col, _ in engine.generate(
            ids, num_samples=1, max_tokens=max_tokens or args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        ):
            toks.append(col[0])
    stopped = len(toks) > 0 and toks[-1] == assistant_end
    body = toks[:-1] if stopped else toks
    return tokenizer.decode(body), stopped, len(toks)

# ---- 1. stop rate ----
print("=" * 70)
print("PART 1: STOP RATE")
print("=" * 70)
groups = [("cook", COOK_PROMPTS), ("general", GENERAL_PROMPTS), ("identity", IDENTITY_PROMPTS)]
transcript = []
for gname, prompts in groups:
    stops, lens = 0, []
    for q in prompts:
        text, stopped, n = ask(q)
        stops += stopped
        lens.append(n)
        transcript.append({"group": gname, "q": q, "a": text, "stopped": stopped, "len": n})
    print(f"{gname:>8}: stop rate {stops}/{len(prompts)}, mean len {sum(lens)/len(lens):.0f} tok")
total_stops = sum(t["stopped"] for t in transcript)
print(f"{'TOTAL':>8}: {total_stops}/{len(transcript)}")

# ---- 2. holdout quiz sample ----
print("\n" + "=" * 70)
print("PART 2: HOLDOUT QUIZ SAMPLE (never-trained books)")
print("=" * 70)
quiz_path = hf_hub_download("hbfreed/cook-posttrain", "eval_quiz.jsonl", repo_type="dataset")
quiz = [json.loads(l) for l in open(quiz_path, encoding="utf-8")]
rng = random.Random(3)
for row in rng.sample(quiz, args.quiz_n):
    ans, stopped, _ = ask(row["question"])
    print(f"\n--- [{row['book']}]")
    print(f"Q: {row['question']}")
    print(f"MODEL{'' if stopped else ' (no stop)'}: {ans}")
    print(f"REF: {row['reference']}")
    print(f"RUBRIC: {row['rubric']}")

# ---- 3. forgetting guard ----
if not args.skip_bpb:
    print("\n" + "=" * 70)
    print("PART 3: FORGETTING GUARD (ClimbMix val bpb; pretrain baseline 0.8278)")
    print("=" * 70)
    seq_len = meta["model_config"]["sequence_len"]
    token_bytes = get_token_bytes(device=device)
    loader = tokenizing_distributed_data_loader(8, seq_len, "val", device=device)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, args.bpb_steps, token_bytes)
    print(f"val bpb: {bpb:.4f} (baseline 0.8278, mid/sft drift expected small)")

out_path = f"/media/henry/MoreFiles/sft_dashboard_{args.source}.json"
json.dump(transcript, open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"\nfull transcript -> {out_path}")
