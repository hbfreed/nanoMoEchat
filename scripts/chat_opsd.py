"""On-policy self-distillation (OPSD) on the cookbook QA set.

The teacher is the SAME model given a privileged reference passage in its
prompt; the student sees only the bare question. We sample rollouts from the
student (on-policy), then minimize per-token reverse KL(student || teacher)
over the rollout tokens — the teacher's passage-conditioned distribution is
the target, and because teacher and student share weights, the target
improves as training progresses (see posttrain/PLAN.md).

Fully on-policy like chat_rl: rollouts are regenerated every step. One
example at a time per rank (all its samples share a prompt prefix, which
keeps the logit slicing trivial).

With --teacher=<model> --teacher-url=<vLLM server> the teacher is instead an
external Qwen model served by vLLM (on-policy distillation a la Thinking
Machines): one teacher copy on its own card serves every student rank, and it
only scores rollouts (prompt_logprobs), never generates. Our tokenizer is
trimmed from Qwen's (scripts/trim_qwen_tokenizer.py), so our token ids map into
Qwen's through teacher_ids.json; no cross-tokenizer alignment. The teacher gets
Qwen's non-thinking header (an empty <think></think> block) and, with
--teacher-passage, the privileged passage too. vLLM returns the teacher's
log-prob of each sampled token rather than its full distribution, so the loss
is the per-token sampled estimator: advantage = log p_teacher - log p_student
on each rollout token, a policy-gradient step on the student (rollouts are
sampled from the full softmax at temperature 1, so no importance weights;
tool outputs the engine forces are masked out). Self-distillation keeps the exact full-vocab KL.

Run:
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_opsd --source=dpo --model-tag=d20 --run=opsd-cook-1
    # vLLM teacher on card 0 (dev-ignore/posttrain/teacher_server.sh), students on cards 1+2:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
        -m scripts.chat_opsd --source=sft --model-tag=d12 --teacher=Qwen/Qwen3.5-4B \
        --teacher-url=http://localhost:8011 --run=opd-qwen4b
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import itertools
import json
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from huggingface_hub import hf_hub_download

from nanochat.checkpoint_manager import load_model, save_checkpoint, wait_for_checkpoint
from tasks.cooking import clean_chunk
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat OPSD training")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--source", type=str, default="dpo", help="sft|dpo — which checkpoint to start from")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=8, help="rollouts per example")
parser.add_argument("--device-batch-size", type=int, default=4, help="sequences per KL forward (student and teacher)")
parser.add_argument("--gen-batch-size", type=int, default=8, help="rollouts per generation pass (no_grad, can exceed device-batch-size)")
parser.add_argument("--examples-per-step", type=int, default=16, help="examples per optimization step across ranks")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature (on-policy wants 1.0)")
parser.add_argument("--top-k", type=int, default=0,
                    help="sample from the top k tokens (<=0: the full distribution; the external-teacher "
                         "estimator assumes rollouts are drawn from the student's full softmax)")
parser.add_argument("--max-passage-tokens", type=int, default=1024, help="truncate privileged passages to this many tokens")
parser.add_argument("--num-epochs", type=int, default=1)
parser.add_argument("--num-iterations", type=int, default=-1, help="override iteration count (-1 = num_epochs)")
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.01)
parser.add_argument("--save-every", type=int, default=40)
parser.add_argument("--teacher", type=str, default=None,
                    help="model name of an external Qwen teacher served by vLLM (default: self-distillation)")
parser.add_argument("--teacher-url", type=str, default="http://localhost:8011",
                    help="vLLM server for --teacher (OpenAI-compatible /v1/completions)")
parser.add_argument("--teacher-passage", type=int, default=1,
                    help="give an external teacher the privileged passage (1) or only the question (0)")
parser.add_argument("--dolci-mix", type=str, default="",
                    help="add Dolci-Instruct-RL prompts as fractions of all prompts, e.g. "
                         "'general:0.45,ifeval:0.3,math:0.05' (rest: cooking). Categories: general, ifeval, "
                         "math, code. Teacher sees them as is (no passage, no verifier). Empty: cooking only")
parser.add_argument("--prompts-file", type=str, default="",
                    help="comma-separated JSONL files of {source, prompt[, teacher_context]} rows "
                         "(scripts/build_opd_prompts.py, build_easy_cooking.py, build_easy_mix.py), mixed "
                         "with the OPSD cooking QA at --cook-frac; teacher_context is shown to the teacher "
                         "only; replaces --dolci-mix")
parser.add_argument("--mask-tool-calls", type=int, default=0,
                    help="1: leave the student's calculator <tool_call> spans out of the loss (the teacher does "
                         "not use our tool format); tool outputs are always masked")
parser.add_argument("--reward-weight", type=float, default=0.0,
                    help="weight of a GRPO-style sequence reward added to the distillation advantage "
                         "(group-normalized over each prompt's rollouts; 0: distillation only)")
parser.add_argument("--format-reward", type=float, default=0.2,
                    help="gsm8k prompts: reward for ending with '#### <number>' (what the GSM8K eval parses)")
parser.add_argument("--correct-reward", type=float, default=1.0,
                    help="gsm8k prompts: reward for a '####' number equal to the gold answer")
parser.add_argument("--marker-kl-free", type=int, default=0,
                    help="1: on gsm8k prompts, leave the '#### <number>' tokens out of the distillation term "
                         "(the teacher never writes the marker and penalizes it hard); the reward still applies")
parser.add_argument("--marker-leak-penalty", type=float, default=0.0,
                    help="non-gsm8k prompts: negative reward for writing '####'")
parser.add_argument("--truncation-penalty", type=float, default=0.0,
                    help="external teacher: extra per-token advantage penalty (nats) on rollouts that hit "
                         "--max-new-tokens without stopping")
parser.add_argument("--cook-frac", type=float, default=0.2, help="cooking share of prompts with --prompts-file")
parser.add_argument("--num-prompts", type=int, default=0,
                    help="total prompts with --dolci-mix (0: all cooking prompts at the cooking fraction)")
parser.add_argument("--max-prompt-tokens", type=int, default=512, help="drop Dolci prompts longer than this")
parser.add_argument("--seed", type=int, default=0, help="prompt sampling and order (same on every rank)")
parser.add_argument("--out-tag", type=str, default=None,
                    help="checkpoint dir name under chatrl_checkpoints (default: --model-tag, else d<n_layer>)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-opsd", name=args.run, config=user_config, save_code=True)
)

model, tokenizer, meta = load_model(
    args.source, device, phase="train", model_tag=args.model_tag, step=args.step
)
# Freeze the SMEBU router biases: in-forward updates use each rank's local counts, so the
# replicas would route differently and only rank 0's biases get saved. A short distillation
# run at a small LR does not need rebalancing.
model.set_defer_bias_updates(True)
engine = Engine(model, tokenizer)
assistant_end = tokenizer.encode_special("<|assistant_end|>")
python_start, python_end = tokenizer.encode_special("<|python_start|>"), tokenizer.encode_special("<|python_end|>")
_m = tokenizer.encode("####")
marker_id = _m[0] if len(_m) == 1 else None  # "####" is one token in the trimmed Qwen vocab
out_tag = args.out_tag or args.model_tag or f"d{model.config.n_layer}"
# one rollout per example per step (rank 0): question, answer, per-rollout KL
samples_path = os.path.join(get_base_dir(), "chatrl_checkpoints", out_tag, "samples.jsonl")
if master_process:
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)

external = args.teacher is not None
if external:
    import requests
    assert hasattr(tokenizer, "render_turn_start"), "an external teacher needs the trimmed Qwen tokenizer"
    with open(os.path.join(tokenizer.tokenizer_dir, "teacher_ids.json")) as f:
        tid = json.load(f)
    teacher_ids = tid["teacher_ids"]  # our id -> Qwen id
    assert len(teacher_ids) == tokenizer.get_vocab_size()
    served = [m["id"] for m in requests.get(f"{args.teacher_url}/v1/models", timeout=30).json()["data"]]
    assert args.teacher in served, f"{args.teacher_url} serves {served}, not {args.teacher}"
    print0(f"teacher {args.teacher} at {args.teacher_url}; our tokenizer was trimmed from {tid['source']}")
    think_header = tokenizer.encode("<think>\n\n</think>\n\n")  # Qwen's non-thinking reply header

# -----------------------------------------------------------------------------
# Data: {question, passage, book, chunk}. Student prompt = bare question;
# teacher prompt = question + privileged passage. Both rendered for
# completion (trailing <|assistant_start|>), rollout tokens appended to each.
_local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "posttrain", "out", "opsd_pairs.jsonl")
if os.path.exists(_local):
    opsd_path = _local
else:
    opsd_path = hf_hub_download("hbfreed/cook-posttrain", "opsd_pairs.jsonl", repo_type="dataset")
examples = []
with open(opsd_path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        # render_for_completion pops a trailing assistant message, then appends
        # <|assistant_start|> to prime the completion — hence the dummy.
        dummy = {"role": "assistant", "content": ""}
        student_prefix = tokenizer.render_for_completion(
            {"messages": [{"role": "user", "content": row["question"]}, dummy]}
        )
        # Clean conversion artifacts so the teacher sees prose, not raw ebook
        # markdown — raw chunks flip the model into document-continuation mode.
        passage_ids = tokenizer.encode(clean_chunk(row["passage"]))[: args.max_passage_tokens]
        passage = tokenizer.decode(passage_ids)
        teacher_content = (
            row["question"]
            + f"\n\nReference (from \"{row['book']}\"):\n"
            + passage
        )
        if external and not args.teacher_passage:
            teacher_content = row["question"]
        teacher_prefix = tokenizer.render_for_completion(
            {"messages": [{"role": "user", "content": teacher_content}, dummy]}
        )
        if external:  # Qwen has no document separator in chat; add its empty think block
            teacher_prefix = teacher_prefix[1:] + think_header
        examples.append((student_prefix, teacher_prefix, row["question"], "cooking", None))

if args.prompts_file:
    import random
    assert not args.dolci_mix, "--prompts-file replaces --dolci-mix"
    rng = random.Random(args.seed)
    extra = []
    for path in args.prompts_file.split(","):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sp = tokenizer.render_for_completion({"messages": [{"role": "user", "content": r["prompt"]}, dummy]})
                if len(sp) > args.max_prompt_tokens:
                    continue
                tp = sp
                if r.get("teacher_context"):  # privileged: only the teacher sees it
                    tp = tokenizer.render_for_completion({"messages": [
                        {"role": "user", "content": r["prompt"] + "\n\n" + r["teacher_context"]}, dummy]})
                extra.append((sp, tp[1:] + think_header if external else tp, r["prompt"], r["source"], r.get("answer")))
    n_cook = min(len(examples), round(len(extra) * args.cook_frac / (1 - args.cook_frac))) if args.cook_frac > 0 else 0
    examples = rng.sample(examples, n_cook) + extra
    rng.shuffle(examples)  # same seed on every rank, so the rank stride splits one shared order
    import collections
    print0("prompt mix: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(e[3] for e in examples).items()))
elif args.dolci_mix:
    import random
    import pyarrow.parquet as pq
    fracs = {k: float(v) for k, v in (kv.split(":") for kv in args.dolci_mix.split(","))}
    cats = {"general": {"general-quality_ref", "general-quality"}, "ifeval": {"ifeval"},
            "math": {"math"}, "code": {"code"}}
    assert set(fracs) <= set(cats), f"unknown Dolci categories {set(fracs) - set(cats)}"
    cook_frac = 1.0 - sum(fracs.values())
    assert 0 < cook_frac <= 1, "Dolci fractions must sum to less than 1 (the rest is cooking)"
    total = args.num_prompts or round(len(examples) / cook_frac)
    rng = random.Random(args.seed)
    pool = {k: [] for k in fracs}
    for i in range(3):
        path = hf_hub_download("allenai/Dolci-Instruct-RL", f"data/train-0000{i}-of-00003.parquet",
                               repo_type="dataset")
        for r in pq.read_table(path, columns=["dataset", "prompt"]).to_pylist():
            d = r["dataset"][0] if isinstance(r["dataset"], list) else r["dataset"]
            text = r["prompt"]
            # single-turn only: the prompt field is "user: ..." with turns inlined
            if not text.startswith("user: ") or "\nassistant:" in text:
                continue
            for k in fracs:
                if d in cats[k]:
                    pool[k].append(text[len("user: "):])
    picked = {"cooking": rng.sample(examples, min(len(examples), round(total * cook_frac)))}
    for k, frac in fracs.items():
        rng.shuffle(pool[k])
        want, chosen = round(total * frac), []
        for q in pool[k]:
            if len(chosen) == want:
                break
            sp = tokenizer.render_for_completion({"messages": [{"role": "user", "content": q}, dummy]})
            if len(sp) > args.max_prompt_tokens:
                continue
            tp = sp[1:] + think_header if external else sp
            chosen.append((sp, tp, q, k, None))
        picked[k] = chosen
    examples = [e for v in picked.values() for e in v]
    rng.shuffle(examples)  # same seed on every rank, so the rank stride splits one shared order
    print0("prompt mix: " + ", ".join(f"{k} {len(v)}" for k, v in picked.items()))
print0(f"OPSD examples: {len(examples)}")

# GSM8K gold answers for the reward (prompt files store only the question)
from tasks.gsm8k import extract_answer
if any(e[3] == "gsm8k" and e[4] is None for e in examples):
    from datasets import load_dataset
    gold_by_q = {r["question"]: extract_answer(r["answer"]) for r in load_dataset("openai/gsm8k", "main", split="train")}
    examples = [e if e[3] != "gsm8k" or e[4] is not None else e[:4] + (gold_by_q.get(e[2]),) for e in examples]
    missing = sum(e[3] == "gsm8k" and e[4] is None for e in examples)
    assert missing == 0, f"{missing} gsm8k prompts without a gold answer"


def sequence_reward(source, gold, text):
    """Verifiable reward for one rollout: (reward, has_marker, correct)."""
    pred = extract_answer(text)
    if source == "gsm8k":
        correct = pred is not None and gold is not None and pred.rstrip(".") == gold.rstrip(".")
        return args.format_reward * (pred is not None) + args.correct_reward * correct, pred is not None, correct
    return -args.marker_leak_penalty * ("####" in text), "####" in text, False

assert args.examples_per_step % ddp_world_size == 0
examples_per_rank = args.examples_per_step // ddp_world_size
assert args.num_samples % args.device_batch_size == 0
assert args.num_samples % args.gen_batch_size == 0
num_iterations = args.num_iterations
if num_iterations == -1:
    num_iterations = (len(examples) // args.examples_per_step) * args.num_epochs
print0(f"iterations: {num_iterations} ({examples_per_rank} examples/rank/step, {args.num_samples} rollouts each)")


def token_logprobs_slice(inputs, prefix_len, n_completion):
    """Log-probs over the completion region: logits at positions
    [prefix_len-1, prefix_len-1+n_completion) predict the completion tokens.
    Returns (B, n_completion, V) float32 log-probs."""
    logits = model(inputs)
    sl = logits[:, prefix_len - 1 : prefix_len - 1 + n_completion]
    return F.log_softmax(sl.float(), dim=-1)


def teacher_token_logprobs(teacher_prefix, completions):
    """The vLLM teacher's log-prob of each completion token, as a list per completion.
    Our ids map to Qwen's; the teacher scores prefix + completion as a prompt
    (prompt_logprobs=0 gives the actual token's log-prob) and generates one dummy token."""
    prompts = [[teacher_ids[t] for t in teacher_prefix + c] for c in completions]
    for attempt in range(6):  # backoff 5..160 s: rides out a teacher restart
        try:
            r = requests.post(f"{args.teacher_url}/v1/completions", timeout=300, json={
                "model": args.teacher, "prompt": prompts, "max_tokens": 1,
                "prompt_logprobs": 0, "temperature": 0.0})
            if 400 <= r.status_code < 500:  # our request is bad (e.g. too long): retrying won't help
                raise RuntimeError(f"teacher rejected the request: {r.status_code} {r.text[:500]}")
            r.raise_for_status()
            choices = sorted(r.json()["choices"], key=lambda c: c["index"])
            break
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt == 5:
                raise
            print(f"[rank {ddp_rank}] teacher request failed ({e!r}), retry {attempt + 1}/5 in {5 * 2 ** attempt} s")
            time.sleep(5 * 2 ** attempt)
    assert len(choices) == len(prompts), f"teacher returned {len(choices)} of {len(prompts)} rows"
    out = []
    for ch, p, c in zip(choices, prompts, completions):
        entries = ch["prompt_logprobs"]
        assert len(entries) == len(p), f"teacher returned {len(entries)} prompt logprobs for {len(p)} tokens"
        out.append([e[str(q)]["logprob"] for e, q in zip(entries[len(p) - len(c):], p[len(p) - len(c):])])
    return out


optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
rank_example_iter = itertools.cycle(range(ddp_rank, len(examples), ddp_world_size))
mean_kl = float("nan")  # defined even when there are no iterations

for step in range(num_iterations):
    step_kl, step_tokens, step_stops, step_rollouts, step_truncated = 0.0, 0, 0, 0, 0
    step_samples = []
    step_marker, step_count, step_correct = {"gsm8k": 0, "other": 0}, {"gsm8k": 0, "other": 0}, 0
    for _ in range(examples_per_rank):
        student_prefix, teacher_prefix, question, prompt_source, gold = examples[next(rank_example_iter)]
        prefix_len = len(student_prefix)

        # ---- rollouts (on-policy, current weights) ----
        model.eval()
        sequences, sampled_masks = [], []  # mask 0: tokens the engine forced (tool outputs)
        with torch.no_grad():
            for pass_idx in range(args.num_samples // args.gen_batch_size):
                seed = hash((step, prefix_len, pass_idx)) & 0x7FFFFFFF
                seqs, gen_masks = engine.generate_batch(
                    student_prefix,
                    keep_stop=True,  # train on the decision to stop, and count it
                    num_samples=args.gen_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if args.top_k > 0 else None,
                    seed=seed,
                )
                sequences.extend(seqs)
                sampled_masks.extend(gen_masks)
        model.train()

        completions = [seq[prefix_len:] for seq in sequences]
        n_comp = max(len(c) for c in completions)
        step_stops += sum(c[-1] == assistant_end for c in completions if c)
        step_rollouts += len(completions)

        # pad completions; padded positions are masked out of the loss
        comp_mask = torch.zeros(len(completions), n_comp, device=device)
        for i, (c, m) in enumerate(zip(completions, sampled_masks)):
            comp_mask[i, : len(c)] = torch.tensor(m[prefix_len:], dtype=torch.float, device=device)
            # Leave calculator calls out of the loss: the teacher does not use our calculator
            # format, so grading them would teach the student to drop the tool.
            inside = False
            for t, tokid in enumerate(c if args.mask_tool_calls else ()):
                if tokid == python_start:
                    inside = True
                if inside:
                    comp_mask[i, t] = 0.0
                if tokid == python_end:
                    inside = False
        truncated = torch.tensor([float(len(c) >= args.max_new_tokens and c[-1] != assistant_end)
                                  for c in completions], device=device)
        texts = [tokenizer.decode(c) for c in completions]
        rewards = [sequence_reward(prompt_source, gold, t) for t in texts]
        r = torch.tensor([x[0] for x in rewards], device=device)
        reward_adv = (r - r.mean()) / (r.std() + 1e-6) if r.std() > 0 else torch.zeros_like(r)
        kind = "gsm8k" if prompt_source == "gsm8k" else "other"
        step_marker[kind] += sum(x[1] for x in rewards)
        step_count[kind] += len(rewards)
        step_correct += sum(x[2] for x in rewards)
        step_truncated += int(truncated.sum().item())
        padded = [c + [assistant_end] * (n_comp - len(c)) for c in completions]

        student_in = torch.tensor(
            [student_prefix + c for c in padded], dtype=torch.long, device=device
        )
        comp_ids = torch.tensor(padded, dtype=torch.long, device=device)
        kl_keep = torch.ones(len(completions), n_comp, device=device)  # 0: no distillation term
        if args.marker_kl_free and prompt_source == "gsm8k":
            for i, c in enumerate(completions):
                if marker_id in c:
                    j = c.index(marker_id)
                    end = len(c) - 1 if c and c[-1] == assistant_end else len(c)  # the stop token keeps its KL
                    kl_keep[i, j:end] = 0.0
        if external:  # one request per example: its rollouts share the prompt
            t_lp = teacher_token_logprobs(teacher_prefix, completions)
            logp_t_tok = torch.zeros(len(completions), n_comp, device=device)
            for i, lp in enumerate(t_lp):
                logp_t_tok[i, : len(lp)] = torch.tensor(lp, device=device)
        else:
            teacher_in = torch.tensor(
                [teacher_prefix + c for c in padded], dtype=torch.long, device=device
            )

        # ---- reverse KL(student || teacher) over completion tokens ----
        num_passes = len(completions) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            mask = comp_mask[b0:b1]
            n_valid = mask.sum().clamp(min=1)
            logp_s = token_logprobs_slice(student_in[b0:b1], prefix_len, n_comp)
            if external:
                # sampled-token estimator: kl_t = log p_s(x_t) - log p_t(x_t); minimize
                # E[kl] by policy gradient with per-token advantage -kl_t
                lp_s = logp_s.gather(-1, comp_ids[b0:b1, :, None]).squeeze(-1)
                kl = lp_s.detach() - logp_t_tok[b0:b1]
                adv_cost = (kl * kl_keep[b0:b1] + args.truncation_penalty * truncated[b0:b1, None]
                            - args.reward_weight * reward_adv[b0:b1, None])
                loss = (adv_cost * lp_s * mask).sum() / n_valid
            else:
                with torch.no_grad():
                    logp_t = token_logprobs_slice(teacher_in[b0:b1], len(teacher_prefix), n_comp)
                kl = (logp_s.exp() * (logp_s - logp_t)).sum(dim=-1)  # exact, (B, n_comp)
                loss = (kl * mask).sum() / n_valid
                del logp_t
            (loss / (num_passes * examples_per_rank)).backward()
            if master_process and pass_idx == 0:  # one rollout per example, for reading
                m0 = mask[0]
                step_samples.append({
                    "step": step, "source": prompt_source, "question": question,
                    "answer": tokenizer.decode(completions[0]),
                    "stopped": bool(completions[0]) and completions[0][-1] == assistant_end,
                    "tokens": len(completions[0]),
                    "kl": round(((kl[0].detach() * m0).sum() / m0.sum().clamp(min=1)).item(), 4),
                    "reward": rewards[0][0],
                })
            step_kl += (kl.detach() * mask).sum().item()
            step_tokens += int(mask.sum().item())
            del logp_s, kl

    lrm = 1.0 - step / num_iterations
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    mean_kl = step_kl / max(step_tokens, 1)
    stop_rate = step_stops / max(step_rollouts, 1)
    print0(
        f"Step {step:05d}/{num_iterations:05d} | mean KL: {mean_kl:.4f} | stop rate: {stop_rate:.2f} | truncated: {step_truncated / max(step_rollouts, 1):.2f} | tokens: {step_tokens} | lrm: {lrm:.4f}"
        f" | gsm8k #### {step_marker['gsm8k']}/{step_count['gsm8k']} correct {step_correct} | other #### {step_marker['other']}/{step_count['other']}"
    )
    if master_process:
        with open(samples_path, "a", encoding="utf-8") as f:
            for row in step_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step % 10 == 0 and step_samples:
            r = step_samples[0]
            print0(f"  sample (KL {r['kl']}, {r['tokens']} tok, stopped={r['stopped']}) Q: {r['question'][:120]!r}\n  A: {r['answer'][:400]!r}")
    wandb_run.log({
        "step": step,
        "mean_kl": mean_kl,
        "stop_rate": stop_rate,
        "truncated_rate": step_truncated / max(step_rollouts, 1),
        "gsm8k_marker_rate": step_marker["gsm8k"] / max(step_count["gsm8k"], 1),
        "gsm8k_correct_rate": step_correct / max(step_count["gsm8k"], 1),
        "other_marker_rate": step_marker["other"] / max(step_count["other"], 1),
        "completion_tokens": step_tokens,
        "lrm": lrm,
    })

    if master_process and args.save_every > 0 and step > 0 and step % args.save_every == 0:
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", out_tag)
        save_checkpoint(checkpoint_dir, step, model.state_dict(), None,
                        {"step": step, "mean_kl": mean_kl, "tokenizer": meta.get("tokenizer"), "model_config": model.config.__dict__})
        print0(f"Saved checkpoint at step {step}")

# -----------------------------------------------------------------------------
if master_process:
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", out_tag)
    save_checkpoint(
        checkpoint_dir,
        num_iterations,
        model.state_dict(),
        None,
        {"step": num_iterations, "mean_kl": mean_kl, "tokenizer": meta.get("tokenizer"), "model_config": model.config.__dict__},
    )
    print(f"Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish()
if master_process:
    wait_for_checkpoint()
compute_cleanup()
