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

Run:
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_opsd -- --source=dpo --model-tag=d20 --run=opsd-cook-1
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import itertools
import json

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
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-passage-tokens", type=int, default=1024, help="truncate privileged passages to this many tokens")
parser.add_argument("--num-epochs", type=int, default=1)
parser.add_argument("--num-iterations", type=int, default=-1, help="override iteration count (-1 = num_epochs)")
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.01)
parser.add_argument("--save-every", type=int, default=40)
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
engine = Engine(model, tokenizer)
assistant_end = tokenizer.encode_special("<|assistant_end|>")

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
        teacher_prefix = tokenizer.render_for_completion(
            {"messages": [{"role": "user", "content": teacher_content}, dummy]}
        )
        examples.append((student_prefix, teacher_prefix))
print0(f"OPSD examples: {len(examples)}")

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

for step in range(num_iterations):
    step_kl, step_tokens, step_stops, step_rollouts = 0.0, 0, 0, 0
    for _ in range(examples_per_rank):
        student_prefix, teacher_prefix = examples[next(rank_example_iter)]
        prefix_len = len(student_prefix)

        # ---- rollouts (on-policy, current weights) ----
        model.eval()
        sequences = []
        with torch.no_grad():
            for pass_idx in range(args.num_samples // args.gen_batch_size):
                seed = hash((step, prefix_len, pass_idx)) & 0x7FFFFFFF
                seqs, _ = engine.generate_batch(
                    student_prefix,
                    num_samples=args.gen_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                )
                sequences.extend(seqs)
        model.train()

        completions = [seq[prefix_len:] for seq in sequences]
        n_comp = max(len(c) for c in completions)
        step_stops += sum(c[-1] == assistant_end for c in completions if c)
        step_rollouts += len(completions)

        # pad completions; padded positions are masked out of the loss
        comp_mask = torch.zeros(len(completions), n_comp, device=device)
        for i, c in enumerate(completions):
            comp_mask[i, : len(c)] = 1.0
        padded = [c + [assistant_end] * (n_comp - len(c)) for c in completions]

        student_in = torch.tensor(
            [student_prefix + c for c in padded], dtype=torch.long, device=device
        )
        teacher_in = torch.tensor(
            [teacher_prefix + c for c in padded], dtype=torch.long, device=device
        )
        comp_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # ---- reverse KL(student || teacher) over completion tokens ----
        num_passes = len(completions) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            with torch.no_grad():
                logp_t = token_logprobs_slice(teacher_in[b0:b1], len(teacher_prefix), n_comp)
            logp_s = token_logprobs_slice(student_in[b0:b1], prefix_len, n_comp)
            p_s = logp_s.exp()
            kl = (p_s * (logp_s - logp_t)).sum(dim=-1)  # (B, n_comp)
            mask = comp_mask[b0:b1]
            n_valid = mask.sum().clamp(min=1)
            loss = (kl * mask).sum() / n_valid
            (loss / (num_passes * examples_per_rank)).backward()
            step_kl += (kl.detach() * mask).sum().item()
            step_tokens += int(mask.sum().item())
            del logp_s, logp_t, p_s, kl

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
        f"Step {step:05d}/{num_iterations:05d} | mean KL: {mean_kl:.4f} | stop rate: {stop_rate:.2f} | tokens: {step_tokens} | lrm: {lrm:.4f}"
    )
    wandb_run.log({
        "step": step,
        "mean_kl": mean_kl,
        "stop_rate": stop_rate,
        "completion_tokens": step_tokens,
        "lrm": lrm,
    })

    if master_process and args.save_every > 0 and step > 0 and step % args.save_every == 0:
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", f"d{model.config.n_layer}")
        save_checkpoint(checkpoint_dir, step, model.state_dict(), None,
                        {"step": step, "mean_kl": mean_kl, "model_config": model.config.__dict__})
        print0(f"Saved checkpoint at step {step}")

# -----------------------------------------------------------------------------
if master_process:
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", f"d{model.config.n_layer}")
    save_checkpoint(
        checkpoint_dir,
        num_iterations,
        model.state_dict(),
        None,
        {"step": num_iterations, "mean_kl": mean_kl, "model_config": model.config.__dict__},
    )
    print(f"Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish()
if master_process:
    wait_for_checkpoint()
compute_cleanup()
