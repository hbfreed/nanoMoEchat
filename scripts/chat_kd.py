"""
Off-policy distillation (soft-label KD) on teacher-written answers from scripts/gen_kd_data.

Each example is a user prompt (as the student sees it) and a teacher answer with the teacher's
top-k log-probs at every answer position. The loss is cross-entropy against the teacher's top-k
distribution renormalized over those k tokens (k=20 holds ~99% of the mass), averaged over answer
tokens; --hard-weight mixes in plain CE on the sampled token. Prompt tokens get no loss.
This is SFT on teacher text, except each position learns the teacher's whole ranking of next
tokens, not only the one it sampled. Router biases stay frozen (as in chat_opsd).

torchrun --standalone --nproc_per_node=2 -m scripts.chat_kd --data=/media/henry/MoreFiles/kd/kd_easy_v1.pt \\
    --source=sft --model-tag=d12 --out-tag=d12-kd-easy
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint, wait_for_checkpoint
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0

parser = argparse.ArgumentParser(description="Off-policy soft-label distillation")
parser.add_argument("--data", required=True, help="torch file from scripts.gen_kd_data")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--source", type=str, default="sft")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--out-tag", type=str, required=True)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--target-examples-per-step", type=int, default=32)
parser.add_argument("--num-epochs", type=int, default=2)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="as chat_sft")
parser.add_argument("--hard-weight", type=float, default=0.0, help="weight of plain CE on the sampled token")
parser.add_argument("--val-frac", type=float, default=0.03)
parser.add_argument("--eval-every", type=int, default=25)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
user_config = vars(args).copy()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
wandb_run = DummyWandb() if args.run == "dummy" or not master_process else wandb.init(
    project="nanochat-kd", name=args.run, config=user_config)

model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.step)
model.set_defer_bias_updates(True)  # never applied: router biases stay frozen
max_len = model.config.sequence_len

# -----------------------------------------------------------------------------
# Data: student prompt ids + teacher answer ids, with top-k teacher log-probs per answer position
dummy = {"role": "assistant", "content": ""}
rows = torch.load(args.data)
examples = []
for r in rows:
    p = tokenizer.render_for_completion({"messages": [{"role": "user", "content": r["prompt"]}, dummy]})
    if len(p) + len(r["ids"]) > max_len:
        continue
    examples.append((p, r))
rng = random.Random(args.seed)
rng.shuffle(examples)
n_val = max(1, int(len(examples) * args.val_frac))
val, train = examples[:n_val], examples[n_val:]
print0(f"{len(rows)} rows, {len(examples)} fit in {max_len} tokens: train {len(train)}, val {len(val)}")
pad_id = tokenizer.encode_special("<|assistant_end|>")
K = rows[0]["topk_ids"].shape[1]


def collate(batch):
    T = max(len(p) + len(r["ids"]) for p, r in batch) - 1
    B = len(batch)
    inputs = torch.full((B, T), pad_id, dtype=torch.long)
    hard = torch.full((B, T), -1, dtype=torch.long)
    tk_ids = torch.zeros((B, T, K), dtype=torch.long)
    tk_lp = torch.full((B, T, K), -1e4, dtype=torch.float32)
    for i, (p, r) in enumerate(batch):
        seq = torch.tensor(p + r["ids"].tolist(), dtype=torch.long)
        n, a = len(seq), len(r["ids"])
        inputs[i, : n - 1] = seq[:-1]
        s = len(p) - 1  # logits at s predict the first answer token
        hard[i, s : s + a] = r["ids"].long()
        tk_ids[i, s : s + a] = r["topk_ids"].long()
        tk_lp[i, s : s + a] = r["topk_lp"].float()
    return inputs.to(device), hard.to(device), tk_ids.to(device), tk_lp.to(device)


def batches(data, epoch_seed):
    order = list(range(ddp_rank, len(data), ddp_world_size))
    random.Random(epoch_seed).shuffle(order)
    for i in range(0, len(order) - args.device_batch_size + 1, args.device_batch_size):
        yield collate([data[j] for j in order[i : i + args.device_batch_size]])


def kd_loss(inputs, hard, tk_ids, tk_lp):
    """Summed soft CE, summed KL to the renormalized top-k teacher, summed hard CE, token count."""
    logits = model(inputs).float()
    logp = F.log_softmax(logits, dim=-1)
    mask = hard >= 0
    lp_t = torch.log_softmax(tk_lp, dim=-1)  # renormalize teacher over its top-k
    p_t = lp_t.exp()
    lp_s = logp.gather(-1, tk_ids)  # student log-probs of the teacher's top-k
    soft = -(p_t * lp_s).sum(-1)
    kl = (p_t * (lp_t - lp_s)).sum(-1)
    hard_ce = -logp.gather(-1, hard.clamp(min=0)[..., None]).squeeze(-1)
    m = mask.float()
    return (soft * m).sum(), (kl * m).sum(), (hard_ce * m).sum(), m.sum()


examples_per_step = args.device_batch_size * ddp_world_size
assert args.target_examples_per_step % examples_per_step == 0
grad_accum = args.target_examples_per_step // examples_per_step
steps_per_epoch = len(train) // args.target_examples_per_step
num_iterations = steps_per_epoch * args.num_epochs
print0(f"grad accum {grad_accum}, {steps_per_epoch} steps/epoch, {num_iterations} steps")

optimizers = model.setup_optimizers(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
                                    matrix_lr=args.matrix_lr, weight_decay=0.0)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]


def evaluate():
    model.eval()
    tot = torch.zeros(4, device=device)
    with torch.no_grad():
        for b in batches(val, 0):
            tot += torch.stack(kd_loss(*b))
    if ddp:
        dist.all_reduce(tot)
    model.train()
    return (tot[1] / tot[3]).item(), (tot[2] / tot[3]).item()


def train_iter():
    epoch = 0
    while True:
        yield from batches(train, args.seed + 1000 + epoch)
        epoch += 1


it = train_iter()
val_kl = val_ce = float("nan")
for step in range(num_iterations + 1):
    if step % args.eval_every == 0 or step == num_iterations:
        val_kl, val_ce = evaluate()
        print0(f"Step {step:04d} | val KL(top{K}) {val_kl:.4f} | val CE(teacher tokens) {val_ce:.4f}")
        wandb_run.log({"step": step, "val_kl": val_kl, "val_ce": val_ce})
    if step == num_iterations:
        break
    # Token mean over the whole step (all micro-batches, all ranks). The optimizers average
    # gradients over ranks (reduce_scatter AVG), so each rank's summed loss is scaled by
    # world_size / total tokens.
    micro = [next(it) for _ in range(grad_accum)]
    n_tok = sum((b[1] >= 0).sum() for b in micro).float()
    if ddp:
        dist.all_reduce(n_tok)
    stats = torch.zeros(4, device=device)
    for b in micro:
        soft, kl, hard_ce, n = kd_loss(*b)
        ((soft + args.hard_weight * hard_ce) * ddp_world_size / n_tok).backward()
        stats += torch.stack([soft.detach(), kl.detach(), hard_ce.detach(), n])
    lrm = 1.0 - step / num_iterations
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    if ddp:
        dist.all_reduce(stats)
    print0(f"Step {step:04d}/{num_iterations} | soft CE {stats[0] / stats[3]:.4f} | KL {stats[1] / stats[3]:.4f} "
           f"| hard CE {stats[2] / stats[3]:.4f} | tokens {int(stats[3])} | lrm {lrm:.3f}")
    wandb_run.log({"step": step, "train_kl": (stats[1] / stats[3]).item(), "lrm": lrm})

if master_process:
    checkpoint_dir = os.path.join(get_base_dir(), "chatrl_checkpoints", args.out_tag)
    save_checkpoint(checkpoint_dir, num_iterations, model.state_dict(), None,
                    {"step": num_iterations, "val_kl": val_kl, "val_ce": val_ce, "kd_data": args.data,
                     "tokenizer": meta.get("tokenizer"), "model_config": model.config.__dict__})
    print(f"Saved model checkpoint to {checkpoint_dir}")
wandb_run.finish()
if master_process:
    wait_for_checkpoint()
compute_cleanup()
