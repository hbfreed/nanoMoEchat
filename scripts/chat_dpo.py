"""DPO on the cookbook preference pairs (hbfreed/cook-posttrain, dpo_pairs).

Chosen = shelf-register recipe answers, rejected = mom-blog register, same
dish/same core ingredients (see posttrain/PLAN.md). Reference log-probs are
precomputed with the frozen starting weights (policy == reference at init),
so no second model is resident during training.

Run:
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_dpo -- --source=sft --model-tag=d20 --run=dpo-cook-1
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from huggingface_hub import hf_hub_download

from nanochat.checkpoint_manager import load_model, save_checkpoint, wait_for_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat DPO training")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--source", type=str, default="sft", help="sft|mid — which checkpoint to start from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--step", type=int, default=None, help="step to load the model from")
parser.add_argument("--device-batch-size", type=int, default=4, help="preference PAIRS per device per micro-step (2x sequences)")
parser.add_argument("--num-epochs", type=int, default=2, help="number of training epochs")
parser.add_argument("--target-pairs-per-step", type=int, default=32, help="preference pairs per optimization step")
parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
parser.add_argument("--val-pairs", type=int, default=64, help="pairs held out for val margin/accuracy")
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.002, help="initial LR as fraction of base LR (~10x below SFT)")
parser.add_argument("--eval-every", type=int, default=20)
parser.add_argument("--num-iterations", type=int, default=-1, help="override iteration count (-1 = num_epochs)")
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
    else wandb.init(project="nanochat-dpo", name=args.run, config=user_config, save_code=True)
)

model, tokenizer, meta = load_model(
    args.source, device, phase="train", model_tag=args.model_tag, step=args.step
)
max_seq_len = meta["model_config"]["sequence_len"]

# -----------------------------------------------------------------------------
# Data: each pair renders as two conversations sharing the user prompt.
dpo_path = hf_hub_download("hbfreed/cook-posttrain", "dpo_pairs.jsonl", repo_type="dataset")
pairs = []
with open(dpo_path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        rendered = []
        ok = True
        for answer in (row["chosen"], row["rejected"]):
            ids, mask = tokenizer.render_conversation({"messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": answer},
            ]})
            if len(ids) > max_seq_len:
                ok = False
                break
            rendered.append((ids, mask))
        if ok:
            pairs.append(rendered)
print0(f"DPO pairs: {len(pairs)} (val holdout: {args.val_pairs})")
val_pairs = pairs[: args.val_pairs]
train_pairs = pairs[args.val_pairs :]

pad_token_id = tokenizer.encode_special("<|assistant_end|>")


def collate(pair_batch):
    # flatten pairs to sequences: [c0, r0, c1, r1, ...]
    seqs = [s for pair in pair_batch for s in pair]
    nrows = len(seqs)
    ncols = max(len(ids) for ids, mask in seqs) - 1
    inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
    targets = torch.full((nrows, ncols), -1, dtype=torch.long)
    for i, (ids, mask) in enumerate(seqs):
        n = len(ids)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        inputs[i, : n - 1] = ids_tensor[:-1]
        row_targets = ids_tensor[1:]
        mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
        row_targets[mask_tensor == 0] = -1
        targets[i, : n - 1] = row_targets
    return inputs.to(device), targets.to(device)


def seq_logprobs(inputs, targets):
    """Per-sequence sum of assistant-token log-probs. Returns (nrows,) tensor
    and the aux loss dict from the forward (None for pure-dense configs)."""
    _, loss, aux = model(inputs, targets, loss_reduction="none")
    loss = loss.view(inputs.size(0), -1)
    return -(loss * (targets >= 0)).sum(dim=1), aux


def rank_indices(n):
    return list(range(ddp_rank, n, ddp_world_size))


# -----------------------------------------------------------------------------
# Reference log-probs: one frozen pass with the starting weights. Each rank
# covers its own training shard (matching the iteration order below) plus the
# full val set.
model.eval()
ref = {}
with torch.no_grad():
    idxs = rank_indices(len(train_pairs)) + [("val", i) for i in range(len(val_pairs))]
    for pos in range(0, len(idxs), args.device_batch_size):
        chunk = idxs[pos : pos + args.device_batch_size]
        batch = [val_pairs[i[1]] if isinstance(i, tuple) else train_pairs[i] for i in chunk]
        inputs, targets = collate(batch)
        lp, _ = seq_logprobs(inputs, targets)
        for j, key in enumerate(chunk):
            ref[key] = (lp[2 * j].item(), lp[2 * j + 1].item())
model.train()
print0(f"Reference log-probs precomputed for {len(ref)} pairs (this rank)")

# -----------------------------------------------------------------------------
device_batch_size = args.device_batch_size
pairs_per_step = device_batch_size * ddp_world_size
assert args.target_pairs_per_step % pairs_per_step == 0
grad_accum_steps = args.target_pairs_per_step // pairs_per_step
num_iterations = args.num_iterations
if num_iterations == -1:
    num_iterations = (len(train_pairs) // args.target_pairs_per_step) * args.num_epochs
print0(f"grad accum: {grad_accum_steps}, iterations: {num_iterations}")


def train_batches():
    while True:
        idxs = rank_indices(len(train_pairs))
        for pos in range(0, len(idxs) - device_batch_size + 1, device_batch_size):
            chunk = idxs[pos : pos + device_batch_size]
            yield chunk, collate([train_pairs[i] for i in chunk])


def dpo_loss_and_stats(inputs, targets, keys, backward_scale=None):
    lp, aux = seq_logprobs(inputs, targets)
    pol_c, pol_r = lp[0::2], lp[1::2]
    ref_cr = torch.tensor([ref[k] for k in keys], device=device)
    reward_c = args.beta * (pol_c - ref_cr[:, 0])
    reward_r = args.beta * (pol_r - ref_cr[:, 1])
    margin = reward_c - reward_r
    loss = -F.logsigmoid(margin).mean()
    # keep the router regularized during DPO: fold the weighted aux losses
    # (reduction="none" leaves them out of the CE by design)
    if aux is not None:
        cfg = model.config
        if cfg.load_balance_loss_weight > 0:
            loss = loss + cfg.load_balance_loss_weight * aux["load_balance_loss"]
        if cfg.router_z_loss_weight > 0:
            loss = loss + cfg.router_z_loss_weight * aux["router_z_loss"]
    if backward_scale is not None:
        (loss * backward_scale).backward()
    return loss.detach(), margin.detach()


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
train_iter = train_batches()
val_stats = {}
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    if last_step or step % args.eval_every == 0:
        model.eval()
        margins = []
        with torch.no_grad():
            for pos in range(0, len(val_pairs), device_batch_size):
                batch = val_pairs[pos : pos + device_batch_size]
                keys = [("val", i) for i in range(pos, pos + len(batch))]
                inputs, targets = collate(batch)
                _, margin = dpo_loss_and_stats(inputs, targets, keys)
                margins.append(margin)
        margins = torch.cat(margins)
        val_stats = {
            "val_margin": margins.mean().item(),
            "val_acc": (margins > 0).float().mean().item(),
        }
        print0(f"Step {step:05d} | val margin: {val_stats['val_margin']:.4f} | val acc: {val_stats['val_acc']:.4f}")
        wandb_run.log({"step": step, **val_stats})
        model.train()

    if last_step:
        break

    losses, margins = [], []
    for micro_step in range(grad_accum_steps):
        keys, (inputs, targets) = next(train_iter)
        loss, margin = dpo_loss_and_stats(inputs, targets, keys, backward_scale=1.0 / grad_accum_steps)
        losses.append(loss)
        margins.append(margin)

    lrm = 1.0 - step / num_iterations
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    loss_item = torch.stack(losses).mean().item()
    margin_cat = torch.cat(margins)
    print0(
        f"Step {step:05d}/{num_iterations:05d} | dpo loss: {loss_item:.6f} | margin: {margin_cat.mean().item():.4f} | acc: {(margin_cat > 0).float().mean().item():.4f} | lrm: {lrm:.4f}"
    )
    wandb_run.log({
        "step": step,
        "train_loss": loss_item,
        "train_margin": margin_cat.mean().item(),
        "train_acc": (margin_cat > 0).float().mean().item(),
        "lrm": lrm,
    })

# -----------------------------------------------------------------------------
if master_process:
    base_dir = get_base_dir()
    model_tag = f"d{model.config.n_layer}"
    checkpoint_dir = os.path.join(base_dir, "chatdpo_checkpoints", model_tag)
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None,
        {
            "step": step,
            **val_stats,
            "model_config": model.config.__dict__,
        },
    )
    print(f"Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish()
if master_process:
    wait_for_checkpoint()
compute_cleanup()
