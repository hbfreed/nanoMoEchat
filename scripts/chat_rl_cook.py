"""RLVR on cooking prompts: GRPO/REINFORCE against verifiable rewards.

Same simplified GRPO loop as scripts/chat_rl.py (no trust region, on-policy,
(r - mu) advantages), but the task is tasks/cooking_rl.py: gates x
(0.3 flavor-matrix + 0.7 pairwise judge vs the DPO chosen reference), with
the judge served locally (COOK_JUDGE_URL, see tasks/cooking_rl.py).

Checkpoints go to chatrlvr_checkpoints (NOT chatrl_checkpoints, which holds
the OPSD artifacts).

Serve the judge first, then:
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl_cook -- --run=rlvr-cook-1
"""

import os
import itertools
import argparse
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model, wait_for_checkpoint
from nanochat.engine import Engine
from tasks.cooking_rl import CookingRL

parser = argparse.ArgumentParser(description="RLVR on cooking prompts")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--source", type=str, default="rl", help="which checkpoint to start from (rl = final OPSD cook)")
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--examples-per-step", type=int, default=8)
parser.add_argument("--num-samples", type=int, default=16, help="rollouts per prompt (GRPO group)")
parser.add_argument("--max-new-tokens", type=int, default=400)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.05)
parser.add_argument("--num-epochs", type=int, default=1)
parser.add_argument("--save-every", type=int, default=30)
parser.add_argument("--eval-every", type=int, default=30)
parser.add_argument("--eval-examples", type=int, default=32)
parser.add_argument("--judge-mode", type=str, default="rank", choices=["rank", "pair"])
args = parser.parse_args()
user_config = vars(args).copy()

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.bfloat16

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rlvr", name=args.run, config=user_config)

model, tokenizer, meta = load_model(args.source, device, phase="eval")
engine = Engine(model, tokenizer)

device_batch_size = args.device_batch_size
examples_per_step = args.examples_per_step
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k

# Last 10% of prompts held out for eval; never trained.
full_task = CookingRL(seed=1234 + ddp_rank, judge_mode=args.judge_mode)
split = int(full_task.num_examples() * 0.9)
train_indices = list(range(split))
val_indices = list(range(split, full_task.num_examples()))
num_steps = (len(train_indices) // examples_per_step) * args.num_epochs
print0(f"prompts: {split} train / {len(val_indices)} val; steps: {num_steps}")


@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = train_indices[ddp_rank::ddp_world_size]
    for example_idx in itertools.cycle(rank_indices):
        conversation = full_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            seq_batch, masks_batch = engine.generate_batch(
                tokens, num_samples=device_batch_size, max_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, seed=seed,
            )
            generated_token_sequences.extend(seq_batch)
            masks.extend(masks_batch)

        texts = [
            tokenizer.decode(sample_tokens[prefix_length:])
            for sample_tokens in generated_token_sequences
        ]
        rewards, ref_score = full_task.reward_group(conversation, texts)
        wandb_run.log({"ref_rank_score": ref_score})

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [m + [0] * (max_length - len(m)) for m in masks]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards - rewards.mean()
        yield generated_token_sequences, inputs, targets, rewards, advantages


@torch.no_grad()
def run_val():
    model.eval()
    total, n = 0.0, 0
    for idx in val_indices[ddp_rank::ddp_world_size][: args.eval_examples // ddp_world_size]:
        conversation = full_task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        seqs, _ = engine.generate_batch(
            tokens, num_samples=1, max_tokens=max_new_tokens, temperature=0.6, top_k=top_k,
        )
        text = tokenizer.decode(seqs[0][len(tokens):])
        total += full_task.reward(conversation, text)
        n += 1
    t = torch.tensor([total, float(n)], device=device)
    if ddp:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1].clamp(min=1)).item()


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

assert examples_per_step % ddp_world_size == 0
examples_per_rank = examples_per_step // ddp_world_size
print0(f"sequences/step: {examples_per_step * num_samples}; examples/rank: {examples_per_rank}")

batch_iterator = get_batch()
for step in range(num_steps):
    if step % args.eval_every == 0:
        val_reward = run_val()
        print0(f"Step {step} | val reward: {val_reward:.4f}")
        wandb_run.log({"step": step, "val_reward": val_reward})

    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs, targets = inputs_all[b0:b1], targets_all[b0:b1]
            advantages = advantages_all[b0:b1]
            _, nll, _ = model(inputs, targets, loss_reduction="none")
            logp = -nll.view_as(inputs)
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            (-pg_obj).backward()
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(s) for s in sequences_all)

    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_len = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        t = torch.tensor([mean_reward, mean_len], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_reward, mean_len = t[0].item(), t[1].item()
    lrm = 1.0 - step / num_steps
    print0(f"Step {step}/{num_steps} | reward: {mean_reward:.4f} | len: {mean_len:.1f} | lrm: {lrm:.4f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_len, "lrm": lrm})

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        checkpoint_dir = os.path.join(
            get_base_dir(), "chatrlvr_checkpoints",
            f"d{model.config.n_layer}-{args.judge_mode}",
        )
        save_checkpoint(checkpoint_dir, step, model.state_dict(), None,
                        {"model_config": model.config.__dict__})
        print(f"Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish()
if master_process:
    wait_for_checkpoint()
compute_cleanup()
