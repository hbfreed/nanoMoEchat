"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json
import time
import argparse

import torch
import torch.distributed as dist

import wandb
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, wait_for_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    print0,
    print_banner,
)
from nanochat.dataloader import TokenStream, pretokenized_data_loader, tokenizing_distributed_data_loader
from nanochat.engine import Engine
from nanochat.flash_attention import ATTN_IMPL
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.muon import DistMuon, FSDPMuon
from nanochat.tokenizer import get_token_bytes, get_tokenizer, get_tokenizer_name
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--wandb-run-id", type=str, default="", help="resume this existing wandb run id instead of creating a new one")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="number of transformer layers")
parser.add_argument("--model-dim", type=int, default=-1, help="model dimension (-1 = derive from depth as depth * 64)")
parser.add_argument("--num-heads", type=int, default=-1, help="number of attention heads (-1 = derive from model_dim)")
parser.add_argument("--num-kv-heads", type=int, default=-1, help="number of kv heads for GQA (-1 = same as num_heads)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context")
# MoE routing
parser.add_argument(
    "--mlp-act",
    type=str,
    default="relu2",
    choices=["relu2", "swiglu"],
    help="expert MLP: relu2 (w1/w2, stk path) or swiglu (w1/w3/w2, grouped Triton path)",
)
parser.add_argument("--expert-sizes", type=json.loads, default=[(64, 256)], help="JSON list of [count, width] tuples, e.g. '[[64,256]]'")
parser.add_argument("--num-active-experts", type=int, default=8, help="top-k experts per token")
parser.add_argument("--load-balance-loss-weight", type=float, default=0.08, help="load balance loss weight")
parser.add_argument("--router-z-loss-weight", type=float, default=0.001, help="router z-loss weight")
parser.add_argument("--compute-loss-weight", type=float, default=0.004, help="compute loss weight")
parser.add_argument("--use-bias-balancing", action="store_true", help="enable bias balancing for expert routing")
parser.add_argument("--bias-update-speed", type=float, default=0.0005, help="SMEBU learning rate (lambda)")
parser.add_argument("--bias-momentum", type=float, default=0.5, help="SMEBU momentum factor (beta)")
parser.add_argument("--bias-kappa", type=float, default=2.0, help="SMEBU tanh saturation speed (kappa)")
parser.add_argument("--fp32-master-weights", action="store_true", help="keep params/optimizer state in fp32, cast to bf16 in forward")
parser.add_argument("--seed", type=int, default=42, help="seed for model init (compute_init seeds 42)")
parser.add_argument("--stochastic-rounding", action="store_true", help="stochastically round bf16 weight updates (Muon and AdamW)")
parser.add_argument("--arch", type=str, default="nanochat", choices=["nanochat", "afmoe"], help="block architecture; afmoe converts to transformers/vLLM AfmoeForCausalLM")
parser.add_argument("--sliding-window", type=int, default=-1, help="window (tokens incl. current) for S layers in --window-pattern (-1: seq_len // 2)")
parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE base")
parser.add_argument("--num-shared-experts", type=int, default=0, help="afmoe: shared experts (each as wide as a routed expert)")
parser.add_argument("--num-dense-layers", type=int, default=0, help="afmoe: first k layers use a dense SwiGLU MLP")
parser.add_argument("--dense-intermediate-size", type=int, default=0, help="afmoe: dense MLP width (0: 4 * model dim)")
parser.add_argument("--norm-lr", type=float, default=0.004, help="AdamW LR for norm weights (afmoe)")
parser.add_argument("--bf16-optim-state", action="store_true", help="store momentum / Adam moments in bf16 (use with --fp32-master-weights)")
parser.add_argument("--replica-check-every", type=int, default=250, help="multi-GPU: every N steps, check all ranks hold identical weights and resync from rank 0 if not (-1 = off)")
parser.add_argument("--loss-impl", type=str, default="chunked", choices=["cce_kahan_full_c", "cce", "compiled", "chunked"], help="lm_head + cross-entropy implementation")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=20, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size (reduce if OOM)")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
# Output
parser.add_argument("--model-tag", type=str, default="", help="override model tag for checkpoint directory name")
parser.add_argument("--tokenizer", type=str, default="", help="tokenizer dir under the base dir (default: $NANOCHAT_TOKENIZER or 'tokenizer'), e.g. tokenizer_qwen49152")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoint every N steps (-1 = only at end)")
parser.add_argument("--keep-last-n", type=int, default=-1, help="prune old checkpoints, keeping only the last N unprotected saves (-1 = keep all)")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--profile-step", type=int, default=-1, help="profile this step: per-phase wall time and GPU time by kernel type (rank 0), plus a chrome trace in the checkpoint dir's parent")
parser.add_argument("--muon-per-expert", action=argparse.BooleanOptionalAction, default=True, help="Muon orthogonalizes each MoE expert's matrices separately rather than the concatenated (n_embd, E*width) matrices (with --fsdp, experts are sharded whole, so no optimizer communication). A/B, depth-4 192x256 top-4, 2000 steps: val bpb 1.1300 vs 1.1398 joint")
parser.add_argument("--activation-checkpointing", action="store_true", help="recompute each block in backward to fit bigger micro-batches")
parser.add_argument("--checkpoint-layers", type=int, default=-1, help="with --activation-checkpointing: checkpoint only the first N blocks (-1: all), trading memory for less recompute")
parser.add_argument("--fsdp", action="store_true", help="FSDP2: shard params, grads and optimizer state across ranks (per block)")
parser.add_argument("--fsdp-keep-weights", action="store_true", help="with --fsdp: keep each block's gathered weights from forward to backward instead of gathering them again (reshard_after_forward=False). Costs about 4/3 bytes per parameter per rank. 3.74B total on 3x3090: 2788 -> 2335 ms/step")
parser.add_argument("--fsdp-copy-engine", action=argparse.BooleanOptionalAction, default=True, help="with --fsdp: all-gather through the GPUs' copy engines (symmetric memory, NCCL_CTA_POLICY=2), so communication stops taking SMs from compute. 3.74B total on 3x3090 with --fsdp-keep-weights: 2350 -> 2120 ms/step")
parser.add_argument("--moe-backend", type=str, default="triton", choices=["triton", "cute"], help="megablocks backend for swiglu experts: triton (any GPU) or cute (CuTe DSL, Ampere; 1.5-3%% faster end to end on 3x3090)")
parser.add_argument("--pretokenized", action="store_true", help="read <base_dir>/tokens/<tokenizer> (scripts/pretokenize.py) instead of tokenizing parquet on the fly")
args = parser.parse_args()
if args.tokenizer:  # get_tokenizer() everywhere, incl. the dataloaders, reads this
    os.environ["NANOCHAT_TOKENIZER"] = args.tokenizer
user_config = vars(args).copy()  # for logging
# Convert expert_sizes from list of lists to list of tuples (JSON gives lists)
args.expert_sizes = [tuple(x) for x in args.expert_sizes]
# -----------------------------------------------------------------------------

# Compute init
if args.fsdp and args.fsdp_copy_engine:
    # zero-CTA collectives; NCCL reads this when it creates the communicators
    os.environ.setdefault("NCCL_CTA_POLICY", "2")
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
torch.manual_seed(args.seed)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(f"Attention: {ATTN_IMPL} ({'sliding windows are dense-masked and slow' if ATTN_IMPL == 'sdpa' else 'real sliding-window kernels'})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    wandb_kwargs = dict(project="nanochat", name=args.run, config=user_config)
    if args.wandb_run_id:
        # Resume the existing wandb run so charts stay continuous across crashes.
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "allow"
    wandb_run = wandb.init(**wandb_kwargs)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs - use provided values or derive from depth
depth = args.depth
num_layers = depth
model_dim = args.model_dim
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
max_seq_len = args.max_seq_len
if model_dim == -1:
    model_dim = depth * 64  # default aspect ratio 64
if num_heads == -1:
    num_heads = max(1, (model_dim + 127) // 128)  # default head dim 128
if num_kv_heads == -1:
    num_kv_heads = num_heads  # default 1:1 GQA ratio (i.e. no GQA)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
device_batch_size = args.device_batch_size
total_batch_size = args.total_batch_size
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    window_pattern=args.window_pattern,
    expert_sizes=args.expert_sizes,
    mlp_act=args.mlp_act,
    num_active_experts=args.num_active_experts,
    load_balance_loss_weight=args.load_balance_loss_weight,
    router_z_loss_weight=args.router_z_loss_weight,
    compute_loss_weight=args.compute_loss_weight,
    use_bias_balancing=args.use_bias_balancing,
    bias_update_speed=args.bias_update_speed,
    bias_momentum=args.bias_momentum,
    bias_kappa=args.bias_kappa,
    fp32_master_weights=args.fp32_master_weights,
    loss_impl=args.loss_impl,
    arch=args.arch,
    sliding_window=args.sliding_window,
    rope_theta=args.rope_theta,
    num_shared_experts=args.num_shared_experts,
    num_dense_layers=args.num_dense_layers,
    dense_intermediate_size=args.dense_intermediate_size,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
# SMEBU bias updates once per step from counts summed over micro-steps and ranks (below),
# so routing is fixed within a step and the bias is identical on every rank
model.set_defer_bias_updates(True)
if args.activation_checkpointing:
    model.set_activation_checkpointing(True, None if args.checkpoint_layers < 0 else args.checkpoint_layers)
model.set_moe_backend(args.moe_backend)
if args.fsdp:
    # FSDP2: each block (and the root: embeddings, lm_head, final norm) is all-gathered
    # for its forward/backward and resharded after (after backward only, with
    # --fsdp-keep-weights); grads are reduce-scattered each backward. Buffers (expert
    # biases) stay replicated.
    from torch.distributed.fsdp import fully_shard
    assert ddp, "--fsdp needs torchrun with more than one rank"
    assert not args.fp32_master_weights, "--fsdp with fp32 master weights is not supported yet"
    from torch.distributed.tensor import Shard
    # per-expert Muon: shard expert matrices along their expert dim so every rank holds
    # whole experts (w1/w3 are (n_embd, E*width): dim 1; w2 already shards whole experts on dim 0)
    layouts = model.expert_layouts() if args.muon_per_expert else {}
    for p, (dim, width) in layouts.items():
        n = p.shape[dim] // width
        assert n % ddp_world_size == 0, (
            f"--fsdp with per-expert Muon gives each rank whole experts, so the {n} experts of "
            f"width {width} must split evenly over {ddp_world_size} ranks"
        )
    placement = lambda p: Shard(layouts[p][0]) if p in layouts else None
    reshard = not args.fsdp_keep_weights
    for module in (*model.transformer.h, model):  # the root last
        fully_shard(module, shard_placement_fn=placement, reshard_after_forward=reshard)
        if args.fsdp_copy_engine:
            module.set_symm_mem_for_comm()
    del layouts, placement  # they reference the unsharded params; keeping them keeps a full copy alive
    print0(f"FSDP: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per rank after sharding")
orig_model = model  # original, uncompiled model, for saving raw model state_dict
if args.fsdp or args.activation_checkpointing:
    # compiling the root graph-breaks at every FSDP2 hook and at the (compile-disabled)
    # MoE inside each checkpointed block, leaving blocks eager: compile the pieces instead
    for block in orig_model.transformer.h:
        block.compile_pieces()
else:
    model = torch.compile(model, dynamic=False)
total_params = sum(p.numel() for p in model.parameters())
if model_config.use_moe:
    total_expert_width = sum(count * size for count, size in model_config.expert_sizes)
    num_experts = sum(count for count, _ in model_config.expert_sizes)
    mlp_mult = 3 if model_config.mlp_act == "swiglu" else 2
    moe_params = model_dim * total_expert_width * mlp_mult * num_layers
    inactive_moe = (
        moe_params * (num_experts - model_config.num_active_experts) // num_experts
    )
    active_params = total_params - inactive_moe
    print0(f"Number of total parameters: {total_params:,}")
    print0(f"Number of active parameters: {active_params:,}")
else:
    print0(f"Number of parameters: {total_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")
num_params = active_params if model_config.use_moe else total_params

# Calculate number of iterations
num_iterations = args.num_iterations
target_flops = args.target_flops
target_param_data_ratio = args.target_param_data_ratio
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    # target_param_data_ratio is a float arg; floor-div of floats is a float
    # and range() downstream rejects it
    num_iterations = int(target_tokens // total_batch_size)
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
    stochastic_rounding=args.stochastic_rounding,
    optim_state_dtype=torch.bfloat16 if args.bf16_optim_state else None,
    norm_lr=args.norm_lr,
    fsdp=args.fsdp,
    muon_per_expert=args.muon_per_expert,
)
adamw_optimizer, muon_optimizer = optimizers

# setup_optimizers only uses the sharded DistMuon/DistAdamW (which reduce-scatter grads
# themselves) for power-of-2 world sizes. Otherwise nothing syncs grads and each rank
# silently trains its own model, so average them before stepping.
sharded_optim = isinstance(muon_optimizer, (DistMuon, FSDPMuon))
sync_grads = ddp and not sharded_optim
# sharded optimizer state lives on every rank, so every rank saves/loads its own shard
optim_shard_rank = ddp_rank if sharded_optim else None
print0(f"Optimizers: {'FSDP2 (FSDPMuon/AdamW on shards)' if args.fsdp else 'sharded (DistMuon/DistAdamW)' if sharded_optim else 'replicated'}")
if args.fsdp:  # these run the model on rank 0 only, or compare replicas FSDP does not have
    for flag in ("sample_every", "core_metric_every", "replica_check_every"):
        if getattr(args, flag) > 0:
            print0(f"--fsdp: disabling --{flag.replace('_', '-')} (not supported with FSDP yet)")
            setattr(args, flag, -1)
# SMEBU expert biases: updated from globally summed counts, so identical on every rank;
# the replica check covers them too.
expert_bias_buffers = [
    b for name, b in orig_model.named_buffers()
    if name.endswith(("expert_bias", "expert_bias_momentum"))
]


def check_replicas_(step):
    """Ranks should hold bit-identical weights; if any drifted, warn and resync from rank 0.
    One small all_gather of per-tensor checksums, so cheap enough to run periodically."""
    state = list(orig_model.parameters()) + expert_bias_buffers
    sums = torch.stack([t.detach().double().sum() for t in state])
    gathered = [torch.empty_like(sums) for _ in range(ddp_world_size)]
    dist.all_gather(gathered, sums)
    drifted = sum(not torch.equal(gathered[0], g) for g in gathered[1:])
    if drifted:
        print0(f"WARNING step {step}: {drifted} rank(s) drifted from rank 0's weights; resyncing")
        for t in state:
            dist.broadcast(t.data, src=0)


def all_reduce_avg_(tensors):
    """Average same-dtype tensors across ranks in place, with a single collective."""
    if not tensors:
        return
    flat = torch.cat([t.reshape(-1) for t in tensors])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for t in tensors:
        t.copy_(flat[offset : offset + t.numel()].view_as(t))
        offset += t.numel()


# Resume from a previous checkpoint if requested (load model + optimizer state)
base_dir = get_base_dir()
start_step = 0
if args.resume_from_step != -1:
    resume_tag = args.model_tag if args.model_tag else f"d{num_layers}"
    resume_dir = os.path.join(base_dir, "base_checkpoints", resume_tag)
    print0(f"Resuming from step {args.resume_from_step} in {resume_dir}")
    model_data, optimizer_data, _meta = load_checkpoint(
        resume_dir, args.resume_from_step, device, load_optimizer=True,
        shard_rank=optim_shard_rank,
    )
    # Strip torch.compile's "_orig_mod." prefix from state_dict keys
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    # assign=False (in-place copy) so optimizer's param references stay valid.
    # assign=True would swap the tensors, orphaning Muon's stored param refs.
    if args.fsdp:  # every rank loaded the full state; each keeps its shards
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
        set_model_state_dict(orig_model, model_data, options=StateDictOptions(full_state_dict=True))
    else:
        orig_model.load_state_dict(model_data, strict=True, assign=False)
    assert isinstance(optimizer_data, list) and len(optimizer_data) == 2, \
        "expected a 2-element list [adamw_state, muon_state]"
    adamw_optimizer.load_state_dict(optimizer_data[0])
    muon_optimizer.load_state_dict(optimizer_data[1])
    start_step = args.resume_from_step
    del model_data, optimizer_data
    # Protect the resume checkpoint from keep_last_n pruning (touch sentinel).
    if master_process:
        resume_model_path = os.path.join(resume_dir, f"model_{args.resume_from_step:06d}.pt")
        if os.path.exists(resume_model_path):
            open(resume_model_path + ".protected", "w").close()

# Initialize the DataLoaders for train/val
if args.pretokenized:
    start_micro = max(args.resume_from_step, 0) * grad_accum_steps
    train_loader = pretokenized_data_loader(
        device_batch_size, max_seq_len, split="train", device=device, start_micro_step=start_micro
    )
    build_val_loader = lambda: pretokenized_data_loader(
        device_batch_size, max_seq_len, split="val", device=device
    )
    train_stream = TokenStream("train")
    tokens_needed = num_iterations * total_batch_size
    print0(f"Pre-tokenized train stream: {train_stream.num_tokens/1e9:.2f}B tokens in "
           f"{len(train_stream.shards)} shards; this run reads {tokens_needed/1e9:.2f}B"
           + (" (WRAPS AROUND: repeats data)" if tokens_needed > train_stream.num_tokens else ""))
    x, y = next(train_loader)
else:
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="train", device=device
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="val", device=device
    )
    x, y = next(train_loader)  # kick off load of the very first batch of data

# On resume, fast-forward the data stream to the checkpoint's position: the
# loader is a deterministic walk over sorted parquet shards (rank-striped), so
# draining the same number of micro-batches reproduces it exactly. Without
# this, a resumed run silently retrains on data from stream position zero --
# including the heavily-repeated SPT mix shards -- and diverges from any
# non-crashed twin run's data schedule. (The pre-tokenized loader starts there.)
if args.resume_from_step > 0 and not args.pretokenized:
    n_skip = args.resume_from_step * grad_accum_steps
    print0(f"Fast-forwarding data stream: draining {n_skip:,} micro-batches...")
    t0 = time.time()
    for _ in range(n_skip - 1):
        next(train_loader)
    x, y = next(train_loader)
    print0(f"Fast-forward done in {time.time() - t0:.1f}s")

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

warmup_ratio = args.warmup_ratio
warmdown_ratio = args.warmdown_ratio
final_lr_frac = args.final_lr_frac

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
val_bpb = None  # stays None if --eval-every=-1
smooth_train_loss = 0  # EMA of training loss
ema_beta = 0.9  # EMA decay factor
total_training_time = 0  # total wall-clock time of training
save_every = args.save_every
eval_every = args.eval_every
eval_tokens = args.eval_tokens
core_metric_every = args.core_metric_every
core_metric_max_per_task = args.core_metric_max_per_task
sample_every = args.sample_every
model_tag = args.model_tag

# note that we run +1 steps only so that we can eval and save at the end
for step in range(start_step, num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if eval_every > 0 and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    results = {}
    if core_metric_every > 0 and (
        last_step or (step > 0 and step % core_metric_every == 0)
    ):
        model.eval()
        results = evaluate_model(
            orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
        )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    if sample_every > 0 and master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend=tokenizer.get_bos_token_id())
            sample, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=16, temperature=0
            )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint periodically and at the end (only on master process).
    # Also always save at the last pre-decay step so WSD resumes can grab a clean
    # stable-phase checkpoint regardless of save_every cadence.
    warmdown_start = num_iterations - round(warmdown_ratio * num_iterations)
    at_warmdown_start = step == warmdown_start and warmdown_ratio > 0
    should_save = last_step or at_warmdown_start or (save_every > 0 and step > 0 and step % save_every == 0)
    if should_save and (master_process or sharded_optim):
        output_dirname = model_tag if model_tag else f"d{depth}"  # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        # Protect "important" saves from pruning: the pre-decay snapshot
        # (so WSD continuations always have a clean stable-phase entry) and
        # the final step.
        protect = at_warmdown_start or last_step
        keep_last_n = args.keep_last_n if args.keep_last_n > 0 else None
        if args.fsdp:  # collective: every rank sends its shards, rank 0 gets the full state
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            model_state = get_model_state_dict(
                orig_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        else:
            model_state = orig_model.state_dict() if master_process else None
        save_checkpoint(
            checkpoint_dir,
            step,
            model_state if master_process else None,
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "tokenizer": get_tokenizer_name(),
            },
            protect=protect,
            keep_last_n=keep_last_n,
            shard_rank=optim_shard_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    profiling = step == args.profile_step
    if profiling:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)
        prof.__enter__()
        torch.cuda.reset_peak_memory_stats()
        mem = {"step start": torch.cuda.memory_allocated()}
    synchronize()
    t0 = time.time()
    expert_counts = None
    for micro_step in range(grad_accum_steps):
        logits, loss, combined_aux_loss = model(x, y)
        if profiling and micro_step == 0:
            mem["after 1st forward"] = torch.cuda.memory_allocated()
            mem["peak in 1st forward"] = torch.cuda.max_memory_allocated()
        if combined_aux_loss is not None and "tokens_per_expert" in combined_aux_loss:
            c = combined_aux_loss["tokens_per_expert"].long()
            expert_counts = c if expert_counts is None else expert_counts + c
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
    if sync_grads:
        for p in model.parameters():
            if p.grad is None:  # keep the flat buffer layout identical on every rank
                p.grad = torch.zeros_like(p)
        all_reduce_avg_([p.grad for p in model.parameters()])
    if profiling:
        synchronize()
        t_fwdbwd = time.time()
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    if expert_counts is not None:
        if ddp:
            dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)
        orig_model.update_expert_biases(expert_counts)
    if ddp:
        if args.replica_check_every > 0 and step % args.replica_check_every == 0:
            check_replicas_(step)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    if profiling:
        prof.__exit__(None, None, None)
        if master_process:
            # GPU kernels only (CPU-side ops' device time would double count their kernels)
            kinds = {"comm (nccl)": ("nccl",), "attention": ("flash", "fmha"),
                     "matmul": ("gemm", "cutlass", "s16816", "s1688", "xmma"),
                     "moe (triton)": ("_sdd", "_dsd", "_dds", "padded_", "gather", "scatter",
                                      "sort", "histogram", "topology", "cumsum", "binned")}
            by_kind, kernels = {}, []
            for e in prof.key_averages():
                if e.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                t = e.device_time_total / 1e3
                kernels.append((t, e.key))
                k = next((k for k, pats in kinds.items() if any(x in e.key.lower() for x in pats)), "other kernels")
                by_kind[k] = by_kind.get(k, 0) + t
            mem["peak in step"] = torch.cuda.max_memory_allocated()
            print0("PROFILE memory (GB): " + ", ".join(f"{k} {v/1e9:.2f}" for k, v in mem.items()))
            print0(f"PROFILE step {step}: wall {dt*1e3:.0f} ms = fwd+bwd {1e3*(t_fwdbwd-t0):.0f} ms "
                   f"+ optimizer/sync {1e3*(t1-t_fwdbwd):.0f} ms | GPU kernel time (comm overlaps compute): "
                   + ", ".join(f"{k} {v:.0f} ms" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])))
            for t, key in sorted(kernels, reverse=True)[:15]:
                print0(f"  {t:8.1f} ms  {key[:110]}")
            trace = os.path.join(base_dir, f"profile_{model_tag or 'run'}_step{step}.json")
            prof.export_chrome_trace(trace)
            print0(f"PROFILE trace: {trace}")
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step - start_step + 1))  # EMA restarts on resume
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)  # dt covers every micro-step
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec = gpu_peak_flops * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec  # in %
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time / 60:.2f}m"
    )
    if step % 100 == 0:
        log_dict = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if combined_aux_loss is not None:
            log_dict["train/ce_loss"] = combined_aux_loss["ce_loss"].item()
            log_dict["train/router_z_loss"] = combined_aux_loss["router_z_loss"].item()
            log_dict["train/load_balance_loss"] = combined_aux_loss["load_balance_loss"].item()
            if "compute_loss" in combined_aux_loss:
                log_dict["train/compute_loss"] = combined_aux_loss["compute_loss"].item()
            if "router_logits_abs_max" in combined_aux_loss:
                log_dict["train/router_logits_abs_max"] = combined_aux_loss["router_logits_abs_max"].item()
                log_dict["train/router_logits_abs_mean"] = combined_aux_loss["router_logits_abs_mean"].item()
            if "expert_bias_abs_max" in combined_aux_loss:
                log_dict["train/expert_bias_abs_max"] = combined_aux_loss["expert_bias_abs_max"].item()
                log_dict["train/expert_bias_abs_mean"] = combined_aux_loss["expert_bias_abs_mean"].item()
            if "expert_usage" in combined_aux_loss:
                for j, val in enumerate(combined_aux_loss["expert_usage"]):
                    log_dict[f"train/expert_usage_{j}"] = val.item()
                # Width-class aggregates: with many experts the per-expert
                # scalars are unreadable; these four numbers are the variable-
                # width experiment. effective_width < mean width = the compute
                # loss is tilting routing toward narrow experts.
                usage = [v.item() for v in combined_aux_loss["expert_usage"]]
                widths, idx = [], 0
                for count, size in model_config.expert_sizes:
                    widths += [size] * count
                total_f = sum(usage) or 1.0
                by_class = {}
                for f_e, w_e in zip(usage, widths):
                    by_class[w_e] = by_class.get(w_e, 0.0) + f_e
                for w_e, share in sorted(by_class.items()):
                    log_dict[f"train/width_share_{w_e}"] = share / total_f
                eff_w = sum(f_e * w_e for f_e, w_e in zip(usage, widths)) / total_f
                mean_w = sum(widths) / len(widths)
                log_dict["train/effective_width"] = eff_w
                log_dict["train/effective_compute_frac"] = eff_w / mean_w
            if step % 1000 == 0 and "expert_bias_per_layer" in combined_aux_loss:
                for i, bias_vec in enumerate(combined_aux_loss["expert_bias_per_layer"]):
                    for j, val in enumerate(bias_vec):
                        log_dict[f"train/expert_bias_layer_{i}_expert_{j}"] = val.item()
            if step % 1000 == 0 and "expert_usage_per_layer" in combined_aux_loss:
                for i, usage_vec in enumerate(combined_aux_loss["expert_usage_per_layer"]):
                    for j, val in enumerate(usage_vec):
                        log_dict[f"train/expert_usage_layer_{i}_expert_{j}"] = val.item()
        wandb_run.log(log_dict)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report

get_report().log(
    section="Base model training",
    data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
            "final_lr_frac": final_lr_frac,
        },
        {
            "Minimum validation bpb": min_val_bpb,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# cleanup
wandb_run.finish()
# Ensure the final async checkpoint save is flushed to disk before we tear
# down the distributed / CUDA runtime.
if master_process or sharded_optim:
    wait_for_checkpoint()
compute_cleanup()
