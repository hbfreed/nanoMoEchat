"""
Profile one training step of a given model shape on a single GPU.

Reports step time, tokens/s, achieved TFLOP/s and MFU against a *measured* bf16
matmul ceiling (not the spec sheet; it moves with the power cap), peak memory,
the forward / backward / optimizer split, and GPU time grouped by kernel type.
Builds the model exactly like scripts/base_train.py, on random tokens.

Run with:
python -m dev.profile_step --depth=4 --model-dim=256 --num-heads=2 --num-kv-heads=2 \
    --expert-sizes='[[192,256]]' --num-active-experts=4 --device-batch-size=32 --json
"""

import argparse
import json
import re
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from nanochat.gpt import GPT, GPTConfig

parser = argparse.ArgumentParser(description="Profile one training step")
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--model-dim", type=int, default=256)
parser.add_argument("--num-heads", type=int, default=2)
parser.add_argument("--num-kv-heads", type=int, default=2)
parser.add_argument("--expert-sizes", type=json.loads, default=[[192, 256]])
parser.add_argument("--num-active-experts", type=int, default=4)
parser.add_argument("--mlp-act", type=str, default="relu2", choices=["relu2", "swiglu"])
parser.add_argument("--vocab-size", type=int, default=50304)
parser.add_argument("--max-seq-len", type=int, default=1024)
parser.add_argument("--window-pattern", type=str, default="L")
parser.add_argument("--sliding-window", type=int, default=-1)
parser.add_argument("--arch", type=str, default="nanochat", choices=["nanochat", "afmoe"])
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--fp32-master-weights", action="store_true")
parser.add_argument("--use-bias-balancing", action="store_true")
parser.add_argument("--no-compile", action="store_true")
parser.add_argument("--loss-impl", type=str, default="chunked", choices=["cce_kahan_full_c", "cce", "compiled", "chunked"])
parser.add_argument("--warmup", type=int, default=8, help="untimed steps (compile, autotune)")
parser.add_argument("--steps", type=int, default=10, help="timed steps")
parser.add_argument("--trace", type=str, default="", help="write a chrome trace of one step here")
parser.add_argument("--json", action="store_true", help="also print one JSON line of results")
args = parser.parse_args()

device = torch.device("cuda")
torch.manual_seed(0)


def matmul_ceiling(n=8192, iters=20):
    """Achievable dense bf16 TFLOP/s on this GPU at its current clocks/power cap."""
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


# kernel name -> category, first match wins
CATEGORIES = [
    ("attention", r"flash|fmha|sdpa|attention|efficient_attention"),
    ("moe (triton/stk)", r"grouped|fused_moe|sdd|dsd|dds|padded_|_moe|topology|histogram|sort|cumsum|gather|scatter"),
    ("matmul", r"gemm|cutlass|ampere_.*(h|s)\d+|sm80|sm86|xmma|cublas|matmul|mm_"),
    ("loss (cce)", r"cce|cross_entropy|lse|_ce_"),
    ("norm / elementwise", r"norm|elementwise|vectorized|triton_poi|triton_per|triton_red|unrolled|copy|fill|cat|where"),
    ("reduction / topk", r"reduce|topk|softmax|sigmoid|argmax|bitonic|radix"),
    ("comm / memcpy", r"nccl|memcpy|memset"),
]


def categorize(name):
    lname = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, lname):
            return cat
    return "other"


config = GPTConfig(
    sequence_len=args.max_seq_len,
    vocab_size=args.vocab_size,
    n_layer=args.depth,
    n_head=args.num_heads,
    n_kv_head=args.num_kv_heads,
    n_embd=args.model_dim,
    window_pattern=args.window_pattern,
    sliding_window=args.sliding_window,
    arch=args.arch,
    expert_sizes=[tuple(x) for x in args.expert_sizes],
    num_active_experts=args.num_active_experts,
    mlp_act=args.mlp_act,
    use_bias_balancing=args.use_bias_balancing,
    fp32_master_weights=args.fp32_master_weights,
    loss_impl=args.loss_impl,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
optimizers = model.setup_optimizers(
    unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
)
flops_per_token = model.estimate_flops()
n_params = sum(p.numel() for p in model.parameters())
if not args.no_compile:
    model = torch.compile(model, dynamic=False)

B, T = args.device_batch_size, args.max_seq_len
x = torch.randint(0, args.vocab_size, (B, T), device=device)
y = torch.randint(0, args.vocab_size, (B, T), device=device)


def step(events=None):
    if events:
        events[0].record()
    _, loss, _ = model(x, y)
    if events:
        events[1].record()
    loss.backward()
    if events:
        events[2].record()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    if events:
        events[3].record()


ceiling = matmul_ceiling()
for _ in range(args.warmup):
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

phase = defaultdict(float)
t0 = time.perf_counter()
for _ in range(args.steps):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    step(ev)
    torch.cuda.synchronize()
    phase["forward"] += ev[0].elapsed_time(ev[1])
    phase["backward"] += ev[1].elapsed_time(ev[2])
    phase["optimizer"] += ev[2].elapsed_time(ev[3])
step_s = (time.perf_counter() - t0) / args.steps
peak_mem = torch.cuda.max_memory_allocated() / 1e9

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()
if args.trace:
    prof.export_chrome_trace(args.trace)
by_cat, by_kernel = defaultdict(float), defaultdict(float)
for e in prof.key_averages():
    t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    if t <= 0:
        continue
    by_cat[categorize(e.key)] += t
    by_kernel[e.key] += t
gpu_total = sum(by_cat.values())

tokens = B * T
tflops = flops_per_token * tokens / step_s / 1e12
from nanochat.flash_attention import ATTN_IMPL
print(f"attention: {ATTN_IMPL} | window pattern {args.window_pattern} (S window {args.sliding_window})")
print(f"shape: d={args.model_dim} L={args.depth} experts={args.expert_sizes} K={args.num_active_experts} "
      f"{args.mlp_act} seq={T} batch={B} {'fp32-master' if args.fp32_master_weights else 'bf16'}")
print(f"params {n_params/1e6:.1f}M | step {step_s*1e3:.1f} ms | {tokens/step_s:,.0f} tok/s | "
      f"{tflops:.1f} TFLOP/s | MFU {100*tflops/ceiling:.1f}% of measured {ceiling:.1f} TFLOP/s | "
      f"peak mem {peak_mem:.2f} GB")
tot = sum(phase.values())
print("phases: " + " | ".join(f"{k} {v/args.steps:.1f} ms ({100*v/tot:.0f}%)" for k, v in phase.items()))
print("GPU time by kernel type (one step):")
for cat, t in sorted(by_cat.items(), key=lambda kv: -kv[1]):
    print(f"  {cat:20s} {t/1e3:8.2f} ms  {100*t/gpu_total:5.1f}%")
print(f"  {'(GPU busy)':20s} {gpu_total/1e3:8.2f} ms  of {step_s*1e3:.1f} ms wall")
print("top kernels:")
for k, t in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {t/1e3:7.2f} ms  {categorize(k):18s} {k[:90]}")
if args.json:
    print("JSON " + json.dumps({
        "args": vars(args), "params": n_params, "step_ms": step_s * 1e3,
        "tok_per_s": tokens / step_s, "tflops": tflops, "ceiling_tflops": ceiling,
        "mfu": tflops / ceiling, "peak_mem_gb": peak_mem,
        "phase_ms": {k: v / args.steps for k, v in phase.items()},
        "gpu_ms_by_category": {k: v / 1e3 for k, v in by_cat.items()},
        "gpu_busy_ms": gpu_total / 1e3,
    }))
