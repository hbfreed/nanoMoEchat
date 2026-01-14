"""
CUDA event timing for MoE MLP internals on a single GPU.

Example:
  python -m scripts.time_moemlp --device=cuda:0 --batch_size=10 --seq_len=1024 --iters=10
"""

import argparse
from contextlib import nullcontext

import torch
from einops import rearrange

import stk
from megablocks import ops
from megablocks.layers.relu_squared import relu_squared
from nanochat.gpt import GPTConfig, MoEMLP
from nanochat.topology_var import topology_var


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--num_active_experts", type=int, default=8)
    parser.add_argument("--expert_sizes", type=str, default="64x256")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--autocast", action="store_true", default=True)
    parser.add_argument("--measure_backward", action="store_true", default=False)
    parser.add_argument("--sweep", action="store_true", default=False)
    parser.add_argument("--batch_sizes", type=str, default="")
    parser.add_argument("--seq_lens", type=str, default="")
    return parser.parse_args()


def parse_expert_sizes(spec: str) -> list[tuple[int, int]]:
    # Format: "64x256,8x512" -> [(64,256),(8,512)]
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        count_str, size_str = part.split("x")
        out.append((int(count_str), int(size_str)))
    return out


def build_moemlp(args: argparse.Namespace, device: torch.device) -> MoEMLP:
    expert_sizes = parse_expert_sizes(args.expert_sizes)
    cfg = GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=65536,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=args.model_dim,
        num_active_experts=args.num_active_experts,
        expert_sizes=expert_sizes,
        use_moe=True,
    )
    return MoEMLP(cfg).to(device)


def time_op(label, fn, events):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    events[label] = (start, end)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("CUDA device required for event timing.")
    torch.cuda.set_device(device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.autocast
        else nullcontext()
    )

    def run_once(mlp: MoEMLP, x: torch.Tensor) -> dict[str, float]:
        events = {}
        with autocast_ctx:
            x_flat = rearrange(
                x, "batch_size seq_len n_embd -> (batch_size seq_len) n_embd "
            )
            router_logits = time_op("router_linear", lambda: mlp.router(x_flat), events)
            router_probs = time_op(
                "router_sigmoid",
                lambda: torch.sigmoid(router_logits.to(torch.float32)),
                events,
            )
            top_k_weights, selected_experts = time_op(
                "topk",
                lambda: torch.topk(router_probs, mlp.num_active_experts, dim=-1),
                events,
            )
            top_k_weights = time_op(
                "topk_norm",
                lambda: top_k_weights
                / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20),
                events,
            )
            top_k_weights = top_k_weights.to(x.dtype)
            top_k_weights_flat = rearrange(top_k_weights, "... -> (...)")
            selected_experts_flat = rearrange(selected_experts, "... -> (...)")

            # _sort_tokens_by_expert breakdown
            bin_ids, indices = time_op(
                "sort_tokens.sort",
                lambda: ops.sort(selected_experts_flat, mlp.sort_end_bit),
                events,
            )
            tokens_per_expert = time_op(
                "sort_tokens.histogram",
                lambda: ops.histogram(selected_experts_flat, mlp.num_experts),
                events,
            )

            # bins for gather/scatter
            bins = time_op(
                "bins_cumsum",
                lambda: ops.inclusive_cumsum(tokens_per_expert, 0).contiguous(),
                events,
            )

            # _create_topology breakdown
            padded_tokens_per_expert = time_op(
                "topology.round_up",
                lambda: ops.round_up(tokens_per_expert, mlp.block_size),
                events,
            )
            padded_bins = time_op(
                "topology.padded_bins_cumsum",
                lambda: ops.inclusive_cumsum(padded_tokens_per_expert, 0).contiguous(),
                events,
            )
            padded_tokens = time_op(
                "topology.clamp_min",
                lambda: padded_bins[-1].clamp_min(mlp.block_size),
                events,
            )
            block_rows = time_op(
                "topology.block_rows",
                lambda: padded_tokens // mlp.block_size,
                events,
            )
            column_indices = time_op(
                "topology.topology_var",
                lambda: topology_var(
                    padded_bins,
                    mlp.expert_size_blocks,
                    mlp.expert_block_offsets,
                    mlp.block_size,
                    block_rows,
                ),
                events,
            )

            # Reconstruct the rest of topology with explicit sub-ops to time
            expert_token_blocks = time_op(
                "topology.expert_token_blocks",
                lambda: padded_tokens_per_expert // mlp.block_size,
                events,
            )
            repeated_sizes = time_op(
                "topology.repeat_interleave",
                lambda: torch.repeat_interleave(
                    mlp.expert_size_blocks, expert_token_blocks
                ),
                events,
            )
            offsets = time_op(
                "topology.offsets_cumsum",
                lambda: torch.cat(
                    [repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)]
                ),
                events,
            )
            column_indices_i32 = time_op(
                "topology.column_i32",
                lambda: column_indices.to(torch.int32),
                events,
            )
            offsets_i32 = time_op(
                "topology.offsets_i32",
                lambda: offsets.to(torch.int32),
                events,
            )
            shape = (padded_tokens, mlp.total_expert_width)
            num_blocks = column_indices_i32.numel()
            data_placeholder = torch.empty(
                num_blocks,
                mlp.block_size,
                mlp.block_size,
                dtype=x.dtype,
                device="meta",
            )
            row_indices = time_op(
                "topology.row_indices",
                lambda: stk.ops.row_indices(
                    shape, data_placeholder, offsets_i32, column_indices_i32
                ),
                events,
            )
            row_indices = time_op(
                "topology.row_indices_i32",
                lambda: row_indices.to(torch.int32),
                events,
            )

            # sparse transpose breakdown
            _, gather_indices = time_op(
                "transpose.sort",
                lambda: ops.sort(column_indices_i32, mlp.transpose_sort_end_bit),
                events,
            )
            column_indices_t = time_op(
                "transpose.gather",
                lambda: row_indices.gather(0, gather_indices.long()),
                events,
            )
            block_offsets_t = time_op(
                "transpose.block_offsets",
                lambda: gather_indices.int(),
                events,
            )
            nnz_per_column = time_op(
                "transpose.histogram",
                lambda: ops.histogram(column_indices_i32, mlp.total_expert_width // mlp.block_size),
                events,
            )
            nnz_per_column = time_op(
                "transpose.cumsum",
                lambda: ops.inclusive_cumsum(nnz_per_column, 0),
                events,
            )
            if nnz_per_column.dim() == 0:
                nnz_per_column = nnz_per_column.unsqueeze(0)
            offsets_t = time_op(
                "transpose.offsets_cat",
                lambda: torch.cat(
                    [torch.zeros((1,), dtype=torch.int32, device=row_indices.device), nnz_per_column]
                ),
                events,
            )

            topology = stk.Matrix(
                shape,
                data_placeholder,
                row_indices,
                column_indices_i32,
                offsets_i32,
                column_indices_t.to(torch.int32),
                offsets_t.to(torch.int32),
                block_offsets_t.to(torch.int32),
            )
            x_permuted = time_op(
                "padded_gather",
                lambda: ops.padded_gather(
                    x_flat,
                    indices,
                    bin_ids,
                    bins,
                    padded_bins,
                    mlp.num_active_experts,
                ),
                events,
            )
            x_permuted = time_op(
                "sdd", lambda: stk.ops.sdd(x_permuted, mlp.w1, topology), events
            )
            x_permuted = time_op("relu_squared", lambda: relu_squared(x_permuted), events)
            x_permuted = time_op(
                "dsd", lambda: stk.ops.dsd(x_permuted, mlp.w2), events
            )
            x_permuted = time_op(
                "padded_scatter",
                lambda: ops.padded_scatter(
                    x_permuted,
                    indices,
                    bin_ids,
                    top_k_weights_flat,
                    bins,
                    padded_bins,
                    mlp.num_active_experts,
                ),
                events,
            )
            if args.measure_backward:
                loss = x_permuted.float().pow(2).mean()
        if args.measure_backward:
            bwd_events = {}
            _ = time_op("backward_total", lambda: loss.backward(), bwd_events)
            events.update(bwd_events)
        torch.cuda.synchronize()
        return {
            name: start.elapsed_time(end) for name, (start, end) in events.items()
        }

    def measure(args_for_run: argparse.Namespace) -> dict[str, float]:
        mlp = build_moemlp(args_for_run, device)
        mlp.train()
        dtype = torch.bfloat16
        x = torch.randn(
            args_for_run.batch_size,
            args_for_run.seq_len,
            args_for_run.model_dim,
            device=device,
            dtype=dtype,
        )
        for _ in range(args_for_run.warmup_steps):
            run_once(mlp, x)
            mlp.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for _ in range(args_for_run.iters):
            times = run_once(mlp, x)
            mlp.zero_grad(set_to_none=True)
            for k, v in times.items():
                totals[k] = totals.get(k, 0.0) + v
        return {k: v / args_for_run.iters for k, v in totals.items()}

    if args.sweep:
        batch_sizes = (
            [int(x) for x in args.batch_sizes.split(",") if x.strip()]
            if args.batch_sizes
            else [args.batch_size]
        )
        seq_lens = (
            [int(x) for x in args.seq_lens.split(",") if x.strip()]
            if args.seq_lens
            else [args.seq_len]
        )
        print("batch_size,seq_len,total_ms,create_topology_ms,sdd_ms,dsd_ms,gather_ms,scatter_ms,relu_ms")
        for bs in batch_sizes:
            for sl in seq_lens:
                args_run = argparse.Namespace(**vars(args))
                args_run.batch_size = bs
                args_run.seq_len = sl
                avg = measure(args_run)
                total = sum(avg.values())
                def get(k):
                    return avg.get(k, 0.0)
                create_topology_ms = (
                    get("topology.topology_var")
                    + get("topology.repeat_interleave")
                    + get("topology.offsets_cumsum")
                    + get("topology.row_indices")
                    + get("topology.round_up")
                    + get("topology.padded_bins_cumsum")
                    + get("transpose.sort")
                    + get("transpose.cumsum")
                    + get("transpose.histogram")
                    + get("transpose.gather")
                    + get("transpose.offsets_cat")
                    + get("topology.expert_token_blocks")
                    + get("topology.clamp_min")
                    + get("topology.block_rows")
                    + get("topology.column_i32")
                    + get("topology.offsets_i32")
                    + get("topology.row_indices_i32")
                    + get("transpose.block_offsets")
                )
                print(
                    f"{bs},{sl},{total:.3f},"
                    f"{create_topology_ms:.3f},"
                    f"{get('sdd'):.3f},{get('dsd'):.3f},"
                    f"{get('padded_gather'):.3f},{get('padded_scatter'):.3f},"
                    f"{get('relu_squared'):.3f}"
                )
        return

    avg = measure(args)
    total = sum(avg.values())
    print("CUDA event timings (ms):")
    for name, ms in sorted(avg.items(), key=lambda x: x[1], reverse=True):
        pct = (ms / total * 100) if total > 0 else 0.0
        print(f"{name:20s} {ms:8.3f} ms  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
