"""Variable-size expert topology construction for MoE with different expert capacities."""

from typing import Any
import torch

# This will be imported after building the extension
try:
    import nanomoe_ops as ops
except ModuleNotFoundError:
    ops = None
    print(
        "Warning: nanomoe_ops not built. Run 'python setup.py install' in nanoMOE directory."
    )


class TopologyVarOp(torch.autograd.Function):
    """Topology construction for variable-size experts.

    Unlike the standard topology op which assumes uniform expert sizes,
    this version accepts per-expert block counts and offsets.
    """

    @staticmethod
    def forward(
        ctx: Any,
        padded_bins: torch.Tensor,
        expert_block_counts: torch.Tensor,  # [num_experts] - blocks per expert
        expert_block_offsets: torch.Tensor,  # [num_experts+1] - cumulative offsets
        block_size: int,
        output_block_rows: int,
    ):
        """
        Args:
            padded_bins: Cumulative token counts per expert (padded to block_size)
            expert_block_counts: Number of blocks in each expert's FFN (variable!)
            expert_block_offsets: Cumulative block offsets for indexing into weight matrix
            block_size: Block size for sparse matrix
            output_block_rows: Total number of token blocks across all experts

        Returns:
            Column indices for sparse topology matrix
        """
        # Calculate total output size and cumulative offsets using TENSOR OPS
        # Avoids Python loop with .item() calls that cause GPU syncs
        # (Reduces from 3N syncs to 1 sync for N experts)

        # Compute token blocks per expert: diff of padded_bins / block_size
        prepend = padded_bins.new_zeros(1)
        bin_sizes = torch.diff(padded_bins, prepend=prepend)
        expert_token_blocks = (bin_sizes // block_size).to(torch.int32)

        # Compute output size per expert (token_blocks * expert_blocks)
        expert_output_sizes = expert_token_blocks * expert_block_counts.to(torch.int32)

        # Single sync to get total (unavoidable for allocation)
        total_nnz = expert_output_sizes.sum().item()

        if total_nnz == 0:
            return torch.empty(0, dtype=torch.int16, device=padded_bins.device)

        # Compute offsets on GPU - cumsum gives [a, a+b, a+b+c, ...], we need [0, a, a+b, ...]
        cumsum = expert_output_sizes.cumsum(0)
        output_offsets_tensor = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=padded_bins.device),
            cumsum[:-1]
        ]).to(torch.int32)

        out = torch.empty(
            total_nnz,
            dtype=torch.int16,
            device=padded_bins.device,
        )

        if ops is not None:
            ops.indices_variable(
                padded_bins,
                expert_block_counts,
                expert_block_offsets,
                output_offsets_tensor,  # NEW: pass cumulative offsets
                block_size,
                output_block_rows,
                out,
            )
        else:
            raise RuntimeError("nanomoe_ops not available. Build the extension first.")

        return out


def topology_var(
    padded_bins: torch.Tensor,
    expert_block_counts: torch.Tensor,
    expert_block_offsets: torch.Tensor,
    block_size: int,
    output_block_rows: int,
) -> torch.Tensor:
    """Construct variable-size expert topology.

    Usage:
        column_indices = topology_var(
            padded_bins,           # From ops.inclusive_cumsum(tokens_per_expert_padded)
            expert_block_counts,   # [8] for 8 experts, e.g., [16, 32, 16, ...]
            expert_block_offsets,  # [9] cumulative, e.g., [0, 16, 48, 64, ...]
            block_size=128,
            output_block_rows=total_token_blocks
        )
    """
    return TopologyVarOp.apply(
        padded_bins,
        expert_block_counts,
        expert_block_offsets,
        block_size,
        output_block_rows,
    )

