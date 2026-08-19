# Persistent grouped GEMM for Ampere (SM86), ported from
# meta-pytorch/applied-ai (Hopper TMA-based kernel).
#
# Changes from upstream:
#   - TMA removed entirely (SM86 has no TMA hardware)
#   - Regular tl.load/tl.store with pointer arithmetic + masks
#   - Autotuning configs adjusted for SM86 SMEM limits (100KB)
#   - FP8 variant removed (focus on BF16)
#
# Kernel layout:
#   x:       [M, K]    — all expert inputs concatenated (row-major)
#   w:       [N*G, K]  — all expert weights packed (expert g at rows [g*N:(g+1)*N])
#   output:  [M, N]    — all expert outputs concatenated
#   m_sizes: [G]       — number of tokens per expert (int32)
#
# The weight layout is [N, K] per expert (transposed), and the kernel computes
# output_tile = a_tile @ b_tile.T for each (M, K) x (N, K) -> (M, N) tile.

import torch
import triton
import triton.language as tl
from triton.runtime import driver


# ---------------------------------------------------------------------------
# Autotuning configs for SM86 (RTX 3090, 100KB SMEM)
# SMEM requirement: (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtype_bytes
# ---------------------------------------------------------------------------
_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
        },
        num_stages=ns,
        num_warps=nw,
    )
    for bm in [32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [32, 64]
    for ns in [2, 3]
    for nw in [4, 8]
]


def _early_config_prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    dtsize = named_args["c_ptr"].element_size()
    pruned = []
    for config in configs:
        kw = config.kwargs
        bm = kw["BLOCK_SIZE_M"]
        bn = kw["BLOCK_SIZE_N"]
        bk = kw["BLOCK_SIZE_K"]
        ns = config.num_stages

        G = named_args["G"]
        M = named_args["M_BUCKET"]
        N = named_args["N"]
        K = named_args["K"]

        # 1. SMEM check (conservative — leave headroom for register spills)
        max_smem = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        required = (bm + bn) * bk * ns * dtsize
        if required > max_smem * 0.9:
            continue

        # 2. K must be divisible by BLOCK_K
        if K % bk != 0:
            continue

        # 3. N must be divisible by BLOCK_N (avoid partial N tiles in B)
        if N % bn != 0:
            continue

        # 4. Don't use oversized M tiles for small per-expert M
        m_per_group = max(1, M // G)
        if bm > 64 and bm > m_per_group * 2:
            continue

        pruned.append(config)
    return pruned


@triton.autotune(
    configs=_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": _early_config_prune},
    use_cuda_graph=True,
)
@triton.jit
def _kernel_grouped_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    # Persistent scheduling: each SM gets a unique ID and loops through
    # the global tile worklist, advancing by NUM_SMS each iteration.
    tidx = tl.program_id(0)
    dtype = c_ptr.dtype.element_ty

    M_end_offset = 0
    iterated_tiles = 0

    for g in tl.range(G):
        # Advance to this group's region in A / output
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            N_start_offset = g * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            # Claim tiles belonging to this SM
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # M-first ordering: consecutive tiles share B rows in L2
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                # -- K-loop: accumulate in FP32 --
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                # a: [M, K] — row-major
                a_ptrs = (
                    a_ptr
                    + (M_start_offset + offs_am[:, None]) * K
                    + offs_k[None, :]
                )
                # b: [N*G, K] — expert g at rows [g*N : (g+1)*N]
                b_ptrs = (
                    b_ptr
                    + (N_start_offset + offs_bn[:, None]) * K
                    + offs_k[None, :]
                )

                tl.static_assert(K % BLOCK_SIZE_K == 0)
                for _k in range(0, K, BLOCK_SIZE_K):
                    a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size, other=0.0)
                    accumulator += tl.dot(a, b.T)
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K

                # -- Store output tile --
                c = accumulator.to(dtype)
                c_ptrs = (
                    c_ptr
                    + (M_start_offset + offs_am[:, None]) * N
                    + offs_bn[None, :]
                )
                mask = (offs_am[:, None] < m_size) & (offs_bn[None, :] < n_size)
                tl.store(c_ptrs, c, mask=mask)

                # Advance to next tile for this SM
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# ---------------------------------------------------------------------------
# dW kernel: dW_g = dY_g.T @ X_g  (jagged contraction along M)
#
# Output is per-expert [N, K], packed as [N*G, K].
# Both dY and X are jagged along M (grouped by expert).
# The inner loop contracts over M_g (variable per expert).
# ---------------------------------------------------------------------------
_DW_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "BLOCK_SIZE_M": bm},
        num_stages=ns, num_warps=nw,
    )
    for bn in [32, 64, 128]
    for bk in [32, 64, 128]
    for bm in [32, 64]
    for ns in [2, 3]
    for nw in [4, 8]
]


def _early_config_prune_dw(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    dtsize = named_args["dw_ptr"].element_size()
    N = named_args["N"]
    K = named_args["K"]
    pruned = []
    for config in configs:
        kw = config.kwargs
        bn, bk, bm = kw["BLOCK_SIZE_N"], kw["BLOCK_SIZE_K"], kw["BLOCK_SIZE_M"]
        ns = config.num_stages
        max_smem = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        required = (bn + bk) * bm * ns * dtsize
        if required > max_smem * 0.9:
            continue
        if N % bn != 0:
            continue
        if K % bk != 0:
            continue
        pruned.append(config)
    return pruned


@triton.autotune(
    configs=_DW_CONFIGS,
    key=["G", "N", "K"],
    prune_configs_by={"early_config_prune": _early_config_prune_dw},
    use_cuda_graph=True,
)
@triton.jit
def _kernel_grouped_gemm_dw(
    dy_ptr,   # [M_total, N]  — grad output, jagged by expert
    x_ptr,    # [M_total, K]  — input, jagged by expert
    dw_ptr,   # [N*G, K]      — grad weight output, packed per expert
    m_sizes,        # [G] — token counts (sorted descending)
    m_offsets,      # [G] — cumulative start offset (matches sorted order)
    output_order,   # [G] — maps sorted index → original expert index for output
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # tile size along contraction (M) dim
) -> None:
    tidx = tl.program_id(0)
    dtype = dw_ptr.dtype.element_ty

    # Output tiles are over [N, K] per expert — both fixed dimensions.
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tiles_per_expert = num_n_tiles * num_k_tiles
    total_tiles = tiles_per_expert * G

    while tidx < total_tiles:
        # Which expert (in sorted order) and which output tile?
        sorted_g = tidx // tiles_per_expert
        tile_in_expert = tidx % tiles_per_expert
        tile_n_idx = tile_in_expert % num_n_tiles
        tile_k_idx = tile_in_expert // num_n_tiles

        m_size = tl.load(m_sizes + sorted_g)
        m_offset = tl.load(m_offsets + sorted_g)
        # Original expert index for writing output to the correct location
        orig_g = tl.load(output_order + sorted_g)

        offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Accumulate dW[n, k] = sum_m dY[m, n] * X[m, k]
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

        offs_m = tl.arange(0, BLOCK_SIZE_M)
        # dy: [M_total, N], row m has stride N
        dy_ptrs = dy_ptr + (m_offset + offs_m[:, None]) * N + offs_n[None, :]
        # x: [M_total, K], row m has stride K
        x_ptrs = x_ptr + (m_offset + offs_m[:, None]) * K + offs_k[None, :]

        # Inner loop over M_g (jagged) in chunks of BLOCK_SIZE_M
        # tl.range enables software pipelining: prefetch next tile while
        # computing current tile's dot product.
        num_m_iters = tl.cdiv(m_size, BLOCK_SIZE_M)
        for m_iter in tl.range(0, num_m_iters, 1, num_stages=2):
            m = m_iter * BLOCK_SIZE_M
            mask_m = offs_m[:, None] < (m_size - m)
            dy_block = tl.load(dy_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M, BLOCK_N]
            x_block = tl.load(x_ptrs, mask=mask_m, other=0.0)    # [BLOCK_M, BLOCK_K]
            # dW += dY.T @ X = [BLOCK_N, BLOCK_M] @ [BLOCK_M, BLOCK_K]
            accumulator += tl.dot(dy_block.T, x_block)
            dy_ptrs += BLOCK_SIZE_M * N
            x_ptrs += BLOCK_SIZE_M * K

        # Store to original expert position (not sorted position)
        dw_tile = accumulator.to(dtype)
        dw_ptrs = dw_ptr + (orig_g * N + offs_n[:, None]) * K + offs_k[None, :]
        tl.store(dw_ptrs, dw_tile)

        tidx += NUM_SMS


def _launch_grouped_gemm_dw(grad_output, x, m_sizes, G, N, K):
    """Launch dW kernel: computes dY.T @ X per expert.

    Sorts experts by descending m_size for better SM load balancing
    (largest experts processed first, avoids tail latency).

    grad_output: [M, N]  — jagged by expert
    x:           [M, K]  — jagged by expert
    m_sizes:     [G]     — int32 token counts
    Returns:     [N*G, K]
    """
    dw = torch.empty(N * G, K, device=x.device, dtype=x.dtype)
    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count

    # Sort experts by descending token count for load balancing
    sorted_indices = m_sizes.argsort(descending=True)
    sorted_m_sizes = m_sizes[sorted_indices]
    # Compute offsets in the ORIGINAL (unsorted) token order
    m_offsets = torch.zeros_like(m_sizes)
    m_offsets[1:] = m_sizes[:-1].cumsum(0)
    sorted_m_offsets = m_offsets[sorted_indices]

    _kernel_grouped_gemm_dw[(NUM_SMS,)](
        grad_output, x, dw, sorted_m_sizes, sorted_m_offsets, sorted_indices,
        G, N, K, NUM_SMS,
    )
    return dw


# ---------------------------------------------------------------------------
# Python wrapper (low-level)
# ---------------------------------------------------------------------------
def _launch_grouped_gemm(a, b, m_sizes, G, N_out, K_contract):
    """Launch the persistent kernel: computes a @ b.T per expert.

    a: [M, K_contract]  — jagged rows, grouped by expert
    b: [N_out*G, K_contract] — expert g at rows [g*N_out : (g+1)*N_out]
    Returns: [M, N_out]
    """
    M = a.shape[0]
    y = torch.empty((M, N_out), device=a.device, dtype=a.dtype)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    M_BUCKET = triton.next_power_of_2(M)

    _kernel_grouped_gemm[(NUM_SMS,)](
        a, b, y, m_sizes,
        G, M_BUCKET, N_out, K_contract, NUM_SMS,
    )
    return y


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    """Persistent grouped GEMM for SM86.

    Computes Y_g = X_g @ W_g.T for each expert g.

    Args:
        x:       [M, K]    input (all experts concatenated, bf16)
        w:       [N*G, K]  weights (expert g at rows [g*N:(g+1)*N], bf16)
        m_sizes: [G]       tokens per expert (int32)

    Returns:
        y:       [M, N]    output (bf16)
    """
    assert x.is_contiguous()
    assert w.is_contiguous()

    G = m_sizes.shape[0]
    M, K = x.shape
    N = w.shape[0] // G
    return _launch_grouped_gemm(x, w, m_sizes, G, N, K)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------
class GroupedGemmFn(torch.autograd.Function):
    """Differentiable grouped GEMM: Y = X @ W.T per expert.

    Forward and dX use the persistent kernel. dW uses a loop for now —
    it's a per-expert reduction (jagged contraction dim) that needs a
    different tiling strategy. Still correct, and dW is typically
    overlapped with other backward computation in practice.
    """

    @staticmethod
    def forward(ctx, x, w, m_sizes):
        ctx.save_for_backward(x, w, m_sizes)
        return grouped_gemm(x, w, m_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, m_sizes = ctx.saved_tensors
        G = m_sizes.shape[0]
        M, K = x.shape
        N = w.shape[0] // G

        # -- dX = dY @ W per expert: [M_g, N] @ [N, K] = [M_g, K] --
        # Rewrite as dY @ (W_repack).T using our kernel:
        #   W is [N*G, K], reshape to [G, N, K], permute to [G, K, N],
        #   flatten to [K*G, N].  Then kernel does dY @ W_repack.T
        #   = [M_g, N] @ [N, K] = [M_g, K].
        w_for_dx = w.view(G, N, K).transpose(1, 2).contiguous().view(K * G, N)
        grad_x = _launch_grouped_gemm(grad_output, w_for_dx, m_sizes, G, K, N)

        # -- dW = dY.T @ X per expert: [N, M_g] @ [M_g, K] = [N, K] --
        # Persistent kernel with variable contraction along M_g.
        grad_w = _launch_grouped_gemm_dw(grad_output, x, m_sizes, G, N, K)

        return grad_x, grad_w, None


def grouped_gemm_autograd(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> torch.Tensor:
    """Differentiable persistent grouped GEMM.

    Same interface as grouped_gemm() but supports autograd.
    """
    return GroupedGemmFn.apply(x, w, m_sizes)
