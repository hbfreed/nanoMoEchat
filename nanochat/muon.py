"""
Muon optimizer from Keller et al.
Also a lot of borrowing of ideas from modded-nanogpt.
"""
import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
import triton
import triton.language as tl

from nanochat.newton_schulz import CUBIC5, inv_norm_from_partials, newton_schulz
from nanochat.precision import apply_update_

@triton.jit
def _lerp(start, end, w, one_minus_w, SMALL: tl.constexpr):
    """torch.lerp's formula (ATen/native/Lerp.h), so results match it bitwise."""
    if SMALL:
        return start + w * (end - start)
    else:
        return end - (end - start) * one_minus_w


@triton.jit
def _nesterov_kernel(g, buf, scale, part, R, C, w1, omw1, w2, omw2, SMALL1: tl.constexpr,
                     SMALL2: tl.constexpr, QUANT: tl.constexpr, NORM: tl.constexpr, DIM: tl.constexpr,
                     WIDTH: tl.constexpr, P: tl.constexpr, NCB: tl.constexpr, BR: tl.constexpr,
                     BC: tl.constexpr):
    """Momentum and Nesterov on a BR x BC tile of the (R, C) matrices g and buf. With NORM, also
    each tile's sum of squares of the new g (as bf16, which Newton-Schulz reads) into
    part[expert, k], where experts are WIDTH-wide slices along DIM (see orthogonalize).
    With QUANT, buf is int8 with one fp32 scale per row per BC columns (scale[row, j]): it is
    dequantized, updated in fp32, used unrounded for Nesterov, and requantized linearly."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    rows = i * BR + tl.arange(0, BR)
    cols = j * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < C)
    offs = rows[:, None].to(tl.int64) * C + cols[None, :]
    graw = tl.load(g + offs, mask=mask, other=0.0)
    if QUANT:
        sc = tl.load(scale + rows * NCB + j, mask=rows < R, other=0.0)
        bv = tl.load(buf + offs, mask=mask, other=0).to(tl.float32) * sc[:, None]
        bv = _lerp(bv, graw.to(tl.float32), w1, omw1, SMALL1)
        amax = tl.max(tl.abs(bv), axis=1)
        sc = tl.where(amax > 0, amax / 127.0, 1.0)
        q = tl.floor(bv / sc[:, None] + 0.5)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(buf + offs, q.to(tl.int8), mask=mask)
        tl.store(scale + rows * NCB + j, sc, mask=rows < R)
        end = bv
    else:
        bv = tl.load(buf + offs, mask=mask, other=0.0).to(tl.float32)
        bv = _lerp(bv, graw.to(buf.dtype.element_ty).to(tl.float32), w1, omw1, SMALL1).to(buf.dtype.element_ty)
        tl.store(buf + offs, bv, mask=mask)
        end = bv.to(g.dtype.element_ty).to(tl.float32)
    gn = _lerp(graw.to(tl.float32), end, w2, omw2, SMALL2).to(g.dtype.element_ty)
    tl.store(g + offs, gn, mask=mask)
    if NORM:
        x = gn.to(tl.bfloat16).to(tl.float32)
        if DIM == 1:  # expert e owns columns [e * WIDTH, (e + 1) * WIDTH)
            e = (j * BC) // WIDTH
            k = i * (WIDTH // BC) + j % (WIDTH // BC)
        else:  # expert e owns rows [e * WIDTH, (e + 1) * WIDTH)
            e = (i * BR) // WIDTH
            k = (i % (WIDTH // BR)) * NCB + j
        tl.store(part + e * P + k, tl.sum(x * x))


def _lerp_args(w):
    w32 = np.float32(w)
    return float(w32), float(np.float32(1) - w32), bool(abs(w32) < 0.5)


def _fusable(g: Tensor, buf: Tensor) -> bool:
    return g.is_cuda and g.ndim == 2 and g.is_contiguous() and buf.is_contiguous() and g.shape == buf.shape


QUANT_BLOCK = 256  # momentum values per int8 scale, along a row


def int8_momentum(g: Tensor):
    """Zeroed int8 momentum for g, with its (rows, cdiv(cols, QUANT_BLOCK)) fp32 scales."""
    R, C = g.shape
    return (torch.zeros((R, C), dtype=torch.int8, device=g.device),
            torch.zeros((R, triton.cdiv(C, QUANT_BLOCK)), dtype=torch.float32, device=g.device))


def _nesterov_int8_torch(g: Tensor, buf: Tensor, scale: Tensor, momentum: float) -> None:
    """nesterov_ with int8 momentum, in torch (CPU or strided tensors); same math as the kernel."""
    R, C = g.shape
    pad = scale.size(1) * QUANT_BLOCK - C
    blocks = lambda t: torch.nn.functional.pad(t, (0, pad)).view(R, -1, QUANT_BLOCK)
    bv = blocks(buf.float()) * scale[..., None]
    bv = bv.lerp(blocks(g.float()), 1 - momentum)
    amax = bv.abs().amax(-1)
    sc = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.floor(bv / sc[..., None] + 0.5).clamp(-127, 127)
    buf.copy_(q.view(R, -1)[:, :C].to(torch.int8))
    scale.copy_(sc)
    g.copy_(g.float().lerp(bv.view(R, -1)[:, :C], momentum).to(g.dtype))


def nesterov_(g: Tensor, buf: Tensor, momentum: float, layout=None, BR=8, BC=256, scale=None):
    """buf.lerp_(g, 1 - momentum); g.lerp_(buf, momentum), in one pass over g and buf (two
    _foreach_lerp_ read and write both twice: 17.4 -> 11.5 ms on 36 bf16 512x65536 grads, RTX
    3090). Bitwise equal to the two lerps, also when buf and g differ in dtype (each is rounded
    to the other's dtype where the lerps convert it).

    With a per-expert layout (see orthogonalize) whose width is a multiple of the tile, it also
    returns 1/(|g_e| + eps) and its square per expert, from the same pass: the norm Newton-Schulz
    starts with, without reading g again (14.4 -> 11.8 ms for momentum and norms of those 36).
    Otherwise it returns None.

    An int8 buf (int8_momentum) is linearly quantized with one fp32 scale per row per 256
    columns, passed as scale: half the memory of bf16 momentum, as in Gupta et al., "Effective
    Quantization of Muon Optimizer States" (arXiv 2509.23106), whose linear 8-bit Muon matched
    full precision within 1-2% of validation loss up to 2.7B parameters."""
    quant = buf.dtype == torch.int8
    assert not quant or (scale is not None and BC == QUANT_BLOCK)
    w1, omw1, small1 = _lerp_args(1 - momentum)
    w2, omw2, small2 = _lerp_args(momentum)
    R, C = g.shape
    if g.numel() == 0:
        return None
    nrb, ncb = triton.cdiv(R, BR), triton.cdiv(C, BC)
    dim, width = layout if layout is not None else (0, 0)
    norm = layout is not None and (width % BC == 0 if dim == 1 else width % BR == 0)
    E, P = 1, 1
    if norm:
        E, P = (C // width, nrb * (width // BC)) if dim == 1 else (R // width, (width // BR) * ncb)
    part = torch.empty((E, P), dtype=torch.float32, device=g.device) if norm else g
    _nesterov_kernel[(nrb, ncb)](g, buf, scale if quant else g, part, R, C, w1, omw1, w2, omw2,
                                 SMALL1=small1, SMALL2=small2, QUANT=quant, NORM=norm, DIM=dim,
                                 WIDTH=width, P=P, NCB=ncb, BR=BR, BC=BC, num_warps=8)
    return inv_norm_from_partials(part) if norm else None


def zeropower_via_newtonschulz5(G: Tensor, steps: int, schedule: str = "quintic") -> Tensor:
    """Newton-Schulz orthogonalization of G (or a batch of matrices). On CUDA the Triton
    kernels in newton_schulz.py run it (474 -> 266 ms per per-expert Muon step at d=512 with
    128x512 experts); elsewhere the torch version below."""
    if G.is_cuda:
        return newton_schulz(G, steps, schedule=schedule)
    return _zeropower_torch(G, steps, schedule)


@torch.compile
def _zeropower_torch(G: Tensor, steps: int, schedule: str = "quintic") -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for i in range(steps):
        A = X @ X.mT
        if schedule == "cubic5":  # see newton_schulz.CUBIC5
            X = CUBIC5[i][0] * X + CUBIC5[i][1] * A @ X
            continue
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def _per_expert(t: Tensor, layout) -> Tensor:
    """(E, rows, cols) view of a per-expert matrix (see orthogonalize); no copy."""
    dim, width = layout
    return t.reshape(t.size(0), -1, width).transpose(0, 1) if dim == 1 else t.reshape(-1, width, t.size(1))


def _aspect(x: Tensor) -> float:
    return max(1.0, x.size(-2) / x.size(-1)) ** 0.5


def orthogonalize(g: Tensor, steps: int, layout=None, schedule: str = "quintic") -> Tensor:
    """Muon's update direction for g: Newton-Schulz orthogonalized, times the aspect-ratio
    scale sqrt(max(1, rows/cols)). layout=(dim, width) treats each width-sized slice of g
    along dim as its own matrix (one per MoE expert: w1/w3 are (n_embd, E*width) with
    dim=1, w2 is (E*width, n_embd) with dim=0) and orthogonalizes them as a batch."""
    x = g if layout is None else _per_expert(g, layout)
    if g.is_cuda:
        # the last product writes the scaled result straight into g's layout
        u = torch.empty(g.shape, dtype=torch.bfloat16, device=g.device)
        newton_schulz(x, steps, out=u if layout is None else _per_expert(u, layout), scale=_aspect(x),
                      schedule=schedule)
        return u
    u = zeropower_via_newtonschulz5(x, steps, schedule) * _aspect(x)
    if layout is None:
        return u
    return u.transpose(0, 1).reshape(g.shape) if layout[0] == 1 else u.reshape(g.shape)


def orthogonalize_add_(p: Tensor, g: Tensor, steps: int, alpha: float, layout=None,
                       stochastic: bool = False, inv_norm=None, schedule: str = "quintic") -> None:
    """p += alpha * orthogonalize(g, steps, layout). On CUDA the last Newton-Schulz product adds
    into p through its per-expert view, accumulating in fp32 and rounding once (no update
    tensor, no copy-back, scale or update passes); stochastic rounding takes the unfused path.
    inv_norm: per-matrix (s, s^2) from nesterov_, if it computed them."""
    if not (p.is_cuda and p.is_contiguous() and not stochastic and p.dtype in (torch.bfloat16, torch.float32)):
        apply_update_(p, orthogonalize(g, steps, layout, schedule), alpha, stochastic)
        return
    x = g if layout is None else _per_expert(g, layout)
    out = p if layout is None else _per_expert(p, layout)
    newton_schulz(x, steps, out=out, scale=alpha * _aspect(x), accumulate=True, inv_norm=inv_norm,
                  schedule=schedule)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
        stochastic_rounding: stochastically round bf16 weight updates.
        state_dtype: dtype for the momentum buffer (default: the grad's dtype); torch.int8
            stores it linearly quantized (see nesterov_), which needs nesterov=True.
        expert_layouts: {param: (dim, width)} for MoE expert matrices to orthogonalize
            per expert (see orthogonalize); other params are orthogonalized whole.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 stochastic_rounding=False, state_dtype=None, expert_layouts=None, ns_schedule="quintic"):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        ns_schedule=ns_schedule)
        self.stochastic_rounding = stochastic_rounding
        self.state_dtype = state_dtype
        self.expert_layouts = expert_layouts or {}
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grads = [p.grad for p in params]
            assert all(g is not None for g in grads)
            bufs = []
            for p, g in zip(params, grads):
                state = self.state[p]
                if "momentum_buffer" not in state:
                    if self.state_dtype == torch.int8:
                        state["momentum_buffer"], state["momentum_scale"] = int8_momentum(g)
                    else:
                        state["momentum_buffer"] = torch.zeros_like(g, dtype=self.state_dtype or g.dtype)
                bufs.append(state["momentum_buffer"])
            momentum = group["momentum"]
            inv_norms = {}
            if self.state_dtype == torch.int8:
                assert group["nesterov"], "int8 momentum is implemented for Nesterov momentum"
                for p, g, buf in zip(params, grads, bufs):
                    if _fusable(g, buf):
                        inv_norms[p] = nesterov_(g, buf, momentum, self.expert_layouts.get(p),
                                                 scale=self.state[p]["momentum_scale"])
                    else:
                        _nesterov_int8_torch(g, buf, self.state[p]["momentum_scale"], momentum)
                updates = grads
            elif group["nesterov"] and all(_fusable(g, b) for g, b in zip(grads, bufs)):
                for p, g, buf in zip(params, grads, bufs):
                    inv_norms[p] = nesterov_(g, buf, momentum, self.expert_layouts.get(p))
                updates = grads
            # one multi-tensor launch per op instead of a kernel per parameter, same lerp math
            elif all(b.dtype == g.dtype for b, g in zip(bufs, grads)):
                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                if group["nesterov"]:
                    torch._foreach_lerp_(grads, bufs, momentum)
                updates = grads if group["nesterov"] else bufs
            else:
                updates = []
                for g, buf in zip(grads, bufs):
                    buf.lerp_(g.to(buf.dtype), 1 - momentum)
                    updates.append(g.lerp_(buf.to(g.dtype), momentum) if group["nesterov"] else buf)
            self._orthogonalize_and_apply(params, updates, group["ns_steps"], -group["lr"], inv_norms,
                                          group["ns_schedule"])

    def _orthogonalize_and_apply(self, params, updates, steps, alpha, inv_norms, schedule):
        """p += alpha * orthogonalize(update) for each parameter. Plain matrices of one shape go
        through Newton-Schulz as one batch (each gets exactly the result it would get alone).
        Each update is applied as soon as it exists, so at most one batch of them is alive."""
        batches = {}
        for p, u in zip(params, updates):
            layout = self.expert_layouts.get(p)
            if layout is None:
                batches.setdefault(tuple(u.shape), []).append((p, u))
            else:
                orthogonalize_add_(p, u, steps, alpha, layout, self.stochastic_rounding, inv_norms.get(p),
                                   schedule)
        for batch in batches.values():
            stacked = orthogonalize(torch.stack([u for _, u in batch]), steps, schedule=schedule)
            for (p, _), u in zip(batch, stacked.unbind(0)):
                apply_update_(p, u, alpha, self.stochastic_rounding)


class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2D update via Newton–Schulz,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    Notes:
      * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        params like embeddings or scalars.
      * Momentum buffers are maintained only on the 'owner' rank for each parameter (rank chosen
        by block-cyclic assignment below). If you checkpoint optimizer state on a single rank,
        consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate
        momentum: momentum coefficient in [0,1)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        ns_steps: number of Newton–Schulz iterations for the orthogonalization
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 stochastic_rounding: bool = False, state_dtype=None, expert_layouts=None,
                 ns_schedule: str = "quintic"):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        ns_schedule=ns_schedule)
        self.stochastic_rounding = stochastic_rounding
        self.state_dtype = state_dtype
        self.expert_layouts = expert_layouts or {}
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        # Group all parameters by their shape
        shapes = sorted({p.shape for p in params}) # sort to ensure consistent / deterministic ordering
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Kick off all the reduce scatter operations to average up the gradients across all ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank
                # each rank stacks up its chunk of world_size params into a list
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # pad rs_input with the zero buffer to complete the group
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # the output buffer gets strided across the group based on the rank
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # reduce scatter the gradients within this group of world_size params
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Now each rank computes the update and gathers
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank # calculate the index of the param that this rank owns
                # Wait for the reduce scatter to complete
                all_reduce_futures[future_idx].wait() # possibly later we could use wait_any polling instead
                future_idx += 1
                # Owner computes the Muon update, result is in its param
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g, dtype=self.state_dtype or g.dtype)
                    buf: Tensor = state["momentum_buffer"]
                    if group["nesterov"] and _fusable(g, buf):
                        nesterov_(g, buf, group["momentum"])
                    else:
                        buf.lerp_(g.to(buf.dtype), 1.0 - group["momentum"])
                        g = g.lerp_(buf.to(g.dtype), group["momentum"]) if group["nesterov"] else buf
                    g = orthogonalize(g, group["ns_steps"], self.expert_layouts.get(p), group["ns_schedule"])
                    apply_update_(p, g, -group["lr"], self.stochastic_rounding)
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()


def _local(t: Tensor) -> Tensor:
    """The local shard of an FSDP2 (DTensor) param or grad; plain tensors pass through."""
    return t.to_local() if hasattr(t, "to_local") else t


class FSDPMuon(torch.optim.Optimizer):
    """
    Muon for FSDP2 (fully_shard) params, which each rank holds as a dim-0 shard and
    whose grads arrive already reduce-scattered. Same math as DistMuon:
      - momentum / Nesterov on each rank's shard (elementwise, so shards are exact)
      - per matrix, an owner rank (block-cyclic, as DistMuon) gathers the full
        Nesterov gradient with an all_to_all, runs Newton-Schulz, and sends each
        rank its rows of the update back with a second all_to_all
      - each rank applies its rows to its shard
    Momentum memory is sharded like the params. Shard sizes follow torch.chunk on dim 0,
    as FSDP2 does, so uneven splits (e.g. dim 0 not divisible by 3) are handled.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 stochastic_rounding: bool = False, state_dtype=None, expert_layouts=None,
                 ns_schedule: str = "quintic"):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        ns_schedule=ns_schedule)
        self.stochastic_rounding = stochastic_rounding
        self.state_dtype = state_dtype
        self.expert_layouts = expert_layouts or {}
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        world_size = dist.get_world_size()

        def expert_aligned(p):
            """Per-expert param sharded along its expert dim into whole experts on every
            rank: its update needs no communication at all."""
            layout = self.expert_layouts.get(p)
            if layout is None or not hasattr(p, "placements"):
                return False
            dim, width = layout
            sizes = [c.size(0) for c in torch.empty(p.size(dim)).chunk(world_size)]
            return p.placements[0].dim == dim and all(n % width == 0 for n in sizes)

        self.local_params = [p for p in params if expert_aligned(p)]
        gathered = [p for p in params if not expert_aligned(p)]
        assert all(p.placements[0].dim == 0 for p in gathered if hasattr(p, "placements"))
        shapes = sorted({tuple(p.shape) for p in gathered})
        param_groups = [dict(params=[p for p in gathered if tuple(p.shape) == s]) for s in shapes]
        if self.local_params:
            param_groups.append(dict(params=self.local_params, local=True))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        # 0. expert-aligned shards: whole experts locally, so no communication
        for group in self.param_groups:
            if not group.get("local"):
                continue
            for p in group["params"]:
                g = self._nesterov_(p, group)
                orthogonalize_add_(_local(p), g, group["ns_steps"], -group["lr"],
                                   self.expert_layouts[p], self.stochastic_rounding,
                                   schedule=group["ns_schedule"])
        # 1. momentum + Nesterov on local shards, then all_to_all the shards to each owner
        pending = []
        for group in self.param_groups:
            if group.get("local"):
                continue
            params = group["params"]
            # FSDP2 shards with torch.chunk, which can return fewer chunks; the rest are empty
            rows = [c.size(0) for c in torch.empty(params[0].size(0)).chunk(world_size)]
            rows += [0] * (world_size - len(rows))
            cols = params[0].size(1)
            for base_i in range(0, len(params), world_size):
                chunk = params[base_i:base_i + world_size]
                send = [self._nesterov_(p, group).contiguous() for p in chunk]
                dtype, device = send[0].dtype, send[0].device
                send += [torch.empty(0, cols, dtype=dtype, device=device)] * (world_size - len(chunk))
                owns = base_i + rank < len(params)
                recv = [torch.empty(r if owns else 0, cols, dtype=dtype, device=device) for r in rows]
                work = dist.all_to_all(recv, send, async_op=True)
                pending.append((group, chunk, rows, recv, work))
        # 2. owners orthogonalize, then all_to_all each rank its rows of the update
        updates = []
        for group, chunk, rows, recv, work in pending:
            work.wait()
            dtype, device, cols = recv[0].dtype, recv[0].device, recv[0].size(1)
            if rank < len(chunk):
                full = torch.cat(recv)
                u = orthogonalize(full, group["ns_steps"], self.expert_layouts.get(chunk[rank]),
                                  group["ns_schedule"])
                send = [s.contiguous() for s in u.split(rows)]
            else:
                send = [torch.empty(0, cols, dtype=dtype, device=device) for _ in rows]
            recv_u = [torch.empty(rows[rank], cols, dtype=dtype, device=device) if j < len(chunk)
                      else torch.empty(0, cols, dtype=dtype, device=device) for j in range(world_size)]
            work = dist.all_to_all(recv_u, send, async_op=True)
            updates.append((group, chunk, recv_u, work))
        # 3. apply each rank's rows to its shards
        for group, chunk, recv_u, work in updates:
            work.wait()
            for p, u in zip(chunk, recv_u):
                apply_update_(_local(p), u, -group["lr"], self.stochastic_rounding)

    def _nesterov_(self, p, group):
        """Momentum update on p's local shard; returns the (Nesterov) update direction."""
        g = _local(p.grad)
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g, dtype=self.state_dtype or g.dtype)
        buf = state["momentum_buffer"]
        if group["nesterov"] and _fusable(g, buf):
            nesterov_(g, buf, group["momentum"])
            return g
        buf.lerp_(g.to(buf.dtype), 1.0 - group["momentum"])
        return g.lerp_(buf.to(g.dtype), group["momentum"]) if group["nesterov"] else buf.to(g.dtype)
