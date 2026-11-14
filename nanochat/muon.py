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

from nanochat.newton_schulz import newton_schulz
from nanochat.precision import apply_update_

@triton.jit
def _lerp(start, end, w, one_minus_w, SMALL: tl.constexpr):
    """torch.lerp's formula (ATen/native/Lerp.h), so results match it bitwise."""
    if SMALL:
        return start + w * (end - start)
    else:
        return end - (end - start) * one_minus_w


@triton.jit
def _nesterov_kernel(g, buf, n, w1, omw1, w2, omw2, SMALL1: tl.constexpr, SMALL2: tl.constexpr,
                     BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    graw = tl.load(g + offs, mask=mask)
    gv = graw.to(tl.float32)
    bv = tl.load(buf + offs, mask=mask).to(tl.float32)
    bv = _lerp(bv, graw.to(buf.dtype.element_ty).to(tl.float32), w1, omw1, SMALL1).to(buf.dtype.element_ty)
    tl.store(buf + offs, bv, mask=mask)
    end = bv.to(g.dtype.element_ty).to(tl.float32)
    tl.store(g + offs, _lerp(gv, end, w2, omw2, SMALL2).to(g.dtype.element_ty), mask=mask)


def _lerp_args(w):
    w32 = np.float32(w)
    return float(w32), float(np.float32(1) - w32), bool(abs(w32) < 0.5)


def _fusable(g: Tensor, buf: Tensor) -> bool:
    return g.is_cuda and g.is_contiguous() and buf.is_contiguous() and g.shape == buf.shape


def nesterov_(g: Tensor, buf: Tensor, momentum: float) -> None:
    """buf.lerp_(g, 1 - momentum); g.lerp_(buf, momentum), in one pass over g and buf (two
    _foreach_lerp_ read and write both twice: 17.4 -> 11.5 ms on 36 bf16 512x65536 grads, RTX
    3090). Bitwise equal to the two lerps, also when buf
    and g differ in dtype (each is rounded to the other's dtype where the lerps convert it)."""
    assert g.is_contiguous() and buf.is_contiguous() and g.shape == buf.shape
    w1, omw1, small1 = _lerp_args(1 - momentum)
    w2, omw2, small2 = _lerp_args(momentum)
    n, BLOCK = g.numel(), 2048
    if n == 0:
        return
    _nesterov_kernel[(triton.cdiv(n, BLOCK),)](g, buf, n, w1, omw1, w2, omw2, SMALL1=small1,
                                               SMALL2=small2, BLOCK=BLOCK, num_warps=8)


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Newton-Schulz orthogonalization of G (or a batch of matrices). On CUDA the Triton
    kernels in newton_schulz.py run it (474 -> 266 ms per per-expert Muon step at d=512 with
    128x512 experts); elsewhere the torch version below."""
    if G.is_cuda:
        return newton_schulz(G, steps)
    return _zeropower_torch(G, steps)


@torch.compile
def _zeropower_torch(G: Tensor, steps: int) -> Tensor:
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
    for _ in range(steps):
        A = X @ X.mT
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


def orthogonalize(g: Tensor, steps: int, layout=None) -> Tensor:
    """Muon's update direction for g: Newton-Schulz orthogonalized, times the aspect-ratio
    scale sqrt(max(1, rows/cols)). layout=(dim, width) treats each width-sized slice of g
    along dim as its own matrix (one per MoE expert: w1/w3 are (n_embd, E*width) with
    dim=1, w2 is (E*width, n_embd) with dim=0) and orthogonalizes them as a batch."""
    x = g if layout is None else _per_expert(g, layout)
    if g.is_cuda:
        # the last product writes the scaled result straight into g's layout
        u = torch.empty(g.shape, dtype=torch.bfloat16, device=g.device)
        newton_schulz(x, steps, out=u if layout is None else _per_expert(u, layout), scale=_aspect(x))
        return u
    u = zeropower_via_newtonschulz5(x, steps) * _aspect(x)
    if layout is None:
        return u
    return u.transpose(0, 1).reshape(g.shape) if layout[0] == 1 else u.reshape(g.shape)


def orthogonalize_add_(p: Tensor, g: Tensor, steps: int, alpha: float, layout=None,
                       stochastic: bool = False) -> None:
    """p += alpha * orthogonalize(g, steps, layout). On CUDA the last Newton-Schulz product adds
    into p through its per-expert view, accumulating in fp32 and rounding once (no update
    tensor, no copy-back, scale or update passes); stochastic rounding takes the unfused path."""
    if not (p.is_cuda and p.is_contiguous() and not stochastic and p.dtype in (torch.bfloat16, torch.float32)):
        apply_update_(p, orthogonalize(g, steps, layout), alpha, stochastic)
        return
    x = g if layout is None else _per_expert(g, layout)
    out = p if layout is None else _per_expert(p, layout)
    newton_schulz(x, steps, out=out, scale=alpha * _aspect(x), accumulate=True)


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
        state_dtype: dtype for the momentum buffer (default: the grad's dtype).
        expert_layouts: {param: (dim, width)} for MoE expert matrices to orthogonalize
            per expert (see orthogonalize); other params are orthogonalized whole.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 stochastic_rounding=False, state_dtype=None, expert_layouts=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
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
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=self.state_dtype or g.dtype)
                bufs.append(state["momentum_buffer"])
            momentum = group["momentum"]
            if group["nesterov"] and all(_fusable(g, b) for g, b in zip(grads, bufs)):
                for g, buf in zip(grads, bufs):
                    nesterov_(g, buf, momentum)
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
            self._orthogonalize_and_apply(params, updates, group["ns_steps"], -group["lr"])

    def _orthogonalize_and_apply(self, params, updates, steps, alpha):
        """p += alpha * orthogonalize(update) for each parameter. Plain matrices of one shape go
        through Newton-Schulz as one batch (each gets exactly the result it would get alone).
        Each update is applied as soon as it exists, so at most one batch of them is alive."""
        batches = {}
        for p, u in zip(params, updates):
            layout = self.expert_layouts.get(p)
            if layout is None:
                batches.setdefault(tuple(u.shape), []).append((p, u))
            else:
                orthogonalize_add_(p, u, steps, alpha, layout, self.stochastic_rounding)
        for batch in batches.values():
            stacked = orthogonalize(torch.stack([u for _, u in batch]), steps)
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
                 stochastic_rounding: bool = False, state_dtype=None, expert_layouts=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
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
                    g = orthogonalize(g, group["ns_steps"], self.expert_layouts.get(p))
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
                 stochastic_rounding: bool = False, state_dtype=None, expert_layouts=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
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
                                   self.expert_layouts[p], self.stochastic_rounding)
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
                u = orthogonalize(full, group["ns_steps"], self.expert_layouts.get(chunk[rank]))
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
