"""
FSDP2 optimizers (FSDPMuon, AdamW on DTensor shards) must take the same steps as the
single-GPU Muon / AdamW on full tensors, given the same weights and gradients.

Shapes include dim 0 not divisible by the world size and smaller than it, which FSDP2
shards unevenly (torch.chunk), and enough same-shape matrices that some owner ranks
get none in a round.

Run with (any world size >= 2):
torchrun --standalone --nproc_per_node=3 -m pytest tests/test_fsdp_optim.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ or not torch.cuda.is_available(), reason="needs torchrun + CUDA"
)

SHAPES = [(256, 128), (256, 128), (256, 128), (256, 128), (100, 64), (2, 32), (64, 300)]


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _setup():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _module(seed):
    torch.manual_seed(seed)
    m = nn.Module()
    m.mats = nn.ParameterList([nn.Parameter(torch.randn(s, device="cuda").bfloat16()) for s in SHAPES])
    m.vec = nn.Parameter(torch.randn(77, device="cuda").bfloat16())
    return m


@pytest.mark.parametrize("stochastic_rounding", [False])
@torch.no_grad()
def test_fsdp_optimizers_match_single_gpu(stochastic_rounding):
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor

    from nanochat.adamw import AdamW
    from nanochat.muon import FSDPMuon, Muon

    _setup()
    ref, sharded = _module(0), _module(0)
    fully_shard(sharded)
    mesh = sharded.vec.device_mesh
    kw = dict(stochastic_rounding=stochastic_rounding)
    ref_opts = [Muon(list(ref.mats), lr=0.02, momentum=0.95, **kw),
                AdamW([dict(params=[ref.vec], lr=0.01)], betas=(0.8, 0.95), weight_decay=0.0, **kw)]
    fsdp_opts = [FSDPMuon(list(sharded.mats), lr=0.02, momentum=0.95, **kw),
                 AdamW([dict(params=[sharded.vec], lr=0.01)], betas=(0.8, 0.95), weight_decay=0.0, **kw)]
    for step in range(3):
        gen = torch.Generator("cuda").manual_seed(100 + step)  # same grads on every rank
        for p_ref, p_sh in zip(ref.parameters(), sharded.parameters()):
            g = torch.randn(p_ref.shape, device="cuda", generator=gen).bfloat16()
            p_ref.grad = g.clone()
            p_sh.grad = distribute_tensor(g, mesh, [Shard(0)])
        for opt in ref_opts + fsdp_opts:
            opt.step()
    for (name, p_ref), p_sh in zip(ref.named_parameters(), sharded.parameters()):
        full = p_sh.full_tensor()
        assert torch.equal(full, p_ref), (name, (full.float() - p_ref.float()).abs().max().item())


def _ns_fp32(G, steps=5):
    """Muon's Newton-Schulz iteration in fp32, as a reference."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X, tall = G.float(), G.size(-2) > G.size(-1)
    X = X.mT if tall else X
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


@torch.no_grad()
def test_orthogonalize_per_expert_is_per_slice():
    """layout=(dim, width) must orthogonalize each expert's slice on its own. bf16 NS is
    itself ~3% from fp32 NS, so compare each slice to an fp32 reference of that slice
    alone: a wrong slicing would be ~100% off."""
    from nanochat.muon import orthogonalize
    torch.manual_seed(0)
    w, E = 64, 6
    for dim, shape in ((1, (96, E * w)), (0, (E * w, 96))):
        g = torch.randn(shape, device="cuda").bfloat16()
        batched = orthogonalize(g, 5, (dim, w))
        for e in range(E):
            sl = (slice(None), slice(e * w, (e + 1) * w)) if dim == 1 else (slice(e * w, (e + 1) * w), slice(None))
            s_ = g[sl]
            ref = _ns_fp32(s_) * max(1.0, s_.size(0) / s_.size(1)) ** 0.5
            assert _rel_err(batched[sl], ref) < 5e-2


@torch.no_grad()
def test_fsdp_muon_per_expert_matches_single_gpu():
    """Expert matrices sharded whole along their expert dim (w1-like on dim 1, w2-like on
    dim 0) are stepped locally; must match single-GPU per-expert Muon. Batched NS over a
    different number of experts may pick different kernels, so compare with a tolerance."""
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor

    from nanochat.muon import FSDPMuon, Muon

    _setup()
    world = dist.get_world_size()
    w, E = 32, 4 * world  # whole experts per rank
    shapes = [(96, E * w), (E * w, 96), (96, E * w), (80, 40)]  # last: a normal matrix

    def build():
        torch.manual_seed(0)
        m = nn.Module()
        m.mats = nn.ParameterList([nn.Parameter(torch.randn(s, device="cuda").bfloat16()) for s in shapes])
        return m

    ref, sharded = build(), build()
    dims = {0: 1, 1: 0, 2: 1}
    fully_shard(sharded, shard_placement_fn=lambda p: next(
        (Shard(dims[i]) for i, q in enumerate(sharded.mats) if q is p and i in dims), None))
    mesh = sharded.mats[0].device_mesh
    ref_opt = Muon(list(ref.mats), lr=0.02, expert_layouts={ref.mats[i]: (d, w) for i, d in dims.items()})
    opt = FSDPMuon(list(sharded.mats), lr=0.02, expert_layouts={sharded.mats[i]: (d, w) for i, d in dims.items()})
    assert len(opt.local_params) == 3
    for step in range(3):
        gen = torch.Generator("cuda").manual_seed(100 + step)
        for i, (p_ref, p_sh) in enumerate(zip(ref.mats, sharded.mats)):
            g = torch.randn(p_ref.shape, device="cuda", generator=gen).bfloat16()
            p_ref.grad = g.clone()
            p_sh.grad = distribute_tensor(g, mesh, [Shard(dims.get(i, 0))])
        ref_opt.step()
        opt.step()
    for p_ref, p_sh in zip(ref.mats, sharded.mats):
        assert _rel_err(p_sh.full_tensor(), p_ref) < 1e-3
