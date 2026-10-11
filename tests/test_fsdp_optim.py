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
