"""
Check MoEMLP's kernels (relu2: stk/megablocks + nanomoe_ops, swiglu: grouped Triton or CuTe)
against a plain per-expert loop. Same routing, experts computed explicitly:

- fp64 loop  = ground truth
- bf16 loop  = the error budget (what bf16 rounding alone costs)

The kernels pass if their error vs. the truth is within a small factor of the bf16
loop's, for the output and the gradients of x, the router and every expert weight.
Covers uniform and mixed expert widths, skewed load, and experts that get 0 tokens.

Run with:
python -m pytest tests/test_moe_kernels.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPTConfig, MoEMLP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MIXED = [[8, 256], [8, 512], [16, 128]]


def reference(m, x, dtype):
    """MoEMLP.forward's routing, with the experts as an explicit loop in `dtype`."""
    xf = x.reshape(-1, x.shape[-1])
    probs = torch.sigmoid(F.linear(xf, m.router.weight).float())
    w, sel = torch.topk(probs, m.num_active_experts, dim=-1)
    w = (w / (w.sum(-1, keepdim=True) + 1e-20)).to(x.dtype)
    xd, wd = xf.to(dtype), w.to(dtype)
    out = torch.zeros_like(xd)
    for e, (lo, hi) in enumerate(zip(m.expert_offsets[:-1], m.expert_offsets[1:])):
        tok, slot = (sel == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = xd[tok] @ m.w1[:, lo:hi].to(dtype)
        if m.mlp_act == "swiglu":
            h = F.silu(h) * (xd[tok] @ m.w3[:, lo:hi].to(dtype))
        else:
            h = F.relu(h).square()
        out.index_add_(0, tok, (h @ m.w2[lo:hi].to(dtype)) * wd[tok, slot, None])
    counts = torch.bincount(sel.flatten(), minlength=m.num_experts)
    return out.reshape(x.shape), counts


def output_and_grads(fn, m, x, g):
    params = [x, m.router.weight, m.w1, m.w2] + ([m.w3] if m.mlp_act == "swiglu" else [])
    for p in params:
        p.grad = None
    out = fn()
    (out.float() * g).sum().backward()
    return [out.detach().double()] + [p.grad.double().clone() for p in params]


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def cute_available():
    if torch.cuda.get_device_capability()[0] != 8:  # the CuTe kernels target Ampere
        return False
    try:
        import megablocks.backend.cute_down_proj  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("act,backend", [("relu2", "stk"), ("swiglu", "triton"), ("swiglu", "cute")])
@pytest.mark.parametrize(
    "sizes,load",
    [
        ([[16, 256]], "uniform"),
        (MIXED, "uniform"),
        (MIXED, "skewed"),
        (MIXED, "starved"),
    ],
)
def test_moe_kernels_match_reference(act, backend, sizes, load):
    if backend == "cute" and not cute_available():
        pytest.skip("the CuTe backend needs megablocks[cute] and an Ampere GPU")
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_layer=1, n_embd=256, n_head=2, n_kv_head=2,
        expert_sizes=sizes, num_active_experts=4, mlp_act=act,
    )
    m = MoEMLP(cfg).cuda().to(torch.bfloat16)
    if act == "swiglu":
        m.moe_backend = backend
    torch.nn.init.normal_(m.router.weight, std=0.02)
    x = torch.randn(4, 512, 256, device="cuda") * 0.5
    with torch.no_grad():
        if load == "skewed":  # pile tokens onto the first few experts
            m.router.weight[:4] += 0.2
        elif load == "starved":  # every 3rd expert (every width) never gets picked
            m.router.weight[::3] = 0
            m.router.weight[::3, 0] = -50
            x[..., 0] += 1.0
    x = x.to(torch.bfloat16).requires_grad_()
    g = torch.randn(4, 512, 256, device="cuda")

    kernel = output_and_grads(lambda: m(x)[0], m, x, g)
    truth = output_and_grads(lambda: reference(m, x, torch.float64)[0], m, x, g)
    naive = output_and_grads(lambda: reference(m, x, torch.bfloat16)[0], m, x, g)
    counts = reference(m, x, torch.float64)[1]
    if load == "starved":
        assert (counts == 0).sum() > 0, "starved case should leave some experts empty"

    names = ["out", "dx", "d_router", "dw1", "dw2", "dw3"]
    for name, k, t, b in zip(names, kernel, truth, naive):
        err_kernel, err_bf16 = rel_err(k, t), rel_err(b, t)
        print(f"{backend} {load} {name}: kernel {err_kernel:.2e}  bf16 loop {err_bf16:.2e}")
        assert err_kernel < 3 * err_bf16, f"{name}: kernel err {err_kernel:.2e} vs bf16 loop {err_bf16:.2e}"
