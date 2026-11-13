"""
The Triton Newton-Schulz (nanochat/newton_schulz.py) must compute Muon's iteration: match
a torch version with the same fp32-epilogue rounding, give identical results for strided
and contiguous inputs and for symmetric and full products, and handle shapes that do not
fill whole tiles.

python -m pytest tests/test_newton_schulz.py -q
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _torch_fused(G, steps=5):
    """The iteration with each product rounded once from fp32, like the kernel."""
    from nanochat.newton_schulz import A_COEF, B_COEF, C_COEF
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    X = X.mT if tall else X
    X = X / (X.float().norm(dim=(-2, -1), keepdim=True) + 1e-7).to(X.dtype)
    for _ in range(steps):
        A = (X.float() @ X.float().mT).bfloat16()
        B = (C_COEF * (A.float() @ A.float()) + B_COEF * A.float()).bfloat16()
        X = (B.float() @ X.float() + A_COEF * X.float()).bfloat16()
    return X.mT if tall else X


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("shape", [(512, 512), (128, 512), (512, 128), (100, 64), (64, 300),
                                   (512, 1024), (4, 256, 256), (16, 512, 512)])
def test_matches_torch_iteration(shape):
    from nanochat.newton_schulz import newton_schulz
    torch.manual_seed(0)
    G = torch.randn(shape, device="cuda").bfloat16()
    out = newton_schulz(G, 5)
    assert out.shape == G.shape and out.dtype == torch.bfloat16
    # accumulation order differs from cuBLAS, so bf16 roundings differ in a few places
    assert _rel(out, _torch_fused(G)) < 1e-2


def test_strided_and_symmetric_are_exact():
    from nanochat import newton_schulz as ns
    torch.manual_seed(0)
    g = torch.randn(256, 8 * 128, device="cuda").bfloat16()  # w1-like: (n_embd, E * width)
    strided = g.reshape(256, 8, 128).transpose(0, 1)  # per-expert views, as orthogonalize makes
    assert torch.equal(ns.newton_schulz(strided, 5), ns.newton_schulz(strided.contiguous(), 5))
    X = strided.contiguous()
    assert torch.equal(ns.ns_gemm(X, X.mT, symmetric=True), ns.ns_gemm(X, X.mT))
    A = ns.ns_gemm(X, X.mT)
    assert torch.equal(ns.ns_gemm(A, A, A, alpha=2.0, beta=-4.0, symmetric=True),
                       ns.ns_gemm(A, A, A, alpha=2.0, beta=-4.0))
