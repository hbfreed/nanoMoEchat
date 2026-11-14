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


def _torch_fused(G, steps=5, fp32_out=False):
    """The iteration with each product rounded once from fp32, like the kernel, and the
    normalization s = 1/|X| folded into the first iteration: A = s^2 X X^T, X1 = s (B X + a X).
    fp32_out leaves the last product unrounded."""
    from nanochat.newton_schulz import A_COEF, B_COEF, C_COEF
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    X = X.mT if tall else X
    s = 1 / (X.float().norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for i in range(steps):
        f = s if i == 0 else 1.0
        A = (f * f * (X.float() @ X.float().mT)).bfloat16()
        B = (C_COEF * (A.float() @ A.float()) + B_COEF * A.float()).bfloat16()
        X = f * (B.float() @ X.float() + A_COEF * X.float())
        X = X if fp32_out and i == steps - 1 else X.bfloat16()
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


@pytest.mark.parametrize("state_dtype", [None, torch.float32])
def test_muon_step_matches_per_parameter_loop(state_dtype):
    """Muon.step batches same-shape matrices through Newton-Schulz and runs momentum and Nesterov
    in one pass; it must match a per-parameter loop bitwise (expert matrices to one bf16 step:
    their norms come from the momentum pass)."""
    from nanochat.muon import Muon, orthogonalize, orthogonalize_add_
    from nanochat.precision import apply_update_

    torch.manual_seed(0)
    shapes = [(256, 256)] * 5 + [(64, 256), (64, 256), (256, 64), (256, 4 * 128), (4 * 128, 256)]
    def build():
        torch.manual_seed(0)
        return [torch.nn.Parameter(torch.randn(s, device="cuda").bfloat16()) for s in shapes]
    new, ref = build(), build()
    layouts = lambda ps: {ps[8]: (1, 128), ps[9]: (0, 128)}  # w1-like and w2-like experts
    opt = Muon(new, lr=0.02, state_dtype=state_dtype, expert_layouts=layouts(new))
    ref_layouts, bufs = layouts(ref), {}
    for step in range(3):
        gen = torch.Generator("cuda").manual_seed(step)
        for a, b in zip(new, ref):
            a.grad = torch.randn(a.shape, device="cuda", generator=gen).bfloat16()
            b.grad = a.grad.clone()
        opt.step()
        with torch.no_grad():
            for p in ref:  # the previous Muon.step, one parameter at a time
                g = p.grad
                buf = bufs.setdefault(p, torch.zeros_like(g, dtype=state_dtype or g.dtype))
                buf.lerp_(g.to(buf.dtype), 1 - 0.95)
                g = g.lerp_(buf.to(g.dtype), 0.95)
                if p in ref_layouts:  # expert matrices: the update is fused into the last product
                    orthogonalize_add_(p, g, 5, -0.02, ref_layouts[p])
                else:
                    apply_update_(p, orthogonalize(g, 5), -0.02, False)
    for i, (a, b) in enumerate(zip(new, ref)):
        if i < 8:
            assert torch.equal(a, b)
        else:  # Muon's momentum pass computes the expert norms, summing in another order
            d = (a.float() - b.float()).abs() - _ulp(b)
            # bf16 steps of p, or a small fraction of the 3 updates of about 1e-3 each
            assert (d <= 1e-4).all()


def _ulp(x):
    """Spacing of bf16 values at |x|."""
    x = x.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(x)) - 7)


@pytest.mark.parametrize("shape,layout", [((256, 4 * 128), (1, 128)), ((4 * 128, 256), (0, 128)),
                                          ((4 * 64, 128), (0, 64)), ((256, 256), None),
                                          ((64, 256), None), ((256, 64), None)])
def test_fused_update_matches_unfused(shape, layout):
    """orthogonalize_add_ adds the last Newton-Schulz product into p in fp32 and rounds once;
    the unfused path rounds the update to bf16 first. Same products otherwise, so the two
    agree to one bf16 rounding of p, and the fused one is the correctly rounded
    p + alpha * u of the fp32 update (up to fp32 rounding of the epilogue)."""
    from nanochat.muon import orthogonalize, orthogonalize_add_
    from nanochat.precision import apply_update_
    torch.manual_seed(0)
    p0 = (torch.randn(shape, device="cuda") * 0.05).bfloat16()
    g = torch.randn(shape, device="cuda").bfloat16()
    lr = -0.02
    fused, unfused, u32 = p0.clone(), p0.clone(), torch.zeros(shape, device="cuda")
    orthogonalize_add_(fused, g, 5, lr, layout)
    apply_update_(unfused, orthogonalize(g, 5, layout), lr, False)
    orthogonalize_add_(u32, g, 5, 1.0, layout)  # the unrounded update, in fp32
    exact = p0.float() + lr * u32
    assert torch.equal(orthogonalize(g, 5, layout), u32.bfloat16())
    # rounding u to bf16 moves the unfused sum by up to half a step at |lr * u|, which can
    # exceed a step of the sum where p and lr * u cancel; the sum is at most 2 * scale
    scale = torch.maximum(p0.float().abs(), (lr * u32).abs())
    assert ((fused.float() - unfused.float()).abs() <= _ulp(2 * scale)).all()
    assert ((fused.float() - exact).abs() <= 0.5 * _ulp(exact)).all()
    assert (fused.float() - exact).norm() <= (unfused.float() - exact).norm()


def test_fused_update_stochastic_rounding_falls_back():
    """stochastic_rounding keeps the unfused path: p's expected step is alpha * u even where
    it is below half a bf16 step (round-to-nearest would leave p unchanged)."""
    from nanochat.muon import orthogonalize, orthogonalize_add_
    torch.manual_seed(0)
    g = torch.randn(256, 4 * 128, device="cuda").bfloat16()
    u = orthogonalize(g, 5, (1, 128)).float()
    p = torch.ones(256, 4 * 128, device="cuda").bfloat16()
    step = torch.zeros_like(u)
    for _ in range(16):
        q = p.clone()
        orthogonalize_add_(q, g, 5, -1e-3, (1, 128), stochastic=True)
        step += (q.float() - 1.0) / 16
    coef = (step * u).sum() / (u * u).sum()  # least-squares fit of step = coef * u
    assert abs(coef.item() + 1e-3) < 1e-4


@pytest.mark.parametrize("buf_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("g_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("momentum", [0.95, 0.3])
def test_nesterov_matches_lerps(buf_dtype, g_dtype, momentum):
    """The one-pass momentum + Nesterov kernel is bitwise equal to Muon's two lerps."""
    from nanochat.muon import nesterov_
    torch.manual_seed(0)
    g = torch.randn(300, 1031, device="cuda").to(g_dtype)
    buf = torch.randn(300, 1031, device="cuda").to(buf_dtype)
    g_ref, buf_ref = g.clone(), buf.clone()
    nesterov_(g, buf, momentum)
    buf_ref.lerp_(g_ref.to(buf_dtype), 1 - momentum)
    g_ref.lerp_(buf_ref.to(g_dtype), momentum)
    assert torch.equal(buf, buf_ref) and torch.equal(g, g_ref)


@pytest.mark.parametrize("shape,layout", [((256, 4 * 512), (1, 512)), ((4 * 512, 256), (0, 512)),
                                          ((100, 3 * 256), (1, 256)), ((3 * 64, 300), (0, 64)),
                                          ((256, 3 * 128), (1, 128))])
def test_nesterov_expert_norms(shape, layout):
    """nesterov_ with an expert layout also returns each expert's 1/|g_e| (of the new g, in bf16),
    matching inv_frobenius on the per-expert view up to fp32 summation order; widths that do
    not fit its tiles return None."""
    from nanochat.muon import _per_expert, nesterov_
    from nanochat.newton_schulz import inv_frobenius
    torch.manual_seed(0)
    g = torch.randn(shape, device="cuda").bfloat16()
    buf = torch.randn(shape, device="cuda").bfloat16()
    g_ref, buf_ref = g.clone(), buf.clone()
    norms = nesterov_(g, buf, 0.95, layout)
    nesterov_(g_ref, buf_ref, 0.95)
    assert torch.equal(g, g_ref) and torch.equal(buf, buf_ref)
    if layout == (1, 128):  # narrower than the 256-column tiles
        assert norms is None
        return
    s, s2 = inv_frobenius(_per_expert(g, layout))
    torch.testing.assert_close(norms[0], s, rtol=1e-6, atol=0)
    torch.testing.assert_close(norms[1], s2, rtol=2e-6, atol=0)


def _torch_cubic5(G, steps=5):
    """cubic5 with the kernel's rounding: each product rounded once from fp32, the
    normalization folded into the first step."""
    from nanochat.newton_schulz import CUBIC5
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    X = X.mT if tall else X
    s = 1 / (X.float().norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for i, (a, b) in enumerate(CUBIC5[:steps]):
        f = s if i == 0 else 1.0
        A = (f * f * (X.float() @ X.float().mT)).bfloat16()
        X = (f * (b * (A.float() @ X.float()) + a * X.float())).bfloat16()
    return X.mT if tall else X


@pytest.mark.parametrize("shape", [(512, 512), (128, 512), (512, 128), (100, 64), (16, 512, 512)])
def test_cubic5_matches_torch_iteration(shape):
    from nanochat.newton_schulz import newton_schulz
    torch.manual_seed(0)
    G = torch.randn(shape, device="cuda").bfloat16()
    out = newton_schulz(G, 5, schedule="cubic5")
    assert _rel(out, _torch_cubic5(G)) < 1e-2
    # added into a buffer through the fused last product, as Muon applies it; the buffer is
    # small so its own bf16 rounding (one step is 2^-8 of its magnitude) stays below the update
    base = (1e-3 * torch.randn(shape, device="cuda")).bfloat16()
    acc = base.clone()
    newton_schulz(G, 5, out=acc, scale=-0.5, accumulate=True, schedule="cubic5")
    assert _rel(acc.float() - base.float(), -0.5 * _torch_cubic5(G).float()) < 2e-2


def test_cubic5_muon_step_runs_all_paths():
    """Muon with ns_schedule="cubic5" on expert layouts, a batched plain matrix and the CPU path."""
    from nanochat.muon import Muon
    for device in ("cuda", "cpu"):
        torch.manual_seed(0)
        ps = [torch.nn.Parameter(torch.randn(s, device=device).bfloat16())
              for s in [(256, 4 * 128), (4 * 128, 256), (256, 256), (256, 256)]]
        opt = Muon(ps, lr=0.02, ns_schedule="cubic5", expert_layouts={ps[0]: (1, 128), ps[1]: (0, 128)})
        before = [p.detach().clone() for p in ps]
        for p in ps:
            p.grad = torch.randn_like(p)
        opt.step()
        for p, b in zip(ps, before):
            assert torch.isfinite(p).all() and not torch.equal(p, b)
