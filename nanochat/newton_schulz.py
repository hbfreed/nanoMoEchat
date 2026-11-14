"""Batched Newton-Schulz for Muon: fused epilogues, symmetric products, strided inputs.

Muon's Newton-Schulz was about half of a training step with per-expert Muon (d=512,
128x512 experts: 473 ms per optimizer step against about 100 ms of forward + backward per
8k-token micro-batch). Under torch.compile, cuBLAS ran the products at about 77 TF/s and
the elementwise b*A + c*A@A and a*X + B@X ran as separate passes. Here each product is one
kernel, which computes for every matrix i of a batch
    out[i] = alpha * (a[i] @ b[i]) + beta * c[i]
accumulating in fp32 and rounding once. a, b and c may be any strided views, so
per-expert slices of w1/w3 (transposed views) need no copies. With SYMMETRIC the
product is known to be symmetric (X X^T, or A A for symmetric A): only tiles on or
above the diagonal are computed and each is also stored transposed.
"""
import torch
import triton
import triton.language as tl

A_COEF, B_COEF, C_COEF = (3.4445, -4.7750, 2.0315)


@triton.jit
def _ns_gemm(
    a, b, c, out, f, alpha, beta,
    M, N, K,
    sab, sam, sak,
    sbb, sbk, sbn,
    scb, scm, scn,
    sob, som, son,
    HAS_C: tl.constexpr, HAS_F: tl.constexpr, ACCUMULATE: tl.constexpr,
    SYMMETRIC: tl.constexpr, TILES_N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    if SYMMETRIC:
        # Upper-triangle tile pairs (tm <= tn), row by row.
        tm = 0
        rem = pid
        row_len = TILES_N
        while rem >= row_len:
            rem -= row_len
            tm += 1
            row_len -= 1
        tn = tm + rem
    else:
        tm = pid // TILES_N
        tn = pid % TILES_N
    rm = tm * BM + tl.arange(0, BM)
    rn = tn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + batch * sab + rm[:, None] * sam + rk[None, :] * sak
    b_ptr = b + batch * sbb + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        av = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        bv = tl.load(b_ptr, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(av, bv, acc)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    acc = acc * alpha
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if HAS_C:
        cv = tl.load(c + batch * scb + rm[:, None] * scm + rn[None, :] * scn, mask=mask, other=0.0)
        acc += beta * cv.to(tl.float32)
    if HAS_F:
        acc *= tl.load(f + batch)
    o_ptr = out + batch * sob + rm[:, None] * som + rn[None, :] * son
    if ACCUMULATE:
        acc += tl.load(o_ptr, mask=mask, other=0.0).to(tl.float32)
    res = acc.to(out.dtype.element_ty)
    tl.store(o_ptr, res, mask=mask)
    if SYMMETRIC:
        if tm != tn:
            tl.store(out + batch * sob + rn[None, :] * som + rm[:, None] * son, res, mask=mask)


# Tiles from a sweep on a 128 x 512 x 512 expert batch (RTX 3090, 275 W): 0.80 ms per
# iteration, about 96 TF/s of computed products; 8 warps took 0.87 ms.
def ns_gemm(a, b, c=None, alpha=1.0, beta=0.0, symmetric=False, out=None, f=None,
            accumulate=False, BM=128, BN=128, BK=32, num_warps=4, num_stages=3):
    """f * (alpha * a @ b + beta * c) over a batch of matrices (3-D strided views), with f an
    optional per-matrix fp32 factor. accumulate=True adds the result to out (in fp32, rounded
    once) instead of overwriting it."""
    nb, M, K = a.shape
    N = b.shape[2]
    if out is None:
        out = torch.empty((nb, M, N), dtype=a.dtype, device=a.device)
    tiles_m, tiles_n = triton.cdiv(M, BM), triton.cdiv(N, BN)
    if symmetric:
        assert M == N and BM == BN and not accumulate
        tiles = tiles_n * (tiles_n + 1) // 2
    else:
        tiles = tiles_m * tiles_n
    cc = c if c is not None else out
    _ns_gemm[(tiles, nb)](
        a, b, cc, out, f if f is not None else out, alpha, beta, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        cc.stride(0), cc.stride(1), cc.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_C=c is not None, HAS_F=f is not None, ACCUMULATE=accumulate, SYMMETRIC=symmetric, TILES_N=tiles_n,
        BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _sumsq_partial(x, part, M, N, sxb, sxm, sxn, P: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """part[b, i] = sum of squares of rows [i*BR, (i+1)*BR) of matrix b, in fp32."""
    i = tl.program_id(0)
    b = tl.program_id(1)
    rm = i * BR + tl.arange(0, BR)
    rn = tl.arange(0, BC)
    acc = tl.zeros((BR, BC), dtype=tl.float32)
    for n in range(0, N, BC):
        mask = (rm[:, None] < M) & (rn[None, :] + n < N)
        v = tl.load(x + b * sxb + rm[:, None] * sxm + (rn[None, :] + n) * sxn, mask=mask, other=0.0)
        v = v.to(tl.float32)
        acc += v * v
    tl.store(part + b * P + i, tl.sum(acc))


@triton.jit
def _inv_norm(part, s, s2, eps, P: tl.constexpr, PP: tl.constexpr):
    """s[b] = 1 / (sqrt(sum of part[b]) + eps), s2[b] = s[b]^2."""
    b = tl.program_id(0)
    r = tl.arange(0, PP)
    v = tl.load(part + b * P + r, mask=r < P, other=0.0)
    inv = 1.0 / (tl.sqrt(tl.sum(v)) + eps)
    tl.store(s + b, inv)
    tl.store(s2 + b, inv * inv)


def inv_frobenius(X, eps=1e-7, BR=32, BC=128):
    """1 / (|X_i|_F + eps) and its square for each matrix of a batch (3-D strided view), in fp32.
    The reduction order depends only on the matrix shape, so a matrix gets the same bits
    whatever batch it is in (torch's norm splits batched reductions differently)."""
    nb, M, N = X.shape
    if X.stride(1) == 1 and X.stride(2) != 1:  # read along the contiguous dim
        X = X.mT
        M, N = N, M
    P = triton.cdiv(M, BR)
    part = torch.empty((nb, P), dtype=torch.float32, device=X.device)
    _sumsq_partial[(P, nb)](X, part, M, N, X.stride(0), X.stride(1), X.stride(2), P=P, BR=BR, BC=BC)
    return inv_norm_from_partials(part, eps)


def inv_norm_from_partials(part, eps=1e-7):
    """(s, s^2) with s = 1/(sqrt(sum of part[i]) + eps) for each row i of (nb, P) partial sums."""
    nb, P = part.shape
    s = torch.empty(nb, dtype=torch.float32, device=part.device)
    s2 = torch.empty_like(s)
    _inv_norm[(nb,)](part, s, s2, eps, P=P, PP=triton.next_power_of_2(P))
    return s, s2


def newton_schulz(G, steps: int, out=None, scale=1.0, accumulate=False, inv_norm=None):
    """Muon's quintic Newton-Schulz on a matrix or a batch of matrices, in bf16.

    The same iteration and coefficients as the torch version in muon.py. Rounding once per
    product in fp32 changes the result by about as much as the torch version moves between
    eager and compiled execution: on decaying-spectrum 512x512 inputs it is 3.3% from the
    fp32 iteration, against 1.7% compiled and 4.8% eager, while perturbing the input by one
    bf16 rounding moves the fp32 iteration itself by 2.3%.

    The Frobenius normalization is folded into the first iteration: with s = 1/(|X| + eps)
    per matrix, A = s^2 X X^T and X1 = s (B X + a X), so X/|X| is never materialized. The
    last product writes scale * result into out (any strided view of G's shape, e.g. a
    parameter's per-expert view), or adds it to out with accumulate=True, rounding once
    from fp32; this saves the copy-back, scale and update passes. inv_norm=(s, s^2) passes
    norms computed elsewhere (Muon's momentum pass computes them for expert matrices). Against the fp64 iteration on
    64 decaying-spectrum 512x512 inputs (singular values exp(-i/50)) this is 4.0% off, the
    unfolded kernel 5.1%, and one bf16 rounding of the input moves the fp64 iteration 4.1%;
    on Gaussian inputs 1.02% vs 1.04%."""
    assert steps >= 1
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    if out is None:
        out = torch.empty(X.shape, dtype=torch.bfloat16, device=G.device)
        result = out.squeeze(0) if squeeze else out
    else:
        assert out.shape == G.shape
        result = out
        out = out.unsqueeze(0) if squeeze else out
    X = X.bfloat16()
    s, s2 = inv_frobenius(X) if inv_norm is None else inv_norm
    if X.size(-2) > X.size(-1):  # iterate on the wide orientation
        X, out = X.mT, out.mT
    for i in range(steps):
        first, last = i == 0, i == steps - 1
        A = ns_gemm(X, X.mT, symmetric=True, f=s2 if first else None)   # X X^T
        B = ns_gemm(A, A, A, alpha=C_COEF, beta=B_COEF, symmetric=True)  # c A A + b A
        if last:  # scale * (B X + a X), written or added into out
            f = s * scale if first else None
            ns_gemm(B, X, X, alpha=1.0 if first else scale, beta=A_COEF if first else A_COEF * scale,
                    out=out, f=f, accumulate=accumulate)
        else:
            X = ns_gemm(B, X, X, alpha=1.0, beta=A_COEF, f=s if first else None)  # B X + a X
    return result
