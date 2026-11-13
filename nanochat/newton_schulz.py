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
    a, b, c, out, alpha, beta,
    M, N, K,
    sab, sam, sak,
    sbb, sbk, sbn,
    scb, scm, scn,
    sob, som, son,
    HAS_C: tl.constexpr, SYMMETRIC: tl.constexpr, TILES_N: tl.constexpr,
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
    res = acc.to(out.dtype.element_ty)
    tl.store(out + batch * sob + rm[:, None] * som + rn[None, :] * son, res, mask=mask)
    if SYMMETRIC:
        if tm != tn:
            tl.store(out + batch * sob + rn[None, :] * som + rm[:, None] * son, res, mask=mask)


# Tiles from a sweep on a 128 x 512 x 512 expert batch (RTX 3090, 275 W): 0.80 ms per
# iteration, about 96 TF/s of computed products; 8 warps took 0.87 ms.
def ns_gemm(a, b, c=None, alpha=1.0, beta=0.0, symmetric=False, out=None, BM=128, BN=128, BK=32,
            num_warps=4, num_stages=3):
    """alpha * a @ b + beta * c over a batch of matrices (3-D strided views)."""
    nb, M, K = a.shape
    N = b.shape[2]
    if out is None:
        out = torch.empty((nb, M, N), dtype=a.dtype, device=a.device)
    tiles_m, tiles_n = triton.cdiv(M, BM), triton.cdiv(N, BN)
    if symmetric:
        assert M == N and BM == BN
        tiles = tiles_n * (tiles_n + 1) // 2
    else:
        tiles = tiles_m * tiles_n
    cc = c if c is not None else out
    _ns_gemm[(tiles, nb)](
        a, b, cc, out, alpha, beta, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        cc.stride(0), cc.stride(1), cc.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_C=c is not None, SYMMETRIC=symmetric, TILES_N=tiles_n,
        BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages,
    )
    return out


def newton_schulz(G, steps: int):
    """Muon's quintic Newton-Schulz on a matrix or a batch of matrices, in bf16.

    The same iteration and coefficients as the torch version in muon.py. Rounding once per
    product in fp32 changes the result by about as much as the torch version moves between
    eager and compiled execution: on decaying-spectrum 512x512 inputs it is 3.3% from the
    fp32 iteration, against 1.7% compiled and 4.8% eager, while perturbing the input by one
    bf16 rounding moves the fp32 iteration itself by 2.3%."""
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X.bfloat16()
    X = X / (X.float().norm(dim=(-2, -1), keepdim=True) + 1e-7).to(X.dtype)
    for _ in range(steps):
        A = ns_gemm(X, X.mT, symmetric=True)                              # X X^T
        B = ns_gemm(A, A, A, alpha=C_COEF, beta=B_COEF, symmetric=True)   # c A A + b A
        X = ns_gemm(B, X, X, alpha=1.0, beta=A_COEF)                      # B X + a X
    if tall:
        X = X.mT
    return X.squeeze(0) if squeeze else X
