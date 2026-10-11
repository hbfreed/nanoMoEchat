"""
Chunked fused lm_head + softcap + cross-entropy (the Liger "fused linear CE" idea).

Tokens are processed in chunks. For each chunk the forward pass computes the
logits, the loss *and* the gradients right away, then drops the logits, so memory
stays at one chunk of (chunk x vocab) instead of all tokens. The heavy work is
three plain cuBLAS matmuls per chunk; the softcap/softmax math is one compiled
elementwise kernel. The lm_head gradient accumulates in fp32 across chunks.

Backward just scales the saved gradients by grad_output, so this only supports a
scalar loss (reduction "mean" or "sum") when gradients are needed.
"""

import torch
import torch.nn.functional as F


@torch.compile(dynamic=False)
def _chunk_loss_and_dlogits(logits, targets, softcap: float):
    """Sum of CE over the chunk and d(sum CE)/d(raw logits), for softcapped logits."""
    z = logits.float()
    if softcap > 0:
        t = torch.tanh(z / softcap)
        capped = softcap * t
    else:  # no softcap
        capped = z
    lse = torch.logsumexp(capped, dim=-1)
    valid = targets != -1
    tgt = targets.clamp(min=0)
    picked = capped.gather(1, tgt[:, None]).squeeze(1)
    loss_sum = ((lse - picked) * valid).sum()
    probs = torch.exp(capped - lse[:, None])
    dcapped = probs - F.one_hot(tgt, logits.size(-1)).to(probs.dtype)
    if softcap > 0:
        dcapped = dcapped * (1 - t * t)  # d(softcap*tanh(z/softcap))/dz
    dlogits = dcapped * valid[:, None]
    return loss_sum, dlogits.to(logits.dtype)


@torch.compile(dynamic=False)
def _chunk_losses(logits, targets, softcap: float):
    """Per-token CE (0 where target == -1), for no-grad evaluation."""
    capped = logits.float()
    if softcap > 0:
        capped = softcap * torch.tanh(capped / softcap)
    return F.cross_entropy(capped, targets, ignore_index=-1, reduction="none")


@torch.compiler.disable(recursive=False)
def _chunked_forward(x, weight, targets, softcap, chunk, reduction, need_dx, need_dw):
    """The chunk loop runs eagerly so each chunk's (chunk x vocab) temporaries are freed
    before the next; only the per-chunk elementwise kernel is compiled. Left to Dynamo,
    the loop gets unrolled into one graph that can keep every chunk's buffers alive."""
    n = x.size(0)
    loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
    dx = torch.empty_like(x) if need_dx else None
    dw = torch.zeros(weight.shape, device=x.device, dtype=torch.float32) if need_dw else None
    for s in range(0, n, chunk):
        xc, tc = x[s : s + chunk], targets[s : s + chunk]
        logits = xc @ weight.t()
        chunk_loss, dlogits = _chunk_loss_and_dlogits(logits, tc, softcap)
        loss_sum += chunk_loss
        if need_dx:
            dx[s : s + chunk] = dlogits @ weight
        if need_dw:
            dw += torch.mm(dlogits.t(), xc, out_dtype=torch.float32)
    denom = (targets != -1).sum().clamp(min=1) if reduction == "mean" else 1
    if need_dx:
        dx = dx / denom
    if need_dw:
        dw = dw / denom
    return loss_sum / denom, dx, dw


class _ChunkedSoftcapCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, softcap, chunk, reduction):
        loss, dx, dw = _chunked_forward(
            x, weight, targets, softcap, chunk, reduction,
            ctx.needs_input_grad[0], ctx.needs_input_grad[1],
        )
        ctx.save_for_backward(dx, dw)
        ctx.weight_dtype = weight.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        dx, dw = ctx.saved_tensors
        dx = dx * grad_out.to(dx.dtype) if dx is not None else None
        dw = (dw * grad_out).to(ctx.weight_dtype) if dw is not None else None
        return dx, dw, None, None, None, None


def chunked_softcap_cross_entropy(x, weight, targets, softcap, reduction="mean", chunk=4096):
    """x: (..., H), weight: (V, H), targets: (...) with -1 = ignore."""
    x2, t2 = x.reshape(-1, x.size(-1)), targets.reshape(-1)
    if reduction == "none":
        assert not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad), \
            "reduction='none' is only supported without gradients"
        out = torch.cat([
            _chunk_losses(x2[s : s + chunk] @ weight.t(), t2[s : s + chunk], softcap)
            for s in range(0, x2.size(0), chunk)
        ])
        return out.view_as(targets)
    assert reduction in ("mean", "sum"), reduction
    return _ChunkedSoftcapCE.apply(x2, weight, t2, softcap, chunk, reduction)
