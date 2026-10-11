"""
Weight-update helpers for low-precision parameters.

Rounding a bf16 weight to nearest after each update drops any update smaller than
half a bf16 step (~0.2% of the weight), which bites late in training when the LR
is small. Stochastic rounding rounds up with probability equal to the fraction of
the gap, so tiny updates survive in expectation at no extra memory.
"""

import torch
from torch import Tensor


def stochastic_round_(dst: Tensor, src: Tensor) -> None:
    """dst (bf16) <- src (fp32), rounded stochastically.

    bf16 is the top 16 bits of fp32. Adding uniform noise to the low 16 bits and
    truncating them rounds the magnitude up with probability equal to the
    discarded fraction (sign-magnitude, so this is correct for negatives too).
    """
    bits = src.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device)
    dst.copy_(((bits + noise) & -65536).view(torch.float32))


def apply_update_(p: Tensor, update: Tensor, alpha: float, stochastic: bool) -> None:
    """p += alpha * update, stochastically rounded when p is bf16 and stochastic=True."""
    if stochastic and p.dtype == torch.bfloat16:
        stochastic_round_(p, p.float().add_(update.float(), alpha=alpha))
    else:
        p.add_(update, alpha=alpha)
