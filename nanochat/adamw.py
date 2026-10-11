"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.precision import stochastic_round_


def _adamw_update_(p: Tensor, g: Tensor, state: dict, lr: float, beta1: float, beta2: float,
                   eps: float, wd_eff: float, stochastic_rounding: bool, state_dtype):
    """One AdamW step on p (a param or a param slice) in place."""
    if not state:
        state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
        state['exp_avg'] = torch.zeros_like(p, dtype=state_dtype or p.dtype)
        state['exp_avg_sq'] = torch.zeros_like(p, dtype=state_dtype or p.dtype)
    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    state['step'] += 1
    t = state['step']
    bias1 = 1 - beta1 ** t
    bias2 = 1 - beta2 ** t
    step_size = lr * (torch.sqrt(bias2) / bias1)
    if not stochastic_rounding and state_dtype is None:
        # original in-place path, in the param's own dtype
        if wd_eff != 0:
            p.mul_(1 - wd_eff)
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        denom = exp_avg_sq.sqrt().add_(eps)
        p.add_(other=exp_avg.div(denom).mul_(step_size), alpha=-1.0)
        return
    # fp32 math; moments stored in state_dtype; weight written back (stochastically rounded)
    g = g.float()
    m = exp_avg.float().mul_(beta1).add_(g, alpha=1 - beta1)
    v = exp_avg_sq.float().mul_(beta2).addcmul_(g, g, value=1 - beta2)
    exp_avg.copy_(m)
    exp_avg_sq.copy_(v)
    new = p.float()
    if wd_eff != 0:
        new.mul_(1 - wd_eff)
    new.sub_(m.div_(v.sqrt_().add_(eps)).mul_(step_size))
    if stochastic_rounding and p.dtype == torch.bfloat16:
        stochastic_round_(p, new)
    else:
        p.copy_(new)


class AdamW(torch.optim.Optimizer):
    """Single-GPU AdamW with the same math as DistAdamW, plus stochastic rounding
    and a separate optimizer-state dtype. Used when either option is on; otherwise
    setup_optimizers keeps torch's fused AdamW."""
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01,
                 stochastic_rounding: bool = False, state_dtype=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)
        self.stochastic_rounding = stochastic_rounding
        self.state_dtype = state_dtype

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                wd_eff = lr * group['weight_decay'] * getattr(p, "wd_mul", 1.0)
                _adamw_update_(p, p.grad, self.state[p], lr, beta1, beta2, group['eps'],
                               wd_eff, self.stochastic_rounding, self.state_dtype)


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction.
    Each param's dim 0 is split across ranks; when it doesn't divide evenly (e.g. a
    65,536-row embedding on 3 GPUs) the last slice is zero-padded.
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01,
                 stochastic_rounding: bool = False, state_dtype=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)
        self.stochastic_rounding = stochastic_rounding
        self.state_dtype = state_dtype

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        padded_params = []  # (param, padded full-size buffer) to copy back after all_gather
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = -(-grad.shape[0] // world_size)  # ceil
                pad = rank_size * world_size - grad.shape[0]
                if pad:
                    grad = torch.cat([grad, grad.new_zeros(pad, *grad.shape[1:])])
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = -(-p.shape[0] // world_size)  # ceil
                pad = rank_size * world_size - p.shape[0]
                p_full = torch.cat([p, p.new_zeros(pad, *p.shape[1:])]) if pad else p
                p_slice = p_full[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                wd_eff = lr * wd * getattr(p, "wd_mul", 1.0) if wd != 0 else 0.0
                _adamw_update_(p_slice, grad_slices[idx], self.state[p], lr, beta1, beta2, eps,
                               wd_eff, self.stochastic_rounding, self.state_dtype)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p_full, p_slice, async_op=True).get_future())
                if pad:
                    padded_params.append((p, p_full))
        torch.futures.collect_all(all_reduce_futures).wait()
        for p, p_full in padded_params:
            p.copy_(p_full[: p.shape[0]])
