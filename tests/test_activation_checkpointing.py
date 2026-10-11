"""
Activation checkpointing must not change the math: same loss and gradients as without,
for an afmoe MoE model with bias balancing (deferred SMEBU updates, so the recompute
routes exactly like the original forward), and the deferred update must match the
in-forward one.

Run with:
python -m pytest tests/test_activation_checkpointing.py -v
"""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_model():
    torch.manual_seed(0)
    cfg = GPTConfig(
        arch="afmoe", sequence_len=256, vocab_size=512, n_layer=3, n_head=4, n_kv_head=2,
        n_embd=256, window_pattern="SSL", sliding_window=64, expert_sizes=[(16, 128)],
        num_active_experts=4, mlp_act="swiglu", rope_theta=10000.0, use_bias_balancing=True,
        num_shared_experts=1, num_dense_layers=1,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    model.init_weights()
    with torch.no_grad():  # init zeroes output projections; make every weight matter
        for p in model.parameters():
            if p.ndim == 2:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
    return model.train()


def loss_and_grads(checkpointing):
    model = make_model()
    model.set_defer_bias_updates(True)
    model.set_activation_checkpointing(checkpointing)
    gen = torch.Generator("cuda").manual_seed(1)
    x = torch.randint(0, 512, (4, 256), device="cuda", generator=gen)
    _, loss, aux = model(x, x)
    loss.backward()
    return loss.detach(), {n: p.grad.float() for n, p in model.named_parameters()}, aux["tokens_per_expert"]


def test_checkpointing_matches():
    loss0, grads0, counts0 = loss_and_grads(False)
    loss1, grads1, counts1 = loss_and_grads(True)
    assert torch.equal(counts0, counts1)
    torch.testing.assert_close(loss1, loss0, rtol=1e-3, atol=1e-3)
    for name in grads0:
        g0, g1 = grads0[name], grads1[name]
        rel = ((g1 - g0).norm() / (g0.norm() + 1e-12)).item()
        assert rel < 2e-2, (name, rel)


@torch.no_grad()
def test_deferred_bias_update_matches_in_forward():
    """One micro-batch: GPT.update_expert_biases(counts) == the in-forward SMEBU update."""
    x = torch.randint(0, 512, (4, 256), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    eager = make_model()
    eager(x, x)  # in-forward update
    deferred = make_model()
    deferred.set_defer_bias_updates(True)
    _, _, aux = deferred(x, x)
    deferred.update_expert_biases(aux["tokens_per_expert"])
    for a, b in zip(eager.bias_balanced_mlps(), deferred.bias_balanced_mlps()):
        assert torch.equal(a.expert_bias, b.expert_bias)
        assert torch.equal(a.expert_bias_momentum, b.expert_bias_momentum)
