"""
--token-rounding nearest: while training, the swiglu MoE layers round each expert's routes to
whole 128-row tiles (dropping the lowest-scoring routes of experts that round down) and report
the dropped fraction; evaluation keeps every route.

python -m pytest tests/test_token_rounding.py -q
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _model():
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=256, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=256,
                    window_pattern="L", expert_sizes=[(8, 256)], num_active_experts=2, mlp_act="swiglu")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    model.init_weights()
    for p in model.parameters():  # nonzero output projections, so the MoE output matters
        torch.nn.init.normal_(p, std=0.02)
    return model


def test_nearest_drops_routes_only_in_training():
    torch.manual_seed(0)
    model = _model()
    x = torch.randint(0, 512, (4, 256), device="cuda")
    y = torch.randint(0, 512, (4, 256), device="cuda")
    model.train()
    _, loss_ceil, aux_ceil = model(x, y)
    model.set_token_rounding("nearest")
    _, loss_near, aux_near = model(x, y)
    assert "dropped_route_fraction" not in aux_ceil
    dropped = aux_near["dropped_route_fraction"].item()
    assert 0 < dropped < 0.5  # 2048 routes over 8 experts: some round down
    assert loss_near.item() != loss_ceil.item()
    loss_near.backward()  # the dropped routes are differentiable no-ops
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        _, loss_eval_near, aux_eval = model(x, y)
        model.set_token_rounding("ceil")
        _, loss_eval_ceil, _ = model(x, y)
    assert "dropped_route_fraction" not in aux_eval
    assert loss_eval_near.item() == loss_eval_ceil.item()
