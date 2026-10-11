"""
Parity: an arch="afmoe" nanochat model and its transformers AfmoeForCausalLM
conversion must produce the same logits, so trained checkpoints serve as-is.

Truth is the converted model in fp32; our model runs in bf16 like training, so
it must be within a small factor of the converted model's own bf16 error.
Covers sliding + full layers, GQA, shared experts, a dense first layer and a
nonzero expert bias.

Run with:
python -m pytest tests/test_afmoe_export.py -v -s
"""

import pytest
import torch

from nanochat.afmoe_export import to_hf
from nanochat.gpt import GPT, GPTConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_model(**overrides):
    defaults = dict(window_pattern="SSSL", num_shared_experts=1, num_dense_layers=1, use_bias_balancing=True, n_layer=4)
    defaults.update(overrides)
    torch.manual_seed(0)
    cfg = GPTConfig(
        arch="afmoe", sequence_len=256, vocab_size=512, n_head=4, n_kv_head=2,
        n_embd=256, sliding_window=32, expert_sizes=[(16, 128)],
        num_active_experts=4, mlp_act="swiglu", rope_theta=10000.0, **defaults,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    model.init_weights()
    # init zeroes lm_head and output projections; randomize everything so the
    # comparison exercises every weight
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
        for block in model.transformer.h:
            if hasattr(block.mlp, "expert_bias"):
                block.mlp.expert_bias.copy_(0.05 * torch.randn_like(block.mlp.expert_bias))
    return model.eval()


@pytest.mark.parametrize("overrides", [
    {},  # sliding + full layers, shared expert, dense first layer, expert bias
    {"num_shared_experts": 0},
    {"num_shared_experts": 0, "num_dense_layers": 0, "use_bias_balancing": False},
    {"window_pattern": "L"},  # all full attention: AFMoE layers are NoPE there
])
@torch.no_grad()
def test_afmoe_logits_match_transformers(overrides):
    """Our bf16 model must be as close to the fp32 HF conversion as HF's own bf16
    run is. MoE routing makes this the right yardstick: bf16 rounding flips
    near-tied expert choices for a few % of tokens in any implementation, which
    dominates the global error when there is no shared expert to dilute it."""
    model = make_model(**overrides)
    idx = torch.randint(0, model.config.vocab_size, (2, 200), device="cuda",
                        generator=torch.Generator("cuda").manual_seed(1))
    ours = model(idx).float()
    truth = to_hf(model, torch.float32, "cuda")(idx).logits.float()
    hf_bf16 = to_hf(model, torch.bfloat16, "cuda")(idx).logits.float()

    def errs(x):
        per_token = (x - truth).norm(dim=-1) / truth.norm(dim=-1)
        return ((x - truth).norm() / truth.norm()).item(), per_token.median().item()

    (g_ours, m_ours), (g_hf, m_hf) = errs(ours), errs(hf_bf16)
    print(f"{overrides}: global rel err ours {g_ours:.3f} / HF bf16 {g_hf:.3f} | "
          f"median per-token ours {m_ours:.4f} / HF bf16 {m_hf:.4f}")
    assert g_ours < 1.5 * g_hf + 1e-3, (g_ours, g_hf)
    assert m_ours < 1.5 * m_hf + 1e-3, (m_ours, m_hf)
