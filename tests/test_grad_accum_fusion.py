"""
GPT.set_fuse_grad_accum must hand the optimizer the gradients it got before: after a
4-micro-batch step, with zero_grad(set_to_none=True) between steps as base_train does,
every non-expert gradient is bitwise unchanged and the expert gradients differ only by
bf16 rounding (the kernels add each micro-batch in fp32 and round once, autograd rounds
the micro-batch's dW and then the sum). Covers activation checkpointing and
torch.compile, and the optimizer step that follows.

Run with:
python -m pytest tests/test_grad_accum_fusion.py -v
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat import flash_attention
from nanochat.gpt import GPT, GPTConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

EXPERT = ("w1", "w2", "w3")
MICRO = 4


@pytest.fixture(autouse=True)
def deterministic_attention(monkeypatch):
    # FA2's backward accumulates dq with atomics, so two identical unfused runs
    # already differ in every gradient upstream of the last attention layer. SDPA
    # (math) is deterministic, which leaves the expert gradients as the only change.
    monkeypatch.setattr(flash_attention, "USE_FA3", False)
    monkeypatch.setattr(flash_attention, "USE_FA2", False)
    with sdpa_kernel(SDPBackend.MATH):
        yield


def make_model(fuse, checkpointing=False, compile=False):
    torch.manual_seed(0)
    cfg = GPTConfig(
        sequence_len=256, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=256,
        window_pattern="SL", expert_sizes=[(16, 256)], num_active_experts=4,
        mlp_act="swiglu", use_bias_balancing=True,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cuda")
    model.init_weights()
    with torch.no_grad():  # init zeroes output projections; make every weight matter
        for p in model.parameters():
            if p.ndim == 2:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
    model.set_defer_bias_updates(True)
    model.set_activation_checkpointing(checkpointing)
    if fuse:
        model.set_fuse_grad_accum(True)
    model.train()
    return model, (torch.compile(model, dynamic=False) if compile else model)


def batches():
    gen = torch.Generator("cuda").manual_seed(1)
    return [torch.randint(0, 512, (2, 256), device="cuda", generator=gen) for _ in range(MICRO)]


def accumulate(model, run, xs):
    for x in xs:
        _, loss, _ = run(x, x)
        (loss / len(xs)).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def is_expert(name):
    return name.rsplit(".", 1)[-1] in EXPERT


def compare(grads0, grads1, first_only=False):
    assert grads0.keys() == grads1.keys()
    for name, g0 in grads0.items():
        g1 = grads1[name]
        assert g1.dtype == g0.dtype and g1.shape == g0.shape, name
        if not is_expert(name) or first_only:
            assert torch.equal(g1, g0), name
        else:
            rel = ((g1.float() - g0.float()).norm() / g0.float().norm()).item()
            assert 0 < rel < 4e-3, (name, rel)  # 0 would mean the fused path never ran


@pytest.mark.parametrize("checkpointing", [False, True])
def test_one_micro_batch_is_bitwise(checkpointing):
    # .grad is None, so the kernels write it, exactly as autograd would have.
    xs = batches()[:1]
    grads = [accumulate(m, run, xs) for m, run in
             (make_model(fuse, checkpointing) for fuse in (False, True))]
    compare(*grads, first_only=True)


@pytest.mark.parametrize("checkpointing,compile", [(False, False), (True, False), (False, True), (True, True)])
def test_four_micro_batches_match(checkpointing, compile):
    xs = batches()
    grads = []
    for fuse in (False, True):
        model, run = make_model(fuse, checkpointing, compile)
        accumulate(model, run, xs)  # step 1 (compiles), then zero_grad as base_train
        model.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in model.parameters())
        grads.append(accumulate(model, run, xs))
    compare(*grads)


def test_optimizer_step_matches():
    # The optimizer state and weights after a full step agree to bf16 rounding.
    xs = batches()
    params = []
    for fuse in (False, True):
        model, run = make_model(fuse)
        opts = model.setup_optimizers(unembedding_lr=0.004, embedding_lr=0.2,
                                      matrix_lr=0.02, weight_decay=0.0)
        accumulate(model, run, xs)
        for opt in opts:
            opt.step()
        model.zero_grad(set_to_none=True)
        params.append({n: p.detach().float().clone() for n, p in model.named_parameters()})
    p0, p1 = params
    for name in p0:
        if is_expert(name):
            rel = ((p1[name] - p0[name]).norm() / p0[name].norm()).item()
            assert rel < 1e-2, (name, rel)
        else:
            assert torch.equal(p1[name], p0[name]), name


def test_replaced_weight_raises():
    # Replacing a registered parameter (as fully_shard does) must fail loudly in
    # backward rather than write into a dead tensor.
    model, run = make_model(fuse=True)
    mlp = model.transformer.h[0].mlp
    mlp.w1 = torch.nn.Parameter(mlp.w1.detach().clone())
    with pytest.raises(RuntimeError, match="freed"):
        _, loss, _ = run(batches()[0], batches()[0])
        loss.backward()
