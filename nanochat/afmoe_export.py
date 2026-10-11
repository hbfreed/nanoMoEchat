"""
Convert an arch="afmoe" nanochat model to Hugging Face AfmoeForCausalLM.

The afmoe arch is built to match transformers' AfmoeForCausalLM (and vLLM's
afmoe.py) exactly, so conversion is a rename plus splitting the concatenated
expert matrices into per-expert Linear weights. tests/test_afmoe_export.py
checks the converted model's logits against ours.

Conventions handled here:
- sliding windows: our (left, 0) counts tokens to the left; HF/vLLM count the
  window including the current token, so sliding_window = left + 1.
- our w1 / w3 / w2 are (n_embd, E*D), (n_embd, E*D), (E*D, n_embd) with expert e
  in columns/rows [e*D, (e+1)*D); HF stores each expert's gate/up/down as
  (D, n_embd), (D, n_embd), (n_embd, D).
"""

import torch


def afmoe_hf_config(config, window_sizes):
    """HF AfmoeConfig kwargs for a nanochat GPTConfig with arch='afmoe'."""
    assert config.arch == "afmoe"
    (num_experts, expert_width), = config.expert_sizes
    windows = {w[0] for w in window_sizes if w[0] >= 0}
    assert len(windows) <= 1, f"AFMoE has one sliding window size, got {windows}"
    head_dim = config.n_embd // config.n_head
    return dict(
        architectures=["AfmoeForCausalLM"],
        model_type="afmoe",
        vocab_size=config.vocab_size,
        hidden_size=config.n_embd,
        intermediate_size=config.dense_intermediate_size or 4 * config.n_embd,
        moe_intermediate_size=expert_width,
        num_hidden_layers=config.n_layer,
        num_dense_layers=config.num_dense_layers,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_kv_head,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=config.sequence_len,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        num_experts=num_experts,
        num_experts_per_tok=config.num_active_experts,
        num_shared_experts=config.num_shared_experts,
        score_func="sigmoid",
        route_norm=True,
        route_scale=1.0,
        n_group=1,
        topk_group=1,
        sliding_window=(windows.pop() + 1) if windows else config.sequence_len,
        layer_types=["sliding_attention" if w[0] >= 0 else "full_attention" for w in window_sizes],
        mup_enabled=False,
        tie_word_embeddings=False,
        attention_bias=False,
    )


def afmoe_hf_state_dict(model):
    """HF AfmoeForCausalLM state dict from a nanochat GPT with arch='afmoe'."""
    cfg = model.config
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    out = {
        "model.embed_tokens.weight": sd["transformer.wte.weight"],
        "model.norm.weight": sd["transformer.norm.weight"],
        "lm_head.weight": sd["lm_head.weight"],
    }
    for i in range(cfg.n_layer):
        src, dst = f"transformer.h.{i}.", f"model.layers.{i}."
        for ours, theirs in [
            ("attn.c_q", "self_attn.q_proj"), ("attn.c_k", "self_attn.k_proj"),
            ("attn.c_v", "self_attn.v_proj"), ("attn.c_proj", "self_attn.o_proj"),
            ("attn.c_gate", "self_attn.gate_proj"), ("attn.q_norm", "self_attn.q_norm"),
            ("attn.k_norm", "self_attn.k_norm"), ("input_norm", "input_layernorm"),
            ("post_attn_norm", "post_attention_layernorm"),
            ("pre_mlp_norm", "pre_mlp_layernorm"), ("post_mlp_norm", "post_mlp_layernorm"),
        ]:
            out[f"{dst}{theirs}.weight"] = sd[f"{src}{ours}.weight"]
        if i < cfg.num_dense_layers:
            for ours, theirs in [("c_gate", "gate_proj"), ("c_fc", "up_proj"), ("c_proj", "down_proj")]:
                out[f"{dst}mlp.{theirs}.weight"] = sd[f"{src}mlp.{ours}.weight"]
            continue
        (num_experts, width), = cfg.expert_sizes
        w1, w3, w2 = sd[f"{src}mlp.w1"], sd[f"{src}mlp.w3"], sd[f"{src}mlp.w2"]
        for e in range(num_experts):
            cols = slice(e * width, (e + 1) * width)
            out[f"{dst}mlp.experts.{e}.gate_proj.weight"] = w1[:, cols].t().contiguous()
            out[f"{dst}mlp.experts.{e}.up_proj.weight"] = w3[:, cols].t().contiguous()
            out[f"{dst}mlp.experts.{e}.down_proj.weight"] = w2[cols, :].t().contiguous()
        out[f"{dst}mlp.router.gate.weight"] = sd[f"{src}mlp.router.weight"]
        bias = sd.get(f"{src}mlp.expert_bias")
        out[f"{dst}mlp.expert_bias"] = (bias if bias is not None
                                        else torch.zeros(num_experts)).float()
        if cfg.num_shared_experts > 0:
            for ours, theirs in [("c_gate", "gate_proj"), ("c_fc", "up_proj"), ("c_proj", "down_proj")]:
                out[f"{dst}mlp.shared_experts.{theirs}.weight"] = sd[f"{src}mlp.shared.{ours}.weight"]
        else:  # transformers builds a zero-width shared expert even when there are none
            n = cfg.n_embd
            out[f"{dst}mlp.shared_experts.gate_proj.weight"] = w1.new_zeros(0, n)
            out[f"{dst}mlp.shared_experts.up_proj.weight"] = w1.new_zeros(0, n)
            out[f"{dst}mlp.shared_experts.down_proj.weight"] = w1.new_zeros(n, 0)
    return out


def to_hf(model, dtype=torch.bfloat16, device="cpu"):
    """Build a transformers AfmoeForCausalLM with the same weights.
    Built on a real device (not meta) so its non-persistent rotary buffers exist."""
    from transformers import AfmoeConfig, AfmoeForCausalLM
    hf_cfg = AfmoeConfig(**{k: v for k, v in afmoe_hf_config(model.config, model.window_sizes).items()
                            if k not in ("architectures", "model_type")})
    with torch.device(device):
        hf = AfmoeForCausalLM(hf_cfg).to(dtype)
    sd = afmoe_hf_state_dict(model)
    sd = {k: (v.float() if k.endswith("expert_bias") else v.to(dtype)) for k, v in sd.items()}
    hf.load_state_dict(sd, strict=True)
    return hf.eval()
