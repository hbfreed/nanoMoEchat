"""
Check that vLLM serves an exported AFMoE checkpoint like transformers does.

Step 1 (this repo's venv): fp32 transformers reference logprobs for random tokens
    python dev/vllm_parity.py ref <hf-export-dir> <ref.pt>
Step 2 (a venv with vLLM, which may pin a different torch): compare vLLM's
prompt logprobs against it
    python dev/vllm_parity.py check <hf-export-dir> <ref.pt>

A healthy bf16 export shows a median per-token |logprob diff| around 0.01 nats.
No tokenizer needed; token ids go in directly.
"""

import sys

import torch


def ref(export_dir, out):
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    hf = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.float32).cuda().eval()
    ids = torch.randint(0, hf.config.vocab_size, (4, 256), generator=torch.Generator().manual_seed(0))
    lp = torch.log_softmax(hf(ids.cuda()).logits.float(), -1)
    tok_lp = lp[:, :-1].gather(-1, ids[:, 1:, None].cuda()).squeeze(-1).cpu()
    torch.save({"ids": ids, "tok_lp": tok_lp}, out)
    print(f"saved reference logprobs {tuple(tok_lp.shape)} to {out}")


def check(export_dir, ref_path):
    from vllm import LLM, SamplingParams
    r = torch.load(ref_path)
    llm = LLM(model=export_dir, skip_tokenizer_init=True, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.3, max_model_len=512)
    outs = llm.generate([{"prompt_token_ids": x.tolist()} for x in r["ids"]],
                        SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False))
    d = torch.cat([
        (torch.tensor([pl[int(t)].logprob for pl, t in zip(o.prompt_logprobs[1:], ids[1:])]) - tlp).abs()
        for o, ids, tlp in zip(outs, r["ids"], r["tok_lp"])
    ])
    print(f"vLLM vs transformers per-token |logprob diff|: median {d.median():.4f} "
          f"mean {d.mean():.4f} p99 {d.quantile(0.99):.4f} max {d.max():.4f}")


if __name__ == "__main__":
    mode, export_dir, path = sys.argv[1:4]
    {"ref": ref, "check": check}[mode](export_dir, path)
