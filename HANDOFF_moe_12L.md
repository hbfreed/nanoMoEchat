# Handoff: 12-layer MoE sizing + SFT mixing + GPT-2 tokenizer + today's baseline run

Conversation happened in the **wrong repo** (`nanochat-moe` next door). File paths/line numbers below refer to `nanochat-moe`'s tree; cross-check against this repo before acting.

User is Henry (hbfreed@protonmail.com). **3× RTX 3090** (SM86, no FA3 → must use `--window-pattern=L`, BF16, no fp8).

---

## 1. Target model shape

**Goal:** 12 layers, ~10M active params, ~320M total params.

**Active/total formulas (per-layer, GQA with n_kv_head = n_head):**
- Attention (always active): `4 · H²`
- Router (always active): `H · E`
- MLP active (`K` experts): `2 · K · H · D`
- MLP total: `2 · E · H · D`

**Constraint we landed on:** `expert_dim` must be a multiple of 128 (grouped-GEMM tile size — `N % BLOCK_N == 0` in `nanochat/grouped_gemm.py:74`, with `BLOCK_N ∈ {32,64,128}`). User initially asked for multiples of 128 for expert dims; `num_experts` doesn't strictly need it kernel-wise but powers-of-2 are cleaner.

**Config chosen: A** (H=256, E=192, K=4, D=256) → **~10.0M active, ~306M total trunk**.

Per-layer breakdown:
- Attention: `4·256² = 262K`
- Router: `256·192 = 49K`
- MLP active: `2·4·256·256 = 524K`
- MLP total: `2·192·256·256 = 25.2M`
- × 12 layers → 3.15M attn + 0.59M router + 6.29M MLP-active = **10.0M active**; + 302M MLP-total = **306M total**.

**CLI translation note** — nanoMoEchat's `scripts/base_train.py` has a **different MoE API** than nanochat-moe:
- No `--use-moe` flag (MoE is always on, controlled by `--expert-sizes`).
- No `--aspect-ratio` / `--head-dim` — instead `--model-dim`, `--num-heads`, `--num-kv-heads`.
- Experts specified as `--expert-sizes '[[count, width], ...]'` JSON, not separate `--num-experts` / `--expert-dim`.
- Has extra `--compute-loss-weight` aux loss weight (default 0.004).
- `--target-param-data-ratio` default is **20** here (was 12 in nanochat-moe).

**CLI for Config A** (H=256 explicit, 2 heads of 128):
```bash
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
  --depth=12 \
  --model-dim=256 --num-heads=2 --num-kv-heads=2 \
  --expert-sizes='[[192,256]]' --num-active-experts=4 \
  --device-batch-size=4 --window-pattern=L \
  --total-batch-size=491520
```

`491520 = 24576 · 20`, divisible by `device_batch_size · max_seq_len · world_size = 4·2048·3 = 24576`. Default `524288` won't divide cleanly across 3 GPUs at this micro-batch shape.

Note: `num_experts=192` isn't a multiple of 128, but the grouped-GEMM kernel only requires `expert_dim % BLOCK_N == 0` — `num_experts` is unconstrained. D=256 satisfies the 128 rule.

The `target-param-data-ratio=12` default picks ~4B training tokens for this size (uses `transformer_matrices + lm_head`).

---

## 2. Ling Efficiency-Leverage paper (arXiv 2507.17702)

Full summary was saved to `nanochat-moe/knowledge/summary_moe_efficiency_leverage.md` — **copy or regenerate it here**. Key bearings for Henry's config:

- **Activation ratio A** is the primary efficiency driver (power-law, sparser=better).
- **Granularity G = 2·d_model/d_expert** has a sweet spot at **G≈8–12** under standard load-balancing (SMEBU might shift this higher).
- **One shared expert** is the "default" for large runs (not currently supported in `nanochat/moe.py`).
- **Dense-first layers** (1–3) help routing stability — also not currently in `GPTConfig`.
- MoE compute-optimal allocation: `M_opt = 0.1915·C^0.5095`, `D_opt = 5.22·C^0.4905` — **implies our `target-param-data-ratio=12` is too low for a well-trained MoE**. Bump it.

**Problem for Henry's 12L/10M-active/320M-total budget:** landing G≈8 with `d_expert % 128 == 0` forces `d_model ≥ 1024` which blows the 10M active ceiling. Concrete options we surfaced:
- H=256, D=128, E=256, K=16 → G=4, A=6.25%, but active ≈ 16M+ (over budget)
- H=512, D=128, E=?, K=? → G=8 ✓ but attention alone is 12.6M (over budget)
- Current 10M-active options (A/B/C above) land at G≈1.3 — **way below paper's optimum**.

**Open question Henry needs to answer:** relax the 10M active cap → G≈8, or relax the `multiple-of-128` D constraint → G≈8 at small H (kernel only strictly needs mult-32, the 128 was his conservative preference).

---

## 3. Dolci SFT data — two paths

User wants to "mix in some dolci SFT data" (`allenai/dolci` on HF). Haven't executed this yet — just mapped the approaches.

**(A) Into pretraining (annealing-style):** render dolci conversations with `render_conversation()` in `nanochat/tokenizer.py:266`, dump as a parquet shard with a `text` column into `base_data_climbmix/shard_0XXXX.parquet`. Gets picked up automatically by `list_parquet_files()` (`nanochat/dataset.py:32`). Duplicate shards to oversample.

**(B) Into SFT (standard):** new `tasks/dolci.py` following `tasks/smoltalk.py:10` pattern, add to `train_tasks` list in `scripts/chat_sft.py:165`. `TaskMixture` (`tasks/common.py:54`) handles proportional mixing — pass the same task multiple times to oversample.

Ask Henry which one before implementing — they're meaningfully different.

---

## 4. GPT-2 tokenizer swap

`RustBPETokenizer.from_pretrained("gpt2")` is already wired at `nanochat/tokenizer.py:200`. To switch:

1. Edit `get_tokenizer()` at `nanochat/tokenizer.py:390` — return `RustBPETokenizer.from_pretrained("gpt2")`.
2. **Regenerate `token_bytes.pt`** (needed by `loss_eval.evaluate_bpb` for bpb metric). It's normally written by `scripts/tok_train.py:88`. Either write a small standalone dumper, or skip bpb with `--eval-every=-1`.
3. **Add chat special tokens.** GPT-2 has `<|endoftext|>` (auto-used as BOS per line 207) but none of `<|user_start|>`, etc. For SFT to work, extend with `tiktoken.Encoding(..., special_tokens={...})` — pushes vocab past 50257. Cleanest as a helper/script.

Suggested deliverable: `scripts/tok_gpt2.py` that saves a GPT-2 + chat-tokens encoding + `token_bytes.pt` into `base_dir/tokenizer/`. Wait for Henry to confirm before building.

---

## 5. "Train something for fun today" — baseline dense d12

User pivoted at end: abandon MoE decisions, just train the dense d12 baseline on whatever data's local, for fun.

**Local state in `~/.cache/nanochat/`:**
- `base_checkpoints/d12/` — fully trained at step 4521 (val_bpb=1.075), but from an older code version (config has `num_heads`, `num_kv_heads`, `grad_clip` that current `base_train.py` doesn't accept). **Retrain fresh is cleaner than resuming.**
- `base_data/` — 71 FinewebEdu shards (6.4GB). Legacy auto-fallback triggers since `base_data_climbmix/` is empty (`nanochat/dataset.py:38`).
- `tokenizer/` — already present (vocab 32768, `token_bytes.pt` included).

**Proposed command** (targets ~2–3 hours on 3×3090):
```bash
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
  --depth=12 \
  --device-batch-size=4 \
  --window-pattern=L \
  --target-param-data-ratio=4 \
  --total-batch-size=491520 \
  --run=dummy
```

This leaves `--expert-sizes` at its default `[[64, 256]]` (just produces *some* dense-baseline-equivalent MoE run in this repo since MoE is always on). If Henry wants a true dense baseline, check whether this repo has a dense-mode escape hatch — I didn't verify.

Flags matter:
- `--window-pattern=L`: no FA3 → SDPA has no sliding-window fast path, would cripple MFU.
- `--device-batch-size=4`: 32 (default) OOMs on 24GB at seq_len=2048.
- `--target-param-data-ratio=4`: full Chinchilla-lite (12) would be ~12+ hours on this hardware; 4 cuts it to today-length at the cost of being undertrained.

**Was about to launch when user realized they were in the wrong repo.** Pending Henry's go-ahead.

---

## Immediate next step for you

Henry's last message asked for this handoff. Once he's in `nanoMoEchat` and reads you in, the most likely next action is to **kick off the baseline d12 run** (section 5) — but **confirm first**, since training jobs are the kind of high-side-effect action worth a sanity check. Everything else (MoE sizing, dolci, gpt-2 tokenizer) is design work that should wait on his direction.
