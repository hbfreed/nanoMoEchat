"""
Round 2: Token-Level Routing Analysis for Null + Variable Expert MoE.

Loads trained MoE checkpoints (B, C), runs inference on fineweb validation data,
and records per-token routing decisions across all layers. Produces:
  - Per-token summary CSVs (avg_expert_size, null_fraction, etc.)
  - Scatter plots and markdown report

Usage:
    uv run python analysis/token_routing_analysis.py
"""

import os
import sys
import json
import functools
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nanochat.gpt import GPT, GPTConfig, MoEMLP
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ─────────────────────────────────────────────────────
CHECKPOINT_BASE = "/media/henry/MoreFiles/base_checkpoints/null_expert_sweep"
RUNS = {
    "B": {
        "checkpoint_dir": os.path.join(CHECKPOINT_BASE, "null-B"),
        "step": 4521,
        "label": "B: 48 uniform + 16 null",
    },
    "C": {
        "checkpoint_dir": os.path.join(CHECKPOINT_BASE, "null-C"),
        "step": 4521,
        "label": "C: 24 large + 24 small + 16 null",
    },
}
N_LAYERS = 12
NUM_ACTIVE = 8  # top-k
EVAL_BATCHES = 3200  # number of batches to evaluate (~full val set, ~52M token positions)
BATCH_SIZE = 16
SEQ_LEN = 1024
MIN_TOKEN_COUNT = 5  # minimum occurrences for inclusion in results
DEVICE = "cuda"

# ── Expert width lookup ───────────────────────────────────────────────
def build_expert_width_map(expert_sizes, num_null_experts):
    """Return list of FFN widths indexed by expert ID. Null experts have width 0."""
    widths = []
    for count, width in expert_sizes:
        widths.extend([width] * count)
    widths.extend([0] * num_null_experts)
    return widths


# ── Monkey-patch MoEMLP to capture routing decisions ──────────────────
_original_forward = MoEMLP.forward

@functools.wraps(_original_forward)
def _tracking_forward(self, x):
    batch_size, seq_len, n_embd = x.shape
    # Run the original forward
    output, aux_loss, f_i = _original_forward(self, x)
    # The selected_experts and top_k_weights are computed inside forward()
    # but not returned. We recompute the routing (cheap — just router + topk)
    # to capture them without modifying the original code.
    from einops import rearrange
    x_flat = rearrange(x, "b s d -> (b s) d")
    router_logits = self.router(x_flat)
    if self.num_null_experts > 0:
        null_logit = router_logits[:, self.num_real_experts:]
        router_logits = torch.cat([
            router_logits[:, :self.num_real_experts],
            null_logit.expand(-1, self.num_null_experts),
        ], dim=-1)
    router_probs = F.sigmoid(router_logits.to(torch.float32))
    if self.use_bias_balancing:
        selection_scores = router_probs + self.expert_bias.unsqueeze(0)
        _, selected_experts = torch.topk(selection_scores, self.num_active_experts, dim=-1)
        top_k_weights = router_probs.gather(-1, selected_experts)
    else:
        top_k_weights, selected_experts = torch.topk(router_probs, self.num_active_experts, dim=-1)
    # Zero out null weights and renormalize (matching training behavior)
    if self.num_null_experts > 0:
        is_null = (selected_experts >= self.num_real_experts)
        top_k_weights = top_k_weights.masked_fill(is_null, 0.0)
    top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
    # Stash for collection — reshape to (batch, seq, k)
    self._tracked_experts = selected_experts.view(batch_size, seq_len, -1).detach().cpu()
    self._tracked_weights = top_k_weights.view(batch_size, seq_len, -1).detach().cpu()
    return output, aux_loss, f_i


def enable_tracking():
    MoEMLP.forward = _tracking_forward

def disable_tracking():
    MoEMLP.forward = _original_forward


# ── Load model from checkpoint ────────────────────────────────────────
def load_model(checkpoint_dir, step):
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, DEVICE)
    # Strip _orig_mod. prefix if present (torch.compile artifact)
    model_data = {k.replace("_orig_mod.", ""): v for k, v in model_data.items()}
    model_config = GPTConfig(**meta_data["model_config"])
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=DEVICE)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    return model, meta_data


# ── Collect routing decisions ─────────────────────────────────────────
def collect_routing(model, run_key, run_meta):
    """Run inference and collect per-token routing decisions."""
    tokenizer = get_tokenizer()

    # Set up DDP env vars for dataloader (single-GPU inference)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    val_loader = tokenizing_distributed_data_loader(
        B=BATCH_SIZE, T=SEQ_LEN, split="val", device=DEVICE
    )

    expert_widths = build_expert_width_map(
        run_meta["expert_sizes"], run_meta["num_null_experts"]
    )
    expert_widths_t = torch.tensor(expert_widths, dtype=torch.float32)
    num_real = sum(c for c, _ in run_meta["expert_sizes"])

    expert_widths_np = np.array(expert_widths, dtype=np.float64)

    # Accumulators per token_id (use numpy arrays, keyed by token_id)
    # For each token: count (layer-occurrences), total_expert_size, total_expert_size_excl_null,
    #                 total_null_count, total_routing_slots, total_real_slots
    # Use vocab-sized arrays for O(1) indexing
    vocab_size = 65536  # nanochat vocab
    acc_count = np.zeros(vocab_size, dtype=np.int64)
    acc_expert_size = np.zeros(vocab_size, dtype=np.float64)
    acc_expert_size_excl_null = np.zeros(vocab_size, dtype=np.float64)
    acc_null_count = np.zeros(vocab_size, dtype=np.int64)
    acc_routing_slots = np.zeros(vocab_size, dtype=np.int64)
    acc_real_slots = np.zeros(vocab_size, dtype=np.int64)

    enable_tracking()

    print(f"  Running inference ({EVAL_BATCHES} batches, {BATCH_SIZE}x{SEQ_LEN})...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx >= EVAL_BATCHES:
                break
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{EVAL_BATCHES}")

            # Forward pass (triggers tracking in each MoEMLP layer)
            _ = model(inputs, targets)

            # Collect from all layers — vectorized
            token_ids = inputs.cpu().numpy().ravel()  # (B*T,)
            for layer_idx, block in enumerate(model.transformer.h):
                mlp = block.mlp
                if not hasattr(mlp, "_tracked_experts"):
                    continue
                experts = mlp._tracked_experts.numpy().reshape(-1, NUM_ACTIVE)  # (B*T, k)

                # Look up widths for all selected experts
                widths = expert_widths_np[experts]   # (B*T, k)
                is_null = experts >= num_real         # (B*T, k)
                is_real = ~is_null

                # Accumulate per token_id using np.add.at
                np.add.at(acc_count, token_ids, 1)
                np.add.at(acc_expert_size, token_ids, widths.sum(axis=1))
                np.add.at(acc_routing_slots, token_ids, NUM_ACTIVE)

                # Real slots and their sizes
                real_widths = widths * is_real  # null widths are 0 anyway, but be explicit
                np.add.at(acc_expert_size_excl_null, token_ids, real_widths.sum(axis=1))
                np.add.at(acc_real_slots, token_ids, is_real.sum(axis=1))
                np.add.at(acc_null_count, token_ids, is_null.sum(axis=1))

    disable_tracking()

    # Convert to per-token summaries
    # count was incremented per (layer, occurrence), so actual token occurrences = count / N_LAYERS
    results = []
    active_tids = np.where(acc_count > 0)[0]
    for tid in active_tids:
        token_occurrences = int(acc_count[tid]) // N_LAYERS
        if token_occurrences < MIN_TOKEN_COUNT:
            continue
        total_slots = int(acc_routing_slots[tid])
        real_slots = int(acc_real_slots[tid])

        avg_expert_size = acc_expert_size[tid] / total_slots if total_slots > 0 else 0
        avg_expert_size_excl_null = (
            acc_expert_size_excl_null[tid] / real_slots if real_slots > 0 else 0
        )
        null_fraction = int(acc_null_count[tid]) / total_slots if total_slots > 0 else 0

        # Decode token
        try:
            token_str = tokenizer.decode([int(tid)])
            # Sanitize for CSV
            token_str = token_str.replace("\n", "\\n").replace("\r", "\\r").replace(",", "<comma>").replace("|", "¦")
            if not token_str.isprintable():
                token_str = f"<ID:{tid}>"
        except Exception:
            token_str = f"<ID:{tid}>"

        results.append({
            "token_id": int(tid),
            "token_str": token_str,
            "count": token_occurrences,
            "avg_expert_size": round(avg_expert_size, 2),
            "avg_expert_size_excluding_null": round(avg_expert_size_excl_null, 2),
            "null_fraction": round(null_fraction, 6),
        })

    return results


# ── Write CSV ─────────────────────────────────────────────────────────
def write_csv(results, filename):
    path = os.path.join(OUT_DIR, filename)
    cols = ["token_id", "token_str", "count", "avg_expert_size",
            "avg_expert_size_excluding_null", "null_fraction"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  Wrote {path} ({len(results)} tokens)")
    return path


# ── Plots ─────────────────────────────────────────────────────────────
def plot_scatter(results, run_key, label):
    """Scatter: null_fraction (x) vs avg_expert_size_excluding_null (y), colored by frequency."""
    nf = [r["null_fraction"] for r in results]
    es = [r["avg_expert_size_excluding_null"] for r in results]
    counts = [r["count"] for r in results]
    log_counts = np.log10(np.array(counts) + 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(nf, es, c=log_counts, cmap="viridis", alpha=0.5, s=8, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log10(count + 1)")
    ax.set_xlabel("Null Fraction")
    ax.set_ylabel("Avg Expert Size (excl. null)")
    ax.set_title(f"Run {run_key}: Token Routing — Null Fraction vs Expert Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"scatter_{run_key}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


def plot_null_fraction_histogram(results, run_key):
    """Histogram of null_fraction distribution across tokens."""
    nf = [r["null_fraction"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nf, bins=50, color="#ff7f0e" if run_key == "B" else "#2ca02c", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Null Fraction")
    ax.set_ylabel("Number of Tokens")
    ax.set_title(f"Run {run_key}: Distribution of Per-Token Null Fraction")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fname = f"null_frac_hist_{run_key}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")
    return fname


# ── Flag specific tokens ─────────────────────────────────────────────
def flag_tokens(results, tokenizer):
    """Find specific tokens of interest and return a dict of {label: result_row}."""
    # Build reverse lookup: token_id -> result
    by_id = {r["token_id"]: r for r in results}

    flagged = {}

    # Token 447 (UTF-8 continuation byte)
    if 447 in by_id:
        flagged["Token 447 (0xbf continuation)"] = by_id[447]

    # Common function words
    for word in [" the", " of", " in", " and", " to", " a", " is", " that", " for", " it"]:
        ids = tokenizer.encode([word], num_threads=1)[0]
        if len(ids) == 1 and ids[0] in by_id:
            flagged[f'"{word.strip()}"'] = by_id[ids[0]]

    # Punctuation
    for punct in [".", ",", "!", "?", ":", ";", "(", ")", '"', "'"]:
        ids = tokenizer.encode([punct], num_threads=1)[0]
        if len(ids) == 1 and ids[0] in by_id:
            flagged[f'"{punct}"'] = by_id[ids[0]]

    # Code-ish tokens
    for code_tok in [" def", " return", " import", " class", " function", " if", " else"]:
        ids = tokenizer.encode([code_tok], num_threads=1)[0]
        if len(ids) == 1 and ids[0] in by_id:
            flagged[f'"{code_tok.strip()}"'] = by_id[ids[0]]

    return flagged


# ── Generate markdown report ──────────────────────────────────────────
def generate_report(all_results, all_flagged, scatter_fnames, hist_fnames):
    md = []
    md.append("# Round 2: Token-Level Routing Analysis\n")
    md.append("## Setup\n")
    md.append(f"- Evaluation: {EVAL_BATCHES} batches x {BATCH_SIZE} sequences x {SEQ_LEN} tokens = "
              f"{EVAL_BATCHES * BATCH_SIZE * SEQ_LEN:,} token positions")
    md.append(f"- Minimum token count for inclusion: {MIN_TOKEN_COUNT}")
    md.append(f"- Routing: sigmoid + top-{NUM_ACTIVE}, weights renormalized after null zeroing")
    md.append(f"- Expert width lookup: null=0, sizes per run config\n")

    for run_key in ["B", "C"]:
        if run_key not in all_results:
            continue
        results = all_results[run_key]
        meta = RUNS[run_key]
        flagged = all_flagged[run_key]

        md.append(f"---\n## Run {run_key}: {meta['label']}\n")

        # Summary statistics
        nfs = [r["null_fraction"] for r in results]
        sizes = [r["avg_expert_size_excluding_null"] for r in results]
        md.append(f"**{len(results)} unique tokens** (min count >= {MIN_TOKEN_COUNT})\n")
        md.append(f"| Statistic | Null Fraction | Avg Expert Size (excl null) |")
        md.append(f"|-----------|--------------|----------------------------|")
        md.append(f"| Mean | {np.mean(nfs):.4f} | {np.mean(sizes):.1f} |")
        md.append(f"| Median | {np.median(nfs):.4f} | {np.median(sizes):.1f} |")
        md.append(f"| Std | {np.std(nfs):.4f} | {np.std(sizes):.1f} |")
        md.append(f"| Min | {np.min(nfs):.4f} | {np.min(sizes):.1f} |")
        md.append(f"| Max | {np.max(nfs):.4f} | {np.max(sizes):.1f} |")
        md.append("")

        # Scatter plot (only for variable-size runs)
        if run_key in scatter_fnames:
            md.append(f"### Null Fraction vs Expert Size\n")
            md.append(f"![Scatter]({scatter_fnames[run_key]})\n")

        # Null fraction histogram
        md.append(f"### Null Fraction Distribution\n")
        md.append(f"![Histogram]({hist_fnames[run_key]})\n")

        # Top tokens by null fraction (most null)
        by_null = sorted(results, key=lambda r: r["null_fraction"], reverse=True)
        md.append(f"### Top 20 Tokens by Null Fraction (most null-routed)\n")
        md.append("| Token | ID | Count | Null Frac | Avg Size | Avg Size (excl null) |")
        md.append("|-------|-----|-------|-----------|----------|---------------------|")
        for r in by_null[:20]:
            md.append(f"| `{r['token_str']}` | {r['token_id']} | {r['count']} | "
                      f"{r['null_fraction']:.4f} | {r['avg_expert_size']:.0f} | "
                      f"{r['avg_expert_size_excluding_null']:.0f} |")
        md.append("")

        # Top tokens by expert size (most compute-demanding)
        by_size = sorted(results, key=lambda r: r["avg_expert_size_excluding_null"], reverse=True)
        md.append(f"### Top 20 Tokens by Avg Expert Size excl null (most compute)\n")
        md.append("| Token | ID | Count | Null Frac | Avg Size | Avg Size (excl null) |")
        md.append("|-------|-----|-------|-----------|----------|---------------------|")
        for r in by_size[:20]:
            md.append(f"| `{r['token_str']}` | {r['token_id']} | {r['count']} | "
                      f"{r['null_fraction']:.4f} | {r['avg_expert_size']:.0f} | "
                      f"{r['avg_expert_size_excluding_null']:.0f} |")
        md.append("")

        # Bottom tokens by null fraction (least null)
        md.append(f"### Bottom 20 Tokens by Null Fraction (least null-routed)\n")
        md.append("| Token | ID | Count | Null Frac | Avg Size | Avg Size (excl null) |")
        md.append("|-------|-----|-------|-----------|----------|---------------------|")
        for r in by_null[-20:]:
            md.append(f"| `{r['token_str']}` | {r['token_id']} | {r['count']} | "
                      f"{r['null_fraction']:.4f} | {r['avg_expert_size']:.0f} | "
                      f"{r['avg_expert_size_excluding_null']:.0f} |")
        md.append("")

        # Flagged tokens
        md.append(f"### Flagged Tokens\n")
        md.append("| Category | Token | ID | Count | Null Frac | Avg Size (excl null) |")
        md.append("|----------|-------|-----|-------|-----------|---------------------|")
        for label, r in sorted(flagged.items()):
            md.append(f"| {label} | `{r['token_str']}` | {r['token_id']} | {r['count']} | "
                      f"{r['null_fraction']:.4f} | {r['avg_expert_size_excluding_null']:.0f} |")
        md.append("")

    # Cross-run comparison for flagged tokens
    md.append("---\n## Cross-Run Comparison: Flagged Tokens\n")
    # Find tokens present in both runs
    b_by_id = {r["token_id"]: r for r in all_results["B"]}
    c_by_id = {r["token_id"]: r for r in all_results["C"]}
    common_flagged = {}
    for label, r in all_flagged["B"].items():
        tid = r["token_id"]
        if tid in c_by_id:
            common_flagged[label] = (r, c_by_id[tid])

    if common_flagged:
        md.append("| Token | B Null Frac | C Null Frac | B Avg Size (incl null) | C Avg Size (incl null) |")
        md.append("|-------|------------|------------|----------------------|----------------------|")
        for label, (rb, rc) in sorted(common_flagged.items()):
            md.append(f"| {label} | {rb['null_fraction']:.4f} | {rc['null_fraction']:.4f} | "
                      f"{rb['avg_expert_size']:.0f} | {rc['avg_expert_size']:.0f} |")
    md.append("")

    report_path = os.path.join(OUT_DIR, "analysis_round2.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"\nReport written to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run key to analyze (e.g. B or C). Default: all runs.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:1)")
    parser.add_argument("--skip-report", action="store_true", help="Skip cross-run report generation")
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = args.device

    tokenizer = get_tokenizer()

    runs_to_process = {args.run: RUNS[args.run]} if args.run else RUNS

    all_results = {}
    all_flagged = {}
    scatter_fnames = {}
    hist_fnames = {}

    for run_key, run_meta in runs_to_process.items():
        print(f"\n{'='*60}")
        print(f"Run {run_key}: {run_meta['label']}")
        print(f"{'='*60}")

        # Check checkpoint exists
        ckpt_path = os.path.join(run_meta["checkpoint_dir"], f"model_{run_meta['step']:06d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP: checkpoint not found at {ckpt_path}")
            continue

        # Load model
        print(f"  Loading checkpoint from {run_meta['checkpoint_dir']} step {run_meta['step']}...")
        model, meta = load_model(run_meta["checkpoint_dir"], run_meta["step"])
        mc = meta["model_config"]
        print(f"  Model loaded. Config: {mc}")

        # Extract expert config from checkpoint meta
        expert_sizes = [tuple(x) for x in mc["expert_sizes"]]
        num_null_experts = mc.get("num_null_experts", 0)
        run_meta["expert_sizes"] = expert_sizes
        run_meta["num_null_experts"] = num_null_experts

        # Collect routing decisions
        results = collect_routing(model, run_key, run_meta)
        all_results[run_key] = results
        print(f"  Collected {len(results)} unique tokens")

        # Write CSV (sorted by null_fraction descending)
        results_sorted = sorted(results, key=lambda r: r["null_fraction"], reverse=True)
        write_csv(results_sorted, f"token_routing_{run_key}.csv")

        # Plots (skip scatter for uniform-size runs — it's just a flat line)
        has_variable_experts = len(run_meta.get("expert_sizes", [])) > 1
        if has_variable_experts:
            scatter_fnames[run_key] = plot_scatter(results, run_key, run_meta["label"])
        hist_fnames[run_key] = plot_null_fraction_histogram(results, run_key)

        # Flag specific tokens
        flagged = flag_tokens(results, tokenizer)
        all_flagged[run_key] = flagged
        print(f"  Flagged {len(flagged)} tokens of interest")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Generate report (skip if single-run mode or --skip-report)
    if all_results and not args.skip_report and len(all_results) > 1:
        generate_report(all_results, all_flagged, scatter_fnames, hist_fnames)

    print("\nDone!")


if __name__ == "__main__":
    main()
