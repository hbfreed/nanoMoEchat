"""
MoE Null + Variable Expert Routing Analysis
Fetches wandb data for runs A/B/C, generates per-layer routing diagnostics,
comparison plots, and a markdown report.
"""

import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json, sys
from collections import defaultdict

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Run metadata ──────────────────────────────────────────────────────
RUNS = {
    "A": {"id": "45a1pu7o", "label": "A: 48 uniform, no nulls",
           "n_real": 48, "n_null": 0, "n_total": 48,
           "expert_desc": "48×256"},
    "B": {"id": "8ysh19et", "label": "B: 48 uniform + 16 null",
           "n_real": 48, "n_null": 16, "n_total": 64,
           "expert_desc": "48×256 + 16 null"},
    "C": {"id": "vugp7ip2", "label": "C: 24 large + 24 small + 16 null",
           "n_real": 48, "n_null": 16, "n_total": 64,
           "expert_desc": "24×384 + 24×128 + 16 null"},
}
PROJECT = "hbfreed/nanochat"
N_LAYERS = 12

api = wandb.Api()

# ── 1. Fetch data ────────────────────────────────────────────────────
print("Fetching wandb data...")

data = {}  # run_key -> dict of metric arrays
for key, meta in RUNS.items():
    run = api.run(f"{PROJECT}/{meta['id']}")

    print(f"  Fetching run {key} ({meta['id']})...")
    # Fetch ALL rows without key filtering (keys= causes rows to be dropped
    # when not all keys are present, since metrics are logged at different intervals)
    rows = list(run.scan_history(page_size=10000))
    print(f"    Got {len(rows)} rows")

    data[key] = {"rows": rows, "meta": meta, "run_obj": run}

print("Data fetched.\n")

# ── Helper: extract metric timeseries ────────────────────────────────
def extract_ts(rows, metric):
    """Return (steps, values) arrays for a metric, skipping None."""
    steps, vals = [], []
    for r in rows:
        v = r.get(metric)
        if v is not None:
            steps.append(r.get("step", r.get("_step", 0)))
            vals.append(v)
    return np.array(steps), np.array(vals)


# ── 2. Training curves plot ──────────────────────────────────────────
print("Plotting training curves...")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
final_metrics = {}

for key in ["A", "B", "C"]:
    rows = data[key]["rows"]
    meta = data[key]["meta"]
    final_metrics.setdefault(key, {})
    steps, bpb = extract_ts(rows, "val/bpb")
    if len(steps) > 0:
        ax.plot(steps, bpb, label=meta["label"], color=colors[key], linewidth=1.5)
        final_metrics[key]["final_bpb"] = bpb[-1]
        final_metrics[key]["final_step"] = int(steps[-1])
    # Get total training time
    _, times = extract_ts(rows, "total_training_time")
    if len(times) > 0:
        final_metrics[key]["total_time"] = times[-1]
    # Get null fraction
    _, nf = extract_ts(rows, "train/null_fraction")
    if len(nf) > 0:
        final_metrics[key]["null_fraction"] = nf[-1]
    # Get real experts per token
    _, rept = extract_ts(rows, "train/real_experts_per_token")
    if len(rept) > 0:
        final_metrics[key]["real_experts_per_token"] = rept[-1]
    # Get zero compute fraction
    _, zcf = extract_ts(rows, "train/zero_compute_token_fraction")
    if len(zcf) > 0:
        final_metrics[key]["zero_compute_frac"] = zcf[-1]

# Zoom to step 2000+ and auto-fit y to visible data
crop_min = 2000
all_cropped_bpb = []
for key in ["A", "B", "C"]:
    s, b = extract_ts(data[key]["rows"], "val/bpb")
    mask = s >= crop_min
    if mask.any():
        all_cropped_bpb.extend(b[mask])
if all_cropped_bpb:
    ymin, ymax = min(all_cropped_bpb), max(all_cropped_bpb)
    pad = (ymax - ymin) * 0.15 or 0.005
    ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlim(left=crop_min)
ax.set_xlabel("Step")
ax.set_ylabel("Validation BPB")
ax.set_title("Validation BPB Over Training (step 2000+)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "val_bpb_curves.png"), dpi=150)
plt.close(fig)
print("  Saved val_bpb_curves.png")


# ── 3. Null fraction over training (B and C) ────────────────────────
print("Plotting null fraction over training...")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
for key in ["B", "C"]:
    rows = data[key]["rows"]
    meta = data[key]["meta"]
    steps, nf = extract_ts(rows, "train/null_fraction")
    if len(steps) > 0:
        ax.plot(steps, nf, label=meta["label"], color=colors[key], linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Null Fraction")
ax.set_title("Null Expert Fraction Over Training (fraction of top-8 slots)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "null_fraction_training.png"), dpi=150)
plt.close(fig)
print("  Saved null_fraction_training.png")


# ── 4. Per-layer analysis (from last 1000-step checkpoint) ───────────
print("Per-layer analysis...")

def get_per_layer_usage(rows, n_layers, n_total):
    """Get the LAST logged per-layer expert usage snapshot.
    Returns shape (n_layers, n_total) or None."""
    # Find rows that have per-layer data (logged every 1000 steps)
    key0 = "moe/expert_usage_layer_0_expert_0"
    valid_rows = [r for r in rows if r.get(key0) is not None]
    if not valid_rows:
        return None, None
    last_row = valid_rows[-1]
    step = last_row.get("step", last_row.get("_step", 0))
    usage = np.zeros((n_layers, n_total))
    for layer in range(n_layers):
        for expert in range(n_total):
            v = last_row.get(f"moe/expert_usage_layer_{layer}_expert_{expert}")
            if v is not None:
                usage[layer, expert] = v
    return usage, step


layer_analysis = {}
for key in ["B", "C"]:
    meta = data[key]["meta"]
    usage, step = get_per_layer_usage(data[key]["rows"], N_LAYERS, meta["n_total"])
    if usage is None:
        print(f"  WARNING: No per-layer data for run {key}")
        continue
    print(f"  Run {key}: per-layer data from step {step}")

    n_real = meta["n_real"]
    n_null = meta["n_null"]

    # Null fraction per layer
    # usage[layer, :] sums to 1.0 (it's f_i = fraction of top-k slots)
    # Sum of null expert fractions = fraction of top-k going to null
    null_frac_per_layer = usage[:, n_real:].sum(axis=1)

    # Real expert usage stats per layer
    real_usage = usage[:, :n_real]
    real_stats = {
        "min": real_usage.min(axis=1),
        "max": real_usage.max(axis=1),
        "mean": real_usage.mean(axis=1),
        "std": real_usage.std(axis=1),
    }

    # Dead experts (usage < 0.1 * uniform)
    uniform = 1.0 / meta["n_total"]
    dead_threshold = 0.1 * uniform
    dead_per_layer = (real_usage < dead_threshold).sum(axis=1)

    la = {
        "usage": usage,
        "step": step,
        "null_frac_per_layer": null_frac_per_layer,
        "real_stats": real_stats,
        "dead_per_layer": dead_per_layer,
        "uniform": uniform,
    }

    # Run C specific: split by size group
    if key == "C":
        # Large experts: 0-23, Small experts: 24-47, Null: 48-63
        large_usage = usage[:, :24].sum(axis=1)
        small_usage = usage[:, 24:48].sum(axis=1)
        null_usage = usage[:, 48:].sum(axis=1)
        la["large_agg"] = large_usage
        la["small_agg"] = small_usage
        la["null_agg"] = null_usage
        # Per-expert means within groups
        la["large_per_expert_mean"] = usage[:, :24].mean(axis=1)
        la["small_per_expert_mean"] = usage[:, 24:48].mean(axis=1)

    layer_analysis[key] = la


# ── 5. Per-layer null fraction bar chart ─────────────────────────────
print("Plotting per-layer null fraction...")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
x = np.arange(N_LAYERS)
width = 0.35
for i, key in enumerate(["B", "C"]):
    if key in layer_analysis:
        la = layer_analysis[key]
        offset = (i - 0.5) * width
        ax.bar(x + offset, la["null_frac_per_layer"],
               width, label=RUNS[key]["label"], color=colors[key], alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Null Fraction (of top-8 slots)")
ax.set_title("Null Expert Fraction by Layer")
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "null_fraction_per_layer.png"), dpi=150)
plt.close(fig)
print("  Saved null_fraction_per_layer.png")


# ── 6. Expert usage heatmaps ────────────────────────────────────────
print("Plotting expert usage heatmaps...")
for key in ["B", "C"]:
    if key not in layer_analysis:
        continue
    la = layer_analysis[key]
    meta = RUNS[key]
    usage = la["usage"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    im = ax.imshow(usage, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title(f"Run {key}: Expert Usage (f_i) per Layer — step {la['step']}")
    ax.set_yticks(range(N_LAYERS))

    # Mark boundary between real and null experts
    n_real = meta["n_real"]
    ax.axvline(x=n_real - 0.5, color="red", linewidth=2, linestyle="--", label="Real|Null boundary")
    if key == "C":
        ax.axvline(x=23.5, color="white", linewidth=1.5, linestyle=":", label="Large|Small boundary")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, label="f_i (fraction of routing)")
    fig.tight_layout()
    fname = f"expert_usage_heatmap_{key}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 7. Run C: size group usage per layer ─────────────────────────────
if "C" in layer_analysis:
    print("Plotting Run C size group analysis...")
    la = layer_analysis["C"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aggregate routing share per group per layer
    ax = axes[0]
    x = np.arange(N_LAYERS)
    ax.bar(x - 0.25, la["large_agg"], 0.25, label="24 Large (384w)", color="#d62728")
    ax.bar(x, la["small_agg"], 0.25, label="24 Small (128w)", color="#9467bd")
    ax.bar(x + 0.25, la["null_agg"], 0.25, label="16 Null", color="#7f7f7f")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Aggregate Routing Share")
    ax.set_title("Run C: Routing Share by Expert Group")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: per-expert mean (normalized within group)
    ax = axes[1]
    ax.plot(x, la["large_per_expert_mean"], "o-", label="Large (per expert mean)", color="#d62728")
    ax.plot(x, la["small_per_expert_mean"], "s-", label="Small (per expert mean)", color="#9467bd")
    uniform_line = 1.0 / 64
    ax.axhline(y=uniform_line, color="gray", linestyle="--", alpha=0.7, label=f"Uniform = {uniform_line:.4f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean f_i per Expert")
    ax.set_title("Run C: Per-Expert Usage by Group")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "run_c_size_groups.png"), dpi=150)
    plt.close(fig)
    print("  Saved run_c_size_groups.png")


# ── 8. Expert usage distribution per layer (box/violin) ─────────────
print("Plotting expert usage distributions...")
for key in ["B", "C"]:
    if key not in layer_analysis:
        continue
    la = layer_analysis[key]
    meta = RUNS[key]
    real_usage = la["usage"][:, :meta["n_real"]]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bp = ax.boxplot([real_usage[layer, :] for layer in range(N_LAYERS)],
                    labels=[str(i) for i in range(N_LAYERS)],
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(colors[key])
        patch.set_alpha(0.5)
    uniform = 1.0 / meta["n_total"]
    ax.axhline(y=uniform, color="gray", linestyle="--", alpha=0.7,
               label=f"Uniform = {uniform:.4f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert Usage (f_i)")
    ax.set_title(f"Run {key}: Real Expert Usage Distribution per Layer — step {la['step']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fname = f"expert_usage_boxplot_{key}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 9. Also get run A expert usage for comparison ────────────────────
print("Run A expert usage...")
usage_A, step_A = get_per_layer_usage(data["A"]["rows"], N_LAYERS, RUNS["A"]["n_total"])
if usage_A is not None:
    print(f"  Run A: per-layer data from step {step_A}")
    real_usage_A = usage_A  # all 48 are real

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bp = ax.boxplot([real_usage_A[layer, :] for layer in range(N_LAYERS)],
                    labels=[str(i) for i in range(N_LAYERS)],
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(colors["A"])
        patch.set_alpha(0.5)
    uniform_A = 1.0 / 48
    ax.axhline(y=uniform_A, color="gray", linestyle="--", alpha=0.7,
               label=f"Uniform = {uniform_A:.4f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert Usage (f_i)")
    ax.set_title(f"Run A: Expert Usage Distribution per Layer — step {step_A}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "expert_usage_boxplot_A.png"), dpi=150)
    plt.close(fig)
    print("  Saved expert_usage_boxplot_A.png")

    # Dead expert count for A
    dead_A = (real_usage_A < 0.1 * uniform_A).sum(axis=1)
    layer_analysis["A"] = {
        "usage": usage_A,
        "step": step_A,
        "real_stats": {
            "min": real_usage_A.min(axis=1),
            "max": real_usage_A.max(axis=1),
            "mean": real_usage_A.mean(axis=1),
            "std": real_usage_A.std(axis=1),
        },
        "dead_per_layer": dead_A,
        "uniform": uniform_A,
    }


# ── 10. Generate markdown report ─────────────────────────────────────
print("\nGenerating markdown report...")

md = []
md.append("# MoE Null + Variable Expert Routing Analysis\n")
md.append("## Experiment Setup\n")
md.append("| Run | Config | Experts | Null Experts | LBL Weight |")
md.append("|-----|--------|---------|--------------|------------|")
md.append("| A | Baseline | 48 × 256 FFN | 0 | 0.08 |")
md.append("| B | Uniform + Null | 48 × 256 FFN | 16 | 0.20 |")
md.append("| C | Variable + Null | 24 × 384 + 24 × 128 FFN | 16 | 0.20 |")
md.append("")
md.append("All models: 125M params, 12 layers, sigmoid routing, top-8 selection, ~4500 steps on OpenWebText.")
md.append("Null experts use a single shared logit expanded to 16 copies; tokens routed to null get zero compute.\n")

# ── Training curves
md.append("---\n## 1. Training Curves\n")
md.append("![Validation BPB](val_bpb_curves.png)\n")

# ── Final metrics table
md.append("## 2. Final Metrics\n")
md.append("| Run | Final val/BPB | Total Time (s) | Null Fraction | Real Experts/Token |")
md.append("|-----|--------------|-----------------|---------------|-------------------|")
for key in ["A", "B", "C"]:
    fm = final_metrics.get(key, {})
    bpb = f"{fm.get('final_bpb', float('nan')):.4f}"
    time_s = fm.get("total_time")
    time_str = f"{time_s:.0f}" if time_s else "N/A"
    nf = fm.get("null_fraction")
    nf_str = f"{nf:.4f}" if nf is not None else "N/A"
    rept = fm.get("real_experts_per_token")
    rept_str = f"{rept:.2f}" if rept is not None else "N/A (8.00)"
    md.append(f"| {key} | {bpb} | {time_str} | {nf_str} | {rept_str} |")
md.append("")

# Zero compute fraction
for key in ["B", "C"]:
    fm = final_metrics.get(key, {})
    zcf = fm.get("zero_compute_frac")
    if zcf is not None:
        md.append(f"- **Run {key}** zero-compute token fraction (all 8 slots null): {zcf:.6f}")
md.append("")

# ── Null fraction over training
md.append("---\n## 3. Null Expert Usage\n")
md.append("### 3.1 Null Fraction Over Training\n")
md.append("![Null Fraction](null_fraction_training.png)\n")

# ── Per-layer null fraction
md.append("### 3.2 Null Fraction by Layer\n")
md.append("![Null Fraction Per Layer](null_fraction_per_layer.png)\n")

for key in ["B", "C"]:
    if key not in layer_analysis:
        continue
    la = layer_analysis[key]
    md.append(f"**Run {key}** — null fraction per layer (step {la['step']}):\n")
    md.append("| Layer | Null Fraction |")
    md.append("|-------|--------------|")
    for layer in range(N_LAYERS):
        md.append(f"| {layer} | {la['null_frac_per_layer'][layer]:.4f} |")
    md.append("")

# ── Expert usage distribution
md.append("---\n## 4. Expert Usage Distribution\n")

for key in ["A", "B", "C"]:
    if key not in layer_analysis:
        continue
    la = layer_analysis[key]
    meta = RUNS[key]
    n_real = meta["n_real"]

    md.append(f"### Run {key}: {meta['label']}\n")
    md.append(f"![Expert Usage Boxplot](expert_usage_boxplot_{key}.png)\n")

    md.append(f"Per-layer statistics for {n_real} real experts (step {la['step']}):\n")
    md.append("| Layer | Min | Max | Mean | Std | Dead (<10% uniform) |")
    md.append("|-------|------|------|-------|------|---------------------|")
    rs = la["real_stats"]
    for layer in range(N_LAYERS):
        md.append(f"| {layer} | {rs['min'][layer]:.5f} | {rs['max'][layer]:.5f} | "
                  f"{rs['mean'][layer]:.5f} | {rs['std'][layer]:.5f} | {la['dead_per_layer'][layer]} |")
    md.append(f"\nTotal dead experts across all layers: {la['dead_per_layer'].sum()}\n")

# ── Heatmaps
md.append("---\n## 5. Expert Usage Heatmaps\n")
for key in ["B", "C"]:
    md.append(f"### Run {key}\n")
    md.append(f"![Heatmap](expert_usage_heatmap_{key}.png)\n")
    md.append("Red dashed line = real/null boundary.")
    if key == "C":
        md.append(" White dotted line = large/small boundary.")
    md.append("\n")

# ── Run C size group analysis
if "C" in layer_analysis:
    md.append("---\n## 6. Run C: Expert Size Group Analysis\n")
    md.append("![Size Groups](run_c_size_groups.png)\n")

    la = layer_analysis["C"]
    md.append("### Aggregate Routing Share per Layer\n")
    md.append("| Layer | Large (24×384) | Small (24×128) | Null (16) |")
    md.append("|-------|----------------|----------------|-----------|")
    for layer in range(N_LAYERS):
        md.append(f"| {layer} | {la['large_agg'][layer]:.4f} | "
                  f"{la['small_agg'][layer]:.4f} | {la['null_agg'][layer]:.4f} |")
    md.append("")

    md.append("### Per-Expert Mean by Group\n")
    md.append("| Layer | Large Mean f_i | Small Mean f_i | Ratio (Large/Small) |")
    md.append("|-------|---------------|---------------|---------------------|")
    for layer in range(N_LAYERS):
        lm = la["large_per_expert_mean"][layer]
        sm = la["small_per_expert_mean"][layer]
        ratio = lm / sm if sm > 0 else float("inf")
        md.append(f"| {layer} | {lm:.5f} | {sm:.5f} | {ratio:.2f}× |")
    md.append("")


# ── Key findings
md.append("---\n## 7. Key Findings\n")
md.append("*(auto-generated summary — review against plots)*\n")

# Compare final BPB
bpbs = {k: final_metrics[k]["final_bpb"] for k in ["A", "B", "C"] if k in final_metrics and "final_bpb" in final_metrics[k]}
if bpbs:
    best = min(bpbs, key=bpbs.get)
    md.append(f"1. **Best val/BPB**: Run {best} ({bpbs[best]:.4f})")
    for k in ["A", "B", "C"]:
        if k != best and k in bpbs:
            delta = bpbs[k] - bpbs[best]
            md.append(f"   - Run {k}: +{delta:.4f} vs {best}")
    md.append("")
else:
    md.append("1. **Best val/BPB**: No data available\n")

# Null usage patterns
for key in ["B", "C"]:
    if key in layer_analysis:
        la = layer_analysis[key]
        nf = la["null_frac_per_layer"]
        md.append(f"2. **Run {key} null usage**: ranges from {nf.min():.4f} (layer {nf.argmin()}) "
                  f"to {nf.max():.4f} (layer {nf.argmax()}). Mean = {nf.mean():.4f}")
md.append("")

# Dead experts
for key in ["A", "B", "C"]:
    if key in layer_analysis:
        total_dead = layer_analysis[key]["dead_per_layer"].sum()
        md.append(f"3. **Run {key} dead experts**: {total_dead} total across all layers "
                  f"(threshold: <10% of uniform)")
md.append("")

# Run C size preference
if "C" in layer_analysis:
    la = layer_analysis["C"]
    mean_large = la["large_per_expert_mean"].mean()
    mean_small = la["small_per_expert_mean"].mean()
    ratio = mean_large / mean_small if mean_small > 0 else float("inf")
    md.append(f"4. **Run C size preference**: Large experts get {ratio:.2f}× more routing "
              f"than small experts on average (per expert). "
              f"Large mean f_i = {mean_large:.5f}, Small mean f_i = {mean_small:.5f}")
md.append("")

# Write report
report_path = os.path.join(OUT_DIR, "routing_analysis.md")
with open(report_path, "w") as f:
    f.write("\n".join(md))
print(f"\nReport written to {report_path}")
print("Done!")
