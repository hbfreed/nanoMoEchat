"""Regenerate analysis_round2.md from existing CSVs (no GPU needed)."""
import csv
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(OUT_DIR))
from nanochat.tokenizer import get_tokenizer


def read_csv(path):
    results = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            # Parse from the right — token_str may contain commas/quotes
            parts = line.strip().split(",")
            # Fields: token_id, token_str, count, avg_expert_size, avg_expert_size_excluding_null, null_fraction
            # token_id is first, last 4 are numeric, everything in between is token_str
            token_id = int(parts[0])
            null_fraction = float(parts[-1])
            avg_expert_size_excluding_null = float(parts[-2])
            avg_expert_size = float(parts[-3])
            count = int(parts[-4])
            token_str = ",".join(parts[1:-4])  # rejoin in case token_str had commas
            # Fix pipes for markdown
            token_str = token_str.replace("|", "¦").strip('"')
            results.append({
                "token_id": token_id,
                "token_str": token_str,
                "count": count,
                "avg_expert_size": avg_expert_size,
                "avg_expert_size_excluding_null": avg_expert_size_excluding_null,
                "null_fraction": null_fraction,
            })
    return results


def fmt_token(r):
    return f'`{r["token_str"]}`'


results_B = read_csv(os.path.join(OUT_DIR, "token_routing_B.csv"))
results_C = read_csv(os.path.join(OUT_DIR, "token_routing_C.csv"))

RUNS = {"B": results_B, "C": results_C}
LABELS = {"B": "B: 48 uniform + 16 null", "C": "C: 24 large + 24 small + 16 null"}

md = []
md.append("# Round 2: Token-Level Routing Analysis\n")
md.append("## Setup\n")
md.append("- Evaluation: 3200 batches x 16 sequences x 1024 tokens = 52,428,800 token positions")
md.append("- Minimum token count for inclusion: 5")
md.append("- Routing: sigmoid + top-8, weights renormalized after null zeroing")
md.append("- Expert width lookup: null=0, sizes per run config")
md.append("- **Note**: Routing analysis uses v2 checkpoints (~1.08 bpb). Training curves use v1 runs (~1.03 bpb).\n")

for run_key in ["B", "C"]:
    results = RUNS[run_key]
    md.append(f"---\n## Run {run_key}: {LABELS[run_key]}\n")

    nfs = [r["null_fraction"] for r in results]
    sizes_incl = [r["avg_expert_size"] for r in results]
    sizes_excl = [r["avg_expert_size_excluding_null"] for r in results]
    md.append(f"**{len(results)} unique tokens** (min count >= 5)\n")
    md.append("| Statistic | Null Fraction | Avg Expert Size (incl null) | Avg Expert Size (excl null) |")
    md.append("|-----------|--------------|---------------------------|----------------------------|")
    md.append(f"| Mean | {np.mean(nfs):.4f} | {np.mean(sizes_incl):.1f} | {np.mean(sizes_excl):.1f} |")
    md.append(f"| Median | {np.median(nfs):.4f} | {np.median(sizes_incl):.1f} | {np.median(sizes_excl):.1f} |")
    md.append(f"| Std | {np.std(nfs):.4f} | {np.std(sizes_incl):.1f} | {np.std(sizes_excl):.1f} |")
    md.append(f"| Min | {np.min(nfs):.4f} | {np.min(sizes_incl):.1f} | {np.min(sizes_excl):.1f} |")
    md.append(f"| Max | {np.max(nfs):.4f} | {np.max(sizes_incl):.1f} | {np.max(sizes_excl):.1f} |")
    md.append("")

    # Scatter only for C
    if run_key == "C":
        md.append("### Null Fraction vs Expert Size\n")
        md.append("![Scatter](scatter_C.png)\n")

    md.append("### Null Fraction Distribution\n")
    md.append(f"![Histogram](null_frac_hist_{run_key}.png)\n")

    # Top 20 by avg expert size incl null (most compute)
    by_size_incl = sorted(results, key=lambda r: r["avg_expert_size"], reverse=True)
    md.append("### Top 20 Tokens by Compute (highest avg expert size incl null)\n")
    md.append("| Token | ID | Count | Null Frac | Avg Size (incl null) | Avg Size (excl null) |")
    md.append("|-------|-----|-------|-----------|---------------------|---------------------|")
    for r in by_size_incl[:20]:
        md.append(f'| {fmt_token(r)} | {r["token_id"]} | {r["count"]} | '
                  f'{r["null_fraction"]:.4f} | {r["avg_expert_size"]:.0f} | '
                  f'{r["avg_expert_size_excluding_null"]:.0f} |')
    md.append("")

    # Bottom 20 by avg expert size incl null (least compute)
    md.append("### Bottom 20 Tokens by Compute (lowest avg expert size incl null)\n")
    md.append("| Token | ID | Count | Null Frac | Avg Size (incl null) | Avg Size (excl null) |")
    md.append("|-------|-----|-------|-----------|---------------------|---------------------|")
    for r in reversed(by_size_incl[-20:]):
        md.append(f'| {fmt_token(r)} | {r["token_id"]} | {r["count"]} | '
                  f'{r["null_fraction"]:.4f} | {r["avg_expert_size"]:.0f} | '
                  f'{r["avg_expert_size_excluding_null"]:.0f} |')
    md.append("")

    # Token frequency vs null fraction — binned table
    bins = [(0, 100, "<100"), (100, 1_000, "100-1k"), (1_000, 10_000, "1k-10k"),
            (10_000, 100_000, "10k-100k"), (100_000, float("inf"), ">100k")]
    md.append("### Token Frequency vs Null Routing\n")
    md.append("| Frequency Bin | # Tokens | Mean Null Frac | Mean Avg Size (incl null) |")
    md.append("|--------------|----------|---------------|--------------------------|")
    for lo, hi, label in bins:
        bucket = [r for r in results if lo <= r["count"] < hi]
        if bucket:
            mn = np.mean([r["null_fraction"] for r in bucket])
            ms = np.mean([r["avg_expert_size"] for r in bucket])
            md.append(f"| {label} | {len(bucket)} | {mn:.4f} | {ms:.0f} |")
    md.append("")

    # Top 20 by expert size (C only — B is always 256)
    if run_key == "C":
        by_size = sorted(results, key=lambda r: r["avg_expert_size_excluding_null"], reverse=True)
        md.append("### Top 20 Tokens by Avg Expert Size excl null (most compute)\n")
        md.append("| Token | ID | Count | Null Frac | Avg Size (incl null) | Avg Size (excl null) |")
        md.append("|-------|-----|-------|-----------|---------------------|---------------------|")
        for r in by_size[:20]:
            md.append(f'| {fmt_token(r)} | {r["token_id"]} | {r["count"]} | '
                      f'{r["null_fraction"]:.4f} | {r["avg_expert_size"]:.0f} | '
                      f'{r["avg_expert_size_excluding_null"]:.0f} |')
        md.append("")

        # Expert size distribution histogram (excl null) — C only
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sizes_excl, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Avg Expert Size (excl null)")
        ax.set_ylabel("Number of Tokens")
        ax.set_title("C: Distribution of Expert Size Preference (excl null)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "expert_size_hist_C.png"), dpi=150)
        plt.close(fig)
        md.append("### Expert Size Distribution (excl null)\n")
        md.append("![Expert Size Histogram](expert_size_hist_C.png)\n")

# Cross-run comparison
md.append("---\n## Cross-Run Comparison: Flagged Tokens\n")
b_by_id = {r["token_id"]: r for r in results_B}
c_by_id = {r["token_id"]: r for r in results_C}

tokenizer = get_tokenizer()
flagged = {}
for word in [" the", " of", " in", " and", " to", " a", " is", " that", " for", " it",
             " def", " return", " import", " class", " function", " if", " else"]:
    ids = tokenizer.encode([word], num_threads=1)[0]
    if len(ids) == 1:
        tid = ids[0]
        if tid in b_by_id and tid in c_by_id:
            flagged[word.strip()] = tid
for punct in [".", ",", "!", "?", ":", ";"]:
    ids = tokenizer.encode([punct], num_threads=1)[0]
    if len(ids) == 1:
        tid = ids[0]
        if tid in b_by_id and tid in c_by_id:
            flagged[punct] = tid
if 65527 in b_by_id and 65527 in c_by_id:
    flagged["BOS"] = 65527

md.append("| Token | B Null Frac | C Null Frac | B Avg Size (incl null) | C Avg Size (incl null) |")
md.append("|-------|------------|------------|----------------------|----------------------|")
for label, tid in sorted(flagged.items()):
    rb, rc = b_by_id[tid], c_by_id[tid]
    md.append(f'| {label} | {rb["null_fraction"]:.4f} | {rc["null_fraction"]:.4f} | '
              f'{rb["avg_expert_size"]:.0f} | {rc["avg_expert_size"]:.0f} |')
md.append("")

# Cross-run: B vs C null fraction scatter + Spearman correlation
common_ids = sorted(set(b_by_id) & set(c_by_id))
b_nfs = np.array([b_by_id[tid]["null_fraction"] for tid in common_ids])
c_nfs = np.array([c_by_id[tid]["null_fraction"] for tid in common_ids])

# Spearman = Pearson on ranks
def spearman_r(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return np.corrcoef(rx, ry)[0, 1]

rho = spearman_r(b_nfs, c_nfs)

c_sizes_excl = np.array([c_by_id[tid]["avg_expert_size_excluding_null"] for tid in common_ids])

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(b_nfs, c_nfs, c=c_sizes_excl, s=3, alpha=0.4, cmap="RdYlBu_r", vmin=128, vmax=384)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("C: Avg Expert Size (excl null)")
ax.set_xlabel("B: Null Fraction")
ax.set_ylabel("C: Null Fraction")
ax.set_title(f"B vs C Null Fraction (Spearman ρ = {rho:.3f}, n = {len(common_ids)})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "null_frac_B_vs_C.png"), dpi=150)
plt.close(fig)

md.append("### B vs C Null Fraction Correlation\n")
md.append(f"![B vs C scatter](null_frac_B_vs_C.png)\n")
md.append(f"**Spearman ρ = {rho:.3f}** across {len(common_ids)} shared tokens.")
md.append("Color = C's avg expert size (excl null). Tokens below the diagonal are less null-routed in C than B; "
          "blue (small expert) below the diagonal = tokens where C substitutes small experts for null routing.\n")

# Null vs Less Compute: bucket by B null fraction, show what C does
md.append("### Null Routing vs Reduced Compute\n")
md.append("Tokens bucketed by B's null fraction. For each bucket, what does C do with the same tokens?\n")
md.append("| B Null Frac Bucket | # Tokens | B Mean Null Frac | C Mean Null Frac | C Mean Size (excl null) | C Mean Size (incl null) |")
md.append("|-------------------|----------|-----------------|-----------------|------------------------|------------------------|")
null_bins = [(0.0, 0.2, "0.0-0.2"), (0.2, 0.4, "0.2-0.4"), (0.4, 0.6, "0.4-0.6"), (0.6, 0.8, "0.6-0.8")]
for lo, hi, label in null_bins:
    mask = (b_nfs >= lo) & (b_nfs < hi)
    if mask.sum() > 0:
        c_nf_bucket = c_nfs[mask]
        c_se_bucket = c_sizes_excl[mask]
        c_si_bucket = np.array([c_by_id[tid]["avg_expert_size"] for tid in np.array(common_ids)[mask]])
        md.append(f"| {label} | {mask.sum()} | {b_nfs[mask].mean():.4f} | "
                  f"{c_nf_bucket.mean():.4f} | {c_se_bucket.mean():.0f} | {c_si_bucket.mean():.0f} |")
md.append("")

# Cross-run: top/bottom overlap
N_OVERLAP = 20
b_by_compute = sorted(results_B, key=lambda r: r["avg_expert_size"])
c_by_compute = sorted(results_C, key=lambda r: r["avg_expert_size"])

b_bottom_ids = {r["token_id"] for r in b_by_compute[:N_OVERLAP]}
c_bottom_ids = {r["token_id"] for r in c_by_compute[:N_OVERLAP]}
b_top_ids = {r["token_id"] for r in b_by_compute[-N_OVERLAP:]}
c_top_ids = {r["token_id"] for r in c_by_compute[-N_OVERLAP:]}

bottom_overlap = b_bottom_ids & c_bottom_ids
top_overlap = b_top_ids & c_top_ids

md.append("### Top/Bottom 20 Overlap Between B and C\n")
md.append(f"- **Lowest-compute overlap**: {len(bottom_overlap)}/{N_OVERLAP} tokens appear in both B and C bottom-20")
md.append(f"- **Highest-compute overlap**: {len(top_overlap)}/{N_OVERLAP} tokens appear in both B and C top-20")
if bottom_overlap:
    shared_bottom = [b_by_id[tid]["token_str"] for tid in bottom_overlap]
    md.append(f'- Shared lowest-compute: {", ".join(f"`{t}`" for t in sorted(shared_bottom))}')
if top_overlap:
    shared_top = [b_by_id[tid]["token_str"] for tid in top_overlap]
    md.append(f'- Shared highest-compute: {", ".join(f"`{t}`" for t in sorted(shared_top))}')
md.append("")

report_path = os.path.join(OUT_DIR, "analysis_round2.md")
with open(report_path, "w") as f:
    f.write("\n".join(md))
print(f"Report written to {report_path}")
