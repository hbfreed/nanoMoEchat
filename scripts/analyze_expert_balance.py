"""
Analyze expert balance from a wandb run.

Usage:
    wandb login  # if not already logged in
    python scripts/analyze_expert_balance.py --run hbfreed/nanochat/094qbitu
"""

import argparse
import json
import numpy as np

def fetch_run_data(run_path):
    import wandb
    api = wandb.Api()
    run = api.run(run_path)

    print(f"Run: {run.name} ({run.state})")
    print(f"Config:")
    for k, v in sorted(run.config.items()):
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    expert_sizes = run.config.get("expert_sizes", [[64, 256]])
    num_experts = sum(count for count, _ in expert_sizes)
    num_active = run.config.get("num_active_experts", 8)
    n_layer = run.config.get("n_layer", 12)
    print(f"\n  Total experts: {num_experts}, Active per token: {num_active}")

    return run, num_experts, num_active, n_layer


def analyze_expert_usage(run, num_experts):
    """Pull expert_usage timeseries and compute balance metrics."""
    # Get the latest expert usage from summary
    summary = run.summary

    # Try to get per-expert usage from summary
    usage = []
    for j in range(num_experts):
        key = f"train/expert_usage_{j}"
        if key in summary:
            usage.append(summary[key])

    if not usage:
        print("\nNo per-expert usage found in run summary. Trying history...")
        # Fall back to history
        keys = [f"train/expert_usage_{j}" for j in range(num_experts)]
        hist = run.history(keys=keys, samples=5)
        if len(hist) > 0:
            last_row = hist.iloc[-1]
            usage = [last_row.get(k, np.nan) for k in keys]
            usage = [v for v in usage if not np.isnan(v)]

    if not usage:
        print("No expert usage data found. Check that the run logged expert_usage metrics.")
        return None

    usage = np.array(usage)
    return usage


def analyze_per_layer_usage(run, num_experts, n_layer):
    """Pull per-layer expert usage (logged every 1000 steps)."""
    keys = []
    for i in range(n_layer):
        for j in range(num_experts):
            keys.append(f"train/expert_usage_layer_{i}_expert_{j}")

    hist = run.history(keys=keys, samples=50)
    if len(hist) == 0 or hist[keys[0]].isna().all():
        return None

    # Get the last row with data
    last_row = hist.dropna(subset=[keys[0]]).iloc[-1] if not hist.dropna(subset=[keys[0]]).empty else None
    if last_row is None:
        return None

    layer_usage = np.zeros((n_layer, num_experts))
    for i in range(n_layer):
        for j in range(num_experts):
            k = f"train/expert_usage_layer_{i}_expert_{j}"
            if k in last_row and not np.isnan(last_row[k]):
                layer_usage[i, j] = last_row[k]

    return layer_usage


def analyze_loss_trends(run):
    """Pull auxiliary loss trends over training."""
    keys = [
        "train/load_balance_loss",
        "train/router_z_loss",
        "train/compute_loss",
        "train/ce_loss",
        "train/loss",
        "train/router_logits_abs_max",
        "train/router_logits_abs_mean",
    ]
    hist = run.history(keys=keys, samples=500)
    return hist


def print_balance_report(usage, num_experts, num_active):
    """Print a detailed balance analysis."""
    ideal = num_active / num_experts  # ideal fraction per expert

    print(f"\n{'='*70}")
    print(f"EXPERT BALANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"  Ideal uniform usage: {ideal:.4f} ({ideal*100:.2f}%)")
    print(f"  Actual mean usage:   {usage.mean():.4f} ({usage.mean()*100:.2f}%)")
    print(f"  Std deviation:       {usage.std():.4f}")
    print(f"  Coefficient of var:  {usage.std()/usage.mean():.4f}")
    print(f"  Min usage:           {usage.min():.4f} (expert {usage.argmin()})")
    print(f"  Max usage:           {usage.max():.4f} (expert {usage.argmax()})")
    print(f"  Max/Min ratio:       {usage.max()/max(usage.min(), 1e-10):.2f}x")

    # How many experts get < 50% of ideal traffic
    underused = (usage < ideal * 0.5).sum()
    overused = (usage > ideal * 1.5).sum()
    dead = (usage < ideal * 0.01).sum()
    print(f"\n  Dead experts (<1% ideal):    {dead}/{num_experts}")
    print(f"  Underused (<50% ideal):      {underused}/{num_experts}")
    print(f"  Overused (>150% ideal):      {overused}/{num_experts}")

    # Entropy-based balance score (1.0 = perfectly balanced)
    usage_norm = usage / usage.sum()
    entropy = -np.sum(usage_norm * np.log(usage_norm + 1e-10))
    max_entropy = np.log(num_experts)
    balance_score = entropy / max_entropy
    print(f"\n  Entropy balance score:       {balance_score:.4f} (1.0 = perfect)")

    # Show distribution
    print(f"\n  Usage distribution (sorted):")
    sorted_usage = np.sort(usage)[::-1]
    sorted_idx = np.argsort(usage)[::-1]
    # Show top 10 and bottom 10
    n_show = min(10, num_experts)
    print(f"  {'Expert':>8} {'Usage':>10} {'vs Ideal':>10} {'Bar'}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*30}")
    for rank in range(n_show):
        idx = sorted_idx[rank]
        val = sorted_usage[rank]
        ratio = val / ideal
        bar_len = int(min(ratio * 20, 40))
        bar = '█' * bar_len
        print(f"  {idx:>8} {val:>10.4f} {ratio:>9.2f}x  {bar}")
    if num_experts > 2 * n_show:
        print(f"  {'...':>8}")
    for rank in range(max(n_show, num_experts - n_show), num_experts):
        idx = sorted_idx[rank]
        val = sorted_usage[rank]
        ratio = val / ideal
        bar_len = int(min(ratio * 20, 40))
        bar = '█' * bar_len if bar_len > 0 else '▏'
        print(f"  {idx:>8} {val:>10.4f} {ratio:>9.2f}x  {bar}")

    return balance_score


def print_layer_report(layer_usage, num_experts, num_active):
    """Per-layer balance analysis."""
    ideal = num_active / num_experts

    print(f"\n{'='*70}")
    print(f"PER-LAYER EXPERT BALANCE")
    print(f"{'='*70}")
    print(f"  {'Layer':>6} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8} {'Dead':>6} {'Balance':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")

    for i in range(layer_usage.shape[0]):
        u = layer_usage[i]
        if u.sum() == 0:
            continue
        u_norm = u / u.sum()
        entropy = -np.sum(u_norm * np.log(u_norm + 1e-10))
        balance = entropy / np.log(num_experts)
        dead = (u < ideal * 0.01).sum()
        print(f"  {i:>6} {u.mean():>8.4f} {u.std():>8.4f} {u.std()/max(u.mean(),1e-10):>8.4f} "
              f"{u.min():>8.4f} {u.max():>8.4f} {dead:>6} {balance:>8.4f}")


def print_loss_report(hist):
    """Print auxiliary loss trends."""
    print(f"\n{'='*70}")
    print(f"AUXILIARY LOSS TRENDS")
    print(f"{'='*70}")

    for col in ["train/load_balance_loss", "train/router_z_loss", "train/compute_loss",
                 "train/ce_loss", "train/loss", "train/router_logits_abs_max", "train/router_logits_abs_mean"]:
        if col in hist.columns:
            vals = hist[col].dropna()
            if len(vals) > 0:
                label = col.replace("train/", "")
                first_val = vals.iloc[:5].mean()
                last_val = vals.iloc[-5:].mean()
                direction = "↑" if last_val > first_val else "↓"
                print(f"  {label:>30}: {first_val:.6f} → {last_val:.6f} {direction}  (min={vals.min():.6f}, max={vals.max():.6f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert balance from a wandb run")
    parser.add_argument("--run", type=str, required=True, help="wandb run path (entity/project/run_id)")
    args = parser.parse_args()

    run, num_experts, num_active, n_layer = fetch_run_data(args.run)

    # 1. Overall expert usage
    usage = analyze_expert_usage(run, num_experts)
    if usage is not None:
        balance_score = print_balance_report(usage, num_experts, num_active)

    # 2. Per-layer breakdown
    layer_usage = analyze_per_layer_usage(run, num_experts, n_layer)
    if layer_usage is not None:
        print_layer_report(layer_usage, num_experts, num_active)

    # 3. Loss trends
    hist = analyze_loss_trends(run)
    if hist is not None and len(hist) > 0:
        print_loss_report(hist)

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if usage is not None:
        balance_score_val = balance_score
        if balance_score_val > 0.95:
            print("  Expert balance is EXCELLENT (>0.95). Routing is well-distributed.")
        elif balance_score_val > 0.85:
            print("  Expert balance is GOOD (0.85-0.95). Minor imbalances exist but are acceptable.")
        elif balance_score_val > 0.70:
            print("  Expert balance is FAIR (0.70-0.85). Some experts are significantly over/under-used.")
            print("  Consider increasing load_balance_loss_weight or enabling bias_balancing.")
        else:
            print("  Expert balance is POOR (<0.70). Severe routing collapse detected.")
            print("  Recommendations:")
            print("    - Increase load_balance_loss_weight (currently likely too low)")
            print("    - Enable --use-bias-balancing for dynamic correction")
            print("    - Check if router_z_loss is keeping logits bounded")


if __name__ == "__main__":
    main()
