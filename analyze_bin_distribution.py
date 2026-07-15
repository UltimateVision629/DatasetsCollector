"""
Analyze per-dimension bin distribution of training actions with histograms.

Plots histograms of action values across 256 discrete bins for each of the
14 action dimensions, showing whether data suffers from mode collapse (most
samples falling into a single bin).

For each dimension, reports:
  - top bin %: fraction of samples in the most common bin
  - top3 bins %: fraction in the 3 most common bins
  - center bin %: fraction in bin 128 (normalized value = 0.0)
  - nonzero bins: how many of 256 bins have at least one sample

Usage:
  cd DatasetsCollector

  # 原始数据
  python analyze_bin_distribution.py -i ./demos -o ./bin_analysis_orig

  # 过滤后数据
  python analyze_bin_distribution.py -i ./demos_filtered -o ./bin_analysis_filtered

  # 过滤后 + step_skip=5 累积
  python analyze_bin_distribution.py -i ./demos_filtered -o ./bin_analysis_step5 --step_skip 5

  # 指定统计文件路径
  python analyze_bin_distribution.py -i ../datasets/trajectories_filtered_v2 -o ./bin_analysis --stats_path ../datasets/dataset_statistics.json

Output:
  - bin_dist_overview.png: 14-dim combined histogram
  - bin_detail_<dim_name>.png: per-dimension individual plot
  - Terminal summary table with verdict for each dimension
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_and_normalize(data_dir: str, stats_path: str, step_skip: int = 1):
    """Load all actions, optionally accumulate deltas, and normalize."""
    with open(stats_path) as f:
        all_stats = json.load(f)
    q01 = np.array(all_stats["block_grasp"]["action"]["q01"])
    q99 = np.array(all_stats["block_grasp"]["action"]["q99"])

    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"ERROR: No .npz files found in {data_dir}")
        sys.exit(1)

    all_actions = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        T = data["length"].item() if "length" in data else 50

        if step_skip == 1:
            for t in range(T):
                all_actions.append(data[f"action/{t}"].astype(np.float32))
        else:
            for t in range(0, T - step_skip, step_skip):
                acc = np.zeros(14, dtype=np.float32)
                for k in range(t, t + step_skip):
                    ak = data[f"action/{k}"].astype(np.float32)
                    acc[0:6] += ak[0:6]
                    acc[7:13] += ak[7:13]
                acc[6] = data[f"action/{t + step_skip - 1}"][6]
                acc[13] = data[f"action/{t + step_skip - 1}"][13]
                all_actions.append(acc)

    all_actions = np.array(all_actions)
    normalized = np.clip(
        2.0 * (all_actions - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0
    )
    return all_actions, normalized, q01, q99


def analyze(data_dir: str, stats_path: str, output_dir: str, step_skip: int = 1):
    """Run full bin distribution analysis."""
    if not HAS_MPL:
        print("ERROR: matplotlib required. pip install matplotlib")
        sys.exit(1)

    all_actions, normalized, q01, q99 = load_and_normalize(data_dir, stats_path, step_skip)

    bins = np.linspace(-1.0, 1.0, 256)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    discretized = np.array([np.digitize(normalized[i], bins) for i in range(len(normalized))])

    dim_names = [
        "R_dx", "R_dy", "R_dz", "R_dRx", "R_dRy", "R_dRz", "R_grip",
        "L_dx", "L_dy", "L_dz", "L_dRx", "L_dRy", "L_dRz", "L_grip",
    ]

    os.makedirs(output_dir, exist_ok=True)

    # ── Combined overview plot ──
    fig, axes = plt.subplots(7, 2, figsize=(18, 22))
    axes = axes.flatten()

    for dim in range(14):
        ax = axes[dim]
        counts = np.bincount(discretized[:, dim], minlength=257)[1:257]
        x = np.arange(1, 257)

        top_bin = counts.argmax() + 1
        top_pct = counts.max() / len(normalized) * 100
        top3_pct = sum(sorted(counts)[-3:]) / len(normalized) * 100
        center_pct = counts[127] / len(normalized) * 100
        nonzero = (counts > 0).sum()

        colors = ["steelblue"] * 256
        colors[top_bin - 1] = "red"
        ax.bar(x, counts, width=1.0, color=colors, alpha=0.8)

        # Bracket the 4 densest bins
        sorted_idx = np.argsort(-counts)
        dense4 = sorted(sorted_idx[:4])
        for b in dense4:
            ax.axvspan(b + 0.5, b + 1.5, alpha=0.4, color="orange")

        ax.axvline(x=128, color="green", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(
            f"{dim_names[dim]}  top={top_bin}({top_pct:.1f}%)  "
            f"top3={top3_pct:.1f}%  center={center_pct:.1f}%  "
            f"nonzero={nonzero}/256",
            fontsize=9,
        )
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.set_xlim(1, 256)

    plt.suptitle(f"Bin Distribution - {data_dir} (step_skip={step_skip}, N={len(normalized)})",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    overview_path = os.path.join(output_dir, "bin_dist_overview.png")
    plt.savefig(overview_path, dpi=120)
    plt.close()
    print(f"Saved: {overview_path}")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"SUMMARY - {data_dir} (step_skip={step_skip}, N={len(normalized)})")
    print(f"{'='*80}")
    print(f"{'Dim':<8} {'top bin':>8} {'top%':>8} {'top3%':>8} {'center%':>9} {'#bins':>7}  {'verdict'}")
    print("-" * 65)
    for dim in range(14):
        counts = np.bincount(discretized[:, dim], minlength=257)[1:257]
        top_pct = counts.max() / len(normalized) * 100
        top3_pct = sum(sorted(counts)[-3:]) / len(normalized) * 100
        center_pct = counts[127] / len(normalized) * 100
        nz = (counts > 0).sum()

        if top_pct < 10:
            v = "EXCELLENT"
        elif top_pct < 20:
            v = "GOOD"
        elif top_pct < 35:
            v = "OK"
        elif top_pct < 50:
            v = "WEAK"
        else:
            v = "BAD (collapse)"

        print(f"{dim_names[dim]:<8} {counts.argmax()+1:>8} {top_pct:>7.1f}% {top3_pct:>7.1f}% "
              f"{center_pct:>8.1f}% {nz:>7}  {v}")

    # ── Per-dim individual plots ──
    for dim in range(14):
        fig, ax = plt.subplots(figsize=(12, 4))
        counts = np.bincount(discretized[:, dim], minlength=257)[1:257]
        x = np.arange(1, 257)

        top_idx = counts.argmax()

        colors = ["steelblue"] * 256
        colors[top_idx] = "red"
        ax.bar(x, counts, width=1.0, color=colors, alpha=0.8)
        ax.axvline(x=128, color="green", linestyle="--", linewidth=1, label="Bin 128 (center)")
        ax.axvline(x=top_idx + 1, color="red", linestyle="-", linewidth=1,
                   label=f"Peak: bin {top_idx+1} ({counts[top_idx]/len(normalized)*100:.1f}%)")

        ax.set_title(f"{dim_names[dim]} - {len(normalized)} samples")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.set_xlim(1, 256)
        ax.legend(fontsize=8)

        out = os.path.join(output_dir, f"bin_detail_{dim_names[dim]}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=100)
        plt.close()

    print(f"\nSaved {14} individual plots to {output_dir}/bin_detail_*.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze bin distribution of training actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_bin_distribution.py -i ./demos -o ./bin_analysis
  python analyze_bin_distribution.py -i ./demos_filtered -o ./bin_analysis --step_skip 5
        """,
    )
    parser.add_argument("-i", "--data_dir", type=str, required=True,
                        help="Directory containing .npz trajectory files")
    parser.add_argument("-o", "--output_dir", type=str, default="./bin_analysis",
                        help="Output directory for plots (default: ./bin_analysis)")
    parser.add_argument("--stats_path", type=str, default="../datasets/dataset_statistics.json",
                        help="Path to dataset_statistics.json (default: ../datasets/dataset_statistics.json)")
    parser.add_argument("--step_skip", type=int, default=1,
                        help="Delta accumulation frames: 1=no accumulation, 5=accumulate 5 frames (default: 1)")
    args = parser.parse_args()

    analyze(args.data_dir, args.stats_path, args.output_dir, args.step_skip)


if __name__ == "__main__":
    main()
