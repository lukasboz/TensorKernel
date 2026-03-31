"""
TensorKernel Profiling Analysis

Reads benchmark CSV results and generates analysis charts:
  1. GFLOPS bar chart per kernel variant
  2. Speedup vs naive line chart
  3. Roofline model plot
  4. OpenMP thread scaling chart

Usage:
    python scripts/profile_analysis.py [benchmark_results.csv]
"""

import sys
import csv
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_csv(filename):
    """Load benchmark results from CSV."""
    results = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "kernel": row["kernel"],
                "variant": row["variant"],
                "M": int(row["M"]) if row["M"] != "0" else 0,
                "N": int(row["N"]) if row["N"] != "0" else 0,
                "K": int(row["K"]) if row["K"] != "0" else 0,
                "elapsed_ms": float(row["elapsed_ms"]),
                "gflops": float(row["gflops"]),
                "bandwidth_gb_s": float(row["bandwidth_gb_s"]),
                "speedup": float(row["speedup"]),
            })
    return results


def plot_gflops_bar(results, kernel_name, output_file):
    """Bar chart of GFLOPS for each variant at each size."""
    # Group by variant and size
    by_variant = defaultdict(list)
    sizes = set()
    for r in results:
        if r["kernel"] == kernel_name:
            size = r["N"] if r["M"] == 0 else r["M"]
            by_variant[r["variant"]].append((size, r["gflops"]))
            sizes.add(size)

    sizes = sorted(sizes)
    variants = sorted(by_variant.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.8 / len(variants)

    for i, variant in enumerate(variants):
        data = dict(by_variant[variant])
        gflops = [data.get(s, 0) for s in sizes]
        ax.bar(x + i * width, gflops, width,
               label=variant, color=colors[i % len(colors)])

    ax.set_xlabel("Problem Size")
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"{kernel_name} — Performance by Variant")
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  Saved {output_file}")
    plt.close()


def plot_speedup(results, kernel_name, output_file):
    """Speedup vs naive line chart."""
    by_variant = defaultdict(list)
    for r in results:
        if r["kernel"] == kernel_name:
            size = r["N"] if r["M"] == 0 else r["M"]
            by_variant[r["variant"]].append((size, r["speedup"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (variant, data) in enumerate(sorted(by_variant.items())):
        data.sort()
        sizes = [d[0] for d in data]
        speedups = [d[1] for d in data]
        ax.plot(sizes, speedups, "o-", label=variant,
                color=colors[i % len(colors)], linewidth=2, markersize=6)

    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Speedup vs Naive")
    ax.set_title(f"{kernel_name} — Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  Saved {output_file}")
    plt.close()


def plot_roofline(results, output_file,
                  peak_gflops=100.0, peak_bw_gb_s=40.0):
    """
    Roofline model: plots achieved GFLOPS vs operational intensity.
    Operational intensity = FLOPS / bytes_moved.

    The roofline ceiling is min(peak_compute, peak_bandwidth * OI).
    Points below the roof are either compute-bound or memory-bound.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw roofline
    oi_range = np.logspace(-2, 3, 500)
    roof = np.minimum(peak_gflops, peak_bw_gb_s * oi_range)
    ax.plot(oi_range, roof, "k-", linewidth=2, label="Roofline")
    ax.fill_between(oi_range, roof, alpha=0.05, color="gray")

    # Plot kernel results
    markers = {"naive": "o", "tiled": "s", "simd": "^",
               "openmp": "D", "cuda": "*", "im2col": "v"}
    colors_map = {"naive": "#2196F3", "tiled": "#4CAF50",
                  "simd": "#FF9800", "openmp": "#F44336",
                  "cuda": "#9C27B0", "im2col": "#795548"}

    for r in results:
        if r["gflops"] <= 0 or r["bandwidth_gb_s"] <= 0:
            continue
        oi = r["gflops"] / r["bandwidth_gb_s"]
        marker = markers.get(r["variant"], "o")
        color = colors_map.get(r["variant"], "#666666")
        ax.scatter(oi, r["gflops"], marker=marker, color=color, s=80,
                   zorder=5, edgecolors="black", linewidth=0.5)

    # Legend for variants
    for variant in markers:
        ax.scatter([], [], marker=markers[variant],
                   color=colors_map.get(variant, "#666"),
                   label=variant, s=60)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOPS/Byte)")
    ax.set_ylabel("Achieved GFLOPS")
    ax.set_title("Roofline Model")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"  Saved {output_file}")
    plt.close()


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        print("Run the C++ benchmark first: ./build/Release/tk_bench")
        sys.exit(1)

    print(f"Loading results from {csv_file}...")
    results = load_csv(csv_file)
    print(f"  {len(results)} results loaded\n")

    # Find which kernels were benchmarked
    kernels = set(r["kernel"] for r in results)
    print("Generating charts...")

    for kernel in sorted(kernels):
        plot_gflops_bar(results, kernel, f"profile_{kernel}_gflops.png")
        plot_speedup(results, kernel, f"profile_{kernel}_speedup.png")

    plot_roofline(results, "profile_roofline.png")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
