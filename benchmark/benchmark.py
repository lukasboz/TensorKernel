"""
TensorKernel Python Benchmark Suite

Runs benchmarks through the Python bindings and generates
performance comparison charts (bar charts + speedup plots).

Usage:
    python benchmark/benchmark.py
"""

import sys
import os
import time
import csv

import numpy as np

# Try importing the compiled module
try:
    import tensorkernel as tk
except ImportError:
    print("Error: tensorkernel module not found.")
    print("Build with: cmake --build build --config Release")
    print("Then copy the .pyd/.so from build/ to this directory or install.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Charts will be skipped.")


def time_kernel(fn, warmup=3, trials=10):
    """Time a kernel function. Returns median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times.sort()
    return times[trials // 2]


def benchmark_matmul():
    """Benchmark matrix multiplication variants."""
    sizes = [128, 256, 512, 1024]
    variants = {
        "naive": tk.cpu.matmul_naive,
        "tiled": lambda A, B, C: tk.cpu.matmul_tiled(A, B, C),
        "simd": tk.cpu.matmul_simd,
        "openmp": lambda A, B, C: tk.cpu.matmul_openmp(A, B, C),
    }

    results = {v: {"sizes": [], "gflops": [], "time_ms": []} for v in variants}

    for N in sizes:
        print(f"  matmul N={N}...", end=" ", flush=True)
        A = tk.Tensor.rand([N, N])
        B = tk.Tensor.rand([N, N])
        C = tk.Tensor.zeros([N, N])
        flops = 2 * N ** 3

        for name, fn in variants.items():
            t = time_kernel(lambda: fn(A, B, C),
                            warmup=1 if name == "naive" else 3,
                            trials=3 if name == "naive" else 10)
            gf = flops / t / 1e9
            results[name]["sizes"].append(N)
            results[name]["gflops"].append(gf)
            results[name]["time_ms"].append(t * 1000)

        print("done")

    return results


def benchmark_elementwise():
    """Benchmark element-wise add: naive vs SIMD."""
    sizes = [100_000, 1_000_000, 10_000_000]
    results = {"naive": [], "simd": []}

    for N in sizes:
        print(f"  add N={N}...", end=" ", flush=True)
        A = tk.Tensor.rand([N])
        B = tk.Tensor.rand([N])
        C = tk.Tensor.zeros([N])

        t_naive = time_kernel(lambda: tk.cpu.add_naive(A, B, C))
        t_simd = time_kernel(lambda: tk.cpu.add_simd(A, B, C))

        bw_naive = 3 * N * 4 / t_naive / 1e9  # GB/s
        bw_simd = 3 * N * 4 / t_simd / 1e9

        results["naive"].append({"N": N, "bw": bw_naive, "time_ms": t_naive * 1000})
        results["simd"].append({"N": N, "bw": bw_simd, "time_ms": t_simd * 1000})
        print("done")

    return results


def save_results_csv(matmul_results, filename="benchmark_results_python.csv"):
    """Save benchmark results to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "variant", "N", "gflops", "time_ms"])
        for variant, data in matmul_results.items():
            for i in range(len(data["sizes"])):
                writer.writerow([
                    "matmul", variant, data["sizes"][i],
                    f"{data['gflops'][i]:.2f}",
                    f"{data['time_ms'][i]:.3f}"
                ])
    print(f"Results saved to {filename}")


def plot_matmul_results(results):
    """Generate bar chart comparing matmul GFLOPS across variants."""
    if not HAS_MATPLOTLIB:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # GFLOPS bar chart
    sizes = results["naive"]["sizes"]
    x = np.arange(len(sizes))
    width = 0.18
    variants = list(results.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, (name, color) in enumerate(zip(variants, colors)):
        ax1.bar(x + i * width, results[name]["gflops"], width,
                label=name, color=color)

    ax1.set_xlabel("Matrix Size (N x N)")
    ax1.set_ylabel("GFLOPS")
    ax1.set_title("Matrix Multiplication Performance")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Speedup chart
    naive_times = results["naive"]["time_ms"]
    for i, (name, color) in enumerate(zip(variants, colors)):
        speedups = [naive_times[j] / results[name]["time_ms"][j]
                    for j in range(len(sizes))]
        ax2.plot(sizes, speedups, "o-", label=name, color=color, linewidth=2)

    ax2.set_xlabel("Matrix Size (N x N)")
    ax2.set_ylabel("Speedup vs Naive")
    ax2.set_title("Optimization Speedup")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("benchmark_matmul.png", dpi=150)
    print("Chart saved to benchmark_matmul.png")
    plt.close()


def main():
    print("=" * 50)
    print("  TensorKernel Python Benchmarks")
    print("=" * 50)

    print("\n--- Matrix Multiplication ---")
    matmul_results = benchmark_matmul()

    print("\n--- Element-wise Operations ---")
    ew_results = benchmark_elementwise()

    # Print summary
    print("\n--- Summary ---")
    print(f"{'Variant':<10} {'N':>6} {'GFLOPS':>10} {'Time(ms)':>12}")
    print("-" * 42)
    for variant, data in matmul_results.items():
        for i in range(len(data["sizes"])):
            print(f"{variant:<10} {data['sizes'][i]:>6} "
                  f"{data['gflops'][i]:>10.2f} {data['time_ms'][i]:>12.3f}")

    save_results_csv(matmul_results)
    plot_matmul_results(matmul_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
