"""
TensorKernel — PyTorch Integration Demo

Demonstrates converting between PyTorch tensors and TensorKernel tensors,
running optimized kernels, and comparing results against PyTorch's own ops.

Usage:
    python scripts/pytorch_integration.py
"""

import sys
import time

import numpy as np

try:
    import tensorkernel as tk
except ImportError:
    print("Error: tensorkernel module not found. Build the project first.")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


def pytorch_to_tensorkernel(pt_tensor):
    """Convert a PyTorch CPU tensor to a TensorKernel tensor."""
    np_arr = pt_tensor.detach().cpu().numpy().astype(np.float32)
    return tk.Tensor.from_numpy(np_arr)


def tensorkernel_to_pytorch(tk_tensor):
    """Convert a TensorKernel tensor to a PyTorch tensor."""
    np_arr = tk_tensor.numpy().copy()
    return torch.from_numpy(np_arr)


def demo_matmul():
    """Compare TensorKernel matmul with PyTorch matmul."""
    print("--- Matrix Multiplication Comparison ---")

    N = 512
    pt_a = torch.randn(N, N, dtype=torch.float32)
    pt_b = torch.randn(N, N, dtype=torch.float32)

    # Convert to TensorKernel
    tk_a = pytorch_to_tensorkernel(pt_a)
    tk_b = pytorch_to_tensorkernel(pt_b)
    tk_c = tk.Tensor.zeros([N, N])

    # TensorKernel matmul (OpenMP + SIMD)
    start = time.perf_counter()
    tk.cpu.matmul_openmp(tk_a, tk_b, tk_c)
    tk_time = time.perf_counter() - start

    # PyTorch matmul
    start = time.perf_counter()
    pt_c = pt_a @ pt_b
    pt_time = time.perf_counter() - start

    # Compare results
    tk_result = tensorkernel_to_pytorch(tk_c)
    max_diff = (tk_result - pt_c).abs().max().item()

    print(f"  Matrix size:       {N}x{N}")
    print(f"  TensorKernel time: {tk_time*1000:.2f} ms")
    print(f"  PyTorch time:      {pt_time*1000:.2f} ms")
    print(f"  Max difference:    {max_diff:.6f}")
    print(f"  Match: {'YES' if max_diff < 0.1 else 'NO'}")
    print()


def demo_relu():
    """Compare TensorKernel ReLU with PyTorch ReLU."""
    print("--- ReLU Comparison ---")

    N = 1_000_000
    pt_x = torch.randn(N, dtype=torch.float32)
    tk_x = pytorch_to_tensorkernel(pt_x)

    # TensorKernel
    start = time.perf_counter()
    tk.cpu.relu_simd(tk_x)
    tk_time = time.perf_counter() - start

    # PyTorch
    start = time.perf_counter()
    pt_result = torch.relu(pt_x)
    pt_time = time.perf_counter() - start

    # Compare
    tk_result = tensorkernel_to_pytorch(tk_x)
    max_diff = (tk_result - pt_result).abs().max().item()

    print(f"  Elements:          {N:,}")
    print(f"  TensorKernel time: {tk_time*1000:.3f} ms")
    print(f"  PyTorch time:      {pt_time*1000:.3f} ms")
    print(f"  Max difference:    {max_diff:.8f}")
    print()


def demo_reduction():
    """Compare TensorKernel reduce_sum with PyTorch sum."""
    print("--- Reduction Sum Comparison ---")

    N = 10_000_000
    pt_x = torch.randn(N, dtype=torch.float32)
    tk_x = pytorch_to_tensorkernel(pt_x)

    # TensorKernel
    start = time.perf_counter()
    tk_sum = tk.cpu.reduce_sum_simd(tk_x)
    tk_time = time.perf_counter() - start

    # PyTorch
    start = time.perf_counter()
    pt_sum = pt_x.sum().item()
    pt_time = time.perf_counter() - start

    diff = abs(tk_sum - pt_sum)
    print(f"  Elements:          {N:,}")
    print(f"  TensorKernel sum:  {tk_sum:.4f} ({tk_time*1000:.3f} ms)")
    print(f"  PyTorch sum:       {pt_sum:.4f} ({pt_time*1000:.3f} ms)")
    print(f"  Difference:        {diff:.6f}")
    print()


def demo_pipeline():
    """Demonstrate a mini ML forward pass using TensorKernel kernels."""
    print("--- Mini ML Forward Pass ---")
    print("  Computing: output = ReLU(X @ W + b)")

    batch = 64
    in_features = 256
    out_features = 128

    # Create weights and input
    pt_x = torch.randn(batch, in_features, dtype=torch.float32)
    pt_w = torch.randn(in_features, out_features, dtype=torch.float32)
    pt_b = torch.randn(out_features, dtype=torch.float32)

    # TensorKernel forward pass
    tk_x = pytorch_to_tensorkernel(pt_x)
    tk_w = pytorch_to_tensorkernel(pt_w)
    tk_xw = tk.Tensor.zeros([batch, out_features])

    start = time.perf_counter()
    # Step 1: matmul
    tk.cpu.matmul_openmp(tk_x, tk_w, tk_xw)
    # Step 2: add bias (broadcast manually for each row)
    bias_np = pt_b.numpy()
    xw_np = tk_xw.numpy()
    for i in range(batch):
        xw_np[i] += bias_np
    # Step 3: ReLU
    tk.cpu.relu_simd(tk_xw)
    tk_time = time.perf_counter() - start

    # PyTorch reference
    start = time.perf_counter()
    pt_out = torch.relu(pt_x @ pt_w + pt_b)
    pt_time = time.perf_counter() - start

    tk_result = tensorkernel_to_pytorch(tk_xw)
    max_diff = (tk_result - pt_out).abs().max().item()

    print(f"  Input:  [{batch}, {in_features}]")
    print(f"  Weight: [{in_features}, {out_features}]")
    print(f"  TensorKernel time: {tk_time*1000:.2f} ms")
    print(f"  PyTorch time:      {pt_time*1000:.2f} ms")
    print(f"  Max difference:    {max_diff:.6f}")
    print(f"  Match: {'YES' if max_diff < 0.5 else 'NO'}")
    print()


def main():
    print("=" * 50)
    print("  TensorKernel — PyTorch Integration Demo")
    print("=" * 50)
    print()

    demo_matmul()
    demo_relu()
    demo_reduction()
    demo_pipeline()

    print("Integration demo complete.")


if __name__ == "__main__":
    main()
