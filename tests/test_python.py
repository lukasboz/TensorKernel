"""
TensorKernel Python Test Suite

Tests the pybind11 bindings for correctness, including NumPy interop.

Usage:
    python tests/test_python.py
    # or: python -m pytest tests/test_python.py -v
"""

import sys
import numpy as np

try:
    import tensorkernel as tk
except ImportError:
    print("Error: tensorkernel module not found. Build the project first.")
    sys.exit(1)

passed = 0
failed = 0


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} -- {msg}")
        failed += 1


def test_tensor_creation():
    t = tk.Tensor.rand([3, 4])
    check("shape", t.shape() == [3, 4])
    check("numel", t.numel() == 12)
    check("ndim", t.ndim() == 2)


def test_tensor_zeros():
    t = tk.Tensor.zeros([10])
    arr = t.numpy()
    check("zeros all zero", np.allclose(arr, 0.0))


def test_tensor_ones():
    t = tk.Tensor.ones([5, 5])
    arr = t.numpy()
    check("ones all one", np.allclose(arr, 1.0))


def test_numpy_roundtrip():
    original = np.random.randn(4, 4).astype(np.float32)
    t = tk.Tensor.from_numpy(original)
    result = t.numpy()
    check("numpy roundtrip",
          np.allclose(original, result, atol=1e-7),
          f"max diff: {np.abs(original - result).max()}")


def test_matmul_vs_numpy():
    N = 64
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)
    expected = A_np @ B_np

    A = tk.Tensor.from_numpy(A_np)
    B = tk.Tensor.from_numpy(B_np)
    C = tk.Tensor.zeros([N, N])

    tk.cpu.matmul_simd(A, B, C)
    result = C.numpy()

    check("matmul vs numpy",
          np.allclose(result, expected, atol=1e-2),
          f"max diff: {np.abs(result - expected).max()}")


def test_matmul_openmp_vs_numpy():
    N = 128
    A_np = np.random.randn(N, N).astype(np.float32)
    B_np = np.random.randn(N, N).astype(np.float32)
    expected = A_np @ B_np

    A = tk.Tensor.from_numpy(A_np)
    B = tk.Tensor.from_numpy(B_np)
    C = tk.Tensor.zeros([N, N])

    tk.cpu.matmul_openmp(A, B, C)
    result = C.numpy()

    check("matmul_openmp vs numpy",
          np.allclose(result, expected, atol=1e-1),
          f"max diff: {np.abs(result - expected).max()}")


def test_add_vs_numpy():
    N = 1000
    A_np = np.random.randn(N).astype(np.float32)
    B_np = np.random.randn(N).astype(np.float32)
    expected = A_np + B_np

    A = tk.Tensor.from_numpy(A_np)
    B = tk.Tensor.from_numpy(B_np)
    C = tk.Tensor.zeros([N])

    tk.cpu.add_simd(A, B, C)
    result = C.numpy()

    check("add_simd vs numpy",
          np.allclose(result, expected, atol=1e-6),
          f"max diff: {np.abs(result - expected).max()}")


def test_relu():
    data = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    t = tk.Tensor.from_numpy(data)
    tk.cpu.relu_simd(t)
    result = t.numpy()
    expected = np.array([0, 0, 0, 1, 2], dtype=np.float32)
    check("relu_simd", np.allclose(result, expected))


def test_reduce_sum():
    data = np.random.randn(10000).astype(np.float32)
    t = tk.Tensor.from_numpy(data)

    expected = float(data.sum())
    naive = tk.cpu.reduce_sum_naive(t)
    simd = tk.cpu.reduce_sum_simd(t)

    check("reduce_sum naive",
          abs(naive - expected) < abs(expected) * 1e-3 + 1.0,
          f"naive={naive}, expected={expected}")
    check("reduce_sum simd",
          abs(simd - expected) < abs(expected) * 1e-3 + 1.0,
          f"simd={simd}, expected={expected}")


def test_conv2d():
    inp = np.random.randn(16, 16).astype(np.float32)
    ker = np.random.randn(3, 3).astype(np.float32)

    # Expected via scipy-style correlation
    from scipy.signal import correlate2d
    expected = correlate2d(inp, ker, mode="valid")

    input_t = tk.Tensor.from_numpy(inp)
    kernel_t = tk.Tensor.from_numpy(ker)
    output_t = tk.Tensor.zeros([14, 14])

    tk.cpu.conv2d_im2col(input_t, kernel_t, output_t)
    result = output_t.numpy()

    check("conv2d_im2col vs scipy",
          np.allclose(result, expected, atol=1e-2),
          f"max diff: {np.abs(result - expected).max()}")


def main():
    print("=== TensorKernel Python Tests ===\n")

    test_tensor_creation()
    test_tensor_zeros()
    test_tensor_ones()
    test_numpy_roundtrip()
    test_matmul_vs_numpy()
    test_matmul_openmp_vs_numpy()
    test_add_vs_numpy()
    test_relu()
    test_reduce_sum()

    try:
        from scipy.signal import correlate2d
        test_conv2d()
    except ImportError:
        print("  SKIP: conv2d test (scipy not installed)")

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 40}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
