# TensorKernel

A high-performance tensor computation library showcasing optimized kernel implementations for neural network operations. Written in modern C++ with Python bindings, featuring CPU SIMD/OpenMP optimizations and optional CUDA support.

## Features

- **Multi-backend support**: CPU (with SIMD & OpenMP) and optional CUDA
- **Multiple kernel implementations**: Compare naive, tiled, SIMD, and parallel variants
- **Core operations**:
  - Matrix multiplication with configurable tile sizes
  - Element-wise operations (add, multiply)
  - Activation functions (ReLU, Sigmoid)
  - Reductions (sum, max)
  - 2D Convolution
- **Python bindings**: Seamless NumPy/PyTorch interoperability
- **Performance profiling**: Built-in metrics (GFLOPS, bandwidth, speedup analysis)
- **Memory optimization**: AVX2-aligned allocation for SIMD efficiency

## Building

### Requirements
- C++17 compiler
- CMake 3.15+
- (Optional) CUDA Toolkit for GPU support

### CPU-Only Build
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### With CUDA Support
```bash
mkdir build && cd build
cmake -DTK_CUDA_ENABLED=ON ..
cmake --build . --config Release
```

### Python Module Installation
```bash
cd build
pip install -e .
# or: python -m pip install .
```

## Quick Start

### Python Usage

```python
import tensorkernel as tk
import numpy as np

# Create tensors
A = tk.Tensor.rand([1024, 512])
B = tk.Tensor.rand([512, 1024])
C = tk.Tensor.zeros([1024, 1024])

# Run matrix multiplication
tk.cpu.matmul(A, B, C)

# Access as NumPy array (zero-copy view)
result = C.numpy()

# Element-wise operations
D = tk.Tensor.ones([1024, 1024])
tk.cpu.add(C, D, C)

# Reductions
total = tk.cpu.reduce_sum(C)
```

### Profiling

```python
results = tk.Profiler()
results.profile(
    "matmul",
    "SIMD",
    flop_count=2 * 1024 * 512 * 1024,
    bytes_moved=1024 * 512 * 4 + 512 * 1024 * 4,
    kernel=lambda: tk.cpu.matmul_simd(A, B, C)
)
results.print_summary()
results.export_csv("results.csv")
```

### PyTorch Integration

```python
import tensorkernel as tk
import torch

# Convert PyTorch → TensorKernel
pt_tensor = torch.randn(4, 4, dtype=torch.float32)
np_array = pt_tensor.detach().cpu().numpy()
tk_tensor = tk.Tensor.from_numpy(np_array)

# Run optimized kernel
tk.cpu.relu(tk_tensor)

# Convert back to PyTorch
result = torch.from_numpy(tk_tensor.numpy().copy())
```

## Project Structure

```
TensorKernel/
├── include/
│   ├── tensor.h           # Core tensor class
│   ├── kernels_cpu.h      # CPU kernel declarations
│   ├── kernels_cuda.h     # CUDA kernel declarations (optional)
│   └── profiler.h         # Performance profiling utilities
├── src/
│   ├── tensor.cpp         # Tensor implementation
│   ├── kernels_cpu.cpp    # CPU kernel implementations
│   ├── kernels_cuda.cu    # CUDA kernels
│   └── profiler.cpp       # Profiler implementation
├── bindings/
│   └── pybind_module.cpp  # Python bindings (pybind11)
├── tests/
│   ├── test_main.cpp      # C++ tests
│   └── test_python.py     # Python tests
├── benchmark/
│   ├── benchmark_main.cpp # C++ benchmarks
│   └── benchmark.py       # Python benchmarks & charts
└── scripts/
    ├── pytorch_integration.py  # PyTorch integration demo
    └── profile_analysis.py     # Analysis utilities
```

## Testing

### Run Python Tests
```bash
python tests/test_python.py
```

### Run C++ Tests
```bash
./build/test_main
```

### Run Benchmarks
```bash
# Python benchmark suite with charts
python benchmark/benchmark.py

# C++ benchmarks
./build/benchmark_main
```

## Performance Notes

- **AVX2 Alignment**: Tensors allocated with 32-byte alignment for SIMD efficiency
- **Tiled Matmul**: Default tile size of 64 balances cache utilization
- **Thread Pool**: OpenMP implementations scale with available cores
- **Memory Layout**: Configurable row-major and column-major layouts

## API Highlights

### Tensor Creation
```cpp
// C++
tk::Tensor t1(std::vector<int64_t>{1024, 512});
auto t2 = tk::Tensor::zeros({512, 256});
auto t3 = tk::Tensor::rand({256, 256});
auto t4 = tk::Tensor::from_data(data_ptr, {rows, cols});
```

### Kernel Implementations
Each major operation offers multiple strategies:
- **Naive**: Reference implementation
- **Tiled**: Cache-friendly with configurable tile size
- **SIMD**: Vectorized with AVX2 intrinsics
- **OpenMP**: Parallel with thread-level decomposition
- **Best**: Auto-selects best variant based on tensor size

### Moving Data

```python
# CPU to GPU (CUDA-enabled builds)
tensor.to_device()

# GPU back to CPU
tensor.to_host()

# Check where data lives
if tensor.is_on_device():
    print("Data is on GPU")
```

## Development

### Adding a New Kernel

1. Declare in `include/kernels_cpu.h` (and `.cu` for CUDA)
2. Implement in `src/kernels_cpu.cpp` (and `.cu` for CUDA)
3. Bind in `bindings/pybind_module.cpp`
4. Add tests to `tests/test_python.py` and `tests/test_main.cpp`

### Profiling Custom Code

```cpp
tk::Profiler prof;
prof.profile(
    "my_kernel",
    "variant_name",
    flop_count,
    bytes_moved,
    []() { /* kernel call */ }
);
prof.print_summary();
```

## License

[Add your license here]

## Contributing

Contributions welcome! Please ensure:
- Code passes all tests
- Benchmarks show improvements or maintain performance
- Documentation is updated