#include "tensor.h"
#include "kernels_cpu.h"
#include "profiler.h"

#ifdef TK_CUDA_ENABLED
#include "kernels_cuda.h"
#endif

#include <iostream>
#include <cstring>

// ============================================================
// Matrix Multiplication Benchmarks
// ============================================================

void benchmark_matmul(tk::Profiler& prof) {
    std::cout << "\n=== Matrix Multiplication Benchmarks ===\n";

    int sizes[] = {128, 256, 512, 1024};

    for (int N : sizes) {
        auto A = tk::Tensor::rand({N, N});
        auto B = tk::Tensor::rand({N, N});
        auto C = tk::Tensor::zeros({N, N});

        // 2*M*N*K FLOP for matmul (multiply + add per output element per K)
        int64_t flops = 2LL * N * N * N;
        int64_t bytes = (2LL * N * N + static_cast<int64_t>(N) * N) * sizeof(float);

        // Naive baseline
        auto r_naive = prof.profile("matmul", "naive", flops, bytes,
            [&]{ tk::cpu::matmul_naive(A, B, C); }, 1, 3);
        r_naive.M = N; r_naive.N = N; r_naive.K = N;
        r_naive.speedup = 1.0;
        prof.add_result(r_naive);

        // Tiled
        auto r_tiled = prof.profile("matmul", "tiled", flops, bytes,
            [&]{ tk::cpu::matmul_tiled(A, B, C); });
        r_tiled.M = N; r_tiled.N = N; r_tiled.K = N;
        r_tiled.speedup = r_naive.elapsed_ms / r_tiled.elapsed_ms;
        prof.add_result(r_tiled);

        // SIMD
        auto r_simd = prof.profile("matmul", "simd", flops, bytes,
            [&]{ tk::cpu::matmul_simd(A, B, C); });
        r_simd.M = N; r_simd.N = N; r_simd.K = N;
        r_simd.speedup = r_naive.elapsed_ms / r_simd.elapsed_ms;
        prof.add_result(r_simd);

        // OpenMP
        auto r_omp = prof.profile("matmul", "openmp", flops, bytes,
            [&]{ tk::cpu::matmul_openmp(A, B, C); });
        r_omp.M = N; r_omp.N = N; r_omp.K = N;
        r_omp.speedup = r_naive.elapsed_ms / r_omp.elapsed_ms;
        prof.add_result(r_omp);

#ifdef TK_CUDA_ENABLED
        // CUDA
        A.to_device();
        B.to_device();
        C.to_device();
        tk::cuda::device_synchronize();

        auto r_cuda = prof.profile("matmul", "cuda", flops, bytes,
            [&]{
                tk::cuda::matmul(A.device_data(), B.device_data(), C.device_data(),
                                 N, N, N);
                tk::cuda::device_synchronize();
            });
        r_cuda.M = N; r_cuda.N = N; r_cuda.K = N;
        r_cuda.speedup = r_naive.elapsed_ms / r_cuda.elapsed_ms;
        prof.add_result(r_cuda);

        A.free_device();
        B.free_device();
        C.free_device();
#endif
    }
}

// ============================================================
// Element-wise Operation Benchmarks
// ============================================================

void benchmark_elementwise(tk::Profiler& prof) {
    std::cout << "\n=== Element-wise Benchmarks ===\n";

    int sizes[] = {100000, 1000000, 10000000};

    for (int N : sizes) {
        auto A = tk::Tensor::rand({static_cast<int64_t>(N)});
        auto B = tk::Tensor::rand({static_cast<int64_t>(N)});
        auto C = tk::Tensor::zeros({static_cast<int64_t>(N)});

        int64_t bytes = 3LL * N * sizeof(float); // 2 reads + 1 write

        // Add
        auto r_add_naive = prof.profile("add", "naive", N, bytes,
            [&]{ tk::cpu::add_naive(A, B, C); });
        r_add_naive.N = N;
        r_add_naive.speedup = 1.0;
        prof.add_result(r_add_naive);

        auto r_add_simd = prof.profile("add", "simd", N, bytes,
            [&]{ tk::cpu::add_simd(A, B, C); });
        r_add_simd.N = N;
        r_add_simd.speedup = r_add_naive.elapsed_ms / r_add_simd.elapsed_ms;
        prof.add_result(r_add_simd);

        // ReLU
        auto X = tk::Tensor::rand({static_cast<int64_t>(N)});
        int64_t relu_bytes = static_cast<int64_t>(N) * sizeof(float);

        auto r_relu_naive = prof.profile("relu", "naive", N, relu_bytes,
            [&]{ tk::cpu::relu_naive(X); });
        r_relu_naive.N = N;
        r_relu_naive.speedup = 1.0;
        prof.add_result(r_relu_naive);

        X = tk::Tensor::rand({static_cast<int64_t>(N)});
        auto r_relu_simd = prof.profile("relu", "simd", N, relu_bytes,
            [&]{ tk::cpu::relu_simd(X); });
        r_relu_simd.N = N;
        r_relu_simd.speedup = r_relu_naive.elapsed_ms / r_relu_simd.elapsed_ms;
        prof.add_result(r_relu_simd);

        // Sigmoid
        X = tk::Tensor::rand({static_cast<int64_t>(N)});
        auto r_sig_naive = prof.profile("sigmoid", "naive", 4LL * N, relu_bytes,
            [&]{ tk::cpu::sigmoid_naive(X); });
        r_sig_naive.N = N;
        r_sig_naive.speedup = 1.0;
        prof.add_result(r_sig_naive);

        X = tk::Tensor::rand({static_cast<int64_t>(N)});
        auto r_sig_simd = prof.profile("sigmoid", "simd", 4LL * N, relu_bytes,
            [&]{ tk::cpu::sigmoid_simd(X); });
        r_sig_simd.N = N;
        r_sig_simd.speedup = r_sig_naive.elapsed_ms / r_sig_simd.elapsed_ms;
        prof.add_result(r_sig_simd);
    }
}

// ============================================================
// Reduction Benchmarks
// ============================================================

void benchmark_reduction(tk::Profiler& prof) {
    std::cout << "\n=== Reduction Benchmarks ===\n";

    int sizes[] = {100000, 1000000, 10000000};

    for (int N : sizes) {
        auto X = tk::Tensor::rand({static_cast<int64_t>(N)});
        int64_t bytes = static_cast<int64_t>(N) * sizeof(float);

        auto r_naive = prof.profile("reduce_sum", "naive", N, bytes,
            [&]{ tk::cpu::reduce_sum_naive(X); });
        r_naive.N = N;
        r_naive.speedup = 1.0;
        prof.add_result(r_naive);

        auto r_simd = prof.profile("reduce_sum", "simd", N, bytes,
            [&]{ tk::cpu::reduce_sum_simd(X); });
        r_simd.N = N;
        r_simd.speedup = r_naive.elapsed_ms / r_simd.elapsed_ms;
        prof.add_result(r_simd);

        auto r_omp = prof.profile("reduce_sum", "openmp", N, bytes,
            [&]{ tk::cpu::reduce_sum_openmp(X); });
        r_omp.N = N;
        r_omp.speedup = r_naive.elapsed_ms / r_omp.elapsed_ms;
        prof.add_result(r_omp);

#ifdef TK_CUDA_ENABLED
        X.to_device();
        tk::cuda::device_synchronize();

        auto r_cuda = prof.profile("reduce_sum", "cuda", N, bytes,
            [&]{
                tk::cuda::reduce_sum(X.device_data(), N);
                tk::cuda::device_synchronize();
            });
        r_cuda.N = N;
        r_cuda.speedup = r_naive.elapsed_ms / r_cuda.elapsed_ms;
        prof.add_result(r_cuda);

        X.free_device();
#endif
    }
}

// ============================================================
// Convolution Benchmarks
// ============================================================

void benchmark_conv2d(tk::Profiler& prof) {
    std::cout << "\n=== 2D Convolution Benchmarks ===\n";

    int img_sizes[] = {64, 128, 256, 512};
    int kernel_size = 3;

    for (int S : img_sizes) {
        auto input = tk::Tensor::rand({static_cast<int64_t>(S), static_cast<int64_t>(S)});
        auto kernel = tk::Tensor::rand({kernel_size, kernel_size});
        int OH = S - kernel_size + 1;
        int OW = S - kernel_size + 1;
        auto output = tk::Tensor::zeros({OH, OW});

        int64_t flops = 2LL * OH * OW * kernel_size * kernel_size;
        int64_t bytes = (static_cast<int64_t>(S) * S + kernel_size * kernel_size +
                         static_cast<int64_t>(OH) * OW) * sizeof(float);

        auto r_naive = prof.profile("conv2d", "naive", flops, bytes,
            [&]{ tk::cpu::conv2d_naive(input, kernel, output); });
        r_naive.M = S; r_naive.N = S; r_naive.K = kernel_size;
        r_naive.speedup = 1.0;
        prof.add_result(r_naive);

        auto r_im2col = prof.profile("conv2d", "im2col", flops, bytes,
            [&]{ tk::cpu::conv2d_im2col(input, kernel, output); });
        r_im2col.M = S; r_im2col.N = S; r_im2col.K = kernel_size;
        r_im2col.speedup = r_naive.elapsed_ms / r_im2col.elapsed_ms;
        prof.add_result(r_im2col);
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  TensorKernel Performance Benchmarks\n";
    std::cout << "========================================\n";

    tk::Profiler prof;

    benchmark_matmul(prof);
    benchmark_elementwise(prof);
    benchmark_reduction(prof);
    benchmark_conv2d(prof);

    prof.print_summary();
    prof.export_csv("benchmark_results.csv");

    std::cout << "\nResults exported to benchmark_results.csv\n";
    return 0;
}
