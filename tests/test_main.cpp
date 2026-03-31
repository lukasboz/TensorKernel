#include "tensor.h"
#include "kernels_cpu.h"

#ifdef TK_CUDA_ENABLED
#include "kernels_cuda.h"
#endif

#include <cmath>
#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <cstring>

// ============================================================
// Minimal Test Framework (no external dependencies)
// ============================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

struct TestFailure {
    std::string file;
    int line;
    std::string msg;
};

static std::vector<TestFailure> g_failures;

#define ASSERT_TRUE(cond) do {                                      \
    if (!(cond)) {                                                  \
        g_failures.push_back({__FILE__, __LINE__,                   \
            std::string("ASSERT_TRUE failed: ") + #cond});          \
        g_tests_failed++; g_tests_run++; return;                    \
    }                                                               \
} while(0)

#define ASSERT_EQ(a, b) do {                                        \
    if ((a) != (b)) {                                               \
        g_failures.push_back({__FILE__, __LINE__,                   \
            std::string("ASSERT_EQ failed: ") + #a + " != " + #b});\
        g_tests_failed++; g_tests_run++; return;                    \
    }                                                               \
} while(0)

#define ASSERT_NEAR(a, b, tol) do {                                 \
    float _a = (a), _b = (b), _t = (tol);                          \
    if (std::fabs(_a - _b) > _t) {                                 \
        g_failures.push_back({__FILE__, __LINE__,                   \
            "ASSERT_NEAR failed: " + std::to_string(_a) +          \
            " vs " + std::to_string(_b) +                          \
            " (diff=" + std::to_string(std::fabs(_a-_b)) +         \
            ", tol=" + std::to_string(_t) + ")"});                  \
        g_tests_failed++; g_tests_run++; return;                    \
    }                                                               \
} while(0)

#define RUN_TEST(fn) do {                                           \
    std::cout << "  " << #fn << "... ";                             \
    fn();                                                           \
    if (g_tests_run == g_tests_passed + g_tests_failed               \
        && g_tests_run > 0) {                                       \
        /* test didn't increment, so it passed */                   \
    }                                                               \
    g_tests_run++; g_tests_passed++;                                \
    std::cout << "PASS\n";                                          \
} while(0)

// Helper: compare two float arrays element-wise
static bool arrays_near(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// ============================================================
// Tensor Tests
// ============================================================

void test_tensor_zeros() {
    auto t = tk::Tensor::zeros({3, 4});
    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.numel(), 12);
    ASSERT_EQ(t.ndim(), 2);
    for (int64_t i = 0; i < t.numel(); ++i)
        ASSERT_NEAR(t.data_f32()[i], 0.0f, 1e-7f);
}

void test_tensor_ones() {
    auto t = tk::Tensor::ones({2, 3});
    for (int64_t i = 0; i < t.numel(); ++i)
        ASSERT_NEAR(t.data_f32()[i], 1.0f, 1e-7f);
}

void test_tensor_rand() {
    auto t = tk::Tensor::rand({100});
    // All values should be in [0, 1)
    for (int64_t i = 0; i < t.numel(); ++i) {
        ASSERT_TRUE(t.data_f32()[i] >= 0.0f);
        ASSERT_TRUE(t.data_f32()[i] < 1.0f);
    }
}

void test_tensor_from_data() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = tk::Tensor::from_data(data, {2, 3});
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 3);
    ASSERT_NEAR(t.data_f32()[0], 1.0f, 1e-7f);
    ASSERT_NEAR(t.data_f32()[5], 6.0f, 1e-7f);
}

void test_tensor_clone() {
    auto t = tk::Tensor::rand({4, 4});
    auto c = t.clone();
    ASSERT_TRUE(arrays_near(t.data_f32(), c.data_f32(), 16, 1e-7f));
    // Modifying clone should not affect original
    c.data_f32()[0] = 999.0f;
    ASSERT_TRUE(t.data_f32()[0] != 999.0f);
}

void test_tensor_move() {
    auto t = tk::Tensor::rand({4, 4});
    float first_val = t.data_f32()[0];
    auto t2 = std::move(t);
    ASSERT_NEAR(t2.data_f32()[0], first_val, 1e-7f);
    ASSERT_EQ(t2.numel(), 16);
}

void test_tensor_at() {
    auto t = tk::Tensor::zeros({3, 4});
    t.at({1, 2}) = 42.0f;
    ASSERT_NEAR(t.at({1, 2}), 42.0f, 1e-7f);
    ASSERT_NEAR(t.data_f32()[1 * 4 + 2], 42.0f, 1e-7f);
}

void test_tensor_fill() {
    auto t = tk::Tensor::zeros({10});
    t.fill(3.14f);
    for (int64_t i = 0; i < 10; ++i)
        ASSERT_NEAR(t.data_f32()[i], 3.14f, 1e-6f);
}

void test_tensor_strides() {
    auto t = tk::Tensor::zeros({3, 4, 5});
    // Row-major strides: [4*5, 5, 1] = [20, 5, 1]
    ASSERT_EQ(t.strides()[0], 20);
    ASSERT_EQ(t.strides()[1], 5);
    ASSERT_EQ(t.strides()[2], 1);
}

// ============================================================
// Matrix Multiplication Tests
// ============================================================

void test_matmul_naive_identity() {
    // Multiply by identity matrix
    float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto A = tk::Tensor::from_data(a_data, {3, 3});
    float eye[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto I = tk::Tensor::from_data(eye, {3, 3});
    auto C = tk::Tensor::zeros({3, 3});

    tk::cpu::matmul_naive(A, I, C);
    ASSERT_TRUE(arrays_near(C.data_f32(), a_data, 9, 1e-5f));
}

void test_matmul_tiled_vs_naive() {
    auto A = tk::Tensor::rand({64, 48});
    auto B = tk::Tensor::rand({48, 64});
    auto C_naive = tk::Tensor::zeros({64, 64});
    auto C_tiled = tk::Tensor::zeros({64, 64});

    tk::cpu::matmul_naive(A, B, C_naive);
    tk::cpu::matmul_tiled(A, B, C_tiled);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_tiled.data_f32(),
                            64 * 64, 1e-3f));
}

void test_matmul_simd_vs_naive() {
    auto A = tk::Tensor::rand({32, 64});
    auto B = tk::Tensor::rand({64, 48});
    auto C_naive = tk::Tensor::zeros({32, 48});
    auto C_simd = tk::Tensor::zeros({32, 48});

    tk::cpu::matmul_naive(A, B, C_naive);
    tk::cpu::matmul_simd(A, B, C_simd);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_simd.data_f32(),
                            32 * 48, 1e-3f));
}

void test_matmul_openmp_vs_naive() {
    auto A = tk::Tensor::rand({128, 64});
    auto B = tk::Tensor::rand({64, 96});
    auto C_naive = tk::Tensor::zeros({128, 96});
    auto C_omp = tk::Tensor::zeros({128, 96});

    tk::cpu::matmul_naive(A, B, C_naive);
    tk::cpu::matmul_openmp(A, B, C_omp);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_omp.data_f32(),
                            128 * 96, 1e-3f));
}

void test_matmul_non_square() {
    // Non-square: A[5 x 7] * B[7 x 3] = C[5 x 3]
    auto A = tk::Tensor::rand({5, 7});
    auto B = tk::Tensor::rand({7, 3});
    auto C_naive = tk::Tensor::zeros({5, 3});
    auto C_simd = tk::Tensor::zeros({5, 3});

    tk::cpu::matmul_naive(A, B, C_naive);
    tk::cpu::matmul_simd(A, B, C_simd);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_simd.data_f32(),
                            15, 1e-3f));
}

void test_matmul_large() {
    // Larger size to test tiling boundary conditions
    auto A = tk::Tensor::rand({100, 73});
    auto B = tk::Tensor::rand({73, 91});
    auto C_naive = tk::Tensor::zeros({100, 91});
    auto C_omp = tk::Tensor::zeros({100, 91});

    tk::cpu::matmul_naive(A, B, C_naive);
    tk::cpu::matmul_openmp(A, B, C_omp);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_omp.data_f32(),
                            100 * 91, 1e-2f));
}

// ============================================================
// Element-wise Tests
// ============================================================

void test_add_simd_vs_naive() {
    auto A = tk::Tensor::rand({1000});
    auto B = tk::Tensor::rand({1000});
    auto C_naive = tk::Tensor::zeros({1000});
    auto C_simd = tk::Tensor::zeros({1000});

    tk::cpu::add_naive(A, B, C_naive);
    tk::cpu::add_simd(A, B, C_simd);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_simd.data_f32(),
                            1000, 1e-6f));
}

void test_multiply_simd_vs_naive() {
    auto A = tk::Tensor::rand({1000});
    auto B = tk::Tensor::rand({1000});
    auto C_naive = tk::Tensor::zeros({1000});
    auto C_simd = tk::Tensor::zeros({1000});

    tk::cpu::multiply_naive(A, B, C_naive);
    tk::cpu::multiply_simd(A, B, C_simd);
    ASSERT_TRUE(arrays_near(C_naive.data_f32(), C_simd.data_f32(),
                            1000, 1e-6f));
}

void test_add_correctness() {
    float a[] = {1, 2, 3, 4};
    float b[] = {10, 20, 30, 40};
    auto A = tk::Tensor::from_data(a, {4});
    auto B = tk::Tensor::from_data(b, {4});
    auto C = tk::Tensor::zeros({4});

    tk::cpu::add_naive(A, B, C);
    ASSERT_NEAR(C.data_f32()[0], 11.0f, 1e-7f);
    ASSERT_NEAR(C.data_f32()[3], 44.0f, 1e-7f);
}

// ============================================================
// Activation Function Tests
// ============================================================

void test_relu_correctness() {
    float data[] = {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f, -0.1f};
    auto X = tk::Tensor::from_data(data, {6});
    tk::cpu::relu_naive(X);

    ASSERT_NEAR(X.data_f32()[0], 0.0f, 1e-7f);
    ASSERT_NEAR(X.data_f32()[1], 0.0f, 1e-7f);
    ASSERT_NEAR(X.data_f32()[2], 0.0f, 1e-7f);
    ASSERT_NEAR(X.data_f32()[3], 0.5f, 1e-7f);
    ASSERT_NEAR(X.data_f32()[4], 2.0f, 1e-7f);
    ASSERT_NEAR(X.data_f32()[5], 0.0f, 1e-7f);
}

void test_relu_simd_vs_naive() {
    auto X_naive = tk::Tensor::rand({1000});
    X_naive.fill(-0.5f); // Make some negatives
    for (int i = 0; i < 500; ++i)
        X_naive.data_f32()[i * 2] = 0.5f;

    auto X_simd = X_naive.clone();
    tk::cpu::relu_naive(X_naive);
    tk::cpu::relu_simd(X_simd);
    ASSERT_TRUE(arrays_near(X_naive.data_f32(), X_simd.data_f32(),
                            1000, 1e-6f));
}

void test_sigmoid_range() {
    auto X = tk::Tensor::rand({100});
    // Scale to [-5, 5]
    for (int64_t i = 0; i < 100; ++i)
        X.data_f32()[i] = X.data_f32()[i] * 10.0f - 5.0f;

    tk::cpu::sigmoid_naive(X);
    for (int64_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(X.data_f32()[i] > 0.0f);
        ASSERT_TRUE(X.data_f32()[i] < 1.0f);
    }
}

void test_sigmoid_zero() {
    float data[] = {0.0f};
    auto X = tk::Tensor::from_data(data, {1});
    tk::cpu::sigmoid_naive(X);
    ASSERT_NEAR(X.data_f32()[0], 0.5f, 1e-5f);
}

void test_sigmoid_simd_vs_naive() {
    auto X_naive = tk::Tensor::rand({1000});
    for (int64_t i = 0; i < 1000; ++i)
        X_naive.data_f32()[i] = X_naive.data_f32()[i] * 10.0f - 5.0f;

    auto X_simd = X_naive.clone();
    tk::cpu::sigmoid_naive(X_naive);
    tk::cpu::sigmoid_simd(X_simd);
    // Looser tolerance for fast exp approximation
    ASSERT_TRUE(arrays_near(X_naive.data_f32(), X_simd.data_f32(),
                            1000, 5e-3f));
}

// ============================================================
// Reduction Tests
// ============================================================

void test_reduce_sum_correctness() {
    float data[] = {1, 2, 3, 4, 5};
    auto X = tk::Tensor::from_data(data, {5});
    float sum = tk::cpu::reduce_sum_naive(X);
    ASSERT_NEAR(sum, 15.0f, 1e-5f);
}

void test_reduce_sum_simd_vs_naive() {
    auto X = tk::Tensor::rand({10000});
    float naive = tk::cpu::reduce_sum_naive(X);
    float simd  = tk::cpu::reduce_sum_simd(X);
    // Relative tolerance: SIMD reorders additions (non-associative)
    float tol = std::fabs(naive) * 1e-4f + 1e-3f;
    ASSERT_NEAR(naive, simd, tol);
}

void test_reduce_sum_openmp_vs_naive() {
    auto X = tk::Tensor::rand({10000});
    float naive = tk::cpu::reduce_sum_naive(X);
    float omp   = tk::cpu::reduce_sum_openmp(X);
    float tol = std::fabs(naive) * 1e-4f + 1e-3f;
    ASSERT_NEAR(naive, omp, tol);
}

void test_reduce_max_correctness() {
    float data[] = {1, 5, 3, 2, 4};
    auto X = tk::Tensor::from_data(data, {5});
    ASSERT_NEAR(tk::cpu::reduce_max_naive(X), 5.0f, 1e-7f);
}

void test_reduce_max_simd_vs_naive() {
    auto X = tk::Tensor::rand({10000});
    float naive = tk::cpu::reduce_max_naive(X);
    float simd  = tk::cpu::reduce_max_simd(X);
    ASSERT_NEAR(naive, simd, 1e-6f);
}

// ============================================================
// Convolution Tests
// ============================================================

void test_conv2d_naive_correctness() {
    // 3x3 input, 2x2 kernel -> 2x2 output
    float in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float ker[] = {1, 0, 0, 1};
    auto input = tk::Tensor::from_data(in, {3, 3});
    auto kernel = tk::Tensor::from_data(ker, {2, 2});
    auto output = tk::Tensor::zeros({2, 2});

    tk::cpu::conv2d_naive(input, kernel, output);
    // out[0][0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // out[0][1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // out[1][0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // out[1][1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    ASSERT_NEAR(output.data_f32()[0], 6.0f, 1e-5f);
    ASSERT_NEAR(output.data_f32()[1], 8.0f, 1e-5f);
    ASSERT_NEAR(output.data_f32()[2], 12.0f, 1e-5f);
    ASSERT_NEAR(output.data_f32()[3], 14.0f, 1e-5f);
}

void test_conv2d_im2col_vs_naive() {
    auto input = tk::Tensor::rand({16, 16});
    auto kernel = tk::Tensor::rand({3, 3});
    auto out_naive = tk::Tensor::zeros({14, 14});
    auto out_im2col = tk::Tensor::zeros({14, 14});

    tk::cpu::conv2d_naive(input, kernel, out_naive);
    tk::cpu::conv2d_im2col(input, kernel, out_im2col);
    ASSERT_TRUE(arrays_near(out_naive.data_f32(), out_im2col.data_f32(),
                            14 * 14, 1e-3f));
}

void test_conv2d_im2col_larger() {
    auto input = tk::Tensor::rand({32, 32});
    auto kernel = tk::Tensor::rand({5, 5});
    int OH = 28, OW = 28;
    auto out_naive = tk::Tensor::zeros({OH, OW});
    auto out_im2col = tk::Tensor::zeros({OH, OW});

    tk::cpu::conv2d_naive(input, kernel, out_naive);
    tk::cpu::conv2d_im2col(input, kernel, out_im2col);
    ASSERT_TRUE(arrays_near(out_naive.data_f32(), out_im2col.data_f32(),
                            OH * OW, 1e-2f));
}

// ============================================================
// CUDA Tests
// ============================================================

#ifdef TK_CUDA_ENABLED
void test_cuda_matmul_vs_cpu() {
    int N = 64;
    auto A = tk::Tensor::rand({N, N});
    auto B = tk::Tensor::rand({N, N});
    auto C_cpu = tk::Tensor::zeros({N, N});
    auto C_gpu = tk::Tensor::zeros({N, N});

    // CPU reference
    tk::cpu::matmul_naive(A, B, C_cpu);

    // GPU
    A.to_device();
    B.to_device();
    C_gpu.to_device();
    tk::cuda::matmul(A.device_data(), B.device_data(), C_gpu.device_data(),
                     N, N, N);
    tk::cuda::device_synchronize();
    C_gpu.to_host();

    ASSERT_TRUE(arrays_near(C_cpu.data_f32(), C_gpu.data_f32(),
                            N * N, 1e-2f));
}

void test_cuda_relu() {
    int N = 1000;
    auto X_cpu = tk::Tensor::rand({static_cast<int64_t>(N)});
    for (int i = 0; i < N; ++i)
        X_cpu.data_f32()[i] = X_cpu.data_f32()[i] * 2.0f - 1.0f;

    auto X_gpu = X_cpu.clone();

    tk::cpu::relu_naive(X_cpu);

    X_gpu.to_device();
    tk::cuda::relu(X_gpu.device_data(), N);
    tk::cuda::device_synchronize();
    X_gpu.to_host();

    ASSERT_TRUE(arrays_near(X_cpu.data_f32(), X_gpu.data_f32(), N, 1e-5f));
}

void test_cuda_reduce_sum() {
    int N = 10000;
    auto X = tk::Tensor::rand({static_cast<int64_t>(N)});
    float cpu_sum = tk::cpu::reduce_sum_naive(X);

    X.to_device();
    float gpu_sum = tk::cuda::reduce_sum(X.device_data(), N);
    tk::cuda::device_synchronize();

    float tol = std::fabs(cpu_sum) * 1e-3f + 1e-2f;
    ASSERT_NEAR(cpu_sum, gpu_sum, tol);
}
#endif

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== TensorKernel Test Suite ===\n\n";

    std::cout << "--- Tensor Tests ---\n";
    RUN_TEST(test_tensor_zeros);
    RUN_TEST(test_tensor_ones);
    RUN_TEST(test_tensor_rand);
    RUN_TEST(test_tensor_from_data);
    RUN_TEST(test_tensor_clone);
    RUN_TEST(test_tensor_move);
    RUN_TEST(test_tensor_at);
    RUN_TEST(test_tensor_fill);
    RUN_TEST(test_tensor_strides);

    std::cout << "\n--- Matrix Multiplication Tests ---\n";
    RUN_TEST(test_matmul_naive_identity);
    RUN_TEST(test_matmul_tiled_vs_naive);
    RUN_TEST(test_matmul_simd_vs_naive);
    RUN_TEST(test_matmul_openmp_vs_naive);
    RUN_TEST(test_matmul_non_square);
    RUN_TEST(test_matmul_large);

    std::cout << "\n--- Element-wise Tests ---\n";
    RUN_TEST(test_add_correctness);
    RUN_TEST(test_add_simd_vs_naive);
    RUN_TEST(test_multiply_simd_vs_naive);

    std::cout << "\n--- Activation Function Tests ---\n";
    RUN_TEST(test_relu_correctness);
    RUN_TEST(test_relu_simd_vs_naive);
    RUN_TEST(test_sigmoid_range);
    RUN_TEST(test_sigmoid_zero);
    RUN_TEST(test_sigmoid_simd_vs_naive);

    std::cout << "\n--- Reduction Tests ---\n";
    RUN_TEST(test_reduce_sum_correctness);
    RUN_TEST(test_reduce_sum_simd_vs_naive);
    RUN_TEST(test_reduce_sum_openmp_vs_naive);
    RUN_TEST(test_reduce_max_correctness);
    RUN_TEST(test_reduce_max_simd_vs_naive);

    std::cout << "\n--- Convolution Tests ---\n";
    RUN_TEST(test_conv2d_naive_correctness);
    RUN_TEST(test_conv2d_im2col_vs_naive);
    RUN_TEST(test_conv2d_im2col_larger);

#ifdef TK_CUDA_ENABLED
    std::cout << "\n--- CUDA Tests ---\n";
    RUN_TEST(test_cuda_matmul_vs_cpu);
    RUN_TEST(test_cuda_relu);
    RUN_TEST(test_cuda_reduce_sum);
#endif

    std::cout << "\n========================================\n";
    std::cout << "Results: " << g_tests_passed << " passed, "
              << g_tests_failed << " failed, "
              << g_tests_run << " total\n";

    if (!g_failures.empty()) {
        std::cout << "\nFailures:\n";
        for (const auto& f : g_failures)
            std::cout << "  " << f.file << ":" << f.line << " - " << f.msg << "\n";
    }

    std::cout << "========================================\n";
    return g_tests_failed;
}
