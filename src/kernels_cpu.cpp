#include "kernels_cpu.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>

#ifdef TK_AVX2_ENABLED
#include <immintrin.h>
#endif

namespace tk { namespace cpu {

// ============================================================
// Matrix Multiplication
// ============================================================

// Naive triple-loop implementation (ijk order).
// Memory-bound: B access pattern b[k*N+j] strides through columns,
// destroying cache line reuse for large matrices.
void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = static_cast<int>(A.shape()[0]);
    const int K = static_cast<int>(A.shape()[1]);
    const int N = static_cast<int>(B.shape()[1]);

    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();

    // Zero output
    std::memset(c, 0, static_cast<size_t>(M) * N * sizeof(float));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// Cache-tiled implementation.
// Tile size 64: a 64x64 float tile = 16 KB, fits in L1 cache (typically 32-48 KB).
// Loop order (ii, kk, jj, i, k, j) ensures B tiles stay hot in L1 across j iterations.
// The ikj inner order broadcasts a[i][k] and streams through b[k][j..j+tile],
// which is sequential in memory (row-major) and cache-friendly.
void matmul_tiled(const Tensor& A, const Tensor& B, Tensor& C, int tile) {
    const int M = static_cast<int>(A.shape()[0]);
    const int K = static_cast<int>(A.shape()[1]);
    const int N = static_cast<int>(B.shape()[1]);

    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();

    std::memset(c, 0, static_cast<size_t>(M) * N * sizeof(float));

    for (int ii = 0; ii < M; ii += tile) {
        const int i_end = std::min(ii + tile, M);
        for (int kk = 0; kk < K; kk += tile) {
            const int k_end = std::min(kk + tile, K);
            for (int jj = 0; jj < N; jj += tile) {
                const int j_end = std::min(jj + tile, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = a[i * K + k];
                        for (int j = jj; j < j_end; ++j) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// AVX2 SIMD implementation.
// Processes 8 floats per iteration using 256-bit registers.
// Uses _mm256_fmadd_ps (fused multiply-add): computes a*b+c in a single
// instruction, doubling throughput vs separate mul + add.
// a[i][k] is broadcast to all 8 lanes, then FMA'd with 8-wide b[k][j..j+7].
void matmul_simd(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = static_cast<int>(A.shape()[0]);
    const int K = static_cast<int>(A.shape()[1]);
    const int N = static_cast<int>(B.shape()[1]);

    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();

    std::memset(c, 0, static_cast<size_t>(M) * N * sizeof(float));

#ifdef TK_AVX2_ENABLED
    const int N8 = N & ~7; // largest multiple of 8 <= N

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            __m256 a_ik = _mm256_set1_ps(a[i * K + k]);
            int j = 0;
            for (; j < N8; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&b[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&c[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);
                _mm256_storeu_ps(&c[i * N + j], c_vec);
            }
            // Scalar tail for remaining elements (N not divisible by 8)
            float a_val = a[i * K + k];
            for (; j < N; ++j) {
                c[i * N + j] += a_val * b[k * N + j];
            }
        }
    }
#else
    // Fallback to tiled if AVX2 not available
    matmul_tiled(A, B, C, 64);
#endif
}

// Combined tiling + OpenMP + SIMD.
// Parallelizes the outer tile loop across threads. Each thread owns
// a contiguous block of rows in C, so no data races without locks.
// Inner loop uses AVX2 FMA for compute throughput.
void matmul_openmp(const Tensor& A, const Tensor& B, Tensor& C, int tile) {
    const int M = static_cast<int>(A.shape()[0]);
    const int K = static_cast<int>(A.shape()[1]);
    const int N = static_cast<int>(B.shape()[1]);

    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();

    std::memset(c, 0, static_cast<size_t>(M) * N * sizeof(float));

#ifdef TK_AVX2_ENABLED
    const int N8 = N & ~7;
#endif

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < M; ii += tile) {
        const int i_end = std::min(ii + tile, M);
        for (int kk = 0; kk < K; kk += tile) {
            const int k_end = std::min(kk + tile, K);
            for (int jj = 0; jj < N; jj += tile) {
                const int j_end = std::min(jj + tile, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = a[i * K + k];
#ifdef TK_AVX2_ENABLED
                        __m256 a_vec = _mm256_set1_ps(a_ik);
                        int j = jj;
                        const int j_end8 = std::min(j_end, N8);
                        for (; j + 8 <= j_end8; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&b[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&c[i * N + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&c[i * N + j], c_vec);
                        }
                        for (; j < j_end; ++j)
                            c[i * N + j] += a_ik * b[k * N + j];
#else
                        for (int j = jj; j < j_end; ++j)
                            c[i * N + j] += a_ik * b[k * N + j];
#endif
                    }
                }
            }
        }
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C, MatmulImpl impl) {
    switch (impl) {
        case MatmulImpl::Naive:  matmul_naive(A, B, C); break;
        case MatmulImpl::Tiled:  matmul_tiled(A, B, C); break;
        case MatmulImpl::SIMD:   matmul_simd(A, B, C); break;
        case MatmulImpl::OpenMP: matmul_openmp(A, B, C); break;
        case MatmulImpl::Best:   matmul_openmp(A, B, C); break;
    }
}

// ============================================================
// Element-wise Operations
// ============================================================

void add_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    const int64_t n = A.numel();
    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();
    for (int64_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

void add_simd(const Tensor& A, const Tensor& B, Tensor& C) {
    const int64_t n = A.numel();
    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();

#ifdef TK_AVX2_ENABLED
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i)
        c[i] = a[i] + b[i];
#else
    add_naive(A, B, C);
#endif
}

void multiply_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    const int64_t n = A.numel();
    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();
    for (int64_t i = 0; i < n; ++i)
        c[i] = a[i] * b[i];
}

void multiply_simd(const Tensor& A, const Tensor& B, Tensor& C) {
    const int64_t n = A.numel();
    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();

#ifdef TK_AVX2_ENABLED
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i)
        c[i] = a[i] * b[i];
#else
    multiply_naive(A, B, C);
#endif
}

// ============================================================
// Activation Functions
// ============================================================

void relu_naive(Tensor& X) {
    const int64_t n = X.numel();
    float* x = X.data_f32();
    for (int64_t i = 0; i < n; ++i)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void relu_simd(Tensor& X) {
    const int64_t n = X.numel();
    float* x = X.data_f32();

#ifdef TK_AVX2_ENABLED
    __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; ++i)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
#else
    relu_naive(X);
#endif
}

void sigmoid_naive(Tensor& X) {
    const int64_t n = X.numel();
    float* x = X.data_f32();
    for (int64_t i = 0; i < n; ++i)
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

// Fast SIMD sigmoid using polynomial approximation of exp.
// Uses Schraudolph's trick adapted for AVX2: interprets float bits
// as a linear approximation of 2^x, then adjusts for base e.
// Accurate to ~0.1% relative error, sufficient for ML inference.
#ifdef TK_AVX2_ENABLED
static inline __m256 fast_exp_avx2(__m256 x) {
    // Clamp to prevent overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e))
    // Use the identity: 2^n = reinterpret(n * 2^23 + 127 * 2^23) as float
    const __m256 log2e = _mm256_set1_ps(1.442695041f);
    const __m256 shift = _mm256_set1_ps(12582912.0f);   // 1.5 * 2^23
    const __m256 magic = _mm256_set1_ps(1065353216.0f);  // 127 * 2^23

    __m256 t = _mm256_fmadd_ps(x, log2e, shift);

    // Extract integer part and fraction
    __m256i ti = _mm256_castps_si256(t);
    __m256 frac = _mm256_sub_ps(t, shift);
    __m256 int_part = _mm256_sub_ps(_mm256_mul_ps(frac, _mm256_set1_ps(8388608.0f)), magic);

    // Polynomial correction for fractional part
    // p(f) ≈ 1 + f*ln(2) + f^2*ln(2)^2/2
    __m256 f = _mm256_sub_ps(_mm256_mul_ps(x, log2e), frac);
    const __m256 ln2 = _mm256_set1_ps(0.6931472f);
    const __m256 half_ln2_sq = _mm256_set1_ps(0.2402265f);
    __m256 poly = _mm256_fmadd_ps(f, half_ln2_sq, ln2);
    poly = _mm256_fmadd_ps(f, poly, _mm256_set1_ps(1.0f));

    // Construct 2^int via bit manipulation
    __m256i int_bits = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(frac), _mm256_set1_epi32(127)),
        23);
    __m256 pow2 = _mm256_castsi256_ps(int_bits);

    return _mm256_mul_ps(pow2, poly);
}
#endif

void sigmoid_simd(Tensor& X) {
    const int64_t n = X.numel();
    float* x = X.data_f32();

#ifdef TK_AVX2_ENABLED
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 neg_v = _mm256_mul_ps(v, neg_one);
        __m256 exp_neg = fast_exp_avx2(neg_v);
        __m256 denom = _mm256_add_ps(one, exp_neg);
        // 1.0 / denom using Newton-Raphson refinement
        __m256 rcp = _mm256_rcp_ps(denom);                          // ~12-bit approx
        rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, _mm256_set1_ps(2.0f))); // refine
        _mm256_storeu_ps(x + i, rcp);
    }
    for (; i < n; ++i)
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
#else
    sigmoid_naive(X);
#endif
}

// ============================================================
// Reduction Operations
// ============================================================

float reduce_sum_naive(const Tensor& X) {
    const int64_t n = X.numel();
    const float* x = X.data_f32();
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i)
        sum += x[i];
    return sum;
}

// SIMD reduction: accumulate 8 partial sums, then horizontally reduce.
// Uses multiple accumulators to hide FP addition latency (4 cycles on modern CPUs).
float reduce_sum_simd(const Tensor& X) {
    const int64_t n = X.numel();
    const float* x = X.data_f32();

#ifdef TK_AVX2_ENABLED
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int64_t i = 0;
    // Unroll 4x to saturate the FP add pipeline
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(x + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(x + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(x + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(x + i + 24));
    }
    // Combine accumulators
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(x + i));

    // Horizontal sum: 8 floats -> 1
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // Scalar tail
    for (; i < n; ++i)
        sum += x[i];
    return sum;
#else
    return reduce_sum_naive(X);
#endif
}

float reduce_sum_openmp(const Tensor& X) {
    const int64_t n = X.numel();
    const float* x = X.data_f32();
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < n; ++i)
        sum += x[i];
    return sum;
}

float reduce_max_naive(const Tensor& X) {
    const int64_t n = X.numel();
    const float* x = X.data_f32();
    float max_val = x[0];
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > max_val) max_val = x[i];
    return max_val;
}

float reduce_max_simd(const Tensor& X) {
    const int64_t n = X.numel();
    const float* x = X.data_f32();

#ifdef TK_AVX2_ENABLED
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }

    // Horizontal max
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0x4E)); // swap hi/lo 64
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 0xB1)); // swap adjacent
    float max_val = _mm_cvtss_f32(max128);

    for (; i < n; ++i)
        if (x[i] > max_val) max_val = x[i];
    return max_val;
#else
    return reduce_max_naive(X);
#endif
}

// ============================================================
// 2D Convolution
// ============================================================

// Direct convolution: 4 nested loops.
// O(OH * OW * KH * KW) with poor spatial locality for large kernels.
void conv2d_naive(const Tensor& input, const Tensor& kernel, Tensor& output) {
    const int H  = static_cast<int>(input.shape()[0]);
    const int W  = static_cast<int>(input.shape()[1]);
    const int KH = static_cast<int>(kernel.shape()[0]);
    const int KW = static_cast<int>(kernel.shape()[1]);
    const int OH = H - KH + 1;
    const int OW = W - KW + 1;

    const float* in  = input.data_f32();
    const float* ker = kernel.data_f32();
    float* out = output.data_f32();

    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    sum += in[(oh + kh) * W + (ow + kw)] * ker[kh * KW + kw];
                }
            }
            out[oh * OW + ow] = sum;
        }
    }
}

// im2col: Rearranges input patches into a column matrix so that
// convolution becomes a single matrix multiply (GEMM).
// Column matrix shape: [KH*KW x OH*OW]
// Each column is a flattened receptive field patch of the input.
void im2col(const float* input, int H, int W, int KH, int KW, float* col_matrix) {
    const int OH = H - KH + 1;
    const int OW = W - KW + 1;

    for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
            const int row = kh * KW + kw;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    col_matrix[row * (OH * OW) + oh * OW + ow] =
                        input[(oh + kh) * W + (ow + kw)];
                }
            }
        }
    }
}

// im2col + GEMM convolution.
// 1. Transform input patches into column matrix via im2col
// 2. Reshape kernel to [1 x KH*KW] row vector
// 3. Multiply: output[1 x OH*OW] = kernel_row[1 x KH*KW] * col[KH*KW x OH*OW]
// This reuses the optimized matmul path and is exactly what Caffe/cuDNN does.
void conv2d_im2col(const Tensor& input, const Tensor& kernel, Tensor& output) {
    const int H  = static_cast<int>(input.shape()[0]);
    const int W  = static_cast<int>(input.shape()[1]);
    const int KH = static_cast<int>(kernel.shape()[0]);
    const int KW = static_cast<int>(kernel.shape()[1]);
    const int OH = H - KH + 1;
    const int OW = W - KW + 1;
    const int col_rows = KH * KW;
    const int col_cols = OH * OW;

    // Allocate im2col buffer
    Tensor col_matrix({col_rows, col_cols});
    im2col(input.data_f32(), H, W, KH, KW, col_matrix.data_f32());

    // Reshape kernel to [1 x KH*KW]
    Tensor kernel_row = Tensor::from_data(kernel.data_f32(), {1, static_cast<int64_t>(col_rows)});

    // Output as [1 x OH*OW]
    Tensor out_row({1, static_cast<int64_t>(col_cols)});

    // GEMM: [1 x KH*KW] * [KH*KW x OH*OW] = [1 x OH*OW]
    matmul_simd(kernel_row, col_matrix, out_row);

    // Copy result to output tensor
    std::memcpy(output.data_f32(), out_row.data_f32(), col_cols * sizeof(float));
}

}} // namespace tk::cpu
