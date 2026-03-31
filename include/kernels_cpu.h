#pragma once

#include "tensor.h"

namespace tk { namespace cpu {

// ============================================================
// Matrix Multiplication: C[M x N] = A[M x K] * B[K x N]
// ============================================================

enum class MatmulImpl { Naive, Tiled, SIMD, OpenMP, Best };

void matmul_naive(const Tensor& A, const Tensor& B, Tensor& C);
void matmul_tiled(const Tensor& A, const Tensor& B, Tensor& C, int tile_size = 64);
void matmul_simd(const Tensor& A, const Tensor& B, Tensor& C);
void matmul_openmp(const Tensor& A, const Tensor& B, Tensor& C, int tile_size = 64);
void matmul(const Tensor& A, const Tensor& B, Tensor& C,
            MatmulImpl impl = MatmulImpl::Best);

// ============================================================
// Element-wise Operations
// ============================================================

void add_naive(const Tensor& A, const Tensor& B, Tensor& C);
void add_simd(const Tensor& A, const Tensor& B, Tensor& C);
void multiply_naive(const Tensor& A, const Tensor& B, Tensor& C);
void multiply_simd(const Tensor& A, const Tensor& B, Tensor& C);

// Activation functions (in-place)
void relu_naive(Tensor& X);
void relu_simd(Tensor& X);
void sigmoid_naive(Tensor& X);
void sigmoid_simd(Tensor& X);

// ============================================================
// Reduction Operations
// ============================================================

float reduce_sum_naive(const Tensor& X);
float reduce_sum_simd(const Tensor& X);
float reduce_sum_openmp(const Tensor& X);
float reduce_max_naive(const Tensor& X);
float reduce_max_simd(const Tensor& X);

// ============================================================
// 2D Convolution
// Input: [H x W], Kernel: [KH x KW], Output: [(H-KH+1) x (W-KW+1)]
// ============================================================

void conv2d_naive(const Tensor& input, const Tensor& kernel, Tensor& output);
void conv2d_im2col(const Tensor& input, const Tensor& kernel, Tensor& output);

// im2col helper
void im2col(const float* input, int H, int W, int KH, int KW, float* col_matrix);

}} // namespace tk::cpu
