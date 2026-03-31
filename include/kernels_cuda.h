#pragma once

#ifdef TK_CUDA_ENABLED

#include <cstddef>

namespace tk { namespace cuda {

// ============================================================
// Matrix Multiplication (shared memory tiled)
// A[M x K] * B[K x N] = C[M x N], all device pointers
// ============================================================

void matmul(float* A, float* B, float* C,
            int M, int K, int N, int tile_size = 16);

// ============================================================
// Element-wise Operations (device pointers, n elements)
// ============================================================

void add(const float* A, const float* B, float* C, int n);
void multiply(const float* A, const float* B, float* C, int n);
void relu(float* X, int n);
void sigmoid(float* X, int n);

// ============================================================
// Reductions (device pointer in, host scalar out)
// ============================================================

float reduce_sum(const float* X, int n);
float reduce_max(const float* X, int n);

// ============================================================
// Memory Management
// ============================================================

void device_malloc(float** ptr, size_t count);
void device_free(float* ptr);
void host_to_device(float* dst, const float* src, size_t count);
void device_to_host(float* dst, const float* src, size_t count);
void device_synchronize();

}} // namespace tk::cuda

#endif // TK_CUDA_ENABLED
