#include "kernels_cuda.h"

#ifdef TK_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdio>

// Macro for checking CUDA errors with file/line context
#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while(0)

namespace tk { namespace cuda {

// ============================================================
// Matrix Multiplication — Shared Memory Tiled
// ============================================================
//
// Each thread block loads a TILE_SIZE x TILE_SIZE tile from A and B
// into shared memory, computes partial dot products, then advances
// to the next tile along the K dimension.
//
// Shared memory reduces global memory traffic by a factor of TILE_SIZE.
// With TILE_SIZE=16: each tile = 16*16*4 = 1024 bytes, two tiles = 2 KB
// total shared memory per block (well under the 48 KB limit).
// 16*16 = 256 threads per block gives good occupancy.

#define TILE_SIZE_MM 16

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
    __shared__ float As[TILE_SIZE_MM][TILE_SIZE_MM];
    __shared__ float Bs[TILE_SIZE_MM][TILE_SIZE_MM];

    int row = blockIdx.y * TILE_SIZE_MM + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_MM + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE_MM - 1) / TILE_SIZE_MM;

    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative load: each thread loads one element of A and B
        int a_col = t * TILE_SIZE_MM + threadIdx.x;
        int b_row = t * TILE_SIZE_MM + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_MM; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul(float* A, float* B, float* C,
            int M, int K, int N, int tile_size) {
    dim3 block(TILE_SIZE_MM, TILE_SIZE_MM);
    dim3 grid((N + TILE_SIZE_MM - 1) / TILE_SIZE_MM,
              (M + TILE_SIZE_MM - 1) / TILE_SIZE_MM);

    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Element-wise Kernels
// ============================================================
//
// Grid-stride loop pattern: each thread processes multiple elements
// if the tensor is larger than grid*block. This is more robust than
// launching exactly n threads and handles arbitrary sizes.

#define BLOCK_SIZE 256

__global__ void add_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += gridDim.x * blockDim.x) {
        C[idx] = A[idx] + B[idx];
    }
}

void add(const float* A, const float* B, float* C, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<grid, BLOCK_SIZE>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void multiply_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += gridDim.x * blockDim.x) {
        C[idx] = A[idx] * B[idx];
    }
}

void multiply(const float* A, const float* B, float* C, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    multiply_kernel<<<grid, BLOCK_SIZE>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_kernel(float* X, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += gridDim.x * blockDim.x) {
        X[idx] = fmaxf(X[idx], 0.0f);
    }
}

void relu(float* X, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<grid, BLOCK_SIZE>>>(X, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sigmoid_kernel(float* X, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += gridDim.x * blockDim.x) {
        X[idx] = 1.0f / (1.0f + expf(-X[idx]));
    }
}

void sigmoid(float* X, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_kernel<<<grid, BLOCK_SIZE>>>(X, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Parallel Reduction — Tree Reduction in Shared Memory
// ============================================================
//
// Each thread loads two elements (halving idle threads in the first step).
// Tree reduction uses sequential addressing to avoid shared memory bank
// conflicts: thread i adds element at [i + stride] to [i].
// Final per-block partial result is written to output[blockIdx.x].
// Host-side launcher does a second pass to reduce block partial sums.

__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads two elements to reduce idle threads
    float val = 0.0f;
    if (i < n)     val += input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction (sequential addressing, no bank conflicts)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

float reduce_sum(const float* X, int n) {
    const int threads = BLOCK_SIZE;
    int blocks = (n + threads * 2 - 1) / (threads * 2);

    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    // First pass: n elements -> blocks partial sums
    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(X, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    // Second pass if needed: reduce partial sums
    while (blocks > 1) {
        int new_blocks = (blocks + threads * 2 - 1) / (threads * 2);
        float* d_partial2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial2, new_blocks * sizeof(float)));
        reduce_sum_kernel<<<new_blocks, threads, threads * sizeof(float)>>>(
            d_partial, d_partial2, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_partial));
        d_partial = d_partial2;
        blocks = new_blocks;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));
    return result;
}

__global__ void reduce_max_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = -INFINITY;
    if (i < n)     val = fmaxf(val, input[i]);
    if (i + blockDim.x < n) val = fmaxf(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

float reduce_max(const float* X, int n) {
    const int threads = BLOCK_SIZE;
    int blocks = (n + threads * 2 - 1) / (threads * 2);

    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    reduce_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(X, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    while (blocks > 1) {
        int new_blocks = (blocks + threads * 2 - 1) / (threads * 2);
        float* d_partial2 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial2, new_blocks * sizeof(float)));
        reduce_max_kernel<<<new_blocks, threads, threads * sizeof(float)>>>(
            d_partial, d_partial2, blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_partial));
        d_partial = d_partial2;
        blocks = new_blocks;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));
    return result;
}

// ============================================================
// Memory Management
// ============================================================

void device_malloc(float** ptr, size_t count) {
    CUDA_CHECK(cudaMalloc(ptr, count * sizeof(float)));
}

void device_free(float* ptr) {
    if (ptr) cudaFree(ptr);
}

void host_to_device(float* dst, const float* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice));
}

void device_to_host(float* dst, const float* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost));
}

void device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

}} // namespace tk::cuda

#endif // TK_CUDA_ENABLED
