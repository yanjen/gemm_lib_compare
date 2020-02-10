#include <cuda_runtime.h>

#include <gemm_CUDA.hpp>

__global__ void gemm_CUDA(int m, int n, int k, double alpha, double *A, int lda,
                          double *B, int ldb, double beta, double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < n; ++j) {
        C[i * n + j] *= beta;
    }

    for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
            C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
        }
    }
}

__global__ void gemm_CUDA2(int m, int n, int k, double alpha, double *A,
                           int lda, double *B, int ldb, double beta, double *C,
                           int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    C[i * n + j] *= beta;

    for (int l = 0; l < k; ++l) {
        C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
    }
}

__global__ void gemm_CUDA3(int m, int n, int k, double alpha, double *A,
                           int lda, double *B, int ldb, double beta, double *C,
                           int ldc)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    C[i * n + j] *= beta;

    for (int l = 0; l < k; ++l) {
        C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
    }
}

__global__ void __launch_bounds__(256)
    gemm_CUDA4(int m, int n, int k, double alpha, double *A, int lda, double *B,
               int ldb, double beta, double *C, int ldc)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    C[i * n + j] *= beta;

    for (int l = 0; l < k; ++l) {
        C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
    }
}

__global__ void __launch_bounds__(1024)
    gemm_CUDA5(int m, int n, int k, double alpha, double *A, int lda, double *B,
               int ldb, double beta, double *C, int ldc)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x;

    C[i * n + j] *= beta;

    int w = threadIdx.x;  // blockDim.x = 32

    double value;
    for (int l = 0; l < k / 32; l += 32) {
        value = alpha * A[i * k + (l + w)] * B[(l + w) * n + j];
        for (int o = 16; o >= 1; o /= 16)
            value += __shfl_xor_sync(0xffffffff, value, o, 32);
        if (w == 0) C[i * n + j] += value;
    }
}
