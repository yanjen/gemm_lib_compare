#include <cuda_runtime.h>

#include <gemm_CUDA.hpp>

__global__ void gemm_CUDA(int m, int n, int k, double alpha, double *A, int lda,
                          double *B, int ldb, double beta, double *C, int ldc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < n; ++j) {
        C[j * m + i] *= beta;
    }

    for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
            C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
        }
    }
}
