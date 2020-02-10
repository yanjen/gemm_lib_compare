#include <cuda_runtime.h>

#ifndef GEMM_CUDA_HPP_
#define GEMM_CUDA_HPP_

__global__ void gemm_CUDA(int m, int n, int k, double alpha, double *A, int lda,
                          double *B, int ldb, double beta, double *C, int ldc);

__global__ void gemm_CUDA2(int m, int n, int k, double alpha, double *A,
                           int lda, double *B, int ldb, double beta, double *C,
                           int ldc);

__global__ void gemm_CUDA3(int m, int n, int k, double alpha, double *A,
                           int lda, double *B, int ldb, double beta, double *C,
                           int ldc);

__global__ void __launch_bounds__(256)
    gemm_CUDA4(int m, int n, int k, double alpha, double *A, int lda, double *B,
               int ldb, double beta, double *C, int ldc);

__global__ void __launch_bounds__(1024)
    gemm_CUDA5(int m, int n, int k, double alpha, double *A, int lda, double *B,
               int ldb, double beta, double *C, int ldc);

#endif  // GEMM_CUDA_HPP_