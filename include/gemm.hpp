#ifndef GEMM_HPP_
#define GEMM_HPP_

#define matrix_size 32

void gemm_reference(int m, int n, int k, double alpha, double *A, int lda,
                    double *B, int ldb, double beta, double *C, int ldc);

void gemm_CUDA(int m, int n, int k, double alpha, double *A, int lda, double *B,
               int ldb, double beta, double *C, int ldc);

void gemm_OpenACC(int m, int n, int k, double alpha, double *A, int lda,
                  double *B, int ldb, double beta, double *C, int ldc);

void gemm_OpenMP(int m, int n, int k, double alpha, double *A, int lda,
                 double *B, int ldb, double beta, double *C, int ldc);

void gemm_OpenMP2(int m, int n, int k, double alpha, double *A, int lda,
                  double *B, int ldb, double beta, double *C, int ldc);

#endif  // GEMM_HPP_