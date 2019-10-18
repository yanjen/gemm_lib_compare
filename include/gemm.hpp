#ifndef GEMM_HPP_
#define GEMM_HPP_

void gemm_reference(int m, int n, int k, double alpha, double *A, int lda,
                    double *B, int ldb, double beta, double *C, int ldc);

#endif  // GEMM_HPP_