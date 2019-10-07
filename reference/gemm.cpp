#include <reference/gemm.hpp>

void gemm_reference(char transa, char transb, int m, int n, int k, double alpha,
                    double *A, int lda, double *B, int ldb, double beta,
                    double *C, int ldc)
{
    // First version: no transpose, no alpha beta, no leading dimension
    // Column major
    for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < m; ++i) {
                C[j * m + i] += A[l * m + i] * B[j * k + l];
            }
        }
    }
}
