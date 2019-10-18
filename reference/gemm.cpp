#include <gemm.hpp>

void gemm_reference(int m, int n, int k, double alpha, double *A, int lda,
                    double *B, int ldb, double beta, double *C, int ldc)
{
    // Current version: no leading dimension
    // Row major
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
            }
        }
    }
}
