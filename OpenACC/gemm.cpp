#include <gemm.hpp>

void gemm_OpenACC(int m, int n, int k, double alpha, double *A, int lda,
                  double *B, int ldb, double beta, double *C, int ldc)
{
    // Current version: no leading dimension
    // Row major
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
        }
    }

#pragma acc kernels copyin(A [0:m * k], B [0:k * n]) copy(C [0:m * n])
#pragma acc loop independent gang
    for (int i = 0; i < m; ++i) {
#pragma acc loop independent vector
        for (int j = 0; j < n; ++j) {
#pragma acc loop seq
            for (int l = 0; l < k; ++l) {
                C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
            }
        }
    }
}
