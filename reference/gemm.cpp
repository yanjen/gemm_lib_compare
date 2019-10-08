#include <reference/gemm.hpp>

void gemm_reference(char transa, char transb, int m, int n, int k, double alpha,
                    double *A, int lda, double *B, int ldb, double beta,
                    double *C, int ldc)
{
    // Current version: no leading dimension
    // Column major
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
        }
    }

    if (transa == 'N' && transb == 'N') {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                for (int i = 0; i < m; ++i) {
                    C[j * m + i] += alpha * A[l * m + i] * B[j * k + l];
                }
            }
        }
    }

    if (transa == 'T' && transb == 'N') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                for (int l = 0; l < k; ++l) {
                    C[j * m + i] += alpha * A[i * k + l] * B[j * k + l];
                }
            }
        }
    }

    if (transa == 'N' && transb == 'T') {
        for (int l = 0; l < k; ++l) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    C[j * m + i] += alpha * A[l * m + i] * B[l * n + j];
                }
            }
        }
    }

    if (transa == 'T' && transb == 'T') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                for (int l = 0; l < k; ++l) {
                    C[j * m + i] += alpha * A[i * k + l] * B[l * n + j];
                }
            }
        }
    }
}
