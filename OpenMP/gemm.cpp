#include <vector>

void gemm_OpenMP(int m, int n, int k, double alpha, double *A, int lda,
                 double *B, int ldb, double beta, double *C, int ldc)
{
// Current version: no leading dimension
// Row major
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void gemm_OpenMP2(int m, int n, int k, double alpha, double *A, int lda,
                  double *B, int ldb, double beta, double *C, int ldc)
{
// Current version: no leading dimension
// Row major
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
        }
    }

    int reduction_N = 4;
    std::vector<double> temp(reduction_N);
#pragma omp parallel for collapse(3) reduction(+ : C[:m * n])
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < reduction_N; ++l) {
                int start = l * k / reduction_N;
                int end = (l + 1) * k / reduction_N;
                for (int ll = start; ll < end; ++ll) {
                    temp[l] = alpha * A[i * k + ll] * B[ll * n + j];
                }
                C[i * n + j] += temp[l];
            }
        }
    }
}

void gemm_OpenMP3(int m, int n, int k, double alpha, double *A, int lda,
                  double *B, int ldb, double beta, double *C, int ldc)
{
// Current version: no leading dimension
// Row major
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[j * m + i] *= beta;
        }
    }

#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                C[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
            }
        }
    }
}
