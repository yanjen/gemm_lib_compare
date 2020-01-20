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
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}
