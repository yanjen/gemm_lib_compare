#include <mkl.h>
#include <sys/time.h>

#include <iostream>

#include <gemm.hpp>

void sequential_timing(int N, double *A, double *B, double *C);
void BLAS_timing(int N, double *A, double *B, double *C);

int main(int argc, char const *argv[])
{
    double *A, *B, *C;
    A = new double[matrix_size * matrix_size]();
    B = new double[matrix_size * matrix_size]();
    C = new double[matrix_size * matrix_size]();

    sequential_timing(matrix_size, A, B, C);

    BLAS_timing(matrix_size, A, B, C);

    return 0;
}

void sequential_timing(int N, double *A, double *B, double *C)
{
    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);
    gemm_reference(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}

void BLAS_timing(int N, double *A, double *B, double *C)
{
    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N,
                B, N, 0.0, C, N);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (BLAS) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}
