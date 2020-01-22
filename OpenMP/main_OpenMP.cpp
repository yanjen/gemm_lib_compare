#include <sys/time.h>

#include <iostream>

#include <gemm.hpp>

void OpenMP_timing(int N, double *A, double *B, double *C);

int main(int argc, char const *argv[])
{
    double *A, *B, *C;
    A = new double[matrix_size * matrix_size]();
    B = new double[matrix_size * matrix_size]();
    C = new double[matrix_size * matrix_size]();

    OpenMP_timing(matrix_size, A, B, C);

    return 0;
}

void OpenMP_timing(int N, double *A, double *B, double *C)
{
    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);
    // Regular OpenMP
    gemm_OpenMP(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (OpenMP) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;

    gettimeofday(&start_time, NULL);
    // OpenMP with array reduction
    gemm_OpenMP2(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (OpenMP2) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}
