#include <sys/time.h>

#include <iostream>

#include <gemm.hpp>

void OpenACC_timing(int N, double *A, double *B, double *C);

int main(int argc, char const *argv[])
{
    double *A, *B, *C;
    A = new double[matrix_size * matrix_size]();
    B = new double[matrix_size * matrix_size]();
    C = new double[matrix_size * matrix_size]();

    OpenACC_timing(matrix_size, A, B, C);

    return 0;
}

void OpenACC_timing(int N, double *A, double *B, double *C)
{
    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);
    gemm_OpenACC(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (OpenACC) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}
