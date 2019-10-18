#include <cblas.h>
#include <sys/time.h>

#include <iostream>

#include <gemm.hpp>

int main(int argc, char const *argv[])
{
    int matrix_size = 2048;
    struct timeval start_time, end_time;

    double *A, *B, *C;
    A = new double[matrix_size * matrix_size]();
    B = new double[matrix_size * matrix_size]();
    C = new double[matrix_size * matrix_size]();

    gettimeofday(&start_time, NULL);
    gemm_reference(matrix_size, matrix_size, matrix_size, 1.0, A, matrix_size,
                   B, matrix_size, 0.0, C, matrix_size);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;

    gettimeofday(&start_time, NULL);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, matrix_size,
                matrix_size, matrix_size, 1.0, A, matrix_size, B, matrix_size,
                0.0, C, matrix_size);
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (BLAS) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;

    return 0;
}
