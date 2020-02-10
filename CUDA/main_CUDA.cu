#include <cuda_runtime.h>
#include <sys/time.h>

#include <iostream>

#include <gemm.hpp>
#include <gemm_CUDA.hpp>

void CUDA_timing(int N, double *A, double *B, double *C);
void CUDA_timing2(int N, double *A, double *B, double *C);
void CUDA_timing3(int N, double *A, double *B, double *C);

int main(int argc, char const *argv[])
{
    double *A, *B, *C;
    A = new double[matrix_size * matrix_size]();
    B = new double[matrix_size * matrix_size]();
    C = new double[matrix_size * matrix_size]();

    CUDA_timing(matrix_size, A, B, C);
    CUDA_timing2(matrix_size, A, B, C);
    CUDA_timing3(matrix_size, A, B, C);

    return 0;
}

void CUDA_timing(int N, double *A, double *B, double *C)
{
    double *dA, *dB, *dC;
    struct timeval start_time, end_time;

    cudaMalloc((void **)&dA, N * N * sizeof(double));
    cudaMalloc((void **)&dB, N * N * sizeof(double));
    cudaMalloc((void **)&dC, N * N * sizeof(double));

    cudaDeviceSynchronize();
    gettimeofday(&start_time, NULL);
    cudaMemcpy(dA, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    gemm_CUDA<<<256, N / 256>>>(N, N, N, 1.0, dA, N, dB, N, 0.0, dC, N);
    cudaMemcpy(C, dC, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (CUDA1) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}

void CUDA_timing2(int N, double *A, double *B, double *C)
{
    double *dA, *dB, *dC;
    struct timeval start_time, end_time;

    cudaMalloc((void **)&dA, N * N * sizeof(double));
    cudaMalloc((void **)&dB, N * N * sizeof(double));
    cudaMalloc((void **)&dC, N * N * sizeof(double));

    cudaDeviceSynchronize();
    gettimeofday(&start_time, NULL);
    cudaMemcpy(dA, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    gemm_CUDA2<<<dim3(16, 16), dim3(N / 16, N / 16)>>>(N, N, N, 1.0, dA, N, dB,
                                                       N, 0.0, dC, N);
    cudaMemcpy(C, dC, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (CUDA2) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}

void CUDA_timing3(int N, double *A, double *B, double *C)
{
    double *dA, *dB, *dC;
    struct timeval start_time, end_time;

    cudaMalloc((void **)&dA, N * N * sizeof(double));
    cudaMalloc((void **)&dB, N * N * sizeof(double));
    cudaMalloc((void **)&dC, N * N * sizeof(double));

    cudaDeviceSynchronize();
    gettimeofday(&start_time, NULL);
    cudaMemcpy(dA, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    gemm_CUDA3<<<dim3(16, 16), dim3(N / 16, N / 16)>>>(N, N, N, 1.0, dA, N, dB,
                                                       N, 0.0, dC, N);
    cudaMemcpy(C, dC, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&end_time, NULL);

    std::cout << "Elapse time for Matrix-Matrix multiplication (CUDA3) is "
              << ((end_time.tv_sec - start_time.tv_sec) * 1000000u +
                  end_time.tv_usec - start_time.tv_usec) /
                     1.e6
              << std::endl;
}
