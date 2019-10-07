#include <reference/gemm.hpp>

void gemm_reference(double *A, double *B, double *C) {
  C[0] = A[0] * B[0];
  C[1] = A[1] * B[1];
  C[2] = A[2] * B[2];
}
