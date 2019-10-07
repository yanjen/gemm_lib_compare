#include <iostream>

// #include "reference/gemm.hpp"

void gemm_reference(double *A, double *B, double *C);

int main(int argc, char const *argv[]) {
  double A[3], B[3], C[3];

  gemm_reference(A, B, C);

  return 0;
}