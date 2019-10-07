#include <iostream>

#include <reference/gemm.hpp>

int main(int argc, char const *argv[]) {
  double A[3], B[3], C[3];

  gemm_reference(A, B, C);

  return 0;
}