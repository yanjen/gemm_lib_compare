#include <iostream>

#include <reference/gemm.hpp>

int main(int argc, char const *argv[]) {
  double *A, *B, *C;

  gemm_reference(A, B, C);

  return 0;
}