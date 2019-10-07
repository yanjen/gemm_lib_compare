#include "gtest/gtest.h"
#include <cstdint>
#include <random>

#include <reference/gemm.hpp>

TEST(GEMM, CanMultiply)
{
    double A[9], B[9], C[9];
    int N = 3;

    gemm_reference('N', 'N', N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}

TEST(GEMM, MultiplyCorrectly)
{
    double A[9], B[9], C[9]{0};
    int N = 3;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < 9; ++i) {
        A[i] = static_cast<double>(engine()) / UINT32_MAX;
        B[i] = static_cast<double>(engine()) / UINT32_MAX;
    }
}
