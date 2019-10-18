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

TEST(GEMM, CanMultiplyWithValues)
{
    double A[9], B[9], C[9]{0};
    int N = 3;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < 9; ++i) {
        A[i] = static_cast<double>(engine()) / UINT32_MAX;
        B[i] = static_cast<double>(engine()) / UINT32_MAX;
    }

    gemm_reference('N', 'N', N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}

TEST(GEMM, MultiplyIdentityCorrectly)
{
    double A[9], B[9]{0}, C[9]{0};
    int N = 3;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < 9; ++i) {
        A[i] = static_cast<double>(engine()) / UINT32_MAX;
    }
    for (int i = 0; i < 3; ++i) {
        B[i * N + i] = 1.0;
    }

    gemm_reference('N', 'N', N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    ASSERT_EQ(C[0], A[0]);
    ASSERT_EQ(C[1], A[1]);
    ASSERT_EQ(C[2], A[2]);
    ASSERT_EQ(C[3], A[3]);
    ASSERT_EQ(C[4], A[4]);
    ASSERT_EQ(C[5], A[5]);
    ASSERT_EQ(C[6], A[6]);
    ASSERT_EQ(C[7], A[7]);
    ASSERT_EQ(C[8], A[8]);
}

TEST(GEMM, MultiplyCorrectly)
{
    // clang-format off
	double A[9] = { 1, 4, 7,
				    2, 5, 8,
				    3, 6, 9};
    double B[9] = {-1,-3, 3,
                    0, 2, 4,
                    1,-1, 5};
    // clang-format on
    double C[9]{0};
    int N = 3;

    gemm_reference('N', 'N', N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    ASSERT_EQ(C[0], 6);
    ASSERT_EQ(C[1], -2);
    ASSERT_EQ(C[2], 54);
    ASSERT_EQ(C[3], 6);
    ASSERT_EQ(C[4], -4);
    ASSERT_EQ(C[5], 66);
    ASSERT_EQ(C[6], 6);
    ASSERT_EQ(C[7], -6);
    ASSERT_EQ(C[8], 78);
}

TEST(GEMM, MultiplyCorrectly2)
{
    // clang-format off
	double A[9] = { 1, 4, 7,
				    2, 5, 8,
				    3, 6, 9};
    double B[9] = {-1,-3, 3,
                    0, 2, 4,
                    1,-1, 5};
	double C[9] = { 1, 1, 1,
				    1, 1, 1,
				    1, 1, 1};
    // clang-format on
    int N = 3;
    double alpha = 2.0;
    double beta = -1.0;

    gemm_reference('N', 'N', N, N, N, alpha, A, N, B, N, beta, C, N);

    ASSERT_EQ(C[0], 11);
    ASSERT_EQ(C[1], -5);
    ASSERT_EQ(C[2], 107);
    ASSERT_EQ(C[3], 11);
    ASSERT_EQ(C[4], -9);
    ASSERT_EQ(C[5], 131);
    ASSERT_EQ(C[6], 11);
    ASSERT_EQ(C[7], -13);
    ASSERT_EQ(C[8], 155);
}

TEST(GEMM, MultiplyWithTransposeA)
{
    // clang-format off
	double A[6] = {1,4,   // trans(A) = |1 4|
				   2,5,   //            |2 5|
				   3,6};  //            |3 6|
	double B[8] = {1,2,   // B = |1 3 5 7|
				   3,4,   //     |2 4 6 8|
				   5,6,
				   7,8};
    // clang-format on
    double C[12]{0};
    int M = 3;
    int N = 4;
    int K = 2;

    gemm_reference('T', 'N', M, N, K, 1.0, A, K, B, K, 0.0, C, M);

    ASSERT_EQ(C[0], 9);
    ASSERT_EQ(C[1], 12);
    ASSERT_EQ(C[2], 15);
    ASSERT_EQ(C[3], 19);
    ASSERT_EQ(C[4], 26);
    ASSERT_EQ(C[5], 33);
    ASSERT_EQ(C[6], 29);
    ASSERT_EQ(C[7], 40);
    ASSERT_EQ(C[8], 51);
    ASSERT_EQ(C[9], 39);
    ASSERT_EQ(C[10], 54);
    ASSERT_EQ(C[11], 69);
}

TEST(GEMM, MultiplyWithTransposeB)
{
    // clang-format off
	double A[6] = {1,2,3,     // A = |1 4|
				   4,5,6};    //     |2 5|
							  //     |3 6|
	double B[8] = {1,3,5,7,   // trans(B) = |1 3 5 7|
				   2,4,6,8};  //            |2 4 6 8|
    // clang-format on
    double C[12]{0};
    int M = 3;
    int N = 4;
    int K = 2;

    gemm_reference('N', 'T', M, N, K, 1.0, A, M, B, N, 0.0, C, M);

    ASSERT_EQ(C[0], 9);
    ASSERT_EQ(C[1], 12);
    ASSERT_EQ(C[2], 15);
    ASSERT_EQ(C[3], 19);
    ASSERT_EQ(C[4], 26);
    ASSERT_EQ(C[5], 33);
    ASSERT_EQ(C[6], 29);
    ASSERT_EQ(C[7], 40);
    ASSERT_EQ(C[8], 51);
    ASSERT_EQ(C[9], 39);
    ASSERT_EQ(C[10], 54);
    ASSERT_EQ(C[11], 69);
}

TEST(GEMM, MultiplyWithTransposeAB)
{
    // clang-format off
	double A[6] = {1,4,       // trans(A) = |1 4|
				   2,5,       //            |2 5|
				   3,6};      //            |3 6|
	double B[8] = {1,3,5,7,   // trans(B) = |1 3 5 7|
				   2,4,6,8};  //            |2 4 6 8|
    // clang-format on
    double C[12]{0};
    int M = 3;
    int N = 4;
    int K = 2;

    gemm_reference('T', 'T', M, N, K, 1.0, A, M, B, N, 0.0, C, M);

    ASSERT_EQ(C[0], 9);
    ASSERT_EQ(C[1], 12);
    ASSERT_EQ(C[2], 15);
    ASSERT_EQ(C[3], 19);
    ASSERT_EQ(C[4], 26);
    ASSERT_EQ(C[5], 33);
    ASSERT_EQ(C[6], 29);
    ASSERT_EQ(C[7], 40);
    ASSERT_EQ(C[8], 51);
    ASSERT_EQ(C[9], 39);
    ASSERT_EQ(C[10], 54);
    ASSERT_EQ(C[11], 69);
}
