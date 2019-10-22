#include "gtest/gtest.h"
#include <cstdint>
#include <random>

#include <gemm.hpp>

TEST(GEMMOpenMP, CanMultiply)
{
    double A[9], B[9], C[9];
    int N = 3;

    gemm_OpenMP(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}

TEST(GEMMOpenMP, CanMultiplyWithValues)
{
    double A[9], B[9], C[9]{0};
    int N = 3;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < 9; ++i) {
        A[i] = static_cast<double>(engine()) / UINT32_MAX;
        B[i] = static_cast<double>(engine()) / UINT32_MAX;
    }

    gemm_OpenMP(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}

TEST(GEMMOpenMP, MultiplyIdentityCorrectly)
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

    gemm_OpenMP(N, N, N, 1.0, A, N, B, N, 0.0, C, N);

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

TEST(GEMMOpenMP, MultiplySameAsReference)
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
    double D[9]{0};
    int N = 3;

    gemm_reference(N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    gemm_OpenMP(N, N, N, 1.0, A, N, B, N, 0.0, D, N);

    ASSERT_EQ(D[0], C[0]);
    ASSERT_EQ(D[1], C[1]);
    ASSERT_EQ(D[2], C[2]);
    ASSERT_EQ(D[3], C[3]);
    ASSERT_EQ(D[4], C[4]);
    ASSERT_EQ(D[5], C[5]);
    ASSERT_EQ(D[6], C[6]);
    ASSERT_EQ(D[7], C[7]);
    ASSERT_EQ(D[8], C[8]);
}

TEST(GEMMOpenMP, MultiplySameAsReference2)
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
    double D[9] = { 1, 1, 1,
                    1, 1, 1,
                    1, 1, 1};
    // clang-format on
    int N = 3;
    double alpha = 2.0;
    double beta = -1.0;

    gemm_reference(N, N, N, alpha, A, N, B, N, beta, C, N);
    gemm_OpenMP(N, N, N, alpha, A, N, B, N, beta, D, N);

    ASSERT_EQ(D[0], C[0]);
    ASSERT_EQ(D[1], C[1]);
    ASSERT_EQ(D[2], C[2]);
    ASSERT_EQ(D[3], C[3]);
    ASSERT_EQ(D[4], C[4]);
    ASSERT_EQ(D[5], C[5]);
    ASSERT_EQ(D[6], C[6]);
    ASSERT_EQ(D[7], C[7]);
    ASSERT_EQ(D[8], C[8]);
}

TEST(GEMMOpenMP, MultiplyNonSquareSameAsReference)
{
    // clang-format off
	double A[6] = {1,4,       // A = |1 4|
				   2,5,       //     |2 5|
				   3,6};      //     |3 6|
	double B[8] = {1,3,5,7,   // B = |1 3 5 7|
				   2,4,6,8};  //     |2 4 6 8|
    // clang-format on
    double C[12]{0};
    double D[12]{0};
    int M = 3;
    int N = 4;
    int K = 2;

    gemm_reference(M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    gemm_OpenMP(M, N, K, 1.0, A, K, B, N, 0.0, D, N);

    ASSERT_EQ(D[0], C[0]);
    ASSERT_EQ(D[1], C[1]);
    ASSERT_EQ(D[2], C[2]);
    ASSERT_EQ(D[3], C[3]);
    ASSERT_EQ(D[4], C[4]);
    ASSERT_EQ(D[5], C[5]);
    ASSERT_EQ(D[6], C[6]);
    ASSERT_EQ(D[7], C[7]);
    ASSERT_EQ(D[8], C[8]);
    ASSERT_EQ(D[9], C[9]);
    ASSERT_EQ(D[10], C[10]);
    ASSERT_EQ(D[11], C[11]);
}

TEST(GEMMOpenMP, MultiplyLargeSameAsReference)
{
    int M = 300, N = 200, K = 100;
    double A[M * K], B[K * N], C[M * N], D[M * N];

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < M * N; ++i) {
        if (i < M * K) {
            A[i] = static_cast<double>(engine()) / UINT32_MAX;
        }
        if (i < K * N) {
            B[i] = static_cast<double>(engine()) / UINT32_MAX;
        }
        C[i] = static_cast<double>(engine()) / UINT32_MAX;
        D[i] = C[i];
    }

    gemm_reference(M, N, K, 1.0, A, K, B, N, 1.0, C, N);
    gemm_OpenMP(M, N, K, 1.0, A, K, B, N, 1.0, D, N);

    ASSERT_EQ(D[0], C[0]);
    ASSERT_EQ(D[5852], C[5852]);
    ASSERT_EQ(D[7619], C[7619]);
    ASSERT_EQ(D[16709], C[16709]);
    ASSERT_EQ(D[32812], C[32812]);
    ASSERT_EQ(D[37941], C[37941]);
    ASSERT_EQ(D[54347], C[54347]);
    ASSERT_EQ(D[54802], C[54802]);
    ASSERT_EQ(D[57450], C[57450]);
    ASSERT_EQ(D[59999], C[59999]);
}
