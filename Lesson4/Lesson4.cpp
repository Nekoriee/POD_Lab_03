#include <iostream>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <chrono>
#include <algorithm>
#include <assert.h>
#include <vector>
#include <memory>
// cols = rows = 0 (mod 8)

void mul_matrix(double* A, std::size_t cA, std::size_t rA, const double* B, std::size_t cB, std::size_t rB, const double* C, std::size_t cC, std::size_t rC)
{
    assert(cB == rC);
    assert(cA == cC);
    assert(rA == rB);
    //assert((cA & 0x3F) == 0);

    for (size_t i = 0; i < cA; ++i)
    {
        for (size_t j = 0; j < rA; ++j)
        {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; ++k)
            {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}
// B =
// Columns: [v0, v1, ... v(cols-1)]
// 
// C =
// [a0, b0]
// [a1, b1]
// [...]
// [a(cols-1), b(cols-1)]
// 
// A =
// [0][0] = v0 * a0 + v0 * a1 ... + v0 * a(cols - 1)
// [1][0] = v1 * a0 + v1 * a1 ... + v1 * a(cols - 1)
// ...
// [0][1] = v0 * b0 + v0 * b1 ... + v0 * b(cols - 1)
void mul_matrix_256(double* A, std::size_t cA, std::size_t rA, const double* B, std::size_t cB, std::size_t rB, const double* C, std::size_t cC, std::size_t rC)
{
    assert(cB == rC);
    assert(cA == cC);
    assert(rA == rB);
    //assert((cA & 0x3F) == 0);
    for (size_t i = 0; i < rB / 4; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                // B[k][4i] (if i<rA/4, i++)
                // B[k][i] (if i<rA, i += 4)
                __m256d bCol = _mm256_loadu_pd(B + rB * k + 4 * i);
                // c[k][j]
                // a[i][j] = sum(k=0, rC-1)(b[4i][j] * c[k][j])
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                sum = _mm256_fmadd_pd(bCol, broadcasted, sum);
            }
            _mm256_storeu_pd(A + j * rA + 4 * i, sum);
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> get_matrix_pair(size_t n)
{
    std::vector<double> permut(n*n);
    for (size_t c = 0; c < n; c++)
        for (size_t r = 0; r < n; r++)
            if (c == r) permut[c * n + r] = 1; else permut[c * n + r] = 0;
    for (size_t i = 0; i < n; i++) {
        permut[i * n + n - 1] = 0;
        permut[i * n] = 0;
    }
    permut[n * n - n] = 1;
    permut[n - 1] = 1;
    std::vector<double> vf(n * n);
    double num = 1;
    for (size_t c = 0; c < n; c++)
        for (size_t r = 0; r < n; r++)
            vf[c * n + r] = num++;
    return std::pair<std::vector<double>, std::vector<double>>{ vf, permut };
}

std::pair<std::vector<double>, std::vector<double>> get_permutation_matrix(size_t n)
{
    std::vector<size_t> permut(n);
    for (size_t i = 0; i < n; i++)
        permut[i] = (n - (1 + 10)) % n;
    std::vector<double> vf(n * n), vi(n * n);
    for (size_t c = 0; c < n; c++)
        for (size_t r = 0; r < n; r++) {
            vf[c * n + r] = 1;
            vi[r * n + c] = 1;
        }
    return std::pair<std::vector<double>, std::vector<double>>{ vf, vi };
}

using namespace std::chrono;
int main(int argc, char** argv)
{
    auto show_matrix = [](const double* M, std::size_t nC, std::size_t nR) {
        for (std::size_t r = 0; r < nR; ++r) {
            std::cout << "[" << M[r];
            for (std::size_t c = 1; c < nC; ++c) {
                std::cout << " " << M[r + nR * c];
            }
            std::cout << "]\n";
        }
    };

    int n = 256;
    std::vector<double> A(n * n);

    auto [B, C] = get_permutation_matrix(n);

    //show_matrix(B.data(), n, n);
    //std::cout << "*\n";
    //show_matrix(C.data(), n, n);
    //std::cout << "=\n";

    for (size_t i = 0; i < 1 + 1; i++) {
        std::cout << "Thread num: " << i << "\n";
        for (size_t j = 0; j < 20; j++) {
            if (i > 0) {
                auto t1 = std::chrono::steady_clock::now();
                mul_matrix_256(A.data(), n, n, B.data(), n, n, C.data(), n, n);
                auto t2 = std::chrono::steady_clock::now();
                std::cout << duration_cast<nanoseconds>(t2 - t1).count() << "\n";
            }
            else {
                auto t1 = std::chrono::steady_clock::now();
                mul_matrix(A.data(), n, n, B.data(), n, n, C.data(), n, n);
                auto t2 = std::chrono::steady_clock::now();
                std::cout << duration_cast<nanoseconds>(t2 - t1).count() << "\n";
            }
        }
    }

    //show_matrix(A.data(), n, n);

    return 0;
}
