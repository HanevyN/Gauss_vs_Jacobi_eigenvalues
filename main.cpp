#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include "test_matrices.h"

using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

constexpr double EPSILON = 1e-10;

void print_vector(const Vector3& v) {
    for (double val : v)
        std::cout << val << " ";
    std::cout << "\n";
}

// Jacobi rotation on indices (p,q)
void jacobi_rotate(Matrix3x3& A, Matrix3x3& V, int p, int q) {
    if (A[p][q] == 0.0) return;

    double theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
    double t = (theta >= 0.0) ? 1.0 / (theta + sqrt(1.0 + theta * theta))
                              : -1.0 / (-theta + sqrt(1.0 + theta * theta));
    double c = 1.0 / sqrt(1 + t * t);
    double s = t * c;

    double app = A[p][p];
    double aqq = A[q][q];

    A[p][p] = c * c * app - 2.0 * s * c * A[p][q] + s * s * aqq;
    A[q][q] = s * s * app + 2.0 * s * c * A[p][q] + c * c * aqq;
    A[p][q] = A[q][p] = 0.0;

    for (int k = 0; k < 3; ++k) {
        if (k != p && k != q) {
            double akp = A[k][p];
            double akq = A[k][q];
            A[k][p] = A[p][k] = c * akp - s * akq;
            A[k][q] = A[q][k] = c * akq + s * akp;
        }
    }

    for (int k = 0; k < 3; ++k) {
        double vkp = V[k][p];
        double vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = c * vkq + s * vkp;
    }
}

void jacobi_eigenvalue(Matrix3x3 A, Vector3& eigenvalues, Matrix3x3& eigenvectors) {
    eigenvectors = {{{1,0,0},{0,1,0},{0,0,1}}};
    for (int iter = 0; iter < 50; ++iter) {
        int p = 0, q = 1;
        double max_offdiag = fabs(A[0][1]);

        if (fabs(A[0][2]) > max_offdiag) { max_offdiag = fabs(A[0][2]); p = 0; q = 2; }
        if (fabs(A[1][2]) > max_offdiag) { max_offdiag = fabs(A[1][2]); p = 1; q = 2; }

        if (max_offdiag < EPSILON) break;

        jacobi_rotate(A, eigenvectors, p, q);
    }

    eigenvalues[0] = A[0][0];
    eigenvalues[1] = A[1][1];
    eigenvalues[2] = A[2][2];
}

void gaussian_elimination_eigenvector(const Matrix3x3& A, double lambda, Vector3& eigenvector) {
    Matrix3x3 M = A;
    for (int i = 0; i < 3; ++i) M[i][i] -= lambda;

    if (fabs(M[0][0]) < EPSILON) {
        if (fabs(M[1][0]) > fabs(M[2][0])) std::swap(M[0], M[1]);
        else std::swap(M[0], M[2]);
    }

    if (fabs(M[0][0]) > EPSILON) {
        double f = M[1][0] / M[0][0];
        for (int j = 0; j < 3; ++j) M[1][j] -= f * M[0][j];

        f = M[2][0] / M[0][0];
        for (int j = 0; j < 3; ++j) M[2][j] -= f * M[0][j];
    }

    if (fabs(M[1][1]) > EPSILON) {
        double f = M[2][1] / M[1][1];
        for (int j = 0; j < 3; ++j) M[2][j] -= f * M[1][j];
    }

    eigenvector = {0, 0, 1};
    if (fabs(M[2][2]) > EPSILON) eigenvector[2] = 0;

    if (fabs(M[1][1]) > EPSILON)
        eigenvector[1] = -M[1][2] * eigenvector[2] / M[1][1];
    else
        eigenvector[1] = 1;

    if (fabs(M[0][0]) > EPSILON)
        eigenvector[0] = -(M[0][1] * eigenvector[1] + M[0][2] * eigenvector[2]) / M[0][0];
    else
        eigenvector[0] = 1;
}

double compute_residual(const Matrix3x3& A, double lambda, const Vector3& v) {
    Vector3 Av = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Av[i] += A[i][j] * v[j];

    double max_residual = 0.0;
    for (int i = 0; i < 3; ++i)
        max_residual = std::max(max_residual, fabs(Av[i] - lambda * v[i]));

    return max_residual;
}

int main() {
    for (int t = 0; t < NUM_TEST_CASES; ++t) {
        Matrix3x3 A;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A[i][j] = TEST_MATRICES[t][i][j];

        Vector3 eigenvalues;
        Matrix3x3 eigenvectors;

        // Time full Jacobi eigen-decomposition (eigenvalues + eigenvectors)
        auto start_jacobi = std::chrono::high_resolution_clock::now();
        jacobi_eigenvalue(A, eigenvalues, eigenvectors);
        auto end_jacobi = std::chrono::high_resolution_clock::now();
        auto duration_jacobi = std::chrono::duration_cast<std::chrono::nanoseconds>(end_jacobi - start_jacobi).count();

        std::cout << "Test Case " << t+1 << ":\n";
        std::cout << "Eigenvalues (Jacobi): ";
        print_vector(eigenvalues);

        std::cout << "Jacobi total Duration: " << duration_jacobi << " ns\n";

        // Time full Gaussian elimination for all eigenvectors
        auto start_gauss = std::chrono::high_resolution_clock::now();
        std::array<Vector3,3> gauss_vectors;
        for (int i = 0; i < 3; ++i) {
            gaussian_elimination_eigenvector(A, eigenvalues[i], gauss_vectors[i]);
        }
        auto end_gauss = std::chrono::high_resolution_clock::now();
        auto duration_gauss = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gauss - start_gauss).count();

        std::cout << "Gaussian elimination total Duration (all 3 eigenvectors): " << duration_gauss << " ns\n";

        // Print residuals for Gaussian elimination eigenvectors
        for (int i = 0; i < 3; ++i) {
            double residual = compute_residual(A, eigenvalues[i], gauss_vectors[i]);
            std::cout << "Residual (Gaussian, lambda=" << eigenvalues[i] << "): " << residual << "\n";
        }

        // Print residuals for Jacobi eigenvectors
        for (int i = 0; i < 3; ++i) {
            Vector3 v = {eigenvectors[0][i], eigenvectors[1][i], eigenvectors[2][i]};
            double residual = compute_residual(A, eigenvalues[i], v);
            std::cout << "Residual (Jacobi, lambda=" << eigenvalues[i] << "): " << residual << "\n";
        }
        std::cout << "\n";

        if (t+1 == 8) {
            std::cout << "Jacobi Vectors \n";
            print_vector(eigenvectors[0]);
            print_vector(eigenvectors[1]);
            print_vector(eigenvectors[2]);

            std::cout << "Gauss Vectors \n";

            print_vector(gauss_vectors[0]);
            print_vector(gauss_vectors[1]);
            print_vector(gauss_vectors[2]);
            std::cout << "\n";
        }

    }
    return 0;
}


