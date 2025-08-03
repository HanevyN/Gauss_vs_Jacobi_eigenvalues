#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include "test_matrices.h"

using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

constexpr double EPSILON = 1e-10;

// Helper function to normalize a vector
void normalize(Vector3 &v) {
    double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (norm > EPSILON) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

// Helper function for the Gram-Schmidt process
void gram_schmidt(std::vector<Vector3> &vectors) {
    if (vectors.empty())
        return;

    // Normalize the first vector
    normalize(vectors[0]);

    for (size_t i = 1; i < vectors.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            double dot_product =
                    vectors[i][0] * vectors[j][0] + vectors[i][1] * vectors[j][1] + vectors[i][2] * vectors[j][2];
            vectors[i][0] -= dot_product * vectors[j][0];
            vectors[i][1] -= dot_product * vectors[j][1];
            vectors[i][2] -= dot_product * vectors[j][2];
        }
        normalize(vectors[i]);
    }
}

// Refactored function: now returns a vector of pairs for clarity
std::vector<std::pair<double, Vector3>> gaussian_elimination_eigenvectors(const Matrix3x3 &A, double lambda) {
    Matrix3x3 M = A;
    for (int i = 0; i < 3; ++i) {
        M[i][i] -= lambda;
    }

    // Gaussian Elimination with Partial Pivoting
    for (int i = 0; i < 3; ++i) {
        int pivot_row = i;
        double max_val = fabs(M[i][i]);
        for (int k = i + 1; k < 3; ++k) {
            if (fabs(M[k][i]) > max_val) {
                max_val = fabs(M[k][i]);
                pivot_row = k;
            }
        }

        if (pivot_row != i) {
            std::swap(M[i], M[pivot_row]);
        }

        if (fabs(M[i][i]) < EPSILON) {
            continue;
        }

        for (int k = i + 1; k < 3; ++k) {
            double factor = M[k][i] / M[i][i];
            for (int j = i; j < 3; ++j) {
                M[k][j] -= factor * M[i][j];
            }
        }
    }

    // Determine the rank to find the number of free variables
    int rank = 0;
    for (int i = 0; i < 3; ++i) {
        bool is_zero_row = true;
        for (int j = 0; j < 3; ++j) {
            if (fabs(M[i][j]) > EPSILON) {
                is_zero_row = false;
                break;
            }
        }
        if (!is_zero_row) {
            rank++;
        }
    }

    std::vector<Vector3> basis;
    int num_free_vars = 3 - rank;

    if (num_free_vars == 1) {
        Vector3 v = {0, 0, 0};
        if (rank == 2) {
            v[2] = 1.0;
            v[1] = (fabs(M[1][1]) > EPSILON) ? -M[1][2] * v[2] / M[1][1] : 1.0;
            v[0] = (fabs(M[0][0]) > EPSILON) ? -(M[0][1] * v[1] + M[0][2] * v[2]) / M[0][0] : 1.0;
        } else if (rank == 1) {
            v[1] = 1.0;
            v[2] = 0.0;
            v[0] = (fabs(M[0][0]) > EPSILON) ? -M[0][1] * v[1] / M[0][0] : 1.0;
        } else {
            v[0] = 1.0;
            v[1] = 0.0;
            v[2] = 0.0;
        }
        normalize(v);
        basis.push_back(v);
    } else if (num_free_vars == 2) {
        Vector3 v1 = {0, 0, 0};
        Vector3 v2 = {0, 0, 0};

        // Find first basis vector (e.g., set two free variables to 1, 0)
        v1[2] = 1.0;
        v1[1] = 0.0;
        v1[0] = (fabs(M[0][0]) > EPSILON) ? -(M[0][1] * v1[1] + M[0][2] * v1[2]) / M[0][0] : 1.0;
        basis.push_back(v1);

        // Find second basis vector (e.g., set two free variables to 0, 1)
        v2[2] = 0.0;
        v2[1] = 1.0;
        v2[0] = (fabs(M[0][0]) > EPSILON) ? -(M[0][1] * v2[1] + M[0][2] * v2[2]) / M[0][0] : 1.0;
        basis.push_back(v2);

        gram_schmidt(basis);
    } else if (num_free_vars == 3) {
        // Trivial case for a 3-D eigenspace
        basis.push_back({1, 0, 0});
        basis.push_back({0, 1, 0});
        basis.push_back({0, 0, 1});
    }

    std::vector<std::pair<double, Vector3>> eigenpairs;
    for (const auto &v: basis) {
        eigenpairs.push_back({lambda, v});
    }
    return eigenpairs;
}


void print_vector(const Vector3 &v) {
    for (double val: v)
        std::cout << val << " ";
    std::cout << "\n";
}

// Jacobi rotation on indices (p,q)
void jacobi_rotate(Matrix3x3 &A, Matrix3x3 &V, int p, int q) {
    if (A[p][q] == 0.0)
        return;

    double theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
    double t = (theta >= 0.0) ? 1.0 / (theta + sqrt(1.0 + theta * theta)) : -1.0 / (-theta + sqrt(1.0 + theta * theta));
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

void jacobi_eigenvalue(Matrix3x3 A, Vector3 &eigenvalues, Matrix3x3 &eigenvectors) {
    eigenvectors = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    for (int iter = 0; iter < 50; ++iter) {
        int p = 0, q = 1;
        double max_offdiag = fabs(A[0][1]);

        if (fabs(A[0][2]) > max_offdiag) {
            max_offdiag = fabs(A[0][2]);
            p = 0;
            q = 2;
        }
        if (fabs(A[1][2]) > max_offdiag) {
            max_offdiag = fabs(A[1][2]);
            p = 1;
            q = 2;
        }

        if (max_offdiag < EPSILON)
            break;

        jacobi_rotate(A, eigenvectors, p, q);
    }

    eigenvalues[0] = A[0][0];
    eigenvalues[1] = A[1][1];
    eigenvalues[2] = A[2][2];
}


double compute_residual(const Matrix3x3 &A, double lambda, const Vector3 &v) {
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

        std::cout << "Test Case " << t + 1 << ":\n";
        std::cout << "Eigenvalues (Jacobi): ";
        print_vector(eigenvalues);

        std::cout << "Jacobi total Duration: " << duration_jacobi
                  << "\n"; // Time full Gaussian elimination for all eigenvectors

        // Refactored Gaussian elimination part
        std::vector<std::pair<double, Vector3>> gauss_eigenpairs;
        std::vector<double> unique_eigenvalues;

        // Find unique eigenvalues from Jacobi's result
        for (double val: eigenvalues) {
            bool is_unique = true;
            for (double unique_val: unique_eigenvalues) {
                if (std::fabs(val - unique_val) < EPSILON) {
                    is_unique = false;
                    break;
                }
            }
            if (is_unique) {
                unique_eigenvalues.push_back(val);
            }
        }

        auto start_gauss = std::chrono::high_resolution_clock::now();
        for (double lambda: unique_eigenvalues) {
            std::vector<std::pair<double, Vector3>> current_pairs = gaussian_elimination_eigenvectors(A, lambda);
            gauss_eigenpairs.insert(gauss_eigenpairs.end(), current_pairs.begin(), current_pairs.end());
        }
        auto end_gauss = std::chrono::high_resolution_clock::now();
        auto duration_gauss = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gauss - start_gauss).count();

        std::cout << "Gaussian elimination with pivoting total Duration (all " << gauss_eigenpairs.size()
                  << " eigenvectors): " << duration_gauss << " ns\n";

        // Print residuals for Gaussian elimination eigenvectors with correct eigenvalue pairing
        for (const auto &pair: gauss_eigenpairs) {
            double residual = compute_residual(A, pair.first, pair.second);
            std::cout << "Residual (Gaussian, lambda=" << pair.first << "): " << residual << "\n";
        }


        // Print residuals for Jacobi eigenvectors
        for (int i = 0; i < 3; ++i) {
            Vector3 v = {eigenvectors[0][i], eigenvectors[1][i], eigenvectors[2][i]};
            double residual = compute_residual(A, eigenvalues[i], v);
            std::cout << "Residual (Jacobi, lambda=" << eigenvalues[i] << "): " << residual << "\n";
        }
        std::cout << "\n";

        if (t + 1 == 8) {
            std::cout << "Jacobi Vectors \n";
            print_vector(eigenvectors[0]);
            print_vector(eigenvectors[1]);
            print_vector(eigenvectors[2]);

            std::cout << "Gauss Vectors \n";

            print_vector(gauss_eigenpairs[0].second);
            print_vector(gauss_eigenpairs[1].second);
            print_vector(gauss_eigenpairs[2].second);
            std::cout << "\n";
        }
    }
    return 0;
}
