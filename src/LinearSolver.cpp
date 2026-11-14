#include "LinearSolver.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

std::vector<double> LinearSolver::solve(
    const DenseMatrix& A,
    const std::vector<double>& b,
    double diagDamping) {
    if (A.empty() || A.rows != A.cols) {
        throw std::invalid_argument("Coefficient matrix A must be non-empty and square.");
    }
    if (static_cast<int>(b.size()) != A.rows) {
        throw std::invalid_argument("Right-hand side vector b size must match A dimensions.");
    }

    const int n = A.rows;
    DenseMatrix M = A;
    std::vector<double> rhs = b;

    if (diagDamping != 0.0) {
        for (int i = 0; i < n; ++i) {
            M(i, i) += diagDamping;
        }
    }

    // Forward elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        int pivotRow = k;
        double maxAbs = std::fabs(M(k, k));
        for (int r = k + 1; r < n; ++r) {
            double val = std::fabs(M(r, k));
            if (val > maxAbs) {
                maxAbs = val;
                pivotRow = r;
            }
        }

        if (maxAbs < 1e-12) {
            throw std::runtime_error("Linear system is singular or nearly singular.");
        }

        if (pivotRow != k) {
            for (int c = k; c < n; ++c) {
                std::swap(M(k, c), M(pivotRow, c));
            }
            std::swap(rhs[static_cast<size_t>(k)], rhs[static_cast<size_t>(pivotRow)]);
        }

        const double pivot = M(k, k);
        for (int r = k + 1; r < n; ++r) {
            const double factor = M(r, k) / pivot;
            M(r, k) = 0.0;
            for (int c = k + 1; c < n; ++c) {
                M(r, c) -= factor * M(k, c);
            }
            rhs[static_cast<size_t>(r)] -= factor * rhs[static_cast<size_t>(k)];
        }
    }

    // Back substitution
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = rhs[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= M(i, j) * x[static_cast<size_t>(j)];
        }
        x[static_cast<size_t>(i)] = sum / M(i, i);
    }

    return x;
}

std::vector<double> LinearSolver::solveCholesky(
    const DenseMatrix& A,
    const std::vector<double>& b,
    double diagDamping) {
    if (A.empty() || A.rows != A.cols) {
        throw std::invalid_argument("Coefficient matrix A must be non-empty and square.");
    }
    if (static_cast<int>(b.size()) != A.rows) {
        throw std::invalid_argument("Right-hand side vector b size must match A dimensions.");
    }

    const int n = A.rows;
    DenseMatrix L(n, n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A(i, j);
            if (i == j) {
                sum += diagDamping;
            }
            for (int k = 0; k < j; ++k) {
                sum -= L(i, k) * L(j, k);
            }

            if (i == j) {
                if (sum <= 1e-12) {
                    throw std::runtime_error("Matrix is not symmetric positive definite.");
                }
                L(i, j) = std::sqrt(sum);
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }

    // Forward substitution: L * y = b
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = b[static_cast<size_t>(i)];
        for (int k = 0; k < i; ++k) {
            sum -= L(i, k) * y[static_cast<size_t>(k)];
        }
        y[static_cast<size_t>(i)] = sum / L(i, i);
    }

    // Backward substitution: L^T * x = y
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[static_cast<size_t>(i)];
        for (int k = i + 1; k < n; ++k) {
            sum -= L(k, i) * x[static_cast<size_t>(k)];
        }
        x[static_cast<size_t>(i)] = sum / L(i, i);
    }

    return x;
}

std::vector<double> LinearSolver::solveWeightedLeastSquares(
    const DenseMatrix& J,
    const std::vector<double>& r,
    const std::vector<double>& weights,
    double diagDamping) {
    if (J.empty()) {
        throw std::invalid_argument("Jacobian matrix J must not be empty.");
    }
    if (static_cast<int>(r.size()) != J.rows) {
        throw std::invalid_argument("Residual vector r size must match J.rows.");
    }
    if (!weights.empty() && static_cast<int>(weights.size()) != J.rows) {
        throw std::invalid_argument("Weights vector size must match J.rows.");
    }

    const int numObs = J.rows;
    const int numParams = J.cols;

    DenseMatrix JtWJ(numParams, numParams, 0.0);
    std::vector<double> JtWr(static_cast<size_t>(numParams), 0.0);

    for (int row = 0; row < numObs; ++row) {
        const double w = weights.empty() ? 1.0 : weights[static_cast<size_t>(row)];
        if (w <= 0.0) continue;

        const double wr = w * r[static_cast<size_t>(row)];
        for (int i = 0; i < numParams; ++i) {
            const double j_ri = J(row, i);
            if (j_ri == 0.0) continue;

            JtWr[static_cast<size_t>(i)] += j_ri * wr;
            const double w_j_ri = w * j_ri;
            for (int j = i; j < numParams; ++j) {
                JtWJ(i, j) += w_j_ri * J(row, j);
            }
        }
    }

    // Mirror upper triangle to lower triangle
    for (int i = 0; i < numParams; ++i) {
        for (int j = 0; j < i; ++j) {
            JtWJ(i, j) = JtWJ(j, i);
        }
    }

    return solveCholesky(JtWJ, JtWr, diagDamping);
}