#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "DenseMatrix.h"
#include <vector>

class LinearSolver {
public:
    // Solves (A + diagDamping * I) * x = b using Gaussian elimination with partial pivoting.
    // Throws std::runtime_error if the system is singular or numerically ill-conditioned.
    static std::vector<double> solve(
        const DenseMatrix& A,
        const std::vector<double>& b,
        double diagDamping = 0.0
    );

    // Solves (A + diagDamping * I) * x = b using Cholesky decomposition (L * L^T).
    // Assumes A is symmetric positive definite.
    // Throws std::runtime_error if A is not positive definite.
    static std::vector<double> solveCholesky(
        const DenseMatrix& A,
        const std::vector<double>& b,
        double diagDamping = 0.0
    );

    // Solves weighted normal equations: (J^T * W * J + diagDamping * I) * x = J^T * W * r.
    // If weights is empty, uniform weights of 1.0 are used.
    static std::vector<double> solveWeightedLeastSquares(
        const DenseMatrix& J,
        const std::vector<double>& r,
        const std::vector<double>& weights = {},
        double diagDamping = 1e-8
    );
};

#endif // LINEAR_SOLVER_H