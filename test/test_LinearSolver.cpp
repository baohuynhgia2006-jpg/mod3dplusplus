#include "LinearSolver.h"
#include <cassert>
#include <cmath>
#include <iostream>

bool approxEqual(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

int main() {
    std::cout << "Testing 3x3 linear system with partial pivoting..." << std::endl;
    {
        // 0*x + 2*y + 1*z = 7
        // 3*x - 1*y + 2*z = 7
        // 1*x + 1*y + 1*z = 6
        // Exact solution: x = 1, y = 2, z = 3
        DenseMatrix A(3, 3);
        A(0, 0) = 0.0; A(0, 1) =  2.0; A(0, 2) = 1.0;
        A(1, 0) = 3.0; A(1, 1) = -1.0; A(1, 2) = 2.0;
        A(2, 0) = 1.0; A(2, 1) =  1.0; A(2, 2) = 1.0;

        std::vector<double> b = {7.0, 7.0, 6.0};
        std::vector<double> x = LinearSolver::solve(A, b);

        assert(x.size() == 3);
        assert(approxEqual(x[0], 1.0));
        assert(approxEqual(x[1], 2.0));
        assert(approxEqual(x[2], 3.0));
    }

    std::cout << "Testing Cholesky vs Gaussian elimination on SPD system..." << std::endl;
    {
        DenseMatrix A(3, 3);
        A(0, 0) = 4.0;  A(0, 1) = 12.0; A(0, 2) = -16.0;
        A(1, 0) = 12.0; A(1, 1) = 37.0; A(1, 2) = -43.0;
        A(2, 0) = -16.0; A(2, 1) = -43.0; A(2, 2) = 98.0;

        std::vector<double> b = {1.0, 2.0, 3.0};
        std::vector<double> xGauss = LinearSolver::solve(A, b);
        std::vector<double> xChol = LinearSolver::solveCholesky(A, b);

        for (size_t i = 0; i < 3; ++i) {
            assert(approxEqual(xGauss[i], xChol[i], 1e-9));
        }

        // Verify A * xChol == b
        std::vector<double> Ax = A * xChol;
        for (size_t i = 0; i < 3; ++i) {
            assert(approxEqual(Ax[i], b[i], 1e-9));
        }
    }

    std::cout << "Testing 12-parameter quadratic flow weighted least squares recovery..." << std::endl;
    {
        // Simulate 12-parameter 2D quadratic motion model:
        // u(x,y) = p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2
        // v(x,y) = p6 + p7*x + p8*y + p9*x^2 + p10*x*y + p11*y^2
        const std::vector<double> trueParams = {
            2.5, -0.05, 0.03, 0.001, -0.002, 0.0015,
            -1.8, 0.04, 0.06, -0.001, 0.0012, -0.0008
        };

        const int gridN = 8;
        const int numObs = gridN * gridN * 2 + 2; // +2 rows for an outlier pixel
        DenseMatrix J(numObs, 12, 0.0);
        std::vector<double> r(static_cast<size_t>(numObs), 0.0);
        std::vector<double> weights(static_cast<size_t>(numObs), 1.0);

        int row = 0;
        for (int iy = 0; iy < gridN; ++iy) {
            for (int ix = 0; ix < gridN; ++ix) {
                double x = (ix - 3.5) / 4.0;
                double y = (iy - 3.5) / 4.0;
                double basis[6] = {1.0, x, y, x * x, x * y, y * y};

                double uVal = 0.0;
                double vVal = 0.0;
                for (int k = 0; k < 6; ++k) {
                    J(row, k) = basis[k];
                    uVal += basis[k] * trueParams[static_cast<size_t>(k)];

                    J(row + 1, k + 6) = basis[k];
                    vVal += basis[k] * trueParams[static_cast<size_t>(k + 6)];
                }
                r[static_cast<size_t>(row)] = uVal;
                r[static_cast<size_t>(row + 1)] = vVal;
                row += 2;
            }
        }

        // Inject a gross outlier (simulating a moving object) suppressed by IRLS weight = 0
        J(row, 0) = 1.0;
        r[static_cast<size_t>(row)] = 9999.0;
        weights[static_cast<size_t>(row)] = 0.0;

        J(row + 1, 6) = 1.0;
        r[static_cast<size_t>(row + 1)] = -9999.0;
        weights[static_cast<size_t>(row + 1)] = 0.0;

        std::vector<double> estParams = LinearSolver::solveWeightedLeastSquares(J, r, weights, 0.0);
        assert(estParams.size() == 12);
        for (size_t i = 0; i < 12; ++i) {
            assert(approxEqual(estParams[i], trueParams[i], 1e-7));
        }
    }

    std::cout << "Testing singular matrix exception handling..." << std::endl;
    {
        DenseMatrix singular(2, 2);
        singular(0, 0) = 1.0; singular(0, 1) = 2.0;
        singular(1, 0) = 2.0; singular(1, 1) = 4.0;
        std::vector<double> b = {3.0, 6.0};

        bool caughtSingular = false;
        try {
            LinearSolver::solve(singular, b);
        } catch (const std::runtime_error&) {
            caughtSingular = true;
        }
        assert(caughtSingular);
    }

    std::cout << "\nAll LinearSolver tests passed successfully!" << std::endl;
    return 0;
}