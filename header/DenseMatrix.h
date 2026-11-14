#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <cstddef>
#include <stdexcept>
#include <vector>

struct DenseMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;

    DenseMatrix() = default;
    DenseMatrix(int r, int c, double initVal = 0.0)
        : rows(r), cols(c), data(static_cast<size_t>(r) * c, initVal) {
        if (r < 0 || c < 0) {
            throw std::invalid_argument("Matrix dimensions must be non-negative.");
        }
    }

    bool empty() const {
        return data.empty() || rows <= 0 || cols <= 0;
    }

    double& operator()(int r, int c) {
        return data[static_cast<size_t>(r) * cols + c];
    }

    const double& operator()(int r, int c) const {
        return data[static_cast<size_t>(r) * cols + c];
    }

    void fill(double val) {
        std::fill(data.begin(), data.end(), val);
    }

    static DenseMatrix identity(int n) {
        DenseMatrix I(n, n, 0.0);
        for (int i = 0; i < n; ++i) {
            I(i, i) = 1.0;
        }
        return I;
    }

    DenseMatrix transpose() const {
        DenseMatrix T(cols, rows, 0.0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                T(c, r) = (*this)(r, c);
            }
        }
        return T;
    }

    std::vector<double> operator*(const std::vector<double>& x) const {
        if (static_cast<int>(x.size()) != cols) {
            throw std::invalid_argument("Matrix-vector dimension mismatch.");
        }
        std::vector<double> y(static_cast<size_t>(rows), 0.0);
        for (int r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (int c = 0; c < cols; ++c) {
                sum += (*this)(r, c) * x[static_cast<size_t>(c)];
            }
            y[static_cast<size_t>(r)] = sum;
        }
        return y;
    }

    DenseMatrix operator*(const DenseMatrix& B) const {
        if (cols != B.rows) {
            throw std::invalid_argument("Matrix-matrix dimension mismatch.");
        }
        DenseMatrix C(rows, B.cols, 0.0);
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < cols; ++k) {
                const double aik = (*this)(r, k);
                if (aik == 0.0) continue;
                for (int c = 0; c < B.cols; ++c) {
                    C(r, c) += aik * B(k, c);
                }
            }
        }
        return C;
    }
};

#endif // DENSE_MATRIX_H