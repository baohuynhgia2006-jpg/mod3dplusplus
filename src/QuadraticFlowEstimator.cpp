#include "QuadraticFlowEstimator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

FloatImage downsample2x(const FloatImage& src) {
    const int dstW = std::max(1, src.width / 2);
    const int dstH = std::max(1, src.height / 2);
    FloatImage dst(dstW, dstH, 1);

    constexpr float k[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dstH; ++y) {
        const int sy = std::min(src.height - 1, y * 2);
        for (int x = 0; x < dstW; ++x) {
            const int sx = std::min(src.width - 1, x * 2);
            float acc = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                const int yy = std::clamp(sy + dy, 0, src.height - 1);
                const float wy = k[dy + 2];
                for (int dx = -2; dx <= 2; ++dx) {
                    const int xx = std::clamp(sx + dx, 0, src.width - 1);
                    acc += wy * k[dx + 2] * src.at(yy, xx);
                }
            }
            dst.at(y, x) = acc;
        }
    }
    return dst;
}

inline float sampleBilinear(const FloatImage& img, float x, float y) {
    const float cx = std::clamp(x, 0.0f, static_cast<float>(img.width - 1));
    const float cy = std::clamp(y, 0.0f, static_cast<float>(img.height - 1));

    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(img.width - 1, x0 + 1);
    const int y1 = std::min(img.height - 1, y0 + 1);

    const float fx = cx - static_cast<float>(x0);
    const float fy = cy - static_cast<float>(y0);

    const float v00 = img.at(y0, x0);
    const float v10 = img.at(y0, x1);
    const float v01 = img.at(y1, x0);
    const float v11 = img.at(y1, x1);

    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10)
         + fy          * ((1.0f - fx) * v01 + fx * v11);
}

void computeGradients(const FloatImage& img, std::vector<float>& gx, std::vector<float>& gy) {
    const int w = img.width;
    const int h = img.height;
    const size_t numPixels = static_cast<size_t>(w) * h;
    gx.assign(numPixels, 0.0f);
    gy.assign(numPixels, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            gx[idx] = 0.5f * (img.at(y, x + 1) - img.at(y, x - 1));
            gy[idx] = 0.5f * (img.at(y + 1, x) - img.at(y - 1, x));
        }
    }
}

inline float sampleVectorBilinear(const std::vector<float>& field, int w, int h, float x, float y) {
    const float cx = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    const float cy = std::clamp(y, 0.0f, static_cast<float>(h - 1));

    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(w - 1, x0 + 1);
    const int y1 = std::min(h - 1, y0 + 1);

    const float fx = cx - static_cast<float>(x0);
    const float fy = cy - static_cast<float>(y0);

    const float v00 = field[static_cast<size_t>(y0) * w + x0];
    const float v10 = field[static_cast<size_t>(y0) * w + x1];
    const float v01 = field[static_cast<size_t>(y1) * w + x0];
    const float v11 = field[static_cast<size_t>(y1) * w + x1];

    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10)
         + fy          * ((1.0f - fx) * v01 + fx * v11);
}

bool solveLinear8x8(std::array<std::array<double, 8>, 8> A, std::array<double, 8> b, std::array<double, 8>& x) {
    for (int col = 0; col < 8; ++col) {
        int pivotRow = col;
        double maxVal = std::fabs(A[col][col]);
        for (int row = col + 1; row < 8; ++row) {
            const double val = std::fabs(A[row][col]);
            if (val > maxVal) {
                maxVal = val;
                pivotRow = row;
            }
        }
        if (maxVal < 1e-14) {
            return false;
        }
        if (pivotRow != col) {
            std::swap(A[col], A[pivotRow]);
            std::swap(b[col], b[pivotRow]);
        }

        const double invPivot = 1.0 / A[col][col];
        for (int row = col + 1; row < 8; ++row) {
            const double factor = A[row][col] * invPivot;
            if (factor == 0.0) continue;
            for (int k = col + 1; k < 8; ++k) {
                A[row][k] -= factor * A[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    for (int row = 7; row >= 0; --row) {
        double sum = b[row];
        for (int k = row + 1; k < 8; ++k) {
            sum -= A[row][k] * x[k];
        }
        x[row] = sum / A[row][row];
    }
    return true;
}

// Convert centered normalized 8-DOF parameters (x_hat = (x - xc)/s, y_hat = (y - yc)/s)
// back to exact pixel-coordinate coefficients:
//   u(x, y) = a + b*x + c*y + g*x^2 + h*x*y
//   v(x, y) = d + e*x + f*y + g*x*y + h*y^2
QuadraticCoeffs unnormalizeCoupledCoeffs(
    const std::array<double, 8>& normTheta,
    double xc,
    double yc,
    double s) {
    const double invS = 1.0 / s;
    const double invS2 = invS * invS;

    const double aHat = normTheta[0];
    const double bHat = normTheta[1];
    const double cHat = normTheta[2];
    const double dHat = normTheta[3];
    const double eHat = normTheta[4];
    const double fHat = normTheta[5];
    const double gHat = normTheta[6];
    const double hHat = normTheta[7];

    const double g = gHat * invS2;
    const double h = hHat * invS2;

    QuadraticCoeffs out;
    out.g = g;
    out.h = h;

    out.a = aHat - bHat * xc * invS - cHat * yc * invS + g * xc * xc + h * xc * yc;
    out.b = bHat * invS - 2.0 * g * xc - h * yc;
    out.c = cHat * invS - h * xc;

    out.d = dHat - eHat * xc * invS - fHat * yc * invS + g * xc * yc + h * yc * yc;
    out.e = eHat * invS - g * yc;
    out.f = fHat * invS - g * xc - 2.0 * h * yc;

    return out;
}

} // namespace

FlowField QuadraticFlowEstimator::estimatePair(
    const FloatImage& ref,
    const FloatImage& target,
    QuadraticCoeffs& outCoeffs,
    const QuadraticFlowParams& params) {
    if (ref.empty() || target.empty()) {
        throw std::invalid_argument("Reference and target frames must not be empty.");
    }
    if (ref.width != target.width || ref.height != target.height) {
        throw std::invalid_argument("Reference and target frames must have identical dimensions.");
    }

    std::vector<FloatImage> pyrRef;
    std::vector<FloatImage> pyrTarget;
    pyrRef.push_back(ref);
    pyrTarget.push_back(target);

    for (int lvl = 1; lvl < std::max(1, params.pyramidLevels); ++lvl) {
        const int curW = pyrRef.back().width;
        const int curH = pyrRef.back().height;
        if (curW / 2 < params.minPyramidDim || curH / 2 < params.minPyramidDim) {
            break;
        }
        pyrRef.push_back(downsample2x(pyrRef.back()));
        pyrTarget.push_back(downsample2x(pyrTarget.back()));
    }

    const int numLevels = static_cast<int>(pyrRef.size());

    // Normalized 8-DOF state: [a_hat, b_hat, c_hat, d_hat, e_hat, f_hat, g_hat, h_hat]
    std::array<double, 8> theta{};

    for (int lvl = numLevels - 1; lvl >= 0; --lvl) {
        const FloatImage& I0 = pyrRef[static_cast<size_t>(lvl)];
        const FloatImage& I1 = pyrTarget[static_cast<size_t>(lvl)];
        const int w = I0.width;
        const int h = I0.height;
        const size_t numPixels = static_cast<size_t>(w) * h;

        if (lvl < numLevels - 1) {
            const int prevW = pyrRef[static_cast<size_t>(lvl + 1)].width;
            const int prevH = pyrRef[static_cast<size_t>(lvl + 1)].height;
            const double scaleX = static_cast<double>(w) / std::max(1, prevW);
            const double scaleY = static_cast<double>(h) / std::max(1, prevH);
            const double scaleAvg = 0.5 * (scaleX + scaleY);

            theta[0] *= scaleX;
            theta[1] *= scaleX;
            theta[2] *= scaleX;
            theta[3] *= scaleY;
            theta[4] *= scaleY;
            theta[5] *= scaleY;
            theta[6] *= scaleAvg;
            theta[7] *= scaleAvg;
        }

        const double xc = 0.5 * static_cast<double>(w - 1);
        const double yc = 0.5 * static_cast<double>(h - 1);
        const double s = static_cast<double>(std::max(1, std::max(w, h)));
        const double invS = 1.0 / s;

        std::vector<float> gx0, gy0, gx1, gy1;
        computeGradients(I0, gx0, gy0);
        computeGradients(I1, gx1, gy1);

        std::vector<float> residualBuf(numPixels, 0.0f);
        std::vector<float> gradXBuf(numPixels, 0.0f);
        std::vector<float> gradYBuf(numPixels, 0.0f);
        std::vector<uint8_t> validBuf(numPixels, 0);
        std::vector<float> absResSamples;
        absResSamples.reserve(numPixels);

        const int margin = std::max(1, params.borderMargin);
        const float minGradSq = static_cast<float>(params.minGradientNorm * params.minGradientNorm);

        for (int iter = 0; iter < params.maxIterationsPerLevel; ++iter) {
            std::fill(validBuf.begin(), validBuf.end(), 0);
            absResSamples.clear();

            // 1. Warp target frame with current 8-DOF coupled quadratic estimate and evaluate residuals
            #pragma omp parallel for schedule(static)
            for (int y = margin; y < h - margin; ++y) {
                const double yHat = (static_cast<double>(y) - yc) * invS;
                const double yHat2 = yHat * yHat;

                for (int x = margin; x < w - margin; ++x) {
                    const double xHat = (static_cast<double>(x) - xc) * invS;
                    const double xHat2 = xHat * xHat;
                    const double xyHat = xHat * yHat;

                    // Coupled 8-DOF planar quadratic flow:
                    //   u = a + b*x + c*y + g*x^2 + h*x*y
                    //   v = d + e*x + f*y + g*x*y + h*y^2
                    const double u = theta[0] + theta[1] * xHat + theta[2] * yHat
                                   + theta[6] * xHat2 + theta[7] * xyHat;
                    const double v = theta[3] + theta[4] * xHat + theta[5] * yHat
                                   + theta[6] * xyHat + theta[7] * yHat2;

                    const float wx = static_cast<float>(static_cast<double>(x) + u);
                    const float wy = static_cast<float>(static_cast<double>(y) + v);

                    if (wx < static_cast<float>(margin) || wx > static_cast<float>(w - 1 - margin) ||
                        wy < static_cast<float>(margin) || wy > static_cast<float>(h - 1 - margin)) {
                        continue;
                    }

                    const size_t idx = static_cast<size_t>(y) * w + x;
                    const float ix = 0.5f * (gx0[idx] + sampleVectorBilinear(gx1, w, h, wx, wy));
                    const float iy = 0.5f * (gy0[idx] + sampleVectorBilinear(gy1, w, h, wx, wy));

                    if (ix * ix + iy * iy < minGradSq) {
                        continue;
                    }

                    const float r = I0.at(y, x) - sampleBilinear(I1, wx, wy);
                    residualBuf[idx] = r;
                    gradXBuf[idx] = ix;
                    gradYBuf[idx] = iy;
                    validBuf[idx] = 1;
                }
            }

            // 2. Estimate robust scale via MAD (after initial L2 lock-on step)
            double tukeyScale = 0.0;
            if (iter > 0) {
                for (size_t i = 0; i < numPixels; ++i) {
                    if (validBuf[i]) {
                        absResSamples.push_back(std::fabs(residualBuf[i]));
                    }
                }
                if (absResSamples.size() < 16) {
                    break;
                }
                const size_t mid = absResSamples.size() / 2;
                std::nth_element(
                    absResSamples.begin(),
                    absResSamples.begin() + static_cast<ptrdiff_t>(mid),
                    absResSamples.end()
                );
                const double sigma = std::max(1e-4, 1.4826 * static_cast<double>(absResSamples[mid]));
                tukeyScale = params.tukeyC * sigma;
            }

            // 3. Accumulate 8x8 normal equations H * delta_theta = g
            std::array<std::array<double, 8>, 8> H{};
            std::array<double, 8> gVec{};

            #pragma omp parallel
            {
                std::array<std::array<double, 8>, 8> localH{};
                std::array<double, 8> localG{};

                #pragma omp for schedule(static) nowait
                for (int y = margin; y < h - margin; ++y) {
                    const double yHat = (static_cast<double>(y) - yc) * invS;

                    for (int x = margin; x < w - margin; ++x) {
                        const size_t idx = static_cast<size_t>(y) * w + x;
                        if (!validBuf[idx]) continue;

                        const double r = static_cast<double>(residualBuf[idx]);
                        double wgt = 1.0;
                        if (iter > 0) {
                            const double uNorm = std::fabs(r) / tukeyScale;
                            if (uNorm >= 1.0) {
                                continue;
                            }
                            const double t = 1.0 - uNorm * uNorm;
                            wgt = t * t;
                        }

                        const double xHat = (static_cast<double>(x) - xc) * invS;
                        const double ix = static_cast<double>(gradXBuf[idx]);
                        const double iy = static_cast<double>(gradYBuf[idx]);
                        const double radialGrad = ix * xHat + iy * yHat;

                        // Coupled 1x8 Brightness Constancy Jacobian:
                        //   J = [Ix, Ix*x, Ix*y, Iy, Iy*x, Iy*y, x*(Ix*x + Iy*y), y*(Ix*x + Iy*y)]
                        const std::array<double, 8> J = {
                            ix,
                            ix * xHat,
                            ix * yHat,
                            iy,
                            iy * xHat,
                            iy * yHat,
                            xHat * radialGrad,
                            yHat * radialGrad
                        };

                        const double wr = wgt * r;
                        for (int row = 0; row < 8; ++row) {
                            const double wj = wgt * J[row];
                            localG[row] += wr * J[row];
                            for (int col = row; col < 8; ++col) {
                                localH[row][col] += wj * J[col];
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int row = 0; row < 8; ++row) {
                        gVec[row] += localG[row];
                        for (int col = row; col < 8; ++col) {
                            H[row][col] += localH[row][col];
                        }
                    }
                }
            }

            for (int row = 0; row < 8; ++row) {
                for (int col = 0; col < row; ++col) {
                    H[row][col] = H[col][row];
                }
                H[row][row] += params.regularizationDamping;
            }

            std::array<double, 8> delta{};
            if (!solveLinear8x8(H, gVec, delta)) {
                break;
            }

            double maxStep = 0.0;
            for (int k = 0; k < 8; ++k) {
                theta[k] += delta[k];
                maxStep = std::max(maxStep, std::fabs(delta[k]));
            }

            if (iter > 0 && maxStep < params.convergenceEps) {
                break;
            }
        }
    }

    const int fullW = ref.width;
    const int fullH = ref.height;
    const double xcFull = 0.5 * static_cast<double>(fullW - 1);
    const double ycFull = 0.5 * static_cast<double>(fullH - 1);
    const double sFull = static_cast<double>(std::max(1, std::max(fullW, fullH)));

    outCoeffs = unnormalizeCoupledCoeffs(theta, xcFull, ycFull, sFull);

    FlowField flow(fullW, fullH);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < fullH; ++y) {
        const double yd = static_cast<double>(y);
        for (int x = 0; x < fullW; ++x) {
            const double xd = static_cast<double>(x);
            const size_t idx = static_cast<size_t>(y) * fullW + x;
            flow.u[idx] = static_cast<float>(outCoeffs.evalU(xd, yd));
            flow.v[idx] = static_cast<float>(outCoeffs.evalV(xd, yd));
        }
    }

    return flow;
}

QuadraticFlowResult QuadraticFlowEstimator::estimate(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowParams& params) {
    if (prev.empty() || curr.empty() || next.empty()) {
        throw std::invalid_argument("All three input frames must not be empty.");
    }
    if (prev.width != curr.width || prev.height != curr.height ||
        next.width != curr.width || next.height != curr.height) {
        throw std::invalid_argument("All three input frames must have identical dimensions.");
    }

    QuadraticFlowResult result;
    result.backwardFlow = estimatePair(curr, prev, result.coeffsPrev, params);
    result.forwardFlow = estimatePair(curr, next, result.coeffsNext, params);
    return result;
}