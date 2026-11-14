#include "QuadraticFlowEstimator.h"
#include "DenseMatrix.h"
#include "GaussianPyramid.h"
#include "ImageOps.h"
#include "LinearSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

inline void computeBasis(int x, int y, double cx, double cy, double invScale, double basis[6]) {
    const double nx = (static_cast<double>(x) - cx) * invScale;
    const double ny = (static_cast<double>(y) - cy) * invScale;
    basis[0] = 1.0;
    basis[1] = nx;
    basis[2] = ny;
    basis[3] = nx * nx;
    basis[4] = nx * ny;
    basis[5] = ny * ny;
}

double computeRobustScaleMAD(const std::vector<double>& residuals) {
    if (residuals.empty()) {
        return 1e-3;
    }
    std::vector<double> absRes(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) {
        absRes[i] = std::fabs(residuals[i]);
    }
    const size_t mid = absRes.size() / 2;
    std::nth_element(absRes.begin(), absRes.begin() + static_cast<std::ptrdiff_t>(mid), absRes.end());
    double mad = absRes[mid];
    if (absRes.size() % 2 == 0 && mid > 0) {
        auto maxLeft = std::max_element(absRes.begin(), absRes.begin() + static_cast<std::ptrdiff_t>(mid));
        mad = 0.5 * (mad + *maxLeft);
    }
    return std::max(1.4826 * mad, 1e-3);
}

inline double tukeyWeight(double r, double cSigma) {
    const double u = std::fabs(r) / cSigma;
    if (u >= 1.0) {
        return 0.0;
    }
    const double t = 1.0 - u * u;
    return t * t;
}

std::array<double, 12> estimatePairwiseQuadratic(
    const std::vector<FloatImage>& pyrCurr,
    const std::vector<FloatImage>& pyrTarget,
    const QuadraticFlowParams& params) {
    const int actualLevels = static_cast<int>(pyrCurr.size());
    std::array<double, 12> coeffs{};

    for (int level = actualLevels - 1; level >= 0; --level) {
        if (level < actualLevels - 1) {
            for (double& c : coeffs) {
                c *= 2.0;
            }
        }

        const FloatImage& imgCurr = pyrCurr[static_cast<size_t>(level)];
        const FloatImage& imgTarget = pyrTarget[static_cast<size_t>(level)];

        const int w = imgCurr.width;
        const int h = imgCurr.height;
        const double cx = 0.5 * static_cast<double>(w - 1);
        const double cy = 0.5 * static_cast<double>(h - 1);
        const double halfMax = 0.5 * static_cast<double>(std::max(w, h));
        const double invScale = (halfMax > 0.0) ? (1.0 / halfMax) : 1.0;

        FloatImage ixCurr, iyCurr;
        ImageOps::computeGradients(imgCurr, ixCurr, iyCurr);

        std::vector<uint8_t> validWarp;
        std::vector<double> residualSamples;
        residualSamples.reserve(static_cast<size_t>(w) * h);

        for (int iter = 0; iter < params.maxIterationsPerLevel; ++iter) {
            FlowField flow = QuadraticFlowEstimator::evaluateFlow(w, h, coeffs, +1.0);
            FloatImage warpedTarget = ImageOps::warpBackward(imgTarget, flow, &validWarp);

            FloatImage ixTarget, iyTarget;
            ImageOps::computeGradients(warpedTarget, ixTarget, iyTarget);

            const bool applyIRLS = params.useRobustWeights && !(level == actualLevels - 1 && iter < 2);
            double cSigma = 1.0;

            if (applyIRLS) {
                residualSamples.clear();
                for (int y = 1; y < h - 1; ++y) {
                    for (int x = 1; x < w - 1; ++x) {
                        const size_t idx = static_cast<size_t>(y) * w + x;
                        if (validWarp[idx]) {
                            residualSamples.push_back(static_cast<double>(imgCurr.at(y, x) - warpedTarget.at(y, x)));
                        }
                    }
                }
                const double sigma = computeRobustScaleMAD(residualSamples);
                cSigma = params.tukeyC * sigma;
            }

            // H / g accumulation loop Parallelism:
            DenseMatrix H(12, 12, 0.0);
            std::vector<double> g(12, 0.0);
            int validObsCount = 0;

            #pragma omp parallel reduction(+:validObsCount)
            {
                double localH[12][12] = {};
                double localG[12] = {};
                double basis[6];
                double jRow[12];

                #pragma omp for schedule(static) nowait
                for (int y = 1; y < h - 1; ++y) {
                    for (int x = 1; x < w - 1; ++x) {
                        const size_t idx = static_cast<size_t>(y) * w + x;
                        if (!validWarp[idx]) {
                            continue;
                        }

                        const double r = static_cast<double>(imgCurr.at(y, x) - warpedTarget.at(y, x));
                        const double weight = applyIRLS ? tukeyWeight(r, cSigma) : 1.0;
                        if (weight <= 0.0) {
                            continue;
                        }

                        computeBasis(x, y, cx, cy, invScale, basis);
                        const double ix = 0.5 * (static_cast<double>(ixTarget.at(y, x)) + static_cast<double>(ixCurr.at(y, x)));
                        const double iy = 0.5 * (static_cast<double>(iyTarget.at(y, x)) + static_cast<double>(iyCurr.at(y, x)));

                        for (int k = 0; k < 6; ++k) {
                            jRow[k]     = ix * basis[k];
                            jRow[k + 6] = iy * basis[k];
                        }

                        const double wr = weight * r;
                        for (int i = 0; i < 12; ++i) {
                            localG[i] += jRow[i] * wr;
                            const double wji = weight * jRow[i];
                            for (int j = i; j < 12; ++j) {
                                localH[i][j] += wji * jRow[j];
                            }
                        }
                        validObsCount++;
                    }
                }

                #pragma omp critical
                {
                    for (int i = 0; i < 12; ++i) {
                        g[static_cast<size_t>(i)] += localG[i];
                        for (int j = i; j < 12; ++j) {
                            H(i, j) += localH[i][j];
                        }
                    }
                }
            }

            if (validObsCount < 12) {
                break;
            }

            for (int i = 0; i < 12; ++i) {
                for (int j = 0; j < i; ++j) {
                    H(i, j) = H(j, i);
                }
            }

            std::vector<double> deltaP;
            try {
                deltaP = LinearSolver::solveCholesky(H, g, params.diagDamping);
            } catch (const std::runtime_error&) {
                break;
            }

            double maxDelta = 0.0;
            for (size_t i = 0; i < 12; ++i) {
                coeffs[i] += deltaP[i];
                maxDelta = std::max(maxDelta, std::fabs(deltaP[i]));
            }

            if (maxDelta < params.convergenceEps) {
                break;
            }
        }
    }

    return coeffs;
}

} // namespace

FlowField QuadraticFlowEstimator::evaluateFlow(
    int width,
    int height,
    const std::array<double, 12>& coeffs,
    double vectorScale) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Flow dimensions must be positive.");
    }

    FlowField flow(width, height);
    const double cx = 0.5 * static_cast<double>(width - 1);
    const double cy = 0.5 * static_cast<double>(height - 1);
    const double halfMax = 0.5 * static_cast<double>(std::max(width, height));
    const double invScale = (halfMax > 0.0) ? (1.0 / halfMax) : 1.0;

    double basis[6];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            computeBasis(x, y, cx, cy, invScale, basis);

            double u = 0.0;
            double v = 0.0;
            for (int k = 0; k < 6; ++k) {
                u += coeffs[static_cast<size_t>(k)] * basis[k];
                v += coeffs[static_cast<size_t>(k + 6)] * basis[k];
            }
            flow.atU(y, x) = static_cast<float>(vectorScale * u);
            flow.atV(y, x) = static_cast<float>(vectorScale * v);
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
        throw std::invalid_argument("Input frames must not be empty.");
    }
    if (prev.width != curr.width || prev.height != curr.height ||
        next.width != curr.width || next.height != curr.height) {
        throw std::invalid_argument("All three frames must have identical dimensions.");
    }
    if (prev.channels != 1 || curr.channels != 1 || next.channels != 1) {
        throw std::invalid_argument("QuadraticFlowEstimator expects single-channel FloatImage inputs.");
    }

    const int numLevels = std::max(1, params.pyramidLevels);
    std::vector<FloatImage> pyrPrev = GaussianPyramid::build(prev, numLevels, params.minPyramidDim);
    std::vector<FloatImage> pyrCurr = GaussianPyramid::build(curr, numLevels, params.minPyramidDim);
    std::vector<FloatImage> pyrNext = GaussianPyramid::build(next, numLevels, params.minPyramidDim);

    QuadraticFlowResult result;
    result.coeffsPrev = estimatePairwiseQuadratic(pyrCurr, pyrPrev, params);
    result.coeffsNext = estimatePairwiseQuadratic(pyrCurr, pyrNext, params);
    result.coeffs = result.coeffsNext;

    result.backwardFlow = evaluateFlow(curr.width, curr.height, result.coeffsPrev, +1.0);
    result.forwardFlow = evaluateFlow(curr.width, curr.height, result.coeffsNext, +1.0);
    return result;
}