#include "SecondaryFlowEstimator.h"
#include "GaussianPyramid.h"
#include "ImageOps.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

std::vector<uint8_t> dilateMask(const std::vector<uint8_t>& mask, int width, int height, int radius) {
    if (radius <= 0) {
        return mask;
    }

    const int r2 = radius * radius;
    std::vector<uint8_t> dilated(static_cast<size_t>(width) * height, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!mask[static_cast<size_t>(y) * width + x]) {
                continue;
            }
            const int yMin = std::max(0, y - radius);
            const int yMax = std::min(height - 1, y + radius);
            const int xMin = std::max(0, x - radius);
            const int xMax = std::min(width - 1, x + radius);

            for (int ny = yMin; ny <= yMax; ++ny) {
                const int dy = ny - y;
                for (int nx = xMin; nx <= xMax; ++nx) {
                    const int dx = nx - x;
                    if (dx * dx + dy * dy <= r2) {
                        dilated[static_cast<size_t>(ny) * width + nx] = 1;
                    }
                }
            }
        }
    }

    return dilated;
}

std::vector<uint8_t> downsampleMask(
    const std::vector<uint8_t>& srcMask,
    int srcW,
    int srcH,
    int dstW,
    int dstH) {
    std::vector<uint8_t> dstMask(static_cast<size_t>(dstW) * dstH, 0);
    const double scaleX = static_cast<double>(srcW) / static_cast<double>(dstW);
    const double scaleY = static_cast<double>(srcH) / static_cast<double>(dstH);

    for (int y = 0; y < dstH; ++y) {
        const int yStart = std::max(0, static_cast<int>(std::floor(y * scaleY)));
        const int yEnd = std::min(srcH, static_cast<int>(std::ceil((y + 1) * scaleY)));
        for (int x = 0; x < dstW; ++x) {
            const int xStart = std::max(0, static_cast<int>(std::floor(x * scaleX)));
            const int xEnd = std::min(srcW, static_cast<int>(std::ceil((x + 1) * scaleX)));

            uint8_t active = 0;
            for (int sy = yStart; sy < yEnd && !active; ++sy) {
                for (int sx = xStart; sx < xEnd; ++sx) {
                    if (srcMask[static_cast<size_t>(sy) * srcW + sx]) {
                        active = 1;
                        break;
                    }
                }
            }
            dstMask[static_cast<size_t>(y) * dstW + x] = active;
        }
    }

    return dstMask;
}

FlowField upsampleResidualFlow(const FlowField& coarse, int targetW, int targetH) {
    FlowField fine(targetW, targetH);
    const float scaleX = static_cast<float>(targetW) / static_cast<float>(coarse.width);
    const float scaleY = static_cast<float>(targetH) / static_cast<float>(coarse.height);

    FloatImage coarseU(coarse.width, coarse.height, 1);
    FloatImage coarseV(coarse.width, coarse.height, 1);
    coarseU.data = coarse.u;
    coarseV.data = coarse.v;

    for (int y = 0; y < targetH; ++y) {
        const float srcY = (static_cast<float>(y) + 0.5f) / scaleY - 0.5f;
        for (int x = 0; x < targetW; ++x) {
            const float srcX = (static_cast<float>(x) + 0.5f) / scaleX - 0.5f;
            fine.atU(y, x) = ImageOps::sampleBilinear(coarseU, srcX, srcY) * scaleX;
            fine.atV(y, x) = ImageOps::sampleBilinear(coarseV, srcX, srcY) * scaleY;
        }
    }

    return fine;
}

void smoothResidualInMask(
    FlowField& residual,
    const std::vector<uint8_t>& activeMask,
    int iterations,
    double alpha) {
    if (iterations <= 0 || alpha <= 0.0) {
        return;
    }

    const int w = residual.width;
    const int h = residual.height;
    const float a = static_cast<float>(std::clamp(alpha, 0.0, 0.25));

    FlowField temp = residual;
    constexpr int dx[4] = {-1, 1, 0, 0};
    constexpr int dy[4] = {0, 0, -1, 1};

    for (int it = 0; it < iterations; ++it) {
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                if (!activeMask[idx]) {
                    temp.atU(y, x) = 0.0f;
                    temp.atV(y, x) = 0.0f;
                    continue;
                }

                float sumU = 0.0f;
                float sumV = 0.0f;
                int activeNeighbors = 0;

                for (int k = 0; k < 4; ++k) {
                    const int nx = x + dx[k];
                    const int ny = y + dy[k];
                    if (activeMask[static_cast<size_t>(ny) * w + nx]) {
                        sumU += residual.atU(ny, nx);
                        sumV += residual.atV(ny, nx);
                        activeNeighbors++;
                    }
                }

                if (activeNeighbors > 0) {
                    const float centerWeight = 1.0f - a * static_cast<float>(activeNeighbors);
                    temp.atU(y, x) = centerWeight * residual.atU(y, x) + a * sumU;
                    temp.atV(y, x) = centerWeight * residual.atV(y, x) + a * sumV;
                }
            }
        }
        residual = temp;
    }
}

void estimatePairwiseResidual(
    const std::vector<FloatImage>& pyrCurr,
    const std::vector<FloatImage>& pyrTarget,
    const std::vector<std::vector<uint8_t>>& pyrMasks,
    const std::vector<uint8_t>& misalignedMask,
    const std::array<double, 12>& baseCoeffs,
    double baseVectorScale,
    const SecondaryFlowParams& params,
    FlowField& outResidual,
    std::vector<uint8_t>& outValidMask) {
    const int actualLevels = static_cast<int>(pyrCurr.size());
    const int baseW = pyrCurr[0].width;
    const int baseH = pyrCurr[0].height;
    const size_t basePixels = static_cast<size_t>(baseW) * baseH;

    const int r = std::max(1, params.windowRadius);
    const int winDim = 2 * r + 1;
    std::vector<double> winWeights(static_cast<size_t>(winDim) * winDim, 1.0);
    const double twoSigmaSq = 2.0 * std::max(0.1, params.windowSigma) * std::max(0.1, params.windowSigma);
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const double distSq = static_cast<double>(dx * dx + dy * dy);
            winWeights[static_cast<size_t>(dy + r) * winDim + (dx + r)] = std::exp(-distSq / twoSigmaSq);
        }
    }

    FlowField residualFlow;
    outValidMask.assign(basePixels, 0);

    for (int level = actualLevels - 1; level >= 0; --level) {
        const FloatImage& imgCurr = pyrCurr[static_cast<size_t>(level)];
        const FloatImage& imgTarget = pyrTarget[static_cast<size_t>(level)];
        const std::vector<uint8_t>& activeMask = pyrMasks[static_cast<size_t>(level)];

        const int w = imgCurr.width;
        const int h = imgCurr.height;
        const size_t levelPixels = static_cast<size_t>(w) * h;

        if (level == actualLevels - 1) {
            residualFlow = FlowField(w, h);
        } else {
            residualFlow = upsampleResidualFlow(residualFlow, w, h);
            for (size_t i = 0; i < levelPixels; ++i) {
                if (!activeMask[i]) {
                    residualFlow.u[i] = 0.0f;
                    residualFlow.v[i] = 0.0f;
                }
            }
        }

        std::array<double, 12> levelCoeffs = baseCoeffs;
        const double levelScaleX = static_cast<double>(w) / static_cast<double>(baseW);
        const double levelScaleY = static_cast<double>(h) / static_cast<double>(baseH);
        for (int k = 0; k < 6; ++k) {
            levelCoeffs[static_cast<size_t>(k)] *= levelScaleX;
            levelCoeffs[static_cast<size_t>(k + 6)] *= levelScaleY;
        }
        FlowField baseFlow = QuadraticFlowEstimator::evaluateFlow(w, h, levelCoeffs, baseVectorScale);

        FloatImage ixCurr, iyCurr;
        ImageOps::computeGradients(imgCurr, ixCurr, iyCurr);

        std::vector<uint8_t> validWarp;
        FlowField totalFlow(w, h);

        std::vector<double> a11(levelPixels, 0.0);
        std::vector<double> a12(levelPixels, 0.0);
        std::vector<double> a22(levelPixels, 0.0);
        std::vector<double> b1(levelPixels, 0.0);
        std::vector<double> b2(levelPixels, 0.0);

        for (int iter = 0; iter < params.maxIterationsPerLevel; ++iter) {
            for (size_t i = 0; i < levelPixels; ++i) {
                totalFlow.u[i] = baseFlow.u[i] + residualFlow.u[i];
                totalFlow.v[i] = baseFlow.v[i] + residualFlow.v[i];
            }

            FloatImage warpedTarget = ImageOps::warpBackward(imgTarget, totalFlow, &validWarp);
            FloatImage ixTarget, iyTarget;
            ImageOps::computeGradients(warpedTarget, ixTarget, iyTarget);

            std::fill(a11.begin(), a11.end(), 0.0);
            std::fill(a12.begin(), a12.end(), 0.0);
            std::fill(a22.begin(), a22.end(), 0.0);
            std::fill(b1.begin(), b1.end(), 0.0);
            std::fill(b2.begin(), b2.end(), 0.0);

            // Parallelize structure-tensor outer-product fill:
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t idx = static_cast<size_t>(y) * w + x;
                    if (!validWarp[idx]) {
                        continue;
                    }
                    const double ix = 0.5 * (static_cast<double>(ixTarget.at(y, x)) + static_cast<double>(ixCurr.at(y, x)));
                    const double iy = 0.5 * (static_cast<double>(iyTarget.at(y, x)) + static_cast<double>(iyCurr.at(y, x)));
                    const double rDiff = static_cast<double>(imgCurr.at(y, x) - warpedTarget.at(y, x));

                    a11[idx] = ix * ix;
                    a12[idx] = ix * iy;
                    a22[idx] = iy * iy;
                    b1[idx] = ix * rDiff;
                    b2[idx] = iy * rDiff;
                }
            }

            double maxDelta = 0.0;

            #pragma omp parallel
            {
                double localMaxDelta = 0.0;

                #pragma omp for schedule(dynamic, 8)
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        const size_t idx = static_cast<size_t>(y) * w + x;
                        if (!activeMask[idx]) {
                            continue;
                        }

                        double s11 = 0.0, s12 = 0.0, s22 = 0.0;
                        double rhs1 = 0.0, rhs2 = 0.0;

                        const int yMin = std::max(0, y - r);
                        const int yMax = std::min(h - 1, y + r);
                        const int xMin = std::max(0, x - r);
                        const int xMax = std::min(w - 1, x + r);

                        for (int wy = yMin; wy <= yMax; ++wy) {
                            const int kRow = (wy - y + r) * winDim;
                            for (int wx = xMin; wx <= xMax; ++wx) {
                                const double gw = winWeights[static_cast<size_t>(kRow + (wx - x + r))];
                                const size_t nIdx = static_cast<size_t>(wy) * w + wx;

                                s11 += gw * a11[nIdx];
                                s12 += gw * a12[nIdx];
                                s22 += gw * a22[nIdx];
                                rhs1 += gw * b1[nIdx];
                                rhs2 += gw * b2[nIdx];
                            }
                        }

                        const double d11 = s11 + params.regularizationDamping;
                        const double d22 = s22 + params.regularizationDamping;
                        const double det = d11 * d22 - s12 * s12;

                        if (det > 1e-12) {
                            const double du = (d22 * rhs1 - s12 * rhs2) / det;
                            const double dv = (-s12 * rhs1 + d11 * rhs2) / det;

                            const double stepU = std::clamp(du, -2.0, 2.0);
                            const double stepV = std::clamp(dv, -2.0, 2.0);

                            residualFlow.u[idx] += static_cast<float>(stepU);
                            residualFlow.v[idx] += static_cast<float>(stepV);

                            localMaxDelta = std::max(localMaxDelta, std::max(std::fabs(stepU), std::fabs(stepV)));
                        }
                    }
                }

                #pragma omp critical
                {
                    maxDelta = std::max(maxDelta, localMaxDelta);
                }
            }

            smoothResidualInMask(residualFlow, activeMask, params.smoothingIterations, params.smoothingAlpha);

            if (maxDelta < params.convergenceEps) {
                break;
            }
        }

        // At full resolution (level 0), evaluate post-convergence conditioning and occlusion consistency
        if (level == 0) {
            for (size_t i = 0; i < levelPixels; ++i) {
                totalFlow.u[i] = baseFlow.u[i] + residualFlow.u[i];
                totalFlow.v[i] = baseFlow.v[i] + residualFlow.v[i];
            }
            FloatImage finalWarped = ImageOps::warpBackward(imgTarget, totalFlow, &validWarp);

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t idx = static_cast<size_t>(y) * w + x;
                    if (!misalignedMask[idx] || !validWarp[idx]) {
                        outValidMask[idx] = 0;
                        continue;
                    }

                    // 1. Post-warp photometric consistency check (filters out occluded pixels)
                    const double postErr = std::fabs(
                        static_cast<double>(imgCurr.at(y, x)) - static_cast<double>(finalWarped.at(y, x))
                    );
                    if (postErr > params.maxPostWarpError) {
                        outValidMask[idx] = 0;
                        continue;
                    }

                    // 2. Structure tensor conditioning (rejects flat asphalt & 1D aperture edges)
                    double s11 = 0.0, s12 = 0.0, s22 = 0.0;
                    const int yMin = std::max(0, y - r);
                    const int yMax = std::min(h - 1, y + r);
                    const int xMin = std::max(0, x - r);
                    const int xMax = std::min(w - 1, x + r);

                    for (int wy = yMin; wy <= yMax; ++wy) {
                        const int kRow = (wy - y + r) * winDim;
                        for (int wx = xMin; wx <= xMax; ++wx) {
                            const double gw = winWeights[static_cast<size_t>(kRow + (wx - x + r))];
                            const size_t nIdx = static_cast<size_t>(wy) * w + wx;
                            s11 += gw * a11[nIdx];
                            s12 += gw * a12[nIdx];
                            s22 += gw * a22[nIdx];
                        }
                    }

                    const double trace = s11 + s22;
                    const double detRaw = s11 * s22 - s12 * s12;
                    const double disc = std::sqrt(std::max(0.0, trace * trace - 4.0 * detRaw));
                    const double lambdaMax = 0.5 * (trace + disc);
                    const double lambdaMin = 0.5 * (trace - disc);
                    const double eigenRatio = (lambdaMax > 1e-12) ? (lambdaMin / lambdaMax) : 0.0;

                    outValidMask[idx] =
                        (lambdaMin >= params.minEigenvalue && eigenRatio >= params.minEigenRatio) ? 1 : 0;
                }
            }
        }
    }

    outResidual = std::move(residualFlow);
}

} // namespace

SecondaryFlowResult SecondaryFlowEstimator::estimate(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const AlignmentResult& alignment,
    const SecondaryFlowParams& params) {
    return estimate(prev, curr, next, principalFlow, alignment.misalignedMask, params);
}

SecondaryFlowResult SecondaryFlowEstimator::estimate(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const std::vector<uint8_t>& misalignedMask,
    const SecondaryFlowParams& params) {
    if (prev.empty() || curr.empty() || next.empty()) {
        throw std::invalid_argument("Input frames must not be empty.");
    }
    if (prev.width != curr.width || prev.height != curr.height ||
        next.width != curr.width || next.height != curr.height) {
        throw std::invalid_argument("All three frames must have identical dimensions.");
    }
    if (prev.channels != 1 || curr.channels != 1 || next.channels != 1) {
        throw std::invalid_argument("SecondaryFlowEstimator expects single-channel FloatImage inputs.");
    }

    const int baseW = curr.width;
    const int baseH = curr.height;
    const size_t basePixels = static_cast<size_t>(baseW) * baseH;

    if (misalignedMask.size() != basePixels) {
        throw std::invalid_argument("misalignedMask size must match frame pixel count.");
    }

    bool hasAnyMisaligned = false;
    for (uint8_t m : misalignedMask) {
        if (m != 0) {
            hasAnyMisaligned = true;
            break;
        }
    }

    if (!hasAnyMisaligned) {
        SecondaryFlowResult emptyRes;
        emptyRes.backwardFlow = principalFlow.backwardFlow;
        emptyRes.forwardFlow = principalFlow.forwardFlow;
        emptyRes.residualFlowPrev = FlowField(baseW, baseH);
        emptyRes.residualFlowNext = FlowField(baseW, baseH);
        emptyRes.validFlowMask.assign(basePixels, 0);
        return emptyRes;
    }

    const std::vector<uint8_t> workingMask0 = dilateMask(
        misalignedMask,
        baseW,
        baseH,
        std::max(0, params.maskDilationRadius)
    );

    const int numLevels = std::max(1, params.pyramidLevels);
    std::vector<FloatImage> pyrPrev = GaussianPyramid::build(prev, numLevels, params.minPyramidDim);
    std::vector<FloatImage> pyrCurr = GaussianPyramid::build(curr, numLevels, params.minPyramidDim);
    std::vector<FloatImage> pyrNext = GaussianPyramid::build(next, numLevels, params.minPyramidDim);

    const int actualLevels = static_cast<int>(pyrCurr.size());
    std::vector<std::vector<uint8_t>> pyrMasks(static_cast<size_t>(actualLevels));
    pyrMasks[0] = workingMask0;
    for (int level = 1; level < actualLevels; ++level) {
        pyrMasks[static_cast<size_t>(level)] = downsampleMask(
            workingMask0,
            baseW,
            baseH,
            pyrCurr[static_cast<size_t>(level)].width,
            pyrCurr[static_cast<size_t>(level)].height
        );
    }

    FlowField rawResPrev, rawResNext;
    std::vector<uint8_t> validPrev, validNext;

    // 1. Estimate independent residual parallax mu^j (curr -> prev)
    estimatePairwiseResidual(
        pyrCurr,
        pyrPrev,
        pyrMasks,
        misalignedMask,
        principalFlow.coeffsPrev,
        +1.0,
        params,
        rawResPrev,
        validPrev
    );

    // 2. Estimate independent residual parallax mu^k (curr -> next)
    estimatePairwiseResidual(
        pyrCurr,
        pyrNext,
        pyrMasks,
        misalignedMask,
        principalFlow.coeffsNext,
        +1.0,
        params,
        rawResNext,
        validNext
    );

    SecondaryFlowResult result;
    result.backwardFlow = FlowField(baseW, baseH);
    result.forwardFlow = FlowField(baseW, baseH);
    result.residualFlowPrev = FlowField(baseW, baseH);
    result.residualFlowNext = FlowField(baseW, baseH);
    result.validFlowMask.assign(basePixels, 0);

    for (int y = 0; y < baseH; ++y) {
        for (int x = 0; x < baseW; ++x) {
            const size_t idx = static_cast<size_t>(y) * baseW + x;

            const float muPrevU = misalignedMask[idx] ? rawResPrev.u[idx] : 0.0f;
            const float muPrevV = misalignedMask[idx] ? rawResPrev.v[idx] : 0.0f;
            const float muNextU = misalignedMask[idx] ? rawResNext.u[idx] : 0.0f;
            const float muNextV = misalignedMask[idx] ? rawResNext.v[idx] : 0.0f;

            result.residualFlowPrev.u[idx] = muPrevU;
            result.residualFlowPrev.v[idx] = muPrevV;
            result.residualFlowNext.u[idx] = muNextU;
            result.residualFlowNext.v[idx] = muNextV;

            result.backwardFlow.u[idx] = principalFlow.backwardFlow.u[idx] + muPrevU;
            result.backwardFlow.v[idx] = principalFlow.backwardFlow.v[idx] + muPrevV;
            result.forwardFlow.u[idx] = principalFlow.forwardFlow.u[idx] + muNextU;
            result.forwardFlow.v[idx] = principalFlow.forwardFlow.v[idx] + muNextV;

            if (validPrev[idx] && validNext[idx]) {
                result.validFlowMask[idx] = 1;
                result.validPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    return result;
}