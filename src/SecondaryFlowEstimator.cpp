#include "SecondaryFlowEstimator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

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

std::vector<uint8_t> downsampleMask2x(const std::vector<uint8_t>& src, int srcW, int srcH, int dstW, int dstH) {
    std::vector<uint8_t> dst(static_cast<size_t>(dstW) * dstH, 0);
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            uint8_t any = 0;
            for (int dy = 0; dy < 2; ++dy) {
                const int sy = std::min(srcH - 1, y * 2 + dy);
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = std::min(srcW - 1, x * 2 + dx);
                    if (src[static_cast<size_t>(sy) * srcW + sx]) {
                        any = 1;
                    }
                }
            }
            dst[static_cast<size_t>(y) * dstW + x] = any;
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

inline void sampleFlowBilinear(const FlowField& flow, float x, float y, float& outU, float& outV) {
    const float cx = std::clamp(x, 0.0f, static_cast<float>(flow.width - 1));
    const float cy = std::clamp(y, 0.0f, static_cast<float>(flow.height - 1));

    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(flow.width - 1, x0 + 1);
    const int y1 = std::min(flow.height - 1, y0 + 1);

    const float fx = cx - static_cast<float>(x0);
    const float fy = cy - static_cast<float>(y0);

    const size_t i00 = static_cast<size_t>(y0) * flow.width + x0;
    const size_t i10 = static_cast<size_t>(y0) * flow.width + x1;
    const size_t i01 = static_cast<size_t>(y1) * flow.width + x0;
    const size_t i11 = static_cast<size_t>(y1) * flow.width + x1;

    outU = (1.0f - fy) * ((1.0f - fx) * flow.u[i00] + fx * flow.u[i10])
         + fy          * ((1.0f - fx) * flow.u[i01] + fx * flow.u[i11]);
    outV = (1.0f - fy) * ((1.0f - fx) * flow.v[i00] + fx * flow.v[i10])
         + fy          * ((1.0f - fx) * flow.v[i01] + fx * flow.v[i11]);
}

void smoothValidFlow(
    FlowField& flow,
    const std::vector<uint8_t>& validMask,
    int iterations,
    double alpha) {
    if (iterations <= 0 || alpha <= 0.0) return;

    const int w = flow.width;
    const int h = flow.height;
    const float a = static_cast<float>(std::clamp(alpha, 0.0, 0.5));

    FlowField temp(w, h);
    constexpr int dx4[4] = {-1, 1, 0, 0};
    constexpr int dy4[4] = {0, 0, -1, 1};

    for (int it = 0; it < iterations; ++it) {
        temp.u = flow.u;
        temp.v = flow.v;

        #pragma omp parallel for schedule(static)
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                if (!validMask[idx]) continue;

                float sumU = 0.0f;
                float sumV = 0.0f;
                int count = 0;

                for (int d = 0; d < 4; ++d) {
                    const size_t nIdx = static_cast<size_t>(y + dy4[d]) * w + (x + dx4[d]);
                    if (validMask[nIdx]) {
                        sumU += temp.u[nIdx];
                        sumV += temp.v[nIdx];
                        count++;
                    }
                }

                if (count > 0) {
                    const float avgU = sumU / static_cast<float>(count);
                    const float avgV = sumV / static_cast<float>(count);
                    flow.u[idx] = (1.0f - a) * temp.u[idx] + a * avgU;
                    flow.v[idx] = (1.0f - a) * temp.v[idx] + a * avgV;
                }
            }
        }
    }
}

void estimateDenseResidualPair(
    const std::vector<FloatImage>& pyrRef,
    const std::vector<FloatImage>& pyrWarped,
    const std::vector<std::vector<uint8_t>>& pyrMask,
    const SecondaryFlowParams& params,
    FlowField& outRawResidual,
    std::vector<uint8_t>& outValidMask) {
    const int numLevels = static_cast<int>(pyrRef.size());
    const int rad = std::max(1, params.windowRadius);
    const int winDim = 2 * rad + 1;

    std::vector<float> winWeights(static_cast<size_t>(winDim * winDim), 1.0f);
    const double twoSigmaSq = 2.0 * std::max(0.25, params.windowSigma * params.windowSigma);
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            const double r2 = static_cast<double>(dx * dx + dy * dy);
            winWeights[static_cast<size_t>((dy + rad) * winDim + (dx + rad))] =
                static_cast<float>(std::exp(-r2 / twoSigmaSq));
        }
    }

    FlowField flowU_V;

    for (int lvl = numLevels - 1; lvl >= 0; --lvl) {
        const FloatImage& I0 = pyrRef[static_cast<size_t>(lvl)];
        const FloatImage& I1 = pyrWarped[static_cast<size_t>(lvl)];
        const std::vector<uint8_t>& mask = pyrMask[static_cast<size_t>(lvl)];

        const int w = I0.width;
        const int h = I0.height;
        const size_t numPixels = static_cast<size_t>(w) * h;

        if (lvl == numLevels - 1) {
            flowU_V = FlowField(w, h);
        } else {
            FlowField upsampled(w, h);
            const int prevW = flowU_V.width;
            const int prevH = flowU_V.height;
            for (int y = 0; y < h; ++y) {
                const int sy = std::min(prevH - 1, y / 2);
                for (int x = 0; x < w; ++x) {
                    const int sx = std::min(prevW - 1, x / 2);
                    const size_t dstIdx = static_cast<size_t>(y) * w + x;
                    const size_t srcIdx = static_cast<size_t>(sy) * prevW + sx;
                    upsampled.u[dstIdx] = 2.0f * flowU_V.u[srcIdx];
                    upsampled.v[dstIdx] = 2.0f * flowU_V.v[srcIdx];
                }
            }
            flowU_V = std::move(upsampled);
        }

        // Precompute reference spatial gradients on I0
        std::vector<float> gx(numPixels, 0.0f);
        std::vector<float> gy(numPixels, 0.0f);

        #pragma omp parallel for schedule(static)
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                gx[idx] = 0.5f * (I0.at(y, x + 1) - I0.at(y, x - 1));
                gy[idx] = 0.5f * (I0.at(y + 1, x) - I0.at(y - 1, x));
            }
        }

        std::vector<uint8_t> levelValid(numPixels, 0);

        #pragma omp parallel for schedule(static)
        for (int y = rad + 1; y < h - rad - 1; ++y) {
            for (int x = rad + 1; x < w - rad - 1; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                if (!mask[idx]) {
                    flowU_V.u[idx] = 0.0f;
                    flowU_V.v[idx] = 0.0f;
                    continue;
                }

                // Assemble 2x2 structure tensor G on reference frame I0
                double gxx = 0.0, gxy = 0.0, gyy = 0.0;
                double weightSum = 0.0;

                for (int wy = -rad; wy <= rad; ++wy) {
                    for (int wx = -rad; wx <= rad; ++wx) {
                        const size_t nIdx = static_cast<size_t>(y + wy) * w + (x + wx);
                        const double wgt = static_cast<double>(
                            winWeights[static_cast<size_t>((wy + rad) * winDim + (wx + rad))]
                        );
                        const double ix = static_cast<double>(gx[nIdx]);
                        const double iy = static_cast<double>(gy[nIdx]);
                        gxx += wgt * ix * ix;
                        gxy += wgt * ix * iy;
                        gyy += wgt * iy * iy;
                        weightSum += wgt;
                    }
                }

                if (weightSum <= 1e-12) continue;
                gxx /= weightSum;
                gxy /= weightSum;
                gyy /= weightSum;

                const double trace = gxx + gyy;
                const double det = gxx * gyy - gxy * gxy;
                const double disc = std::sqrt(std::max(0.0, 0.25 * trace * trace - det));
                const double lambdaMax = 0.5 * trace + disc;
                const double lambdaMin = 0.5 * trace - disc;

                if (lambdaMin < params.minEigenvalue ||
                    lambdaMax <= 1e-12 ||
                    (lambdaMin / lambdaMax) < params.minEigenRatio) {
                    flowU_V.u[idx] = 0.0f;
                    flowU_V.v[idx] = 0.0f;
                    continue;
                }

                const double regGxx = gxx + params.regularizationDamping;
                const double regGyy = gyy + params.regularizationDamping;
                const double regDet = regGxx * regGyy - gxy * gxy;
                if (std::fabs(regDet) <= 1e-15) continue;
                const double invDet = 1.0 / regDet;

                double u = static_cast<double>(flowU_V.u[idx]);
                double v = static_cast<double>(flowU_V.v[idx]);
                bool inside = true;

                for (int it = 0; it < params.maxIterationsPerLevel; ++it) {
                    const double cx = static_cast<double>(x) + u;
                    const double cy = static_cast<double>(y) + v;

                    if (cx < rad + 1 || cx > w - rad - 2 || cy < rad + 1 || cy > h - rad - 2) {
                        inside = false;
                        break;
                    }

                    double bx = 0.0, by = 0.0;
                    for (int wy = -rad; wy <= rad; ++wy) {
                        for (int wx = -rad; wx <= rad; ++wx) {
                            const size_t nIdx = static_cast<size_t>(y + wy) * w + (x + wx);
                            const double wgt = static_cast<double>(
                                winWeights[static_cast<size_t>((wy + rad) * winDim + (wx + rad))]
                            );
                            const float i1Val = sampleBilinear(
                                I1,
                                static_cast<float>(cx + wx),
                                static_cast<float>(cy + wy)
                            );
                            const double diff = static_cast<double>(I0.at(y + wy, x + wx) - i1Val);
                            bx += wgt * diff * static_cast<double>(gx[nIdx]);
                            by += wgt * diff * static_cast<double>(gy[nIdx]);
                        }
                    }

                    bx /= weightSum;
                    by /= weightSum;

                    const double du = invDet * ( regGyy * bx - gxy * by);
                    const double dv = invDet * (-gxy * bx + regGxx * by);
                    u += du;
                    v += dv;

                    if (std::hypot(du, dv) < params.convergenceEps) {
                        break;
                    }
                }

                if (!inside) {
                    flowU_V.u[idx] = 0.0f;
                    flowU_V.v[idx] = 0.0f;
                    continue;
                }

                // Check post-warp photometric match at finest level
                if (lvl == 0) {
                    const float finalVal = sampleBilinear(
                        I1,
                        static_cast<float>(x + u),
                        static_cast<float>(y + v)
                    );
                    if (std::fabs(I0.at(y, x) - finalVal) > static_cast<float>(params.maxPostWarpError)) {
                        flowU_V.u[idx] = 0.0f;
                        flowU_V.v[idx] = 0.0f;
                        continue;
                    }
                }

                flowU_V.u[idx] = static_cast<float>(u);
                flowU_V.v[idx] = static_cast<float>(v);
                levelValid[idx] = 1;
            }
        }

        if (lvl == 0) {
            outValidMask = std::move(levelValid);
        }
    }

    smoothValidFlow(flowU_V, outValidMask, params.smoothingIterations, params.smoothingAlpha);
    outRawResidual = std::move(flowU_V);
}

} // namespace

SecondaryFlowResult SecondaryFlowEstimator::estimateFromWarped(
    const FloatImage& curr,
    const AlignmentResult& alignment,
    const QuadraticFlowResult& principalFlow,
    const SecondaryFlowParams& params) {
    if (curr.empty() || alignment.warpedPrev.empty() || alignment.warpedNext.empty()) {
        throw std::invalid_argument("Reference and warped frames must not be empty.");
    }

    const int w = curr.width;
    const int h = curr.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    if (alignment.warpedPrev.width != w || alignment.warpedPrev.height != h ||
        alignment.warpedNext.width != w || alignment.warpedNext.height != h ||
        alignment.inBoundsMask.size() != numPixels) {
        throw std::invalid_argument("Warped frames and inBoundsMask must match curr dimensions.");
    }

    std::vector<FloatImage> pyrCurr;
    std::vector<FloatImage> pyrWarpedPrev;
    std::vector<FloatImage> pyrWarpedNext;
    std::vector<std::vector<uint8_t>> pyrMask;

    pyrCurr.push_back(curr);
    pyrWarpedPrev.push_back(alignment.warpedPrev);
    pyrWarpedNext.push_back(alignment.warpedNext);
    pyrMask.push_back(alignment.inBoundsMask);

    for (int lvl = 1; lvl < std::max(1, params.pyramidLevels); ++lvl) {
        const int curW = pyrCurr.back().width;
        const int curH = pyrCurr.back().height;
        if (curW / 2 < params.minPyramidDim || curH / 2 < params.minPyramidDim) {
            break;
        }
        pyrCurr.push_back(downsample2x(pyrCurr.back()));
        pyrWarpedPrev.push_back(downsample2x(pyrWarpedPrev.back()));
        pyrWarpedNext.push_back(downsample2x(pyrWarpedNext.back()));
        pyrMask.push_back(downsampleMask2x(
            pyrMask.back(), curW, curH, pyrCurr.back().width, pyrCurr.back().height
        ));
    }

    FlowField rawResPrev;
    FlowField rawResNext;
    std::vector<uint8_t> validPrev;
    std::vector<uint8_t> validNext;

    estimateDenseResidualPair(pyrCurr, pyrWarpedPrev, pyrMask, params, rawResPrev, validPrev);
    estimateDenseResidualPair(pyrCurr, pyrWarpedNext, pyrMask, params, rawResNext, validNext);

    SecondaryFlowResult result;
    result.backwardFlow = FlowField(w, h);
    result.forwardFlow = FlowField(w, h);
    result.residualFlowPrev = FlowField(w, h);
    result.residualFlowNext = FlowField(w, h);
    result.validFlowMask.assign(numPixels, 0);

    // Apply exact quadratic composition correction:
    // mu^j(p) = mu_tilde^j(p) + u_0^j(p + mu_tilde^j(p)) - u_0^j(p)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (!alignment.inBoundsMask[idx] || !validPrev[idx] || !validNext[idx]) {
                result.backwardFlow.u[idx] = principalFlow.backwardFlow.u[idx];
                result.backwardFlow.v[idx] = principalFlow.backwardFlow.v[idx];
                result.forwardFlow.u[idx] = principalFlow.forwardFlow.u[idx];
                result.forwardFlow.v[idx] = principalFlow.forwardFlow.v[idx];
                continue;
            }

            const float rawUxPrev = rawResPrev.u[idx];
            const float rawUyPrev = rawResPrev.v[idx];
            float u0PrevShiftedX = 0.0f, u0PrevShiftedY = 0.0f;
            sampleFlowBilinear(
                principalFlow.backwardFlow,
                static_cast<float>(x) + rawUxPrev,
                static_cast<float>(y) + rawUyPrev,
                u0PrevShiftedX,
                u0PrevShiftedY
            );

            const float muJx = rawUxPrev + (u0PrevShiftedX - principalFlow.backwardFlow.u[idx]);
            const float muJy = rawUyPrev + (u0PrevShiftedY - principalFlow.backwardFlow.v[idx]);

            const float rawUxNext = rawResNext.u[idx];
            const float rawUyNext = rawResNext.v[idx];
            float u0NextShiftedX = 0.0f, u0NextShiftedY = 0.0f;
            sampleFlowBilinear(
                principalFlow.forwardFlow,
                static_cast<float>(x) + rawUxNext,
                static_cast<float>(y) + rawUyNext,
                u0NextShiftedX,
                u0NextShiftedY
            );

            const float muKx = rawUxNext + (u0NextShiftedX - principalFlow.forwardFlow.u[idx]);
            const float muKy = rawUyNext + (u0NextShiftedY - principalFlow.forwardFlow.v[idx]);

            result.residualFlowPrev.u[idx] = muJx;
            result.residualFlowPrev.v[idx] = muJy;
            result.residualFlowNext.u[idx] = muKx;
            result.residualFlowNext.v[idx] = muKy;

            result.backwardFlow.u[idx] = principalFlow.backwardFlow.u[idx] + muJx;
            result.backwardFlow.v[idx] = principalFlow.backwardFlow.v[idx] + muJy;
            result.forwardFlow.u[idx] = principalFlow.forwardFlow.u[idx] + muKx;
            result.forwardFlow.v[idx] = principalFlow.forwardFlow.v[idx] + muKy;

            result.validFlowMask[idx] = 1;
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (result.validFlowMask[static_cast<size_t>(y) * w + x]) {
                result.validPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    return result;
}

SecondaryFlowResult SecondaryFlowEstimator::estimate(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const AlignmentResult& alignment,
    const SecondaryFlowParams& params) {
    if (!alignment.warpedPrev.empty() && !alignment.warpedNext.empty() && !alignment.inBoundsMask.empty()) {
        return estimateFromWarped(curr, alignment, principalFlow, params);
    }
    AlignmentResult warped = FrameAligner::warpToReference(prev, curr, next, principalFlow);
    return estimateFromWarped(curr, warped, principalFlow, params);
}

SecondaryFlowResult SecondaryFlowEstimator::estimate(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const std::vector<uint8_t>& activeMask,
    const SecondaryFlowParams& params) {
    AlignmentResult warped = FrameAligner::warpToReference(prev, curr, next, principalFlow);
    if (activeMask.size() == warped.inBoundsMask.size()) {
        for (size_t i = 0; i < activeMask.size(); ++i) {
            warped.inBoundsMask[i] = (warped.inBoundsMask[i] && activeMask[i]) ? 1 : 0;
        }
    }
    return estimateFromWarped(curr, warped, principalFlow, params);
}