#include "ParallaxRigidityTester.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

struct StaticSeedData {
    int x = 0;
    int y = 0;
    double pWarpedJx = 0.0; // p1 + u0^j(p1)
    double pWarpedJy = 0.0;
    double pWarpedKx = 0.0; // p1 + u0^k(p1)
    double pWarpedKy = 0.0;
    double muJx = 0.0;      // mu_1^j
    double muJy = 0.0;
    double muKx = 0.0;      // mu_1^k
    double muKy = 0.0;
};

void averageResidualAround(
    const FlowField& residualFlow,
    const std::vector<uint8_t>& validFlowMask,
    int cx,
    int cy,
    int radius,
    double& outMuX,
    double& outMuY) {
    const int w = residualFlow.width;
    const int h = residualFlow.height;

    if (radius <= 0) {
        outMuX = static_cast<double>(residualFlow.atU(cy, cx));
        outMuY = static_cast<double>(residualFlow.atV(cy, cx));
        return;
    }

    double sumX = 0.0;
    double sumY = 0.0;
    int count = 0;

    const int yMin = std::max(0, cy - radius);
    const int yMax = std::min(h - 1, cy + radius);
    const int xMin = std::max(0, cx - radius);
    const int xMax = std::min(w - 1, cx + radius);

    for (int y = yMin; y <= yMax; ++y) {
        for (int x = xMin; x <= xMax; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (validFlowMask[idx]) {
                sumX += static_cast<double>(residualFlow.atU(y, x));
                sumY += static_cast<double>(residualFlow.atV(y, x));
                count++;
            }
        }
    }

    if (count > 0) {
        outMuX = sumX / static_cast<double>(count);
        outMuY = sumY / static_cast<double>(count);
    } else {
        outMuX = static_cast<double>(residualFlow.atU(cy, cx));
        outMuY = static_cast<double>(residualFlow.atV(cy, cx));
    }
}

} // namespace

ParallaxRigidityResult ParallaxRigidityTester::test(
    const QuadraticFlowResult& principalFlow,
    const AlignmentResult& alignment,
    const SecondaryFlowResult& secondaryFlow,
    const PixelCoord& staticRefPixel,
    const ParallaxRigidityParams& params) {
    return test(
        principalFlow.backwardFlow,
        principalFlow.forwardFlow,
        secondaryFlow.residualFlowPrev,
        secondaryFlow.residualFlowNext,
        alignment.misalignedMask,
        secondaryFlow.validFlowMask,
        std::vector<PixelCoord>{staticRefPixel},
        params
    );
}

ParallaxRigidityResult ParallaxRigidityTester::test(
    const QuadraticFlowResult& principalFlow,
    const AlignmentResult& alignment,
    const SecondaryFlowResult& secondaryFlow,
    const std::vector<PixelCoord>& staticRefPixels,
    const ParallaxRigidityParams& params) {
    return test(
        principalFlow.backwardFlow,
        principalFlow.forwardFlow,
        secondaryFlow.residualFlowPrev,
        secondaryFlow.residualFlowNext,
        alignment.misalignedMask,
        secondaryFlow.validFlowMask,
        staticRefPixels,
        params
    );
}

ParallaxRigidityResult ParallaxRigidityTester::test(
    const FlowField& principalBackwardFlow,
    const FlowField& principalForwardFlow,
    const FlowField& residualFlowPrev,
    const FlowField& residualFlowNext,
    const std::vector<uint8_t>& misalignedMask,
    const std::vector<uint8_t>& validFlowMask,
    const std::vector<PixelCoord>& staticRefPixels,
    const ParallaxRigidityParams& params) {
    if (principalBackwardFlow.empty() || principalForwardFlow.empty() ||
        residualFlowPrev.empty() || residualFlowNext.empty()) {
        throw std::invalid_argument("Input flow fields must not be empty.");
    }

    const int w = principalBackwardFlow.width;
    const int h = principalBackwardFlow.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    if (principalForwardFlow.width != w || principalForwardFlow.height != h ||
        residualFlowPrev.width != w || residualFlowPrev.height != h ||
        residualFlowNext.width != w || residualFlowNext.height != h) {
        throw std::invalid_argument("All flow fields must have identical dimensions.");
    }
    if (misalignedMask.size() != numPixels || validFlowMask.size() != numPixels) {
        throw std::invalid_argument("Mask dimensions must match flow field dimensions.");
    }
    if (staticRefPixels.empty()) {
        throw std::invalid_argument("At least one verified static reference pixel must be provided.");
    }

    std::vector<StaticSeedData> seeds;
    seeds.reserve(staticRefPixels.size());
    for (const auto& pt : staticRefPixels) {
        if (pt.x < 0 || pt.x >= w || pt.y < 0 || pt.y >= h) {
            throw std::out_of_range("staticRefPixel coordinates are out of image bounds.");
        }
        StaticSeedData s;
        s.x = pt.x;
        s.y = pt.y;
        s.pWarpedJx = static_cast<double>(pt.x) + static_cast<double>(principalBackwardFlow.atU(pt.y, pt.x));
        s.pWarpedJy = static_cast<double>(pt.y) + static_cast<double>(principalBackwardFlow.atV(pt.y, pt.x));
        s.pWarpedKx = static_cast<double>(pt.x) + static_cast<double>(principalForwardFlow.atU(pt.y, pt.x));
        s.pWarpedKy = static_cast<double>(pt.y) + static_cast<double>(principalForwardFlow.atV(pt.y, pt.x));

        averageResidualAround(residualFlowPrev, validFlowMask, pt.x, pt.y, params.refWindowRadius, s.muJx, s.muJy);
        averageResidualAround(residualFlowNext, validFlowMask, pt.x, pt.y, params.refWindowRadius, s.muKx, s.muKy);
        seeds.push_back(s);
    }

    ParallaxRigidityResult result;
    result.rigidityErrorMap = FloatImage(w, h, 1);
    result.movingMask.assign(numPixels, 0);
    result.rigidStaticMask.assign(numPixels, 0);

    // Pixels already aligned by the principal planar flow are static background seeds
    for (size_t i = 0; i < numPixels; ++i) {
        if (!misalignedMask[i]) {
            result.rigidStaticMask[i] = 1;
        }
    }

    // Verified static reference pixels are strictly static
    for (const auto& s : seeds) {
        const size_t sIdx = static_cast<size_t>(s.y) * w + s.x;
        result.rigidStaticMask[sIdx] = 1;
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            const bool isCandidate = params.evaluateAllMisaligned
                ? (misalignedMask[idx] != 0)
                : (misalignedMask[idx] != 0 && validFlowMask[idx] != 0);

            if (!isCandidate) {
                continue;
            }

            bool isSeed = false;
            for (const auto& s : seeds) {
                if (s.x == x && s.y == y) {
                    isSeed = true;
                    break;
                }
            }
            if (isSeed) {
                continue;
            }

            // Candidate point p2 warped positions and parallax vectors mu_2^j, mu_2^k
            const double p2WarpedJx = static_cast<double>(x) + static_cast<double>(principalBackwardFlow.atU(y, x));
            const double p2WarpedJy = static_cast<double>(y) + static_cast<double>(principalBackwardFlow.atV(y, x));
            const double p2WarpedKx = static_cast<double>(x) + static_cast<double>(principalForwardFlow.atU(y, x));
            const double p2WarpedKy = static_cast<double>(y) + static_cast<double>(principalForwardFlow.atV(y, x));

            const double mu2Jx = static_cast<double>(residualFlowPrev.atU(y, x));
            const double mu2Jy = static_cast<double>(residualFlowPrev.atV(y, x));
            const double mu2Kx = static_cast<double>(residualFlowNext.atU(y, x));
            const double mu2Ky = static_cast<double>(residualFlowNext.atV(y, x));

            double bestMinPerpDenom = -1.0;
            double bestEq8Error = std::numeric_limits<double>::infinity();
            double bestCollinearOffEpipolar = std::numeric_limits<double>::infinity();

            for (const auto& s : seeds) {
                const double dpwJx = p2WarpedJx - s.pWarpedJx;
                const double dpwJy = p2WarpedJy - s.pWarpedJy;
                const double normJ = std::hypot(dpwJx, dpwJy);

                const double dpwKx = p2WarpedKx - s.pWarpedKx;
                const double dpwKy = p2WarpedKy - s.pWarpedKy;
                const double normK = std::hypot(dpwKx, dpwKy);

                if (normJ < 1e-6 || normK < 1e-6) {
                    continue;
                }

                // Orthogonal vector (Delta p_w)_perp = (-Delta y_w, Delta x_w)
                const double perpJx = -dpwJy;
                const double perpJy =  dpwJx;
                const double perpKx = -dpwKy;
                const double perpKy =  dpwKx;

                // Numerators and denominators of Equation (8)
                const double numJ = mu2Jx * perpJx + mu2Jy * perpJy;
                const double denJ = s.muJx * perpJx + s.muJy * perpJy;

                const double numK = mu2Kx * perpKx + mu2Ky * perpKy;
                const double denK = s.muKx * perpKx + s.muKy * perpKy;

                const double perpDenJ = std::fabs(denJ) / normJ;
                const double perpDenK = std::fabs(denK) / normK;
                const double minPerpDen = std::min(perpDenJ, perpDenK);

                if (minPerpDen > bestMinPerpDenom) {
                    bestMinPerpDenom = minPerpDen;
                    if (minPerpDen >= params.minDenominatorParallax) {
                        const double ratioJ = numJ / denJ;
                        const double ratioK = numK / denK;
                        const double scaleNorm = std::max(1.0, 0.5 * (std::fabs(ratioJ) + std::fabs(ratioK)));
                        bestEq8Error = std::fabs(ratioJ - ratioK) / scaleNorm;
                    } else {
                        const double perpNumJ = std::fabs(numJ) / normJ;
                        const double perpNumK = std::fabs(numK) / normK;
                        bestCollinearOffEpipolar = std::max(perpNumJ, perpNumK);
                    }
                }
            }

            bool isMoving = false;
            if (bestMinPerpDenom >= params.minDenominatorParallax) {
                result.rigidityErrorMap.at(y, x) = static_cast<float>(bestEq8Error);
                isMoving = (bestEq8Error > params.rigidityThreshold);
            } else if (bestMinPerpDenom >= 0.0) {
                result.rigidityErrorMap.at(y, x) = static_cast<float>(bestCollinearOffEpipolar);
                isMoving = (bestCollinearOffEpipolar > params.collinearMotionThreshold);
            }

            if (isMoving) {
                result.movingMask[idx] = 1;
                result.rigidStaticMask[idx] = 0;
                result.movingPixels.push_back(PixelCoord{x, y});
            } else {
                result.rigidStaticMask[idx] = 1;
            }
        }
    }

    return result;
}