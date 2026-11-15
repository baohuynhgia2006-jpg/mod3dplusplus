#include "ParallaxRigidityTester.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

struct StaticSeedData {
    int x = 0;
    int y = 0;
    double pWarpedJx = 0.0;
    double pWarpedJy = 0.0;
    double pWarpedKx = 0.0;
    double pWarpedKy = 0.0;
    double muJx = 0.0;
    double muJy = 0.0;
    double muKx = 0.0;
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

StaticSeedData makeSeedData(
    const FlowField& principalBackwardFlow,
    const FlowField& principalForwardFlow,
    const FlowField& residualFlowPrev,
    const FlowField& residualFlowNext,
    const std::vector<uint8_t>& validFlowMask,
    int x,
    int y,
    int radius) {
    StaticSeedData s;
    s.x = x;
    s.y = y;
    s.pWarpedJx = static_cast<double>(x) + static_cast<double>(principalBackwardFlow.atU(y, x));
    s.pWarpedJy = static_cast<double>(y) + static_cast<double>(principalBackwardFlow.atV(y, x));
    s.pWarpedKx = static_cast<double>(x) + static_cast<double>(principalForwardFlow.atU(y, x));
    s.pWarpedKy = static_cast<double>(y) + static_cast<double>(principalForwardFlow.atV(y, x));

    averageResidualAround(residualFlowPrev, validFlowMask, x, y, radius, s.muJx, s.muJy);
    averageResidualAround(residualFlowNext, validFlowMask, x, y, radius, s.muKx, s.muKy);
    return s;
}

bool isSymmetricEq8Consistent(
    const StaticSeedData& s,
    int x,
    int y,
    const FlowField& principalBackwardFlow,
    const FlowField& principalForwardFlow,
    const FlowField& residualFlowPrev,
    const FlowField& residualFlowNext,
    double minEvalDenom,
    double rigidityThreshold) {
    const double p2WarpedJx = static_cast<double>(x) + static_cast<double>(principalBackwardFlow.atU(y, x));
    const double p2WarpedJy = static_cast<double>(y) + static_cast<double>(principalBackwardFlow.atV(y, x));
    const double p2WarpedKx = static_cast<double>(x) + static_cast<double>(principalForwardFlow.atU(y, x));
    const double p2WarpedKy = static_cast<double>(y) + static_cast<double>(principalForwardFlow.atV(y, x));

    const double dpwJx = p2WarpedJx - s.pWarpedJx;
    const double dpwJy = p2WarpedJy - s.pWarpedJy;
    const double normJ = std::hypot(dpwJx, dpwJy);

    const double dpwKx = p2WarpedKx - s.pWarpedKx;
    const double dpwKy = p2WarpedKy - s.pWarpedKy;
    const double normK = std::hypot(dpwKx, dpwKy);

    if (normJ < 1e-6 || normK < 1e-6) {
        return false;
    }

    const double perpJx = -dpwJy;
    const double perpJy =  dpwJx;
    const double perpKx = -dpwKy;
    const double perpKy =  dpwKx;

    const double mu2Jx = static_cast<double>(residualFlowPrev.atU(y, x));
    const double mu2Jy = static_cast<double>(residualFlowPrev.atV(y, x));
    const double mu2Kx = static_cast<double>(residualFlowNext.atU(y, x));
    const double mu2Ky = static_cast<double>(residualFlowNext.atV(y, x));

    const double numJ = mu2Jx * perpJx + mu2Jy * perpJy;
    const double denJ = s.muJx * perpJx + s.muJy * perpJy;

    const double numK = mu2Kx * perpKx + mu2Ky * perpKy;
    const double denK = s.muKx * perpKx + s.muKy * perpKy;

    const double perpDenJ = std::fabs(denJ) / normJ;
    const double perpDenK = std::fabs(denK) / normK;
    if (std::min(perpDenJ, perpDenK) < minEvalDenom) {
        return false;
    }

    const double ratioJ = numJ / denJ;
    const double ratioK = numK / denK;

    if (ratioJ * ratioK <= 0.0) {
        return false;
    }
    const double absJ = std::fabs(ratioJ);
    const double absK = std::fabs(ratioK);
    if (std::min(absJ, absK) < 0.25 || std::max(absJ, absK) > 4.0) {
        return false;
    }

    const double meanAbs = 0.5 * (absJ + absK);
    return (std::fabs(ratioJ - ratioK) / meanAbs) <= rigidityThreshold;
}

} // namespace

std::vector<PixelCoord> ParallaxRigidityTester::selectStaticSeeds(
    const QuadraticFlowResult& principalFlow,
    const AlignmentResult& alignment,
    const SecondaryFlowResult& secondaryFlow,
    unsigned int rngSeed,
    const ParallaxRigidityParams& params) {
    const int w = principalFlow.backwardFlow.width;
    const int h = principalFlow.backwardFlow.height;

    const std::vector<uint8_t>& activeMask = !alignment.inBoundsMask.empty()
        ? alignment.inBoundsMask
        : alignment.misalignedMask;

    struct CandidatePoint {
        PixelCoord pt;
        double minMag;
    };

    std::vector<CandidatePoint> offPlaneValid;
    offPlaneValid.reserve(secondaryFlow.validPixels.size());

    PixelCoord fallbackPt{w / 2, h / 2};
    double fallbackBest = -1.0;

    const double minSeedMag = std::max(0.45, params.minDenominatorParallax);

    for (const auto& pt : secondaryFlow.validPixels) {
        const size_t idx = static_cast<size_t>(pt.y) * w + pt.x;
        if (!activeMask.empty() && !activeMask[idx]) continue;
        if (!secondaryFlow.validFlowMask[idx]) continue;

        const double muJx = static_cast<double>(secondaryFlow.residualFlowPrev.atU(pt.y, pt.x));
        const double muJy = static_cast<double>(secondaryFlow.residualFlowPrev.atV(pt.y, pt.x));
        const double muKx = static_cast<double>(secondaryFlow.residualFlowNext.atU(pt.y, pt.x));
        const double muKy = static_cast<double>(secondaryFlow.residualFlowNext.atV(pt.y, pt.x));

        const double minMag = std::min(std::hypot(muJx, muJy), std::hypot(muKx, muKy));
        if (minMag > fallbackBest) {
            fallbackBest = minMag;
            fallbackPt = pt;
        }

        if (minMag < minSeedMag) continue;

        // Static 3D parallax vectors across (t -> t-1) and (t -> t+1) oppose each other
        if (muJx * muKx + muJy * muKy >= 0.0) continue;

        offPlaneValid.push_back(CandidatePoint{pt, minMag});
    }

    if (offPlaneValid.empty()) {
        return std::vector<PixelCoord>{fallbackPt};
    }

    // Cap static seed candidates at the 40th percentile of off-plane residual speeds
    // (clamped to [0.55, 2.5] px) so fast-moving vehicles are excluded from becoming seeds.
    std::vector<double> mags;
    mags.reserve(offPlaneValid.size());
    for (const auto& c : offPlaneValid) {
        mags.push_back(c.minMag);
    }
    const size_t p40Idx = (mags.size() * 40) / 100;
    std::nth_element(mags.begin(), mags.begin() + static_cast<ptrdiff_t>(p40Idx), mags.end());
    const double maxStaticSpeed = std::max(minSeedMag + 0.15, std::min(mags[p40Idx], 2.5));

    std::vector<PixelCoord> seedPool;
    seedPool.reserve(offPlaneValid.size());
    for (const auto& c : offPlaneValid) {
        if (c.minMag <= maxStaticSpeed) {
            seedPool.push_back(c.pt);
        }
    }
    if (seedPool.empty()) {
        for (const auto& c : offPlaneValid) {
            seedPool.push_back(c.pt);
        }
    }

    constexpr size_t kMaxEval = 450;
    std::vector<PixelCoord> evalSet;
    if (offPlaneValid.size() <= kMaxEval) {
        evalSet.reserve(offPlaneValid.size());
        for (const auto& c : offPlaneValid) {
            evalSet.push_back(c.pt);
        }
    } else {
        evalSet.reserve(kMaxEval);
        const double step = static_cast<double>(offPlaneValid.size()) / static_cast<double>(kMaxEval);
        for (size_t i = 0; i < kMaxEval; ++i) {
            evalSet.push_back(offPlaneValid[static_cast<size_t>(i * step)].pt);
        }
    }

    const double minDistSq = params.minInlierDistance * params.minInlierDistance;
    const double minEvalDenom = std::min(0.15, params.minDenominatorParallax);

    auto scoreCandidate = [&](const StaticSeedData& seed) -> int {
        static constexpr int kGridCols = 8;
        static constexpr int kGridRows = 6;

        std::array<bool, kGridRows * kGridCols> cellHit{};
        int occupiedCells = 0;
        int distantInliers = 0;
        int minX = w, maxX = -1, minY = h, maxY = -1;

        for (const auto& pt : evalSet) {
            const double dx = static_cast<double>(pt.x - seed.x);
            const double dy = static_cast<double>(pt.y - seed.y);
            if (dx * dx + dy * dy < minDistSq) continue;

            if (isSymmetricEq8Consistent(
                    seed, pt.x, pt.y,
                    principalFlow.backwardFlow, principalFlow.forwardFlow,
                    secondaryFlow.residualFlowPrev, secondaryFlow.residualFlowNext,
                    minEvalDenom, params.rigidityThreshold)) {
                distantInliers++;
                minX = std::min(minX, pt.x);
                maxX = std::max(maxX, pt.x);
                minY = std::min(minY, pt.y);
                maxY = std::max(maxY, pt.y);

                const int gx = std::min(kGridCols - 1, (pt.x * kGridCols) / std::max(1, w));
                const int gy = std::min(kGridRows - 1, (pt.y * kGridRows) / std::max(1, h));
                const size_t cellIdx = static_cast<size_t>(gy * kGridCols + gx);
                if (!cellHit[cellIdx]) {
                    cellHit[cellIdx] = true;
                    occupiedCells++;
                }
            }
        }

        double spanRatio = 0.0;
        if (distantInliers > 0) {
            spanRatio = (static_cast<double>(maxX - minX) / std::max(1, w))
                      + (static_cast<double>(maxY - minY) / std::max(1, h));
        }

        return occupiedCells * 1000 + static_cast<int>(500.0 * spanRatio) + distantInliers;
    };

    std::mt19937 rng(rngSeed);
    std::uniform_int_distribution<size_t> dist(0, seedPool.size() - 1);
    const int trials = std::min(std::max(1, params.seedSearchTrials), static_cast<int>(seedPool.size()));

    int bestScore = -1;
    StaticSeedData bestSeed{};
    for (int t = 0; t < trials; ++t) {
        const PixelCoord cand = seedPool[dist(rng)];
        StaticSeedData s = makeSeedData(
            principalFlow.backwardFlow, principalFlow.forwardFlow,
            secondaryFlow.residualFlowPrev, secondaryFlow.residualFlowNext,
            secondaryFlow.validFlowMask, cand.x, cand.y, params.refWindowRadius
        );
        const int score = scoreCandidate(s);
        if (score > bestScore) {
            bestScore = score;
            bestSeed = s;
        }
    }

    std::vector<PixelCoord> chosenSeeds;
    chosenSeeds.push_back(PixelCoord{bestSeed.x, bestSeed.y});

    // Add up to 2 spatially separated companion seeds from the same rigid static group
    // to eliminate collinearity blind spots across the image.
    const double minSepSq = std::pow(0.22 * std::max(w, h), 2.0);
    for (int extra = 0; extra < 2; ++extra) {
        int nextBestScore = -1;
        PixelCoord nextSeed{-1, -1};

        for (int t = 0; t < trials; ++t) {
            const PixelCoord cand = seedPool[dist(rng)];
            bool tooClose = false;
            for (const auto& existing : chosenSeeds) {
                const double dx = static_cast<double>(cand.x - existing.x);
                const double dy = static_cast<double>(cand.y - existing.y);
                if (dx * dx + dy * dy < minSepSq) {
                    tooClose = true;
                    break;
                }
            }
            if (tooClose) continue;

            if (!isSymmetricEq8Consistent(
                    bestSeed, cand.x, cand.y,
                    principalFlow.backwardFlow, principalFlow.forwardFlow,
                    secondaryFlow.residualFlowPrev, secondaryFlow.residualFlowNext,
                    minEvalDenom, params.rigidityThreshold)) {
                continue;
            }

            StaticSeedData s = makeSeedData(
                principalFlow.backwardFlow, principalFlow.forwardFlow,
                secondaryFlow.residualFlowPrev, secondaryFlow.residualFlowNext,
                secondaryFlow.validFlowMask, cand.x, cand.y, params.refWindowRadius
            );
            const int score = scoreCandidate(s);
            if (score > nextBestScore) {
                nextBestScore = score;
                nextSeed = cand;
            }
        }

        if (nextBestScore > 0) {
            chosenSeeds.push_back(nextSeed);
        } else {
            break;
        }
    }

    return chosenSeeds;
}

ParallaxRigidityResult ParallaxRigidityTester::test(
    const QuadraticFlowResult& principalFlow,
    const AlignmentResult& alignment,
    const SecondaryFlowResult& secondaryFlow,
    const PixelCoord& staticRefPixel,
    const ParallaxRigidityParams& params) {
    return test(
        principalFlow,
        alignment,
        secondaryFlow,
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
    const std::vector<uint8_t>& activeMask = !alignment.inBoundsMask.empty()
        ? alignment.inBoundsMask
        : alignment.misalignedMask;

    return test(
        principalFlow.backwardFlow,
        principalFlow.forwardFlow,
        secondaryFlow.residualFlowPrev,
        secondaryFlow.residualFlowNext,
        activeMask,
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
    const std::vector<uint8_t>& inBoundsMask,
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
    if (inBoundsMask.size() != numPixels || validFlowMask.size() != numPixels) {
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
        seeds.push_back(makeSeedData(
            principalBackwardFlow, principalForwardFlow,
            residualFlowPrev, residualFlowNext,
            validFlowMask, pt.x, pt.y, params.refWindowRadius
        ));
    }

    ParallaxRigidityResult result;
    result.rigidityErrorMap = FloatImage(w, h, 1);
    result.movingMask.assign(numPixels, 0);
    result.rigidStaticMask.assign(numPixels, 0);

    // Explicit static reference seeds are strictly static
    for (const auto& s : seeds) {
        const size_t sIdx = static_cast<size_t>(s.y) * w + s.x;
        result.rigidStaticMask[sIdx] = 1;
    }

    std::vector<uint8_t> rawMovingMask(numPixels, 0);

    // Three-way Irani & Anandan classification:
    // 1) validFlowMask == 0 or !inBoundsMask -> left NEUTRAL (0 in both masks) so Stage 5
    //    can flood across low-texture car doors without fighting false static pins.
    // 2) validFlowMask == 1 and max(||mu^j||, ||mu^k||) < planarParallaxThreshold -> planar static Pi_1.
    // 3) validFlowMask == 1 and max(||mu^j||, ||mu^k||) >= planarParallaxThreshold -> Eq. (8) test.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (result.rigidStaticMask[idx]) {
                continue;
            }
            if (!inBoundsMask[idx] || !validFlowMask[idx]) {
                continue;
            }

            const double mu2Jx = static_cast<double>(residualFlowPrev.atU(y, x));
            const double mu2Jy = static_cast<double>(residualFlowPrev.atV(y, x));
            const double mu2Kx = static_cast<double>(residualFlowNext.atU(y, x));
            const double mu2Ky = static_cast<double>(residualFlowNext.atV(y, x));

            const double maxResMag = std::max(std::hypot(mu2Jx, mu2Jy), std::hypot(mu2Kx, mu2Ky));
            if (maxResMag < params.planarParallaxThreshold) {
                result.rigidStaticMask[idx] = 1;
                continue;
            }

            const double p2WarpedJx = static_cast<double>(x) + static_cast<double>(principalBackwardFlow.atU(y, x));
            const double p2WarpedJy = static_cast<double>(y) + static_cast<double>(principalBackwardFlow.atV(y, x));
            const double p2WarpedKx = static_cast<double>(x) + static_cast<double>(principalForwardFlow.atU(y, x));
            const double p2WarpedKy = static_cast<double>(y) + static_cast<double>(principalForwardFlow.atV(y, x));

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

                const double perpJx = -dpwJy;
                const double perpJy =  dpwJx;
                const double perpKx = -dpwKy;
                const double perpKy =  dpwKx;

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
                rawMovingMask[idx] = 1;
            } else {
                result.rigidStaticMask[idx] = 1;
            }
        }
    }

    // Spatial consistency filter: prune isolated single-pixel moving spikes in 5x5
    const int minNeighbors = std::max(0, params.minMovingNeighbors);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (!rawMovingMask[idx]) continue;

            if (minNeighbors == 0) {
                result.movingMask[idx] = 1;
                result.movingPixels.push_back(PixelCoord{x, y});
                continue;
            }

            int neighbors = 0;
            const int y0 = std::max(0, y - 2);
            const int y1 = std::min(h - 1, y + 2);
            const int x0 = std::max(0, x - 2);
            const int x1 = std::min(w - 1, x + 2);
            for (int ny = y0; ny <= y1; ++ny) {
                for (int nx = x0; nx <= x1; ++nx) {
                    if (nx == x && ny == y) continue;
                    if (rawMovingMask[static_cast<size_t>(ny) * w + nx]) {
                        neighbors++;
                    }
                }
            }

            if (neighbors >= minNeighbors) {
                result.movingMask[idx] = 1;
                result.movingPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    return result;
}