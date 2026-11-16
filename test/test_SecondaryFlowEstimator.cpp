#include "SecondaryFlowEstimator.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

inline float texturedPattern(double x, double y) {
    return static_cast<float>(
        0.5 + 0.25 * std::sin(0.75 * x) * std::cos(0.65 * y)
            + 0.20 * std::cos(0.55 * x + 0.70 * y)
    );
}

void testEstimateFromWarpedAndCompositionCorrection() {
    const int w = 96;
    const int h = 96;

    // Principal flow with linear scale term so composition correction is exercised:
    //   u_0^j(x, y) = 1.0 + 0.02 * x
    //   v_0^j(x, y) = -0.5 + 0.01 * y
    QuadraticFlowResult principal;
    principal.backwardFlow = FlowField(w, h);
    principal.forwardFlow = FlowField(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            principal.backwardFlow.u[idx] = 1.0f + 0.02f * static_cast<float>(x);
            principal.backwardFlow.v[idx] = -0.5f + 0.01f * static_cast<float>(y);
            principal.forwardFlow.u[idx] = -1.0f - 0.02f * static_cast<float>(x);
            principal.forwardFlow.v[idx] = 0.5f - 0.01f * static_cast<float>(y);
        }
    }

    // Raw residual displacement between curr and warpedPrev / warpedNext
    const double rawMuPrevX = 2.0, rawMuPrevY = -1.0;
    const double rawMuNextX = -2.0, rawMuNextY = 1.0;

    FloatImage curr(w, h, 1);
    AlignmentResult alignment;
    alignment.warpedPrev = FloatImage(w, h, 1);
    alignment.warpedNext = FloatImage(w, h, 1);
    alignment.inBoundsMask.assign(static_cast<size_t>(w) * h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Left half: textured; Right half: flat constant intensity (untextured)
            if (x < w / 2) {
                alignment.warpedPrev.at(y, x) = texturedPattern(x, y);
                alignment.warpedNext.at(y, x) = texturedPattern(x + rawMuPrevX - rawMuNextX,
                                                                y + rawMuPrevY - rawMuNextY);
                curr.at(y, x) = texturedPattern(x + rawMuPrevX, y + rawMuPrevY);
            } else {
                alignment.warpedPrev.at(y, x) = 0.4f;
                alignment.warpedNext.at(y, x) = 0.4f;
                curr.at(y, x) = 0.4f;
            }
        }
    }

    SecondaryFlowParams params;
    params.smoothingIterations = 0; // Disable smoothing so exact point displacement is checked

    SecondaryFlowResult res = SecondaryFlowEstimator::estimateFromWarped(
        curr, alignment, principal, params
    );

    // Textured pixel (24, 48): check composition-corrected residual parallax
    // mu^j = rawMu^j + u_0^j(p + rawMu^j) - u_0^j(p)
    const int tx = 24;
    const int ty = 48;
    const size_t tIdx = static_cast<size_t>(ty) * w + tx;
    if (res.validFlowMask[tIdx] != 1) {
        throw std::runtime_error("Expected validFlowMask == 1 on textured region.");
    }

    const double expectedMuPrevX = rawMuPrevX + 0.02 * rawMuPrevX;
    const double expectedMuPrevY = rawMuPrevY + 0.01 * rawMuPrevY;
    const double err = std::hypot(res.residualFlowPrev.atU(ty, tx) - expectedMuPrevX,
                                  res.residualFlowPrev.atV(ty, tx) - expectedMuPrevY);
    if (err > 0.08) {
        throw std::runtime_error("Composition-corrected residual flow error too large: " + std::to_string(err));
    }

    // Flat untextured pixel (72, 48): structure tensor eigenvalues are 0 -> validFlowMask == 0
    const size_t flatIdx = static_cast<size_t>(48) * w + 72;
    if (res.validFlowMask[flatIdx] != 0) {
        throw std::runtime_error("Expected validFlowMask == 0 on flat untextured region.");
    }
}

} // namespace

int main() {
    try {
        testEstimateFromWarpedAndCompositionCorrection();
        std::cout << "[PASS] test_SecondaryFlowEstimator\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] test_SecondaryFlowEstimator: " << e.what() << "\n";
        return 1;
    }
}