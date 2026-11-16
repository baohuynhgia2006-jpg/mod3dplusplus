#include "FrameAligner.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

inline float smoothField(double x, double y) {
    return static_cast<float>(0.008 * x + 0.005 * y + 0.15);
}

void testWarpToReferenceAndInBoundsMask() {
    const int w = 64;
    const int h = 64;

    const float uPrev = 3.0f;
    const float vPrev = -2.0f;
    const float uNext = -2.5f;
    const float vNext = 2.0f;

    FloatImage prev(w, h, 1);
    FloatImage curr(w, h, 1);
    FloatImage next(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            prev.at(y, x) = smoothField(x, y);
            next.at(y, x) = smoothField(x, y);
            curr.at(y, x) = smoothField(x + uPrev, y + vPrev);
        }
    }
    // Make next consistent with (x + uNext, y + vNext) -> curr(x, y)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            next.at(y, x) = smoothField(x + uPrev - uNext, y + vPrev - vNext);
        }
    }

    QuadraticFlowResult qFlow;
    qFlow.backwardFlow = FlowField(w, h);
    qFlow.forwardFlow = FlowField(w, h);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        qFlow.backwardFlow.u[i] = uPrev;
        qFlow.backwardFlow.v[i] = vPrev;
        qFlow.forwardFlow.u[i] = uNext;
        qFlow.forwardFlow.v[i] = vNext;
    }

    AlignmentParams params;
    params.borderMargin = 1.0f;

    AlignmentResult res = FrameAligner::warpToReference(prev, curr, next, qFlow, params);

    // Border pixels warped out of bounds must have inBoundsMask == 0
    if (res.inBoundsMask[0] != 0 || res.inBoundsMask[static_cast<size_t>(h - 1) * w + (w - 1)] != 0) {
        throw std::runtime_error("Expected out-of-bounds border pixels to have inBoundsMask == 0.");
    }

    // Interior valid pixels must match curr closely after bilinear warping
    for (int y = 8; y < h - 8; ++y) {
        for (int x = 8; x < w - 8; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (res.inBoundsMask[idx] != 1) {
                throw std::runtime_error("Expected interior pixel to have inBoundsMask == 1.");
            }
            if (std::fabs(res.warpedPrev.at(y, x) - curr.at(y, x)) > 1e-4f ||
                std::fabs(res.warpedNext.at(y, x) - curr.at(y, x)) > 1e-4f) {
                throw std::runtime_error("Warped frame intensity mismatch at interior pixel.");
            }
        }
    }
}

void testLegacyAlignAndDetectMisaligned() {
    const int w = 64;
    const int h = 64;

    FloatImage prev(w, h, 1);
    FloatImage curr(w, h, 1);
    FloatImage next(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float val = smoothField(x, y);
            prev.at(y, x) = val;
            curr.at(y, x) = val;
            next.at(y, x) = val;
        }
    }

    // Inject localized moving intensity block in curr
    for (int y = 26; y < 38; ++y) {
        for (int x = 26; x < 38; ++x) {
            curr.at(y, x) = 0.95f;
        }
    }

    QuadraticFlowResult zeroFlow;
    zeroFlow.backwardFlow = FlowField(w, h);
    zeroFlow.forwardFlow = FlowField(w, h);

    AlignmentResult res = FrameAligner::alignAndDetectMisaligned(prev, curr, next, zeroFlow);
    if (res.misalignedMask[static_cast<size_t>(32) * w + 32] != 1) {
        throw std::runtime_error("Legacy alignAndDetectMisaligned failed to detect anomalous block.");
    }
    if (res.misalignedMask[static_cast<size_t>(10) * w + 10] != 0) {
        throw std::runtime_error("Legacy alignAndDetectMisaligned falsely flagged static background.");
    }
}

} // namespace

int main() {
    try {
        testWarpToReferenceAndInBoundsMask();
        testLegacyAlignAndDetectMisaligned();
        std::cout << "[PASS] test_FrameAligner\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] test_FrameAligner: " << e.what() << "\n";
        return 1;
    }
}