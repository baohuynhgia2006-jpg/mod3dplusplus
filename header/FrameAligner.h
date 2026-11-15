#ifndef FRAME_ALIGNER_H
#define FRAME_ALIGNER_H

#include "FloatImage.h"
#include "FlowField.h"
#include "QuadraticFlowEstimator.h"
#include <cstdint>
#include <vector>

struct PixelCoord {
    int x = 0;
    int y = 0;
};

struct AlignmentParams {
    // Minimum distance (in pixels) from image borders for a warped lookup to be valid
    float borderMargin = 1.0f;

    // Legacy parameters retained for backward compatibility with existing unit tests
    double madMultiplier = 3.5;
    double minResidualThreshold = 0.02;
    double maxResidualThreshold = 0.25;
    int morphCloseRadius = 1;
    int morphOpenRadius = 1;
};

struct AlignmentResult {
    FloatImage warpedPrev;                    // I_j warped into curr frame: I_j(p + u_0^j(p))
    FloatImage warpedNext;                    // I_k warped into curr frame: I_k(p + u_0^k(p))
    std::vector<uint8_t> inBoundsMask;        // 1 = both warped lookups lie inside valid image bounds
    std::vector<PixelCoord> inBoundsPixels;   // Coordinates where inBoundsMask == 1

    // Legacy fields populated only when alignAndDetectMisaligned() is invoked
    FloatImage residualErrorMap;
    std::vector<uint8_t> misalignedMask;
    std::vector<PixelCoord> misalignedPixels;
    double adaptiveThreshold = 0.0;
};

class FrameAligner {
public:
    // Irani & Anandan Stage 2: Warps prev (j) and next (k) into reference frame curr (i)
    // using the principal quadratic flow fields without intensity thresholding.
    static AlignmentResult warpToReference(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowResult& principalFlow,
        const AlignmentParams& params = AlignmentParams{}
    );

    static AlignmentResult warpToReference(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const FlowField& backwardFlow,
        const FlowField& forwardFlow,
        const AlignmentParams& params = AlignmentParams{}
    );

    // Backward-compatible wrapper that performs warping + optional residual thresholding
    static AlignmentResult alignAndDetectMisaligned(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowResult& principalFlow,
        const AlignmentParams& params = AlignmentParams{}
    );
};

#endif // FRAME_ALIGNER_H