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
    // Effective threshold = max(minResidualThreshold, madMultiplier * 1.4826 * MAD(residuals))
    bool useAdaptiveThreshold = true;
    double madMultiplier = 3.5;
    double minResidualThreshold = 0.03; // Assumes [0, 1] normalized float intensities

    // Optional spatial gradient normalization to suppress sub-pixel interpolation
    // artifacts along sharp static edges: |I_curr - I_warped| / sqrt(Ix^2 + Iy^2 + gradientEps^2)
    bool normalizeByGradient = false;
    double gradientEps = 0.05;

    // If true, pixels warped out-of-bounds in either prev or next are excluded from detection.
    // If false, pixels valid in at least one neighboring frame are evaluated using valid frame(s).
    bool requireBothValid = false;
};

struct AlignmentResult {
    FloatImage warpedPrev;                    // prev (t-1) aligned to curr (t)
    FloatImage warpedNext;                    // next (t+1) aligned to curr (t)
    std::vector<uint8_t> validPrev;           // 1 where backward warp fell inside prev bounds
    std::vector<uint8_t> validNext;           // 1 where forward warp fell inside next bounds
    FloatImage residualMap;                   // Per-pixel alignment residual error
    double computedThreshold = 0.0;           // Threshold applied to classify misalignment
    std::vector<uint8_t> misalignedMask;      // 1 = misaligned pixel, 0 = aligned or out-of-bounds
    std::vector<PixelCoord> misalignedPixels; // Coordinates (x, y) where misalignedMask == 1
};

class FrameAligner {
public:
    static AlignmentResult alignAndDetectMisaligned(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowResult& principalFlow,
        const AlignmentParams& params = AlignmentParams{}
    );

    static AlignmentResult alignAndDetectMisaligned(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const FlowField& forwardFlow,
        const FlowField& backwardFlow,
        const AlignmentParams& params = AlignmentParams{}
    );
};

#endif // FRAME_ALIGNER_H