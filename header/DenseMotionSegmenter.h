#ifndef DENSE_MOTION_SEGMENTER_H
#define DENSE_MOTION_SEGMENTER_H

#include "FloatImage.h"
#include "FrameAligner.h"
#include "Image.h"
#include "ParallaxRigidityTester.h"
#include "SecondaryFlowEstimator.h"
#include <cstdint>
#include <vector>

struct DenseSegmentationParams {
    // Smoothness weight lambda controlling spatial coherence vs. unary priors
    double lambda = 18.0;

    // Soft t-link capacity for pixels flagged moving by Stage 4.
    double movingSeedWeight = 12.0;

    // Hard t-link capacity for verified static pixels
    double staticSeedWeight = 1e5;

    // Static prior for pixels already aligned by Stage 1's principal flow (!misalignedMask).
    double alignedBackgroundPrior = 0.55;

    // Smaller static prior for untextured pixels inside Stage 2's misaligned region
    double unknownStaticPrior = 0.05;

    // Floor on mean squared color difference (in [0, 1] normalized units)
    double minMeanColorDiffSq = 0.005;

    // Padding (in pixels) around the active ROI bounding box for Graph Cut construction
    int roiPadding = 32;

    // Use 8-connected pixel grid (true) vs 4-connected grid (false)
    bool use8Connectivity = true;
};

struct DenseSegmentationResult {
    std::vector<uint8_t> denseMovingMask; // 1 = moving object pixel, 0 = static background
    std::vector<PixelCoord> movingPixels; // Dense coordinates (x, y) of segmented moving object(s)
    double minCutEnergy = 0.0;            // Optimal MRF cut energy
    double computedBeta = 0.0;            // Contrast normalization factor 1 / (2 * <||Ip - Iq||^2>)
};

class DenseMotionSegmenter {
public:
    static DenseSegmentationResult segment(
        const Image& referenceFrame,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        const ParallaxRigidityResult& rigidity,
        const std::vector<PixelCoord>& staticRefPixels = {},
        const DenseSegmentationParams& params = DenseSegmentationParams{}
    );

    static DenseSegmentationResult segment(
        const FloatImage& referenceFrame,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        const ParallaxRigidityResult& rigidity,
        const std::vector<PixelCoord>& staticRefPixels = {},
        const DenseSegmentationParams& params = DenseSegmentationParams{}
    );

    static DenseSegmentationResult segment(
        const FloatImage& referenceFrame,
        const std::vector<uint8_t>& movingSeedMask,
        const std::vector<uint8_t>& verifiedStaticMask,
        const DenseSegmentationParams& params = DenseSegmentationParams{},
        const std::vector<uint8_t>& misalignedMask = {}
    );
};

#endif // DENSE_MOTION_SEGMENTER_H