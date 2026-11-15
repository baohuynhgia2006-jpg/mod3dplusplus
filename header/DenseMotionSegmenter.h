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
    double lambda = 16.0;
    double movingSeedWeight = 18.0;
    double staticSeedWeight = 1e5;

    // Sink unary weight for pixels verified static by Stage 4 (planar Pi_1 or Eq. 8 rigid 3D)
    double verifiedStaticWeight = 3.5;

    // Tiny Sink bias for low-texture neutral pixels so smooth car interiors flood cleanly
    double unknownStaticPrior = 0.02;

    double minMeanColorDiffSq = 0.005;
    int roiPadding = 48;
    bool use8Connectivity = true;

    // Retained for backward compatibility
    double alignedBackgroundPrior = 0.55;
};

struct DenseSegmentationResult {
    std::vector<uint8_t> denseMovingMask;
    std::vector<PixelCoord> movingPixels;
    double minCutEnergy = 0.0;
    double computedBeta = 0.0;
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