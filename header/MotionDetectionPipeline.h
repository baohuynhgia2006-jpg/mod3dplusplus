#ifndef MOTION_DETECTION_PIPELINE_H
#define MOTION_DETECTION_PIPELINE_H

#include "DenseMotionSegmenter.h"
#include "FloatImage.h"
#include "FrameAligner.h"
#include "Image.h"
#include "ParallaxRigidityTester.h"
#include "QuadraticFlowEstimator.h"
#include "SecondaryFlowEstimator.h"
#include <vector>

struct MotionPipelineParams {
    QuadraticFlowParams quadraticParams;
    AlignmentParams alignmentParams;
    SecondaryFlowParams secondaryParams;
    ParallaxRigidityParams rigidityParams;
    DenseSegmentationParams segmentationParams;
};

struct MotionPipelineResult {
    QuadraticFlowResult principalFlow;
    AlignmentResult alignment;
    SecondaryFlowResult secondaryFlow;
    ParallaxRigidityResult rigidity;
    DenseSegmentationResult segmentation;
    std::vector<PixelCoord> usedStaticSeeds;
};

class MotionDetectionPipeline {
public:
    // Automatically selects verified static seeds via symmetric mutual Eq. (8) if none are supplied
    static MotionPipelineResult run(
        const Image& prevFrame,
        const Image& currFrame,
        const Image& nextFrame,
        const MotionPipelineParams& params = MotionPipelineParams{},
        unsigned int rngSeed = 42
    );

    static MotionPipelineResult run(
        const Image& prevFrame,
        const Image& currFrame,
        const Image& nextFrame,
        const PixelCoord& staticRefPixel,
        const MotionPipelineParams& params = MotionPipelineParams{}
    );

    static MotionPipelineResult run(
        const Image& prevFrame,
        const Image& currFrame,
        const Image& nextFrame,
        const std::vector<PixelCoord>& staticRefPixels,
        const MotionPipelineParams& params = MotionPipelineParams{},
        unsigned int rngSeed = 42
    );

    static Image renderBinaryMask(
        const DenseSegmentationResult& segmentation,
        int width,
        int height
    );

    static Image renderOverlay(
        const Image& referenceFrame,
        const DenseSegmentationResult& segmentation,
        const std::vector<PixelCoord>& staticRefPixels = {},
        float overlayAlpha = 0.45f
    );
};

#endif // MOTION_DETECTION_PIPELINE_H