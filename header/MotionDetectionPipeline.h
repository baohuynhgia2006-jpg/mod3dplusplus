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
    // Executes Stages 1 -> 5 on a 3-frame RGB/grayscale tuple using a single verified
    // static reference pixel selected from the misaligned set.
    static MotionPipelineResult run(
        const Image& prevFrame,
        const Image& currFrame,
        const Image& nextFrame,
        const PixelCoord& staticRefPixel,
        const MotionPipelineParams& params = MotionPipelineParams{}
    );

    // Executes Stages 1 -> 5 using one or more verified static reference pixels.
    static MotionPipelineResult run(
        const Image& prevFrame,
        const Image& currFrame,
        const Image& nextFrame,
        const std::vector<PixelCoord>& staticRefPixels,
        const MotionPipelineParams& params = MotionPipelineParams{}
    );

    // Renders a single-channel 8-bit binary mask (0 = static, 255 = moving).
    static Image renderBinaryMask(
        const DenseSegmentationResult& segmentation,
        int width,
        int height
    );

    // Renders an RGB visualization blending a red tint over detected moving object pixels
    // and optional green markers at the verified static seed coordinates.
    static Image renderOverlay(
        const Image& referenceFrame,
        const DenseSegmentationResult& segmentation,
        const std::vector<PixelCoord>& staticRefPixels = {},
        float overlayAlpha = 0.45f
    );
};

#endif // MOTION_DETECTION_PIPELINE_H