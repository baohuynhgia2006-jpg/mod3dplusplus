#include "MotionDetectionPipeline.h"
#include "ImageConverter.h"
#include <algorithm>
#include <stdexcept>

MotionPipelineResult MotionDetectionPipeline::run(
    const Image& prevFrame,
    const Image& currFrame,
    const Image& nextFrame,
    const PixelCoord& staticRefPixel,
    const MotionPipelineParams& params) {
    return run(
        prevFrame,
        currFrame,
        nextFrame,
        std::vector<PixelCoord>{staticRefPixel},
        params
    );
}

MotionPipelineResult MotionDetectionPipeline::run(
    const Image& prevFrame,
    const Image& currFrame,
    const Image& nextFrame,
    const std::vector<PixelCoord>& staticRefPixels,
    const MotionPipelineParams& params) {
    if (prevFrame.empty() || currFrame.empty() || nextFrame.empty()) {
        throw std::invalid_argument("All three input frames must be non-empty.");
    }
    if (prevFrame.width != currFrame.width || prevFrame.height != currFrame.height ||
        nextFrame.width != currFrame.width || nextFrame.height != currFrame.height) {
        throw std::invalid_argument("All three input frames must have identical dimensions.");
    }
    if (staticRefPixels.empty()) {
        throw std::invalid_argument("At least one verified static reference pixel is required.");
    }

    // Pre-processing: Convert 8-bit RGB/grayscale frames to normalized [0, 1] float luminance
    FloatImage prevGray = ImageConverter::toFloatGrayscale(prevFrame, true);
    FloatImage currGray = ImageConverter::toFloatGrayscale(currFrame, true);
    FloatImage nextGray = ImageConverter::toFloatGrayscale(nextFrame, true);

    MotionPipelineResult result;
    result.usedStaticSeeds = staticRefPixels;

    // Stage 1: Fit principal 12-parameter quadratic optical flow field
    result.principalFlow = QuadraticFlowEstimator::estimate(
        prevGray,
        currGray,
        nextGray,
        params.quadraticParams
    );

    // Stage 2: Warp frames using principal flow and extract misaligned pixels
    result.alignment = FrameAligner::alignAndDetectMisaligned(
        prevGray,
        currGray,
        nextGray,
        result.principalFlow,
        params.alignmentParams
    );

    // Stage 3: Fit secondary pairwise optical flow fields over the misaligned region
    result.secondaryFlow = SecondaryFlowEstimator::estimate(
        prevGray,
        currGray,
        nextGray,
        result.principalFlow,
        result.alignment,
        params.secondaryParams
    );

    // Stage 4: Perform 3-frame planar-parallax rigidity test (Theorem 2, Eq. 8)
    result.rigidity = ParallaxRigidityTester::test(
        result.principalFlow,
        result.alignment,
        result.secondaryFlow,
        staticRefPixels,
        params.rigidityParams
    );

    // Stage 5: Generalize sparse moving detections across the full object via Graph Cut MRF
    result.segmentation = DenseMotionSegmenter::segment(
        currFrame,
        result.alignment,
        result.secondaryFlow,
        result.rigidity,
        staticRefPixels,
        params.segmentationParams
    );

    return result;
}

Image MotionDetectionPipeline::renderBinaryMask(
    const DenseSegmentationResult& segmentation,
    int width,
    int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Output mask dimensions must be positive.");
    }
    if (segmentation.denseMovingMask.size() != static_cast<size_t>(width) * height) {
        throw std::invalid_argument("Segmentation mask size does not match width * height.");
    }

    Image maskImg(width, height, 1);
    for (size_t i = 0; i < maskImg.data.size(); ++i) {
        maskImg.data[i] = segmentation.denseMovingMask[i] ? 255 : 0;
    }
    return maskImg;
}

Image MotionDetectionPipeline::renderOverlay(
    const Image& referenceFrame,
    const DenseSegmentationResult& segmentation,
    const std::vector<PixelCoord>& staticRefPixels,
    float overlayAlpha) {
    if (referenceFrame.empty()) {
        throw std::invalid_argument("Reference frame must not be empty.");
    }

    const int w = referenceFrame.width;
    const int h = referenceFrame.height;
    if (segmentation.denseMovingMask.size() != static_cast<size_t>(w) * h) {
        throw std::invalid_argument("Segmentation mask size does not match referenceFrame.");
    }

    const float alpha = std::clamp(overlayAlpha, 0.0f, 1.0f);
    const float invAlpha = 1.0f - alpha;

    Image vis(w, h, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t r = referenceFrame.at(y, x, 0);
            const uint8_t g = (referenceFrame.channels >= 3) ? referenceFrame.at(y, x, 1) : r;
            const uint8_t b = (referenceFrame.channels >= 3) ? referenceFrame.at(y, x, 2) : r;

            const size_t idx = static_cast<size_t>(y) * w + x;
            if (segmentation.denseMovingMask[idx]) {
                vis.at(y, x, 0) = static_cast<uint8_t>(std::min(255.0f, invAlpha * r + alpha * 255.0f));
                vis.at(y, x, 1) = static_cast<uint8_t>(invAlpha * g);
                vis.at(y, x, 2) = static_cast<uint8_t>(invAlpha * b);
            } else {
                vis.at(y, x, 0) = r;
                vis.at(y, x, 1) = g;
                vis.at(y, x, 2) = b;
            }
        }
    }

    // Draw small 3x3 green crosshairs at the verified static seed coordinates
    for (const auto& seed : staticRefPixels) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx != 0 && dy != 0) continue;
                const int px = seed.x + dx;
                const int py = seed.y + dy;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    vis.at(py, px, 0) = 0;
                    vis.at(py, px, 1) = 255;
                    vis.at(py, px, 2) = 0;
                }
            }
        }
    }

    return vis;
}