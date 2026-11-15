#include "MotionDetectionPipeline.h"
#include "ImageConverter.h"
#include <algorithm>
#include <stdexcept>

MotionPipelineResult MotionDetectionPipeline::run(
    const Image& prevFrame,
    const Image& currFrame,
    const Image& nextFrame,
    const MotionPipelineParams& params,
    unsigned int rngSeed) {
    return run(prevFrame, currFrame, nextFrame, std::vector<PixelCoord>{}, params, rngSeed);
}

MotionPipelineResult MotionDetectionPipeline::run(
    const Image& prevFrame,
    const Image& currFrame,
    const Image& nextFrame,
    const PixelCoord& staticRefPixel,
    const MotionPipelineParams& params) {
    return run(prevFrame, currFrame, nextFrame, std::vector<PixelCoord>{staticRefPixel}, params);
}

MotionPipelineResult MotionDetectionPipeline::run(
    const Image& prevFrame,
    const Image& currFrame,
    const Image& nextFrame,
    const std::vector<PixelCoord>& staticRefPixels,
    const MotionPipelineParams& params,
    unsigned int rngSeed) {
    if (prevFrame.empty() || currFrame.empty() || nextFrame.empty()) {
        throw std::invalid_argument("All three input frames must be non-empty.");
    }
    if (prevFrame.width != currFrame.width || prevFrame.height != currFrame.height ||
        nextFrame.width != currFrame.width || nextFrame.height != currFrame.height) {
        throw std::invalid_argument("All three input frames must have identical dimensions.");
    }

    FloatImage prevGray = ImageConverter::toFloatGrayscale(prevFrame, true);
    FloatImage currGray = ImageConverter::toFloatGrayscale(currFrame, true);
    FloatImage nextGray = ImageConverter::toFloatGrayscale(nextFrame, true);

    MotionPipelineResult result;

    // Stage 1: Principal 12-parameter quadratic flow (curr -> prev, curr -> next)
    result.principalFlow = QuadraticFlowEstimator::estimate(
        prevGray,
        currGray,
        nextGray,
        params.quadraticParams
    );

    // Stage 2: Warp prev and next into curr's reference frame (Irani & Anandan)
    result.alignment = FrameAligner::warpToReference(
        prevGray,
        currGray,
        nextGray,
        result.principalFlow,
        params.alignmentParams
    );

    // Stage 3: Dense residual parallax between curr and warped frames + composition correction
    result.secondaryFlow = SecondaryFlowEstimator::estimateFromWarped(
        currGray,
        result.alignment,
        result.principalFlow,
        params.secondaryParams
    );

    // Select static reference seed(s) if not explicitly provided
    result.usedStaticSeeds = staticRefPixels.empty()
        ? ParallaxRigidityTester::selectStaticSeeds(
              result.principalFlow,
              result.alignment,
              result.secondaryFlow,
              rngSeed,
              params.rigidityParams)
        : staticRefPixels;

    // Stage 4: Three-way planar-parallax & Theorem 2 Eq. (8) rigidity test
    result.rigidity = ParallaxRigidityTester::test(
        result.principalFlow,
        result.alignment,
        result.secondaryFlow,
        result.usedStaticSeeds,
        params.rigidityParams
    );

    // Stage 5: ROI-cropped Graph Cut dense segmentation over neutral & moving regions
    result.segmentation = DenseMotionSegmenter::segment(
        currFrame,
        result.alignment,
        result.secondaryFlow,
        result.rigidity,
        result.usedStaticSeeds,
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