#include "MotionDetectionPipeline.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline uint8_t bgTexture(double x, double y) {
    const double v = 85.0
        + 35.0 * std::sin(0.24 * x + 0.11 * y)
        + 30.0 * std::cos(0.15 * x - 0.22 * y);
    return static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
}

Image renderSyntheticFrame(
    int w,
    int h,
    double camShiftX,
    double camShiftY,
    double carCenterX,
    double carCenterY) {
    Image img(w, h, 3);

    // 1. Dominant planar ground background
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t g = bgTexture(static_cast<double>(x) + camShiftX,
                                        static_cast<double>(y) + camShiftY);
            img.at(y, x, 0) = g;
            img.at(y, x, 1) = g;
            img.at(y, x, 2) = g;
        }
    }

    // 2. Static 3D off-plane structure (upper-left landmark) with rigid depth parallax
    const double parallaxScale = 1.45;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double lx = static_cast<double>(x) + parallaxScale * camShiftX;
            const double ly = static_cast<double>(y) + parallaxScale * camShiftY;
            if (lx >= 14.0 && lx <= 44.0 && ly >= 12.0 && ly <= 36.0) {
                const double pat = 110.0 + 45.0 * std::sin(0.55 * lx) * std::cos(0.55 * ly);
                const uint8_t val = static_cast<uint8_t>(std::clamp(pat, 0.0, 255.0));
                img.at(y, x, 0) = val;
                img.at(y, x, 1) = static_cast<uint8_t>(std::min(255, val + 15));
                img.at(y, x, 2) = val;
            }
        }
    }

    // 3. Independently moving vehicle with 2D-textured rim and uniform untextured interior
    const double halfW = 18.0;
    const double halfH = 12.0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double dx = static_cast<double>(x) - carCenterX;
            const double dy = static_cast<double>(y) - carCenterY;
            if (std::fabs(dx) <= halfW && std::fabs(dy) <= halfH) {
                const bool isInterior = (std::fabs(dx) <= halfW - 6.0) && (std::fabs(dy) <= halfH - 5.0);
                const uint8_t redVal = isInterior
                    ? 235
                    : static_cast<uint8_t>(185 + 55 * std::sin(0.85 * dx) * std::cos(0.85 * dy));
                img.at(y, x, 0) = redVal;
                img.at(y, x, 1) = 40;
                img.at(y, x, 2) = 40;
            }
        }
    }

    return img;
}

void testEndToEndWarpedPipeline() {
    const int w = 160;
    const int h = 120;

    // Camera translates (-1.5, +0.8) -> (0, 0) -> (+1.5, -0.8);
    // Vehicle follows a curved trajectory: (80.0, 73.0) -> (84.0, 76.0) -> (88.0, 74.5)
    Image prev = renderSyntheticFrame(w, h, -1.5,  0.8, 80.0, 73.0);
    Image curr = renderSyntheticFrame(w, h,  0.0,  0.0, 84.0, 76.0);
    Image next = renderSyntheticFrame(w, h,  1.5, -0.8, 88.0, 74.5);

    MotionPipelineParams params;
    params.rigidityParams.minInlierDistance = 20.0;

    // 1. Run with explicit static reference seed on the 3D landmark (28, 24)
    MotionPipelineResult resExplicit = MotionDetectionPipeline::run(
        prev, curr, next, PixelCoord{28, 24}, params
    );

    if (resExplicit.segmentation.denseMovingMask[static_cast<size_t>(76) * w + 84] != 1) {
        throw std::runtime_error("End-to-end pipeline missed center of moving vehicle.");
    }
    if (resExplicit.segmentation.denseMovingMask[static_cast<size_t>(95) * w + 120] != 0 ||
        resExplicit.segmentation.denseMovingMask[static_cast<size_t>(24) * w + 28] != 0) {
        throw std::runtime_error("End-to-end pipeline falsely marked static background as moving.");
    }

    // 2. Run with automatic epipole-free static seed selection (no explicit seed passed)
    MotionPipelineResult resAuto = MotionDetectionPipeline::run(prev, curr, next, params);
    if (resAuto.usedStaticSeeds.empty()) {
        throw std::runtime_error("End-to-end pipeline failed to select automatic static seeds.");
    }
    if (resAuto.segmentation.denseMovingMask[static_cast<size_t>(76) * w + 84] != 1) {
        throw std::runtime_error("Auto-seeded pipeline missed center of moving vehicle.");
    }

    // 3. Verify rendering helpers
    Image maskImg = MotionDetectionPipeline::renderBinaryMask(resExplicit.segmentation, w, h);
    Image overlayImg = MotionDetectionPipeline::renderOverlay(
        curr, resExplicit.segmentation, resExplicit.usedStaticSeeds
    );
    if (maskImg.channels != 1 || overlayImg.channels != 3) {
        throw std::runtime_error("Rendered mask/overlay channel counts are incorrect.");
    }
}

} // namespace

int main() {
    try {
        testEndToEndWarpedPipeline();
        std::cout << "[PASS] test_MotionDetectionPipeline\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] test_MotionDetectionPipeline: " << e.what() << "\n";
        return 1;
    }
}