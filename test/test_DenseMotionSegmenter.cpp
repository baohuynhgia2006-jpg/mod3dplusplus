#include "DenseMotionSegmenter.h"
#include <iostream>
#include <stdexcept>

namespace {

void testNeutralInteriorFloodAndHardSeedPinning() {
    const int w = 96;
    const int h = 96;
    const size_t numPixels = static_cast<size_t>(w) * h;

    // Dark static road background (0.15) with a bright uniform vehicle rectangle (0.85)
    FloatImage ref(w, h, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const bool onCar = (x >= 32 && x <= 62 && y >= 36 && y <= 58);
            const float val = onCar ? 0.85f : 0.15f;
            ref.at(y, x, 0) = val;
            ref.at(y, x, 1) = val;
            ref.at(y, x, 2) = val;
        }
    }

    AlignmentResult alignment;
    SecondaryFlowResult secondary;
    ParallaxRigidityResult rigidity;
    rigidity.movingMask.assign(numPixels, 0);
    rigidity.rigidStaticMask.assign(numPixels, 0);

    // Mark road outside car as verified static
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const bool onCar = (x >= 32 && x <= 62 && y >= 36 && y <= 58);
            if (!onCar) {
                rigidity.rigidStaticMask[static_cast<size_t>(y) * w + x] = 1;
            }
        }
    }

    // Seed only a thin strip on the left/right edges of the car; leave center interior neutral
    for (int y = 38; y <= 56; ++y) {
        rigidity.movingMask[static_cast<size_t>(y) * w + 33] = 1;
        rigidity.movingMask[static_cast<size_t>(y) * w + 34] = 1;
        rigidity.movingMask[static_cast<size_t>(y) * w + 60] = 1;
        rigidity.movingMask[static_cast<size_t>(y) * w + 61] = 1;
    }

    DenseSegmentationParams params;
    params.lambda = 16.0;
    params.movingSeedWeight = 18.0;
    params.verifiedStaticWeight = 3.5;
    params.unknownStaticPrior = 0.02;

    std::vector<PixelCoord> staticSeeds = {PixelCoord{10, 10}};
    DenseSegmentationResult seg = DenseMotionSegmenter::segment(
        ref, alignment, secondary, rigidity, staticSeeds, params
    );

    // Center of uniform bright car (47, 47) must be flooded as moving
    if (seg.denseMovingMask[static_cast<size_t>(47) * w + 47] != 1) {
        throw std::runtime_error("Graph Cut failed to flood across neutral uniform car interior.");
    }

    // Dark background outside car (20, 47) and static seed (10, 10) must stay 0
    if (seg.denseMovingMask[static_cast<size_t>(47) * w + 20] != 0 ||
        seg.denseMovingMask[static_cast<size_t>(10) * w + 10] != 0) {
        throw std::runtime_error("Graph Cut leaked across high-contrast boundary into static background.");
    }
}

void testEmptyMovingMaskEarlyExit() {
    const int w = 32;
    const int h = 32;
    FloatImage ref(w, h, 1);
    AlignmentResult alignment;
    SecondaryFlowResult secondary;
    ParallaxRigidityResult rigidity;
    rigidity.movingMask.assign(static_cast<size_t>(w) * h, 0);
    rigidity.rigidStaticMask.assign(static_cast<size_t>(w) * h, 1);

    DenseSegmentationResult seg = DenseMotionSegmenter::segment(ref, alignment, secondary, rigidity);
    if (!seg.movingPixels.empty()) {
        throw std::runtime_error("Expected empty movingPixels when movingMask is all zeros.");
    }
}

} // namespace

int main() {
    try {
        testNeutralInteriorFloodAndHardSeedPinning();
        testEmptyMovingMaskEarlyExit();
        std::cout << "[PASS] test_DenseMotionSegmenter\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] test_DenseMotionSegmenter: " << e.what() << "\n";
        return 1;
    }
}