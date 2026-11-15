#ifndef PARALLAX_RIGIDITY_TESTER_H
#define PARALLAX_RIGIDITY_TESTER_H

#include "FloatImage.h"
#include "FlowField.h"
#include "FrameAligner.h"
#include "QuadraticFlowEstimator.h"
#include "SecondaryFlowEstimator.h"
#include <cstdint>
#include <vector>

struct ParallaxRigidityParams {
    // Residual parallax below this threshold (in px) lies on dominant static plane Pi_1
    double planarParallaxThreshold = 0.30;

    // Normalized Theorem 2 Eq. (8) rigidity residual threshold for off-plane pixels
    double rigidityThreshold = 0.25;

    // Off-epipolar residual threshold (in px) when candidate is collinear with all static seeds
    double collinearMotionThreshold = 0.70;

    // Minimum perpendicular denominator parallax (in px) required to evaluate Eq. (8) ratio
    double minDenominatorParallax = 0.30;

    // Radius for averaging residual parallax around each static reference seed
    int refWindowRadius = 1;

    // Minimum moving neighbors in a 5x5 window to prune isolated single-pixel outliers
    int minMovingNeighbors = 2;

    // Automatic epipole-free static seed selection parameters (used when --seed is omitted)
    int seedSearchTrials = 64;
    double minInlierDistance = 80.0;

    // Legacy flag retained for backward compatibility
    bool evaluateAllMisaligned = false;
};

struct ParallaxRigidityResult {
    FloatImage rigidityErrorMap;
    std::vector<uint8_t> movingMask;       // 1 = confirmed independently moving pixel
    std::vector<uint8_t> rigidStaticMask;  // 1 = confirmed planar or rigid 3D static pixel; 0 = moving or low-texture neutral
    std::vector<PixelCoord> movingPixels;
};

class ParallaxRigidityTester {
public:
    static ParallaxRigidityResult test(
        const QuadraticFlowResult& principalFlow,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        const PixelCoord& staticRefPixel,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );

    static ParallaxRigidityResult test(
        const QuadraticFlowResult& principalFlow,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        const std::vector<PixelCoord>& staticRefPixels,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );

    static ParallaxRigidityResult test(
        const FlowField& principalBackwardFlow,
        const FlowField& principalForwardFlow,
        const FlowField& residualFlowPrev,
        const FlowField& residualFlowNext,
        const std::vector<uint8_t>& inBoundsMask,
        const std::vector<uint8_t>& validFlowMask,
        const std::vector<PixelCoord>& staticRefPixels,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );

    // Epipole-free automatic static seed selection using symmetric mutual Eq. (8) clustering
    static std::vector<PixelCoord> selectStaticSeeds(
        const QuadraticFlowResult& principalFlow,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        unsigned int rngSeed = 42,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );
};

#endif // PARALLAX_RIGIDITY_TESTER_H