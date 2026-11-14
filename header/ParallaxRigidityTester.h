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
    // Dimensionless threshold on Equation (8):
    //   | (mu_2^j . dp_w_perp^j) / (mu_1^j . dp_w_perp^j)
    //   - (mu_2^k . dp_w_perp^k) / (mu_1^k . dp_w_perp^k) | > rigidityThreshold
    double rigidityThreshold = 0.25;

    // Minimum perpendicular parallax component (in pixels) of the static seed p1
    // relative to the warped baseline dp_w: |mu_1 . dp_w_perp| / ||dp_w||.
    // Guards against division by near-zero when p2 is epipolar-collinear with p1.
    double minDenominatorParallax = 0.15;

    // When a pixel p2 is epipolar-collinear with p1 (denominator below minDenominatorParallax),
    // tests off-epipolar motion |mu_2 . dp_w_perp| / ||dp_w|| > collinearMotionThreshold (in pixels).
    double collinearMotionThreshold = 0.5;

    // Local neighborhood radius around static reference pixel(s) for averaging mu_1^j, mu_1^k
    int refWindowRadius = 1;

    // If true, evaluates all misaligned pixels; if false, restricts to Stage 3 validFlowMask.
    bool evaluateAllMisaligned = false;
};

struct ParallaxRigidityResult {
    FloatImage rigidityErrorMap;          // Equation (8) rigidity residual map
    std::vector<uint8_t> movingMask;      // 1 = moving pixel (failed Eq. 8), 0 = otherwise
    std::vector<PixelCoord> movingPixels; // Sparse set of detected moving pixels
    std::vector<uint8_t> rigidStaticMask; // 1 = static pixel (aligned background or satisfied Eq. 8)
};

class ParallaxRigidityTester {
public:
    // Evaluates Theorem 2, Equation (8) using a single verified static pixel p1.
    static ParallaxRigidityResult test(
        const QuadraticFlowResult& principalFlow,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        const PixelCoord& staticRefPixel,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );

    // Evaluates Theorem 2, Equation (8) using one or more verified static pixels.
    // When multiple static seeds are supplied, each candidate pixel p2 automatically uses
    // the static seed p1 with the strongest non-collinear baseline denominator.
    static ParallaxRigidityResult test(
        const QuadraticFlowResult& principalFlow,
        const AlignmentResult& alignment,
        const SecondaryFlowResult& secondaryFlow,
        const std::vector<PixelCoord>& staticRefPixels,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );

    // Low-level interface taking raw flow fields and masks directly.
    static ParallaxRigidityResult test(
        const FlowField& principalBackwardFlow,
        const FlowField& principalForwardFlow,
        const FlowField& residualFlowPrev,
        const FlowField& residualFlowNext,
        const std::vector<uint8_t>& misalignedMask,
        const std::vector<uint8_t>& validFlowMask,
        const std::vector<PixelCoord>& staticRefPixels,
        const ParallaxRigidityParams& params = ParallaxRigidityParams{}
    );
};

#endif // PARALLAX_RIGIDITY_TESTER_H