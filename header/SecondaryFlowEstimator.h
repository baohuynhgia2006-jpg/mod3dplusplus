#ifndef SECONDARY_FLOW_ESTIMATOR_H
#define SECONDARY_FLOW_ESTIMATOR_H

#include "FloatImage.h"
#include "FlowField.h"
#include "FrameAligner.h"
#include "QuadraticFlowEstimator.h"
#include <cstdint>
#include <vector>

struct SecondaryFlowParams {
    int pyramidLevels = 3;
    int minPyramidDim = 16;
    int windowRadius = 3;              // (2*r + 1) x (2*r + 1) Gaussian-weighted LK window
    double windowSigma = 1.5;          // Spatial Gaussian weight sigma inside the LK window
    int maskDilationRadius = 4;        // Dilates misalignedMask to cover object boundaries
    int maxIterationsPerLevel = 10;
    double convergenceEps = 1e-3;
    double minEigenvalue = 2.5e-3;     // Shi-Tomasi threshold rejecting flat/noisy asphalt
    double minEigenRatio = 0.15;       // lambda_min / lambda_max threshold rejecting 1D edges
    double maxPostWarpError = 0.05;    // Rejects pixels still occluded/unmatched after warping
    double regularizationDamping = 1e-5; // Tikhonov damping during iterative updates
    int smoothingIterations = 2;       // Light masked Jacobi smoothing on residual flow per level
    double smoothingAlpha = 0.25;      // Spatial smoothness weight in [0, 0.25]
};

struct SecondaryFlowResult {
    FlowField backwardFlow;               // Total flow curr (t) -> prev (t-1): u0_bwd + mu^j
    FlowField forwardFlow;                // Total flow curr (t) -> next (t+1): u0_fwd + mu^k
    FlowField residualFlowPrev;           // Pairwise parallax vector mu^j (curr -> prev)
    FlowField residualFlowNext;           // Pairwise parallax vector mu^k (curr -> next)
    std::vector<uint8_t> validFlowMask;   // 1 = misaligned pixel with valid 2D flow in both pairs
    std::vector<PixelCoord> validPixels;  // Coordinates (x, y) where validFlowMask == 1
};

class SecondaryFlowEstimator {
public:
    static SecondaryFlowResult estimate(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowResult& principalFlow,
        const AlignmentResult& alignment,
        const SecondaryFlowParams& params = SecondaryFlowParams{}
    );

    static SecondaryFlowResult estimate(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowResult& principalFlow,
        const std::vector<uint8_t>& misalignedMask,
        const SecondaryFlowParams& params = SecondaryFlowParams{}
    );
};

#endif // SECONDARY_FLOW_ESTIMATOR_H