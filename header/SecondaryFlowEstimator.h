#ifndef SECONDARY_FLOW_ESTIMATOR_H
#define SECONDARY_FLOW_ESTIMATOR_H

#include "FloatImage.h"
#include "FlowField.h"
#include "FrameAligner.h"
#include "QuadraticFlowEstimator.h"
#include <cstdint>
#include <vector>

struct SecondaryFlowParams {
    int pyramidLevels = 4;
    int minPyramidDim = 16;
    int windowRadius = 3;
    double windowSigma = 1.5;
    int maxIterationsPerLevel = 12;
    double convergenceEps = 1e-3;
    double minEigenvalue = 1.5e-3;
    double minEigenRatio = 0.08;
    double maxPostWarpError = 0.06;
    double regularizationDamping = 1e-5;
    int smoothingIterations = 4;
    double smoothingAlpha = 0.25;

    // Retained for compatibility with callers passing a mask dilation radius
    int maskDilationRadius = 4;
};

struct SecondaryFlowResult {
    FlowField backwardFlow;          // Total backward flow: u_0^j + mu^j
    FlowField forwardFlow;           // Total forward flow:  u_0^k + mu^k
    FlowField residualFlowPrev;      // Exact residual parallax mu^j relative to warped grid p_w^j
    FlowField residualFlowNext;      // Exact residual parallax mu^k relative to warped grid p_w^k
    std::vector<uint8_t> validFlowMask;
    std::vector<PixelCoord> validPixels;
};

class SecondaryFlowEstimator {
public:
    // Irani & Anandan Stage 3: Estimates dense residual parallax between curr (I_i)
    // and the warped frames (warpedPrev, warpedNext) across alignment.inBoundsMask,
    // applying the exact quadratic composition correction for Theorem 2 Eq. (8).
    static SecondaryFlowResult estimateFromWarped(
        const FloatImage& curr,
        const AlignmentResult& alignment,
        const QuadraticFlowResult& principalFlow,
        const SecondaryFlowParams& params = SecondaryFlowParams{}
    );

    // Backward-compatible overloads
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
        const std::vector<uint8_t>& activeMask,
        const SecondaryFlowParams& params = SecondaryFlowParams{}
    );
};

#endif // SECONDARY_FLOW_ESTIMATOR_H