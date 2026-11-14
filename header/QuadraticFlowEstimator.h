#ifndef QUADRATIC_FLOW_ESTIMATOR_H
#define QUADRATIC_FLOW_ESTIMATOR_H

#include "FloatImage.h"
#include "FlowField.h"
#include <array>

struct QuadraticFlowParams {
    int pyramidLevels = 4;
    int minPyramidDim = 16;
    int maxIterationsPerLevel = 15;
    double convergenceEps = 1e-4;
    double diagDamping = 1e-6;
    bool useRobustWeights = true;
    double tukeyC = 4.685;
};

struct QuadraticFlowResult {
    // [a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5]
    std::array<double, 12> coeffs{};     // Alias for coeffsNext (curr -> next)
    std::array<double, 12> coeffsPrev{}; // Quadratic flow coefficients for curr (t) -> prev (t-1)
    std::array<double, 12> coeffsNext{}; // Quadratic flow coefficients for curr (t) -> next (t+1)
    FlowField forwardFlow;               // Displacement from curr (t) to next (t+1)
    FlowField backwardFlow;              // Displacement from curr (t) to prev (t-1)
};

class QuadraticFlowEstimator {
public:
    // Fits 12-parameter quadratic flow fields from curr (middle frame) to prev and next.
    static QuadraticFlowResult estimate(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowParams& params = QuadraticFlowParams{}
    );

    // Evaluates a 12-parameter quadratic model onto a FlowField of size (width x height),
    // multiplying the resulting vectors by vectorScale.
    static FlowField evaluateFlow(
        int width,
        int height,
        const std::array<double, 12>& coeffs,
        double vectorScale = 1.0
    );
};

#endif // QUADRATIC_FLOW_ESTIMATOR_H