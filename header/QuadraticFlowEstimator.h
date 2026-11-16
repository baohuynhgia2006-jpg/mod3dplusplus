#ifndef QUADRATIC_FLOW_ESTIMATOR_H
#define QUADRATIC_FLOW_ESTIMATOR_H

#include "FloatImage.h"
#include "FlowField.h"

struct QuadraticCoeffs {
    // 8-DOF coupled planar-quadratic flow model:
    //   u(x, y) = a + b*x + c*y + g*x^2 + h*x*y
    //   v(x, y) = d + e*x + f*y + g*x*y + h*y^2
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double e = 0.0;
    double f = 0.0;
    double g = 0.0;
    double h = 0.0;

    inline double evalU(double x, double y) const {
        return a + b * x + c * y + g * x * x + h * x * y;
    }

    inline double evalV(double x, double y) const {
        return d + e * x + f * y + g * x * y + h * y * y;
    }
};

struct QuadraticFlowParams {
    int pyramidLevels = 4;
    int minPyramidDim = 16;
    int maxIterationsPerLevel = 15;
    double convergenceEps = 1e-4;
    double tukeyC = 4.685;
    double minGradientNorm = 1e-4;
    double regularizationDamping = 1e-6;
    int borderMargin = 2;
};

struct QuadraticFlowResult {
    FlowField backwardFlow;       // curr -> prev: u_0^j(x, y), v_0^j(x, y)
    FlowField forwardFlow;        // curr -> next: u_0^k(x, y), v_0^k(x, y)
    QuadraticCoeffs coeffsPrev;   // 8 coupled pixel-space coefficients for curr -> prev
    QuadraticCoeffs coeffsNext;   // 8 coupled pixel-space coefficients for curr -> next
};

class QuadraticFlowEstimator {
public:
    static QuadraticFlowResult estimate(
        const FloatImage& prev,
        const FloatImage& curr,
        const FloatImage& next,
        const QuadraticFlowParams& params = QuadraticFlowParams{}
    );

    static FlowField estimatePair(
        const FloatImage& ref,
        const FloatImage& target,
        QuadraticCoeffs& outCoeffs,
        const QuadraticFlowParams& params = QuadraticFlowParams{}
    );
};

#endif // QUADRATIC_FLOW_ESTIMATOR_H