#include "QuadraticFlowEstimator.h"
#include <string>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

inline float syntheticPattern(double x, double y) {
    const double v = 0.5
        + 0.22 * std::sin(0.11 * x + 0.04 * y)
        + 0.18 * std::cos(0.05 * x - 0.12 * y)
        + 0.10 * std::sin(0.19 * x - 0.17 * y);
    return static_cast<float>(v);
}

void testCoupledEvaluationFormula() {
    QuadraticCoeffs c;
    c.a = 1.5;
    c.b = -0.01;
    c.c = 0.02;
    c.d = -2.0;
    c.e = 0.015;
    c.f = -0.008;
    c.g = 3.0e-5;
    c.h = -2.5e-5;

    const double x = 42.0;
    const double y = 73.0;
    const double expectedU = c.a + c.b * x + c.c * y + c.g * x * x + c.h * x * y;
    const double expectedV = c.d + c.e * x + c.f * y + c.g * x * y + c.h * y * y;

    if (std::fabs(c.evalU(x, y) - expectedU) > 1e-12 ||
        std::fabs(c.evalV(x, y) - expectedV) > 1e-12) {
        throw std::runtime_error("QuadraticCoeffs::evalU/evalV violated 8-DOF coupled formula.");
    }
}

void testCoupledQuadraticRecovery() {
    const int w = 128;
    const int h = 128;

    QuadraticCoeffs gt;
    gt.a = 1.6;
    gt.b = -0.008;
    gt.c = 0.005;
    gt.d = -1.3;
    gt.e = 0.004;
    gt.f = -0.007;
    gt.g = 2.0e-5;
    gt.h = -1.5e-5;

    FloatImage target(w, h, 1);
    FloatImage ref(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            target.at(y, x) = syntheticPattern(static_cast<double>(x), static_cast<double>(y));
            const double u = gt.evalU(static_cast<double>(x), static_cast<double>(y));
            const double v = gt.evalV(static_cast<double>(x), static_cast<double>(y));
            ref.at(y, x) = syntheticPattern(static_cast<double>(x) + u, static_cast<double>(y) + v);
        }
    }

    QuadraticFlowParams params;
    params.pyramidLevels = 4;
    params.maxIterationsPerLevel = 20;

    QuadraticCoeffs est;
    FlowField flow = QuadraticFlowEstimator::estimatePair(ref, target, est, params);

    double sumErr = 0.0;
    int count = 0;
    for (int y = 10; y < h - 10; ++y) {
        for (int x = 10; x < w - 10; ++x) {
            const double gtU = gt.evalU(x, y);
            const double gtV = gt.evalV(x, y);
            const double err = std::hypot(flow.atU(y, x) - gtU, flow.atV(y, x) - gtV);
            sumErr += err;
            count++;
        }
    }

    const double meanEpe = sumErr / static_cast<double>(count);
    if (meanEpe > 0.05) {
        throw std::runtime_error("Coupled 8-DOF flow mean EPE too high: " + std::to_string(meanEpe));
    }
    if (std::fabs(est.g - gt.g) > 1e-5 || std::fabs(est.h - gt.h) > 1e-5) {
        throw std::runtime_error("Coupled quadratic terms g/h did not converge accurately.");
    }
}

void testOutlierRobustnessThreeFrame() {
    const int w = 128;
    const int h = 128;

    const double uPrev = 1.5, vPrev = -1.0;
    const double uNext = -1.5, vNext = 1.0;

    FloatImage prev(w, h, 1);
    FloatImage curr(w, h, 1);
    FloatImage next(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            prev.at(y, x) = syntheticPattern(x, y);
            curr.at(y, x) = syntheticPattern(x + uPrev, y + vPrev);
            next.at(y, x) = syntheticPattern(x + uPrev - uNext, y + vPrev - vNext);
        }
    }

    // Inject an independently moving outlier square into curr
    for (int y = 50; y < 74; ++y) {
        for (int x = 50; x < 74; ++x) {
            curr.at(y, x) = 0.95f;
        }
    }

    QuadraticFlowResult res = QuadraticFlowEstimator::estimate(prev, curr, next);

    const double errPrev = std::hypot(res.backwardFlow.atU(20, 20) - uPrev,
                                      res.backwardFlow.atV(20, 20) - vPrev);
    const double errNext = std::hypot(res.forwardFlow.atU(20, 20) - uNext,
                                      res.forwardFlow.atV(20, 20) - vNext);

    if (errPrev > 0.15 || errNext > 0.15) {
        throw std::runtime_error("Tukey IRLS failed to reject moving foreground outlier block.");
    }
}

void testInputValidation() {
    FloatImage emptyImg;
    FloatImage validImg(32, 32, 1);
    bool threw = false;
    try {
        QuadraticFlowEstimator::estimate(emptyImg, validImg, validImg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error("Expected std::invalid_argument on empty frame.");
    }
}

} // namespace

int main() {
    try {
        testCoupledEvaluationFormula();
        testCoupledQuadraticRecovery();
        testOutlierRobustnessThreeFrame();
        testInputValidation();
        std::cout << "[PASS] test_QuadraticFlowEstimator\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] test_QuadraticFlowEstimator: " << e.what() << "\n";
        return 1;
    }
}