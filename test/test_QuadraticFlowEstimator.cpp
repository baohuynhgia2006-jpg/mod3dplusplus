#include "QuadraticFlowEstimator.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

namespace {

// Continuous multi-frequency texture for exact sub-pixel synthetic frame generation
float evalTexture(double x, double y) {
    double val = 0.5
        + 0.18 * std::sin(0.11 * x + 0.05 * y)
        + 0.14 * std::cos(0.07 * x - 0.13 * y)
        + 0.10 * std::sin(0.23 * x) * std::cos(0.19 * y)
        + 0.08 * std::cos(0.03 * (x * x * 0.02 + y));
    return static_cast<float>(val);
}

void evalQuadraticAt(
    double x,
    double y,
    int w,
    int h,
    const std::array<double, 12>& coeffs,
    double& u,
    double& v) {
    const double cx = 0.5 * (w - 1);
    const double cy = 0.5 * (h - 1);
    const double invScale = 1.0 / (0.5 * std::max(w, h));
    const double nx = (x - cx) * invScale;
    const double ny = (y - cy) * invScale;
    const double b[6] = {1.0, nx, ny, nx * nx, nx * ny, ny * ny};

    u = 0.0;
    v = 0.0;
    for (int k = 0; k < 6; ++k) {
        u += coeffs[static_cast<size_t>(k)] * b[k];
        v += coeffs[static_cast<size_t>(k + 6)] * b[k];
    }
}

// Synthesizes prev (t-1), curr (t), and next (t+1) such that:
// next(x + u(x,y), y + v(x,y)) == curr(x,y)
// prev(x - u(x,y), y - v(x,y)) == curr(x,y)
void generateSyntheticTuple(
    int w,
    int h,
    const std::array<double, 12>& trueCoeffs,
    FloatImage& prev,
    FloatImage& curr,
    FloatImage& next) {
    prev = FloatImage(w, h, 1);
    curr = FloatImage(w, h, 1);
    next = FloatImage(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            curr.at(y, x) = evalTexture(x, y);

            // Invert Phi_+(x, y) = (x + u(x,y), y + v(x,y)) via fixed-point iteration
            double sxNext = static_cast<double>(x);
            double syNext = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxNext, syNext, w, h, trueCoeffs, u, v);
                sxNext = static_cast<double>(x) - u;
                syNext = static_cast<double>(y) - v;
            }
            next.at(y, x) = evalTexture(sxNext, syNext);

            // Invert Phi_-(x, y) = (x - u(x,y), y - v(x,y)) via fixed-point iteration
            double sxPrev = static_cast<double>(x);
            double syPrev = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxPrev, syPrev, w, h, trueCoeffs, u, v);
                sxPrev = static_cast<double>(x) + u;
                syPrev = static_cast<double>(y) + v;
            }
            prev.at(y, x) = evalTexture(sxPrev, syPrev);
        }
    }
}

double computeFlowRMSE(const FlowField& est, const FlowField& gt, int borderMargin = 10) {
    double sqSum = 0.0;
    int count = 0;
    for (int y = borderMargin; y < gt.height - borderMargin; ++y) {
        for (int x = borderMargin; x < gt.width - borderMargin; ++x) {
            double du = static_cast<double>(est.atU(y, x) - gt.atU(y, x));
            double dv = static_cast<double>(est.atV(y, x) - gt.atV(y, x));
            sqSum += du * du + dv * dv;
            count++;
        }
    }
    return std::sqrt(sqSum / static_cast<double>(count));
}

} // namespace

int main() {
    const int w = 128;
    const int h = 128;

    // Ground-truth quadratic flow: translation + affine rotation/zoom + quadratic curvature
    const std::array<double, 12> trueCoeffs = {
        3.2,  -0.45,  0.60,  0.25, -0.18,  0.12,
       -2.4,  -0.55, -0.35, -0.15,  0.22, -0.20
    };

    std::cout << "Generating synthetic 3-frame sequence with quadratic camera motion..." << std::endl;
    FloatImage prev, curr, next;
    generateSyntheticTuple(w, h, trueCoeffs, prev, curr, next);

    FlowField gtForward = QuadraticFlowEstimator::evaluateFlow(w, h, trueCoeffs, +1.0);
    FlowField gtBackward = QuadraticFlowEstimator::evaluateFlow(w, h, trueCoeffs, -1.0);

    std::cout << "Testing Stage 1 quadratic flow estimation on clean sequence..." << std::endl;
    {
        QuadraticFlowResult res = QuadraticFlowEstimator::estimate(prev, curr, next);

        double fwdRmse = computeFlowRMSE(res.forwardFlow, gtForward);
        double bwdRmse = computeFlowRMSE(res.backwardFlow, gtBackward);
        std::cout << "  Forward Flow RMSE:  " << fwdRmse << " px" << std::endl;
        std::cout << "  Backward Flow RMSE: " << bwdRmse << " px" << std::endl;

        assert(fwdRmse < 0.05);
        assert(bwdRmse < 0.05);
    }

    std::cout << "Testing robust outlier rejection with an independently moving foreground object..." << std::endl;
    {
        FloatImage prevObj = prev;
        FloatImage currObj = curr;
        FloatImage nextObj = next;

        // Paste a 28x28 bright foreground block moving in the opposite direction
        // (leftward by 4 px/frame instead of rightward by ~3.2 px/frame)
        auto drawBlock = [](FloatImage& img, int x0, int y0) {
            for (int dy = 0; dy < 28; ++dy) {
                for (int dx = 0; dx < 28; ++dx) {
                    int px = x0 + dx;
                    int py = y0 + dy;
                    if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                        img.at(py, px) = 0.95f - 0.1f * ((dx + dy) % 2);
                    }
                }
            }
        };

        drawBlock(prevObj, 54, 50);
        drawBlock(currObj, 50, 50);
        drawBlock(nextObj, 46, 50);

        QuadraticFlowResult resObj = QuadraticFlowEstimator::estimate(prevObj, currObj, nextObj);
        double bgRmse = computeFlowRMSE(resObj.forwardFlow, gtForward);
        std::cout << "  Background Flow RMSE with moving object present: " << bgRmse << " px" << std::endl;

        assert(bgRmse < 0.08);
    }

    std::cout << "Testing invalid input handling..." << std::endl;
    {
        FloatImage emptyImg;
        bool caught = false;
        try {
            QuadraticFlowEstimator::estimate(emptyImg, curr, next);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    std::cout << "\nAll Stage 1 QuadraticFlowEstimator tests passed successfully!" << std::endl;
    return 0;
}