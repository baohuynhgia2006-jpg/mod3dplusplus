#include "FrameAligner.h"
#include "QuadraticFlowEstimator.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

namespace {

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

            double sxNext = static_cast<double>(x);
            double syNext = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxNext, syNext, w, h, trueCoeffs, u, v);
                sxNext = static_cast<double>(x) - u;
                syNext = static_cast<double>(y) - v;
            }
            next.at(y, x) = evalTexture(sxNext, syNext);

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

} // namespace

int main() {
    const int w = 128;
    const int h = 128;

    const std::array<double, 12> cameraCoeffs = {
        2.8,  -0.35,  0.40,  0.15, -0.10,  0.08,
       -2.1,  -0.40, -0.25, -0.10,  0.14, -0.12
    };

    FloatImage prev, curr, next;
    generateSyntheticTuple(w, h, cameraCoeffs, prev, curr, next);

    std::cout << "Testing Stage 2 alignment on pure background camera motion..." << std::endl;
    {
        QuadraticFlowResult qFlow = QuadraticFlowEstimator::estimate(prev, curr, next);
        AlignmentResult alignRes = FrameAligner::alignAndDetectMisaligned(prev, curr, next, qFlow);

        std::cout << "  Misaligned pixels detected (expected 0): "
                  << alignRes.misalignedPixels.size() << std::endl;
        assert(alignRes.misalignedPixels.empty());
    }

    std::cout << "Testing Stage 2 misalignment extraction with an independently moving object..." << std::endl;
    {
        FloatImage prevObj = prev;
        FloatImage currObj = curr;
        FloatImage nextObj = next;

        // Moving 24x24 textured square centered around (50..73, 50..73) in curr
        auto drawMovingObject = [](FloatImage& img, int x0, int y0) {
            for (int dy = 0; dy < 24; ++dy) {
                for (int dx = 0; dx < 24; ++dx) {
                    int px = x0 + dx;
                    int py = y0 + dy;
                    if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                        img.at(py, px) = ((dx / 4 + dy / 4) % 2 == 0) ? 0.95f : 0.10f;
                    }
                }
            }
        };

        drawMovingObject(prevObj, 55, 50);
        drawMovingObject(currObj, 50, 50);
        drawMovingObject(nextObj, 45, 50);

        QuadraticFlowResult qFlow = QuadraticFlowEstimator::estimate(prevObj, currObj, nextObj);
        AlignmentResult alignRes = FrameAligner::alignAndDetectMisaligned(prevObj, currObj, nextObj, qFlow);

        std::cout << "  Computed threshold: " << alignRes.computedThreshold << std::endl;
        std::cout << "  Total misaligned pixels: " << alignRes.misalignedPixels.size() << std::endl;

        int insideObjHits = 0;
        int farBackgroundFalsePositives = 0;

        for (const auto& pt : alignRes.misalignedPixels) {
            // Allow small 5-pixel halo around the object's sweep across [t-1, t+1]
            bool nearObjectSweep = (pt.x >= 40 && pt.x <= 84 && pt.y >= 45 && pt.y <= 79);
            if (nearObjectSweep) {
                insideObjHits++;
            } else {
                farBackgroundFalsePositives++;
            }
        }

        std::cout << "  Object region hits: " << insideObjHits
                  << ", Far-background false positives: " << farBackgroundFalsePositives << std::endl;

        assert(insideObjHits > 350);
        assert(farBackgroundFalsePositives == 0);
    }

    std::cout << "Testing gradient-normalized residual mode..." << std::endl;
    {
        AlignmentParams normParams;
        normParams.normalizeByGradient = true;
        normParams.minResidualThreshold = 0.15;

        QuadraticFlowResult qFlow = QuadraticFlowEstimator::estimate(prev, curr, next);
        AlignmentResult alignRes = FrameAligner::alignAndDetectMisaligned(prev, curr, next, qFlow, normParams);
        assert(alignRes.misalignedPixels.empty());
    }

    std::cout << "Testing invalid input handling..." << std::endl;
    {
        FlowField badFlow(64, 64);
        FlowField goodFlow(w, h);
        bool caught = false;
        try {
            FrameAligner::alignAndDetectMisaligned(prev, curr, next, badFlow, goodFlow);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    std::cout << "\nAll Stage 2 FrameAligner tests passed successfully!" << std::endl;
    return 0;
}