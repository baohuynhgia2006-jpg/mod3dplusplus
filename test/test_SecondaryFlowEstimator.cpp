#include "FrameAligner.h"
#include "QuadraticFlowEstimator.h"
#include "SecondaryFlowEstimator.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

namespace {

float evalBgTexture(double x, double y) {
    double val = 0.5
        + 0.16 * std::sin(0.11 * x + 0.05 * y)
        + 0.14 * std::cos(0.07 * x - 0.13 * y)
        + 0.10 * std::sin(0.23 * x) * std::cos(0.19 * y);
    return static_cast<float>(val);
}

float evalObjTexture(double lx, double ly) {
    double val = 0.5
        + 0.25 * std::sin(0.45 * lx + 0.20 * ly)
        + 0.20 * std::cos(0.35 * lx - 0.40 * ly);
    return static_cast<float>(std::clamp(val, 0.05, 0.95));
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

void generateBackgroundTuple(
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
            curr.at(y, x) = evalBgTexture(x, y);

            double sxNext = static_cast<double>(x);
            double syNext = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxNext, syNext, w, h, trueCoeffs, u, v);
                sxNext = static_cast<double>(x) - u;
                syNext = static_cast<double>(y) - v;
            }
            next.at(y, x) = evalBgTexture(sxNext, syNext);

            double sxPrev = static_cast<double>(x);
            double syPrev = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxPrev, syPrev, w, h, trueCoeffs, u, v);
                sxPrev = static_cast<double>(x) + u;
                syPrev = static_cast<double>(y) + v;
            }
            prev.at(y, x) = evalBgTexture(sxPrev, syPrev);
        }
    }
}

} // namespace

int main() {
    const int w = 128;
    const int h = 128;

    const std::array<double, 12> cameraCoeffs = {
        2.2,  -0.25,  0.30,  0.10, -0.08,  0.05,
       -1.5,  -0.30, -0.20, -0.08,  0.10, -0.08
    };

    FloatImage prev, curr, next;
    generateBackgroundTuple(w, h, cameraCoeffs, prev, curr, next);

    // Superimpose an accelerating 30x30 textured object with distinct pairwise displacements:
    // curr -> prev: (-2.5, +1.5)
    // curr -> next: (+3.8, -2.2)
    const int objX0 = 48;
    const int objY0 = 48;
    const int objSize = 30;
    const double trueBwdU = -2.5;
    const double trueBwdV = +1.5;
    const double trueFwdU = +3.8;
    const double trueFwdV = -2.2;

    auto stampObject = [&](FloatImage& img, double shiftX, double shiftY) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const double lx = static_cast<double>(x) - (objX0 + shiftX);
                const double ly = static_cast<double>(y) - (objY0 + shiftY);
                if (lx >= 0.0 && lx < objSize && ly >= 0.0 && ly < objSize) {
                    img.at(y, x) = evalObjTexture(lx, ly);
                }
            }
        }
    };

    stampObject(prev, trueBwdU, trueBwdV);
    stampObject(curr, 0.0, 0.0);
    stampObject(next, trueFwdU, trueFwdV);

    std::cout << "Running Stages 1 -> 2 -> 3 with independent pairwise residual flow..." << std::endl;

    QuadraticFlowResult qFlow = QuadraticFlowEstimator::estimate(prev, curr, next);
    AlignmentResult alignRes = FrameAligner::alignAndDetectMisaligned(prev, curr, next, qFlow);
    assert(alignRes.misalignedPixels.size() > 300);

    SecondaryFlowResult secFlow = SecondaryFlowEstimator::estimate(
        prev, curr, next, qFlow, alignRes
    );

    std::cout << "  Misaligned pixels from Stage 2: " << alignRes.misalignedPixels.size() << std::endl;
    std::cout << "  Valid pairwise flow pixels in Stage 3: " << secFlow.validPixels.size() << std::endl;
    assert(secFlow.validPixels.size() > 150);

    double sqErrPrev = 0.0;
    double sqErrNext = 0.0;
    int interiorCount = 0;

    for (const auto& pt : secFlow.validPixels) {
        if (pt.x >= objX0 + 4 && pt.x < objX0 + objSize - 4 &&
            pt.y >= objY0 + 4 && pt.y < objY0 + objSize - 4) {
            const double estBwdU = static_cast<double>(secFlow.backwardFlow.atU(pt.y, pt.x));
            const double estBwdV = static_cast<double>(secFlow.backwardFlow.atV(pt.y, pt.x));
            sqErrPrev += (estBwdU - trueBwdU) * (estBwdU - trueBwdU) + (estBwdV - trueBwdV) * (estBwdV - trueBwdV);

            const double estFwdU = static_cast<double>(secFlow.forwardFlow.atU(pt.y, pt.x));
            const double estFwdV = static_cast<double>(secFlow.forwardFlow.atV(pt.y, pt.x));
            sqErrNext += (estFwdU - trueFwdU) * (estFwdU - trueFwdU) + (estFwdV - trueFwdV) * (estFwdV - trueFwdV);

            // Verify mu^j == backwardFlow - principalFlow.backwardFlow
            const double expResPrevU = estBwdU - static_cast<double>(qFlow.backwardFlow.atU(pt.y, pt.x));
            const double expResPrevV = estBwdV - static_cast<double>(qFlow.backwardFlow.atV(pt.y, pt.x));
            assert(std::fabs(static_cast<double>(secFlow.residualFlowPrev.atU(pt.y, pt.x)) - expResPrevU) < 1e-5);
            assert(std::fabs(static_cast<double>(secFlow.residualFlowPrev.atV(pt.y, pt.x)) - expResPrevV) < 1e-5);

            // Verify mu^k == forwardFlow - principalFlow.forwardFlow
            const double expResNextU = estFwdU - static_cast<double>(qFlow.forwardFlow.atU(pt.y, pt.x));
            const double expResNextV = estFwdV - static_cast<double>(qFlow.forwardFlow.atV(pt.y, pt.x));
            assert(std::fabs(static_cast<double>(secFlow.residualFlowNext.atU(pt.y, pt.x)) - expResNextU) < 1e-5);
            assert(std::fabs(static_cast<double>(secFlow.residualFlowNext.atV(pt.y, pt.x)) - expResNextV) < 1e-5);

            interiorCount++;
        }
    }

    assert(interiorCount > 100);
    const double rmsePrev = std::sqrt(sqErrPrev / static_cast<double>(interiorCount));
    const double rmseNext = std::sqrt(sqErrNext / static_cast<double>(interiorCount));
    std::cout << "  Backward Pair (curr->prev) RMSE: " << rmsePrev << " px" << std::endl;
    std::cout << "  Forward Pair  (curr->next) RMSE: " << rmseNext << " px" << std::endl;

    assert(rmsePrev < 0.25);
    assert(rmseNext < 0.25);

    std::cout << "\nAll updated Stage 3 SecondaryFlowEstimator tests passed successfully!" << std::endl;
    return 0;
}