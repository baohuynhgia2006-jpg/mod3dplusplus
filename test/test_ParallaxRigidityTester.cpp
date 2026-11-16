#include "ParallaxRigidityTester.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void testThreeWayPartitionAndSeedSelection() {
    const int w = 96;
    const int h = 96;
    const size_t numPixels = static_cast<size_t>(w) * h;

    QuadraticFlowResult principal;
    principal.backwardFlow = FlowField(w, h);
    principal.forwardFlow = FlowField(w, h);

    SecondaryFlowResult secondary;
    secondary.residualFlowPrev = FlowField(w, h);
    secondary.residualFlowNext = FlowField(w, h);
    secondary.validFlowMask.assign(numPixels, 1);

    AlignmentResult alignment;
    alignment.inBoundsMask.assign(numPixels, 1);

    // Pure camera translation: u_0^j = (2, 1), u_0^k = (-2, -1)
    for (size_t i = 0; i < numPixels; ++i) {
        principal.backwardFlow.u[i] = 2.0f;
        principal.backwardFlow.v[i] = 1.0f;
        principal.forwardFlow.u[i] = -2.0f;
        principal.forwardFlow.v[i] = -1.0f;
    }

    // Static 3D projective parallax field across the scene:
    //   mu^j(p) = gamma(p) * (p_w^j - e^j),  mu^k(p) = gamma(p) * (p_w^k - e^k)
    // with e^j = (5, 5) and e^k = (90, 90).
    const double ejX = 5.0, ejY = 5.0;
    const double ekX = 90.0, ekY = 90.0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            const double pwJx = static_cast<double>(x) + 2.0;
            const double pwJy = static_cast<double>(y) + 1.0;
            const double pwKx = static_cast<double>(x) - 2.0;
            const double pwKy = static_cast<double>(y) - 1.0;

            const double gamma = (x > 60) ? 0.03 : 0.04;
            secondary.residualFlowPrev.u[idx] = static_cast<float>(gamma * (pwJx - ejX));
            secondary.residualFlowPrev.v[idx] = static_cast<float>(gamma * (pwJy - ejY));
            secondary.residualFlowNext.u[idx] = static_cast<float>(gamma * (pwKx - ekX));
            secondary.residualFlowNext.v[idx] = static_cast<float>(gamma * (pwKy - ekY));
        }
    }

    // Region 1: Planar static patch Pi_1 at x in [10, 25], y in [70, 85] (||mu|| < 0.30)
    for (int y = 70; y <= 85; ++y) {
        for (int x = 10; x <= 25; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            secondary.residualFlowPrev.u[idx] = 0.05f;
            secondary.residualFlowPrev.v[idx] = -0.04f;
            secondary.residualFlowNext.u[idx] = -0.05f;
            secondary.residualFlowNext.v[idx] = 0.04f;
        }
    }

    // Region 2: Independently moving object at x in [34, 48], y in [54, 68]
    for (int y = 54; y <= 68; ++y) {
        for (int x = 34; x <= 48; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            secondary.residualFlowPrev.u[idx] = -3.5f;
            secondary.residualFlowPrev.v[idx] = 2.5f;
            secondary.residualFlowNext.u[idx] = 3.5f;
            secondary.residualFlowNext.v[idx] = 2.5f;
        }
    }

    // Region 3: Low-texture interior inside moving object at x in [38, 44], y in [58, 64]
    for (int y = 58; y <= 64; ++y) {
        for (int x = 38; x <= 44; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            secondary.validFlowMask[idx] = 0;
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (secondary.validFlowMask[static_cast<size_t>(y) * w + x]) {
                secondary.validPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    // 1. Test automatic epipole-free static seed selection
    ParallaxRigidityParams params;
    params.minInlierDistance = 30.0;
    std::vector<PixelCoord> autoSeeds = ParallaxRigidityTester::selectStaticSeeds(
        principal, alignment, secondary, 42, params
    );
    if (autoSeeds.empty()) {
        throw std::runtime_error("selectStaticSeeds returned empty seed vector.");
    }
    for (const auto& s : autoSeeds) {
        if (s.x >= 34 && s.x <= 48 && s.y >= 54 && s.y <= 68) {
            throw std::runtime_error("selectStaticSeeds placed a static seed on the moving object.");
        }
    }

    // 2. Test three-way classification using verified static seed (20, 20)
    ParallaxRigidityResult res = ParallaxRigidityTester::test(
        principal, alignment, secondary, PixelCoord{20, 20}, params
    );

    // Planar static Pi_1 -> rigidStaticMask == 1, movingMask == 0
    const size_t planarIdx = static_cast<size_t>(78) * w + 18;
    if (res.rigidStaticMask[planarIdx] != 1 || res.movingMask[planarIdx] != 0) {
        throw std::runtime_error("Planar static pixel Pi_1 was not classified as rigid static.");
    }

    // Rigid 3D static structure at (70, 24) -> rigidStaticMask == 1, movingMask == 0
    const size_t rigid3dIdx = static_cast<size_t>(24) * w + 70;
    if (res.rigidStaticMask[rigid3dIdx] != 1 || res.movingMask[rigid3dIdx] != 0) {
        throw std::runtime_error("Rigid 3D static structure failed Theorem 2 Eq. (8) test.");
    }

    // Moving object boundary pixel at (36, 56) -> movingMask == 1, rigidStaticMask == 0
    const size_t movingIdx = static_cast<size_t>(56) * w + 36;
    if (res.movingMask[movingIdx] != 1 || res.rigidStaticMask[movingIdx] != 0) {
        throw std::runtime_error("Independently moving pixel was not detected.");
    }

    // Low-texture interior at (41, 61) -> neutral (0 in both masks)
    const size_t neutralIdx = static_cast<size_t>(61) * w + 41;
    if (res.movingMask[neutralIdx] != 0 || res.rigidStaticMask[neutralIdx] != 0) {
        throw std::runtime_error("Low-texture pixel was not left neutral for Stage 5 Graph Cut.");
    }
}

} // namespace

int main() {
    try {
        testThreeWayPartitionAndSeedSelection();
        std::cout << "[PASS] test_ParallaxRigidityTester\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] test_ParallaxRigidityTester: " << e.what() << "\n";
        return 1;
    }
}