#include "ParallaxRigidityTester.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

struct CameraPose3D {
    double tx, ty, tz;
    double wx, wy, wz;
};

// Projects a 3D point P = (X, Y, Z) in the reference camera frame into a camera
// moved by small rotation W = (wx, wy, wz) and translation T = (tx, ty, tz).
void projectPoint(
    double X,
    double Y,
    double Z,
    const CameraPose3D& pose,
    double f,
    double cx,
    double cy,
    double& outU,
    double& outV) {
    // Apparent point motion in camera frame: P' = P - T - W x P
    const double Xp = X - pose.tx - (pose.wy * Z - pose.wz * Y);
    const double Yp = Y - pose.ty - (pose.wz * X - pose.wx * Z);
    const double Zp = Z - pose.tz - (pose.wx * Y - pose.wy * X);

    outU = f * (Xp / Zp) + cx;
    outV = f * (Yp / Zp) + cy;
}

} // namespace

int main() {
    const int w = 100;
    const int h = 100;
    const size_t numPixels = static_cast<size_t>(w) * h;

    const double f = 100.0;
    const double cx = 50.0;
    const double cy = 50.0;
    const double Z_plane = 20.0; // Dominant background plane depth

    // Non-uniform 3D camera motion across (j = t-1, curr = t, k = t+1)
    // Includes 3D translation (Tx, Ty, Tz) and 3D rotation (Wx, Wy, Wz)
    const CameraPose3D poseJ{-1.6,  1.1, -0.45, -0.012,  0.008, -0.015};
    const CameraPose3D poseK{ 2.3, -1.4,  0.65,  0.015, -0.010,  0.020};

    // Exact principal planar flow u0^j and u0^k induced by the reference plane Z = Z_plane
    FlowField u0J(w, h);
    FlowField u0K(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double X = (x - cx) * Z_plane / f;
            const double Y = (y - cy) * Z_plane / f;

            double xj, yj, xk, yk;
            projectPoint(X, Y, Z_plane, poseJ, f, cx, cy, xj, yj);
            projectPoint(X, Y, Z_plane, poseK, f, cx, cy, xk, yk);

            u0J.atU(y, x) = static_cast<float>(xj - x);
            u0J.atV(y, x) = static_cast<float>(yj - y);
            u0K.atU(y, x) = static_cast<float>(xk - x);
            u0K.atV(y, x) = static_cast<float>(yk - y);
        }
    }

    FlowField muJ(w, h);
    FlowField muK(w, h);
    std::vector<uint8_t> misalignedMask(numPixels, 0);
    std::vector<uint8_t> validFlowMask(numPixels, 0);

    // Helper to populate a static 3D structure at arbitrary depth Z_obj != Z_plane
    auto addStatic3DRegion = [&](int x0, int y0, int bw, int bh, double Z_obj) {
        for (int y = y0; y < y0 + bh; ++y) {
            for (int x = x0; x < x0 + bw; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                misalignedMask[idx] = 1;
                validFlowMask[idx] = 1;

                const double X = (x - cx) * Z_obj / f;
                const double Y = (y - cy) * Z_obj / f;

                double xj, yj, xk, yk;
                projectPoint(X, Y, Z_obj, poseJ, f, cx, cy, xj, yj);
                projectPoint(X, Y, Z_obj, poseK, f, cx, cy, xk, yk);

                // Total flow minus planar flow = planar-parallax vector mu
                muJ.atU(y, x) = static_cast<float>((xj - x) - u0J.atU(y, x));
                muJ.atV(y, x) = static_cast<float>((yj - y) - u0J.atV(y, x));
                muK.atU(y, x) = static_cast<float>((xk - x) - u0K.atU(y, x));
                muK.atV(y, x) = static_cast<float>((yk - y) - u0K.atV(y, x));
            }
        }
    };

    // Helper to populate an independently moving 3D object
    auto addMoving3DRegion = [&](
        int x0, int y0, int bw, int bh, double Z_obj,
        double dXj, double dYj, double dZj,
        double dXk, double dYk, double dZk) {
        for (int y = y0; y < y0 + bh; ++y) {
            for (int x = x0; x < x0 + bw; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                misalignedMask[idx] = 1;
                validFlowMask[idx] = 1;

                const double X = (x - cx) * Z_obj / f;
                const double Y = (y - cy) * Z_obj / f;

                double xj, yj, xk, yk;
                projectPoint(X + dXj, Y + dYj, Z_obj + dZj, poseJ, f, cx, cy, xj, yj);
                projectPoint(X + dXk, Y + dYk, Z_obj + dZk, poseK, f, cx, cy, xk, yk);

                muJ.atU(y, x) = static_cast<float>((xj - x) - u0J.atU(y, x));
                muJ.atV(y, x) = static_cast<float>((yj - y) - u0J.atV(y, x));
                muK.atU(y, x) = static_cast<float>((xk - x) - u0K.atU(y, x));
                muK.atV(y, x) = static_cast<float>((yk - y) - u0K.atV(y, x));
            }
        }
    };

    std::cout << "Testing Theorem 2 Eq. (8) under full 3D projective camera motion (Tx,Ty,Tz,Wx,Wy,Wz)..." << std::endl;
    {
        // Static 3D structure A at depth Z = 8.0 (contains verified static seed p1)
        addStatic3DRegion(15, 20, 10, 10, 8.0);

        // Static 3D structure B at a completely different depth Z = 12.5
        addStatic3DRegion(20, 68, 12, 12, 12.5);

        // Independently moving 3D object at depth Z = 10.0 breaking projective rigidity
        addMoving3DRegion(
            62, 42, 14, 14, 10.0,
            0.80, -0.80,  0.15,
            0.80, -0.80, -0.20
        );

        PixelCoord staticSeed{20, 25};
        ParallaxRigidityParams params;
        params.rigidityThreshold = 0.05;

        ParallaxRigidityResult res = ParallaxRigidityTester::test(
            u0J, u0K, muJ, muK, misalignedMask, validFlowMask,
            std::vector<PixelCoord>{staticSeed},
            params
        );

        const float staticErr = res.rigidityErrorMap.at(72, 25);
        const float movingErr = res.rigidityErrorMap.at(48, 68);

        std::cout << "  Eq. (8) residual on static 3D block B (Z=12.5): " << staticErr << std::endl;
        std::cout << "  Eq. (8) residual on moving 3D object:           " << movingErr << std::endl;
        std::cout << "  Detected moving pixels: " << res.movingPixels.size()
                  << " (expected " << 14 * 14 << ")" << std::endl;

        assert(staticErr < 0.025f);
        assert(movingErr > 1.0f);
        assert(res.movingPixels.size() == 14 * 14);
        assert(res.movingMask[25 * w + 20] == 0);
        assert(res.movingMask[72 * w + 25] == 0);
        assert(res.movingMask[48 * w + 68] == 1);
    }

    std::cout << "Testing multi-seed epipolar collinearity resolution..." << std::endl;
    {
        std::vector<PixelCoord> twoSeeds = {
            PixelCoord{20, 25},
            PixelCoord{25, 72}
        };

        ParallaxRigidityParams params;
        params.rigidityThreshold = 0.05;

        ParallaxRigidityResult res = ParallaxRigidityTester::test(
            u0J, u0K, muJ, muK, misalignedMask, validFlowMask,
            twoSeeds,
            params
        );

        assert(res.movingPixels.size() == 14 * 14);
    }

    std::cout << "\nAll updated Stage 4 ParallaxRigidityTester (Eq. 8) tests passed successfully!" << std::endl;
    return 0;
}