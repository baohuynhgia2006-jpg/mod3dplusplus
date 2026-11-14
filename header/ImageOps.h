#ifndef IMAGE_OPS_H
#define IMAGE_OPS_H

#include "FloatImage.h"
#include "FlowField.h"
#include <cstdint>
#include <vector>

class ImageOps {
public:
    // Computes spatial derivatives Ix and Iy using a 5-tap derivative kernel
    // [-1, 8, 0, -8, 1] / 12.0f with reflect-101 boundary handling.
    static void computeGradients(const FloatImage& src, FloatImage& ix, FloatImage& iy);

    // Returns true if (x, y) lies within [0, width - 1] x [0, height - 1].
    static bool isInside(const FloatImage& img, float x, float y);

    // Bilinearly interpolates channel c at sub-pixel coordinates (x, y).
    // Out-of-bounds coordinates are clamped to the nearest border pixel.
    static float sampleBilinear(const FloatImage& img, float x, float y, int c = 0);

    // Backward warps src according to flow: warped(y, x) = src(x + u(y, x), y + v(y, x)).
    // If validMask is non-null, it is resized to width * height and filled with 1 where
    // (x + u, y + v) falls inside src bounds, and 0 otherwise.
    static FloatImage warpBackward(
        const FloatImage& src,
        const FlowField& flow,
        std::vector<uint8_t>* validMask = nullptr
    );
};

#endif // IMAGE_OPS_H