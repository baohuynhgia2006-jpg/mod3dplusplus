#ifndef GAUSSIAN_PYRAMID_H
#define GAUSSIAN_PYRAMID_H

#include "FloatImage.h"
#include <vector>

class GaussianPyramid {
public:
    // Applies a 5-tap separable Burt-Adelson Gaussian filter ([1, 4, 6, 4, 1] / 16)
    // with reflect-101 boundary handling and subsamples by a factor of 2.
    // Output dimensions are ((width + 1) / 2) x ((height + 1) / 2).
    static FloatImage pyrDown(const FloatImage& src);

    // Builds a multi-scale pyramid where level 0 is the full-resolution baseImage
    // and each subsequent level is downsampled by 2x. Stops early if either dimension
    // would drop below minDimension.
    static std::vector<FloatImage> build(
        const FloatImage& baseImage,
        int numLevels,
        int minDimension = 8
    );
};

#endif // GAUSSIAN_PYRAMID_H