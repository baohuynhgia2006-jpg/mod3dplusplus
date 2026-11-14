#include "GaussianPyramid.h"
#include <cassert>
#include <cmath>
#include <iostream>

bool approxEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    std::cout << "Testing DC gain (constant image preservation)..." << std::endl;
    {
        FloatImage constantImg(32, 24, 1);
        for (float& val : constantImg.data) {
            val = 0.75f;
        }

        FloatImage down = GaussianPyramid::pyrDown(constantImg);
        assert(down.width == 16 && down.height == 12 && down.channels == 1);
        for (float val : down.data) {
            assert(approxEqual(val, 0.75f));
        }
    }

    std::cout << "Testing anti-aliasing on Nyquist checkerboard pattern..." << std::endl;
    {
        // Alternating 0.0 and 1.0 pixels. Naive subsampling would yield all 0s or all 1s;
        // proper Gaussian filtering attenuates the Nyquist frequency to ~0.5.
        FloatImage checker(16, 16, 1);
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                checker.at(y, x) = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
            }
        }

        FloatImage down = GaussianPyramid::pyrDown(checker);
        assert(down.width == 8 && down.height == 8);
        // Interior pixel (y=4, x=4) has full 5x5 support without boundary reflection
        assert(approxEqual(down.at(4, 4), 0.5f));
    }

    std::cout << "Testing odd-dimension downsampling..." << std::endl;
    {
        FloatImage oddImg(15, 11, 1);
        FloatImage down = GaussianPyramid::pyrDown(oddImg);
        assert(down.width == 8 && down.height == 6);
    }

    std::cout << "Testing multi-level pyramid construction and minDimension stopping..." << std::endl;
    {
        FloatImage base(64, 48, 1);
        // Request 10 levels with minDimension = 8:
        // Level 0: 64x48
        // Level 1: 32x24
        // Level 2: 16x12
        // Level 3: 8x6 -> height 6 < 8, stops at 3 levels (0, 1, 2)
        std::vector<FloatImage> pyr = GaussianPyramid::build(base, 10, 8);
        assert(pyr.size() == 3);
        assert(pyr[0].width == 64 && pyr[0].height == 48);
        assert(pyr[1].width == 32 && pyr[1].height == 24);
        assert(pyr[2].width == 16 && pyr[2].height == 12);
    }

    std::cout << "Testing invalid input handling..." << std::endl;
    {
        FloatImage emptyImg;
        bool caughtEmpty = false;
        try {
            GaussianPyramid::pyrDown(emptyImg);
        } catch (const std::invalid_argument&) {
            caughtEmpty = true;
        }
        assert(caughtEmpty);

        FloatImage valid(16, 16, 1);
        bool caughtLevels = false;
        try {
            GaussianPyramid::build(valid, 0);
        } catch (const std::invalid_argument&) {
            caughtLevels = true;
        }
        assert(caughtLevels);
    }

    std::cout << "\nAll GaussianPyramid tests passed successfully!" << std::endl;
    return 0;
}