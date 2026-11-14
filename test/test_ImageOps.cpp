#include "ImageOps.h"
#include <cassert>
#include <cmath>
#include <iostream>

bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    std::cout << "Testing 5-tap spatial gradients on a 2D quadratic surface..." << std::endl;
    {
        // I(x, y) = 2x^2 + 3xy - y^2 + 4x - 5y + 10
        // Analytical derivatives:
        // Ix(x, y) = 4x + 3y + 4
        // Iy(x, y) = 3x - 2y - 5
        const int w = 16;
        const int h = 16;
        FloatImage img(w, h, 1);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float fx = static_cast<float>(x);
                float fy = static_cast<float>(y);
                img.at(y, x) = 2.0f * fx * fx + 3.0f * fx * fy - fy * fy + 4.0f * fx - 5.0f * fy + 10.0f;
            }
        }

        FloatImage ix, iy;
        ImageOps::computeGradients(img, ix, iy);

        assert(ix.width == w && ix.height == h);
        assert(iy.width == w && iy.height == h);

        // Check interior pixels (away from boundary reflection)
        for (int y = 3; y < h - 3; ++y) {
            for (int x = 3; x < w - 3; ++x) {
                float expectedIx = 4.0f * x + 3.0f * y + 4.0f;
                float expectedIy = 3.0f * x - 2.0f * y - 5.0f;
                assert(approxEqual(ix.at(y, x), expectedIx, 1e-3f));
                assert(approxEqual(iy.at(y, x), expectedIy, 1e-3f));
            }
        }
    }

    std::cout << "Testing sub-pixel bilinear interpolation..." << std::endl;
    {
        // Linear ramp: I(x, y) = 10*x + 20*y + 5
        // Bilinear interpolation must be exact everywhere inside [0, w-1] x [0, h-1]
        FloatImage ramp(10, 10, 1);
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                ramp.at(y, x) = 10.0f * x + 20.0f * y + 5.0f;
            }
        }

        float sx = 3.25f;
        float sy = 6.75f;
        float expected = 10.0f * sx + 20.0f * sy + 5.0f;
        assert(approxEqual(ImageOps::sampleBilinear(ramp, sx, sy), expected));
    }

    std::cout << "Testing backward image warping and validMask generation..." << std::endl;
    {
        const int w = 12;
        const int h = 12;
        FloatImage src(w, h, 1);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                src.at(y, x) = 2.0f * x + 5.0f * y;
            }
        }

        // Constant sub-pixel shift: u = +1.5, v = -2.25
        FlowField flow(w, h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                flow.atU(y, x) = 1.5f;
                flow.atV(y, x) = -2.25f;
            }
        }

        std::vector<uint8_t> validMask;
        FloatImage warped = ImageOps::warpBackward(src, flow, &validMask);

        assert(warped.width == w && warped.height == h);
        assert(validMask.size() == static_cast<size_t>(w) * h);

        // Pixel (y=0, x=0) samples (-2.25, 1.5) -> out of bounds (y < 0)
        assert(validMask[0] == 0);

        // Pixel (y=4, x=3) samples (y=1.75, x=4.5) -> inside bounds
        assert(validMask[4 * w + 3] == 1);
        float expectedVal = 2.0f * (3.0f + 1.5f) + 5.0f * (4.0f - 2.25f);
        assert(approxEqual(warped.at(4, 3), expectedVal));

        // Pixel (y=5, x=11) samples (x=12.5) -> out of bounds (x > 11)
        assert(validMask[5 * w + 11] == 0);
    }

    std::cout << "Testing error handling on invalid inputs..." << std::endl;
    {
        FloatImage emptyImg, ix, iy;
        bool caughtGrad = false;
        try {
            ImageOps::computeGradients(emptyImg, ix, iy);
        } catch (const std::invalid_argument&) {
            caughtGrad = true;
        }
        assert(caughtGrad);

        FloatImage img(8, 8, 1);
        FlowField badFlow(4, 4);
        bool caughtWarp = false;
        try {
            ImageOps::warpBackward(img, badFlow);
        } catch (const std::invalid_argument&) {
            caughtWarp = true;
        }
        assert(caughtWarp);
    }

    std::cout << "\nAll ImageOps tests passed successfully!" << std::endl;
    return 0;
}