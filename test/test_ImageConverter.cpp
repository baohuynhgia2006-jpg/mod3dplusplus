#include "ImageConverter.h"
#include <cassert>
#include <cmath>
#include <iostream>

bool approxEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    std::cout << "Testing RGB to normalized float grayscale conversion..." << std::endl;
    {
        Image rgb(2, 2, 3);
        // (0, 0): Pure Red
        rgb.at(0, 0, 0) = 255; rgb.at(0, 0, 1) = 0;   rgb.at(0, 0, 2) = 0;
        // (0, 1): Pure Green
        rgb.at(0, 1, 0) = 0;   rgb.at(0, 1, 1) = 255; rgb.at(0, 1, 2) = 0;
        // (1, 0): Pure Blue
        rgb.at(1, 0, 0) = 0;   rgb.at(1, 0, 1) = 0;   rgb.at(1, 0, 2) = 255;
        // (1, 1): Arbitrary color (100, 150, 200)
        rgb.at(1, 1, 0) = 100; rgb.at(1, 1, 1) = 150; rgb.at(1, 1, 2) = 200;

        FloatImage grayNorm = ImageConverter::toFloatGrayscale(rgb, true);
        assert(!grayNorm.empty());
        assert(grayNorm.width == 2 && grayNorm.height == 2 && grayNorm.channels == 1);

        assert(approxEqual(grayNorm.at(0, 0), 0.299f));
        assert(approxEqual(grayNorm.at(0, 1), 0.587f));
        assert(approxEqual(grayNorm.at(1, 0), 0.114f));

        float expectedMixed = (0.299f * 100.0f + 0.587f * 150.0f + 0.114f * 200.0f) / 255.0f;
        assert(approxEqual(grayNorm.at(1, 1), expectedMixed));
    }

    std::cout << "Testing unnormalized [0, 255] float grayscale conversion..." << std::endl;
    {
        Image rgb(1, 1, 3);
        rgb.at(0, 0, 0) = 255; rgb.at(0, 0, 1) = 255; rgb.at(0, 0, 2) = 255;

        FloatImage grayUnnorm = ImageConverter::toFloatGrayscale(rgb, false);
        assert(approxEqual(grayUnnorm.at(0, 0), 255.0f, 1e-4f));
    }

    std::cout << "Testing 1-channel and 4-channel (RGBA) passthrough/conversion..." << std::endl;
    {
        Image single(1, 1, 1);
        single.at(0, 0) = 128;
        FloatImage grayFromSingle = ImageConverter::toFloatGrayscale(single, true);
        assert(approxEqual(grayFromSingle.at(0, 0), 128.0f / 255.0f));

        Image rgba(1, 1, 4);
        rgba.at(0, 0, 0) = 255; rgba.at(0, 0, 1) = 0; rgba.at(0, 0, 2) = 0; rgba.at(0, 0, 3) = 50;
        FloatImage grayFromRgba = ImageConverter::toFloatGrayscale(rgba, true);
        assert(approxEqual(grayFromRgba.at(0, 0), 0.299f));
    }

    std::cout << "Testing error handling on invalid inputs..." << std::endl;
    {
        Image emptyImg;
        bool caughtEmpty = false;
        try {
            ImageConverter::toFloatGrayscale(emptyImg);
        } catch (const std::invalid_argument&) {
            caughtEmpty = true;
        }
        assert(caughtEmpty);

        Image twoChannel(2, 2, 2);
        bool caughtChannels = false;
        try {
            ImageConverter::toFloatGrayscale(twoChannel);
        } catch (const std::invalid_argument&) {
            caughtChannels = true;
        }
        assert(caughtChannels);
    }

    std::cout << "\nAll ImageConverter tests passed successfully!" << std::endl;
    return 0;
}