#include "GaussianPyramid.h"
#include <stdexcept>

namespace {

inline int reflect101(int idx, int len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx;
        }
        if (idx >= len) {
            idx = 2 * len - 2 - idx;
        }
    }
    return idx;
}

} // namespace

FloatImage GaussianPyramid::pyrDown(const FloatImage& src) {
    if (src.empty()) {
        throw std::invalid_argument("Cannot downsample an empty FloatImage.");
    }

    constexpr float k[5] = {
        1.0f / 16.0f,
        4.0f / 16.0f,
        6.0f / 16.0f,
        4.0f / 16.0f,
        1.0f / 16.0f
    };

    const int inW = src.width;
    const int inH = src.height;
    const int channels = src.channels;

    const int outW = (inW + 1) / 2;
    const int outH = (inH + 1) / 2;

    // Pass 1: Horizontal smoothing evaluated only at even x columns (2 * x)
    FloatImage temp(outW, inH, channels);
    for (int y = 0; y < inH; ++y) {
        for (int x = 0; x < outW; ++x) {
            const int srcX = 2 * x;
            const int x0 = reflect101(srcX - 2, inW);
            const int x1 = reflect101(srcX - 1, inW);
            const int x2 = reflect101(srcX,     inW);
            const int x3 = reflect101(srcX + 1, inW);
            const int x4 = reflect101(srcX + 2, inW);

            for (int c = 0; c < channels; ++c) {
                temp.at(y, x, c) =
                    k[0] * src.at(y, x0, c) +
                    k[1] * src.at(y, x1, c) +
                    k[2] * src.at(y, x2, c) +
                    k[3] * src.at(y, x3, c) +
                    k[4] * src.at(y, x4, c);
            }
        }
    }

    // Pass 2: Vertical smoothing on temp evaluated only at even y rows (2 * y)
    FloatImage dst(outW, outH, channels);
    for (int y = 0; y < outH; ++y) {
        const int srcY = 2 * y;
        const int y0 = reflect101(srcY - 2, inH);
        const int y1 = reflect101(srcY - 1, inH);
        const int y2 = reflect101(srcY,     inH);
        const int y3 = reflect101(srcY + 1, inH);
        const int y4 = reflect101(srcY + 2, inH);

        for (int x = 0; x < outW; ++x) {
            for (int c = 0; c < channels; ++c) {
                dst.at(y, x, c) =
                    k[0] * temp.at(y0, x, c) +
                    k[1] * temp.at(y1, x, c) +
                    k[2] * temp.at(y2, x, c) +
                    k[3] * temp.at(y3, x, c) +
                    k[4] * temp.at(y4, x, c);
            }
        }
    }

    return dst;
}

std::vector<FloatImage> GaussianPyramid::build(
    const FloatImage& baseImage,
    int numLevels,
    int minDimension) {
    if (baseImage.empty()) {
        throw std::invalid_argument("Cannot build pyramid from an empty FloatImage.");
    }
    if (numLevels < 1) {
        throw std::invalid_argument("numLevels must be at least 1.");
    }
    if (minDimension < 1) {
        minDimension = 1;
    }

    std::vector<FloatImage> pyramid;
    pyramid.reserve(static_cast<size_t>(numLevels));
    pyramid.push_back(baseImage);

    for (int level = 1; level < numLevels; ++level) {
        const FloatImage& prev = pyramid.back();
        const int nextW = (prev.width + 1) / 2;
        const int nextH = (prev.height + 1) / 2;

        if (nextW < minDimension || nextH < minDimension || (prev.width == 1 && prev.height == 1)) {
            break;
        }

        pyramid.push_back(pyrDown(prev));
    }

    return pyramid;
}