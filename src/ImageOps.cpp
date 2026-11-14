#include "ImageOps.h"
#include <algorithm>
#include <cmath>
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

void ImageOps::computeGradients(const FloatImage& src, FloatImage& ix, FloatImage& iy) {
    if (src.empty()) {
        throw std::invalid_argument("Cannot compute gradients of an empty FloatImage.");
    }

    const int w = src.width;
    const int h = src.height;
    const int channels = src.channels;

    ix = FloatImage(w, h, channels);
    iy = FloatImage(w, h, channels);

    constexpr float inv12 = 1.0f / 12.0f;

    // Horizontal derivative Ix: [-1, 8, 0, -8, 1] / 12 (applied left-to-right as +1*I(x-2) - 8*I(x-1) + 8*I(x+1) - 1*I(x+2))
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int xm2 = reflect101(x - 2, w);
            const int xm1 = reflect101(x - 1, w);
            const int xp1 = reflect101(x + 1, w);
            const int xp2 = reflect101(x + 2, w);

            for (int c = 0; c < channels; ++c) {
                ix.at(y, x, c) = (
                    src.at(y, xm2, c)
                    - 8.0f * src.at(y, xm1, c)
                    + 8.0f * src.at(y, xp1, c)
                    - src.at(y, xp2, c)
                ) * inv12;
            }
        }
    }

    // Vertical derivative Iy
    for (int y = 0; y < h; ++y) {
        const int ym2 = reflect101(y - 2, h);
        const int ym1 = reflect101(y - 1, h);
        const int yp1 = reflect101(y + 1, h);
        const int yp2 = reflect101(y + 2, h);

        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                iy.at(y, x, c) = (
                    src.at(ym2, x, c)
                    - 8.0f * src.at(ym1, x, c)
                    + 8.0f * src.at(yp1, x, c)
                    - src.at(yp2, x, c)
                ) * inv12;
            }
        }
    }
}

bool ImageOps::isInside(const FloatImage& img, float x, float y) {
    if (img.empty()) {
        return false;
    }
    return x >= 0.0f && x <= static_cast<float>(img.width - 1) &&
           y >= 0.0f && y <= static_cast<float>(img.height - 1);
}

float ImageOps::sampleBilinear(const FloatImage& img, float x, float y, int c) {
    const float maxX = static_cast<float>(img.width - 1);
    const float maxY = static_cast<float>(img.height - 1);

    const float cx = std::clamp(x, 0.0f, maxX);
    const float cy = std::clamp(y, 0.0f, maxY);

    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(x0 + 1, img.width - 1);
    const int y1 = std::min(y0 + 1, img.height - 1);

    const float dx = cx - static_cast<float>(x0);
    const float dy = cy - static_cast<float>(y0);

    const float v00 = img.at(y0, x0, c);
    const float v10 = img.at(y0, x1, c);
    const float v01 = img.at(y1, x0, c);
    const float v11 = img.at(y1, x1, c);

    const float top = v00 + dx * (v10 - v00);
    const float bottom = v01 + dx * (v11 - v01);
    return top + dy * (bottom - top);
}

FloatImage ImageOps::warpBackward(
    const FloatImage& src,
    const FlowField& flow,
    std::vector<uint8_t>* validMask) {
    if (src.empty() || flow.empty()) {
        throw std::invalid_argument("Source image and flow field must not be empty.");
    }
    if (src.width != flow.width || src.height != flow.height) {
        throw std::invalid_argument("Source image and flow field dimensions must match.");
    }

    const int w = src.width;
    const int h = src.height;
    const int channels = src.channels;

    FloatImage warped(w, h, channels);
    if (validMask) {
        validMask->assign(static_cast<size_t>(w) * h, 0);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float srcX = static_cast<float>(x) + flow.atU(y, x);
            const float srcY = static_cast<float>(y) + flow.atV(y, x);

            if (validMask) {
                (*validMask)[static_cast<size_t>(y) * w + x] = isInside(src, srcX, srcY) ? 1 : 0;
            }

            for (int c = 0; c < channels; ++c) {
                warped.at(y, x, c) = sampleBilinear(src, srcX, srcY, c);
            }
        }
    }

    return warped;
}