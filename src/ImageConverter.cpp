#include "ImageConverter.h"
#include <stdexcept>

FloatImage ImageConverter::toFloatGrayscale(const Image& src, bool normalize) {
    if (src.empty()) {
        throw std::invalid_argument("Cannot convert an empty image to grayscale.");
    }

    if (src.channels != 1 && src.channels != 3 && src.channels != 4) {
        throw std::invalid_argument("Unsupported channel count for grayscale conversion (expected 1, 3, or 4).");
    }

    constexpr float wR = 0.299f;
    constexpr float wG = 0.587f;
    constexpr float wB = 0.114f;
    const float scale = normalize ? (1.0f / 255.0f) : 1.0f;

    FloatImage dst(src.width, src.height, 1);
    const size_t pixelCount = static_cast<size_t>(src.width) * src.height;

    if (src.channels == 1) {
        for (size_t i = 0; i < pixelCount; ++i) {
            dst.data[i] = static_cast<float>(src.data[i]) * scale;
        }
    } else {
        const size_t stride = static_cast<size_t>(src.channels);
        for (size_t i = 0; i < pixelCount; ++i) {
            const size_t idx = i * stride;
            const float r = static_cast<float>(src.data[idx]);
            const float g = static_cast<float>(src.data[idx + 1]);
            const float b = static_cast<float>(src.data[idx + 2]);
            dst.data[i] = (wR * r + wG * g + wB * b) * scale;
        }
    }

    return dst;
}