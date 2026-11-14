#ifndef FLOAT_IMAGE_H
#define FLOAT_IMAGE_H

#include <cstddef>
#include <vector>

struct FloatImage {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<float> data;

    FloatImage() = default;
    FloatImage(int w, int h, int c = 1)
        : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, 0.0f) {}

    bool empty() const {
        return data.empty() || width <= 0 || height <= 0 || channels <= 0;
    }

    float& at(int y, int x, int c = 0) {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }

    const float& at(int y, int x, int c = 0) const {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }
};

#endif // FLOAT_IMAGE_H