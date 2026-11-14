#ifndef IMAGE_H
#define IMAGE_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, int c)
        : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, 0) {}

    bool empty() const {
        return data.empty() || width <= 0 || height <= 0 || channels <= 0;
    }

    uint8_t& at(int y, int x, int c = 0) {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }

    const uint8_t& at(int y, int x, int c = 0) const {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }
};

#endif // IMAGE_H