#ifndef IMAGE_CONVERTER_H
#define IMAGE_CONVERTER_H

#include "FloatImage.h"
#include "Image.h"

class ImageConverter {
public:
    // Converts an 8-bit image (1, 3, or 4 channels) to a single-channel float image
    // using ITU-R BT.601 luminance weights: Y = 0.299*R + 0.587*G + 0.114*B.
    // If normalize is true, output values are scaled to [0.0f, 1.0f]; otherwise [0.0f, 255.0f].
    static FloatImage toFloatGrayscale(const Image& src, bool normalize = true);
};

#endif // IMAGE_CONVERTER_H