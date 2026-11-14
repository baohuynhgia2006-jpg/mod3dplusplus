#include "ImageSequenceReader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

// Natural sort comparator so frame2.png precedes frame10.png
bool naturalCompare(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::isdigit(static_cast<unsigned char>(a[i])) &&
            std::isdigit(static_cast<unsigned char>(b[j]))) {
            // Skip leading zeros
            size_t startA = i;
            while (i < a.size() && a[i] == '0') ++i;
            size_t numStartA = i;
            while (i < a.size() && std::isdigit(static_cast<unsigned char>(a[i]))) ++i;

            size_t startB = j;
            while (j < b.size() && b[j] == '0') ++j;
            size_t numStartB = j;
            while (j < b.size() && std::isdigit(static_cast<unsigned char>(b[j]))) ++j;

            size_t lenA = i - numStartA;
            size_t lenB = j - numStartB;
            if (lenA != lenB) {
                return lenA < lenB;
            }
            int cmp = a.compare(numStartA, lenA, b, numStartB, lenB);
            if (cmp != 0) {
                return cmp < 0;
            }
            // Tie-break by total digit length (including leading zeros)
            size_t totalDigitsA = i - startA;
            size_t totalDigitsB = j - startB;
            if (totalDigitsA != totalDigitsB) {
                return totalDigitsA < totalDigitsB;
            }
        } else {
            if (a[i] != b[j]) {
                return static_cast<unsigned char>(a[i]) < static_cast<unsigned char>(b[j]);
            }
            ++i;
            ++j;
        }
    }
    return a.size() < b.size();
}

} // namespace

ImageSequenceReader::ImageSequenceReader(
    const std::string& directoryPath,
    const std::vector<std::string>& extensions)
    : currentIndex(0) {
    if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
        throw std::invalid_argument("Directory does not exist: " + directoryPath);
    }

    std::vector<std::string> normalizedExts;
    normalizedExts.reserve(extensions.size());
    for (const auto& ext : extensions) {
        std::string lowerExt = toLower(ext);
        if (!lowerExt.empty() && lowerExt[0] != '.') {
            lowerExt = "." + lowerExt;
        }
        normalizedExts.push_back(lowerExt);
    }

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string fileExt = toLower(entry.path().extension().string());
        if (std::find(normalizedExts.begin(), normalizedExts.end(), fileExt) != normalizedExts.end()) {
            filepaths.push_back(entry.path().string());
        }
    }

    std::sort(filepaths.begin(), filepaths.end(), naturalCompare);
}

ImageSequenceReader::ImageSequenceReader(const std::string& directoryPath, const std::string& extension)
    : ImageSequenceReader(directoryPath, std::vector<std::string>{extension}) {}

bool ImageSequenceReader::hasNext() const {
    return currentIndex < filepaths.size();
}

Image ImageSequenceReader::nextFrame(int desiredChannels) {
    if (!hasNext()) {
        std::cerr << "Warning: End of sequence reached." << std::endl;
        return Image();
    }

    Image frame = readImage(filepaths[currentIndex], desiredChannels);
    if (frame.empty()) {
        std::cerr << "Failed to read image: " << filepaths[currentIndex] << std::endl;
    }

    currentIndex++;
    return frame;
}

void ImageSequenceReader::reset() {
    currentIndex = 0;
}

size_t ImageSequenceReader::getTotalFrames() const {
    return filepaths.size();
}

std::string ImageSequenceReader::getCurrentFilepath() const {
    if (currentIndex >= filepaths.size()) {
        return "";
    }
    return filepaths[currentIndex];
}

Image ImageSequenceReader::readImage(const std::string& filepath, int desiredChannels) {
    int width = 0;
    int height = 0;
    int origChannels = 0;

    unsigned char* rawData = stbi_load(filepath.c_str(), &width, &height, &origChannels, desiredChannels);
    if (!rawData) {
        std::cerr << "stbi_load error (" << filepath << "): " << stbi_failure_reason() << std::endl;
        return Image();
    }

    int outChannels = (desiredChannels > 0) ? desiredChannels : origChannels;
    Image img(width, height, outChannels);
    std::memcpy(img.data.data(), rawData, img.data.size());

    stbi_image_free(rawData);
    return img;
}

bool ImageSequenceReader::writeImage(const std::string& filepath, const Image& img, int jpgQuality) {
    if (img.empty()) {
        return false;
    }

    std::string ext = toLower(fs::path(filepath).extension().string());
    int result = 0;

    if (ext == ".png") {
        int stride = img.width * img.channels;
        result = stbi_write_png(filepath.c_str(), img.width, img.height, img.channels, img.data.data(), stride);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        result = stbi_write_jpg(filepath.c_str(), img.width, img.height, img.channels, img.data.data(), jpgQuality);
    } else {
        std::cerr << "Unsupported output extension: " << ext << std::endl;
        return false;
    }

    return result != 0;
}