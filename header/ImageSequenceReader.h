#ifndef IMAGE_SEQUENCE_READER_H
#define IMAGE_SEQUENCE_READER_H

#include "Image.h"
#include <string>
#include <vector>

class ImageSequenceReader {
private:
    std::vector<std::string> filepaths;
    size_t currentIndex;

public:
    explicit ImageSequenceReader(
        const std::string& directoryPath,
        const std::vector<std::string>& extensions = {".jpg", ".jpeg", ".png"}
    );

    ImageSequenceReader(const std::string& directoryPath, const std::string& extension);

    bool hasNext() const;
    Image nextFrame(int desiredChannels = 0);
    void reset();
    size_t getTotalFrames() const;
    std::string getCurrentFilepath() const;

    static Image readImage(const std::string& filepath, int desiredChannels = 0);
    static bool writeImage(const std::string& filepath, const Image& img, int jpgQuality = 95);
};

#endif // IMAGE_SEQUENCE_READER_H