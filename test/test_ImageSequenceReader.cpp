#include "ImageSequenceReader.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

void createDummyImages(const std::string& dirPath) {
    if (!fs::exists(dirPath)) {
        fs::create_directory(dirPath);
    }

    // Frame 0: Lossless PNG
    Image img0(100, 100, 3);
    img0.at(20, 10, 0) = 40;
    img0.at(20, 10, 1) = 128;
    img0.at(20, 10, 2) = 250;
    assert(ImageSequenceReader::writeImage(dirPath + "/frame_0.png", img0));

    // Frame 1: Lossy JPG (fill uniform block so DCT compression stays close to original values)
    Image img1(100, 100, 3);
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            img1.at(y, x, 0) = 100;
            img1.at(y, x, 1) = 150;
            img1.at(y, x, 2) = 200;
        }
    }
    assert(ImageSequenceReader::writeImage(dirPath + "/frame_1.jpg", img1, 100));

    // Frame 2: Uppercase extension .PNG & natural sort check (frame_10 > frame_2)
    Image img2(100, 100, 3);
    img2.at(20, 10, 0) = 210;
    img2.at(20, 10, 1) = 90;
    img2.at(20, 10, 2) = 30;
    assert(ImageSequenceReader::writeImage(dirPath + "/frame_10.PNG", img2));
}

void cleanupDummyImages(const std::string& dirPath) {
    if (fs::exists(dirPath)) {
        fs::remove_all(dirPath);
    }
}

int main() {
    const std::string testDir = "./test_frames";

    std::cout << "Setting up test environment..." << std::endl;
    cleanupDummyImages(testDir);
    createDummyImages(testDir);

    try {
        std::cout << "Testing multi-format initialization (.jpg, .jpeg, .png)..." << std::endl;
        ImageSequenceReader reader(testDir);
        assert(reader.getTotalFrames() == 3);

        // 1. Verify frame_0.png (exact lossless match)
        std::cout << "Testing PNG reading and pixel integrity..." << std::endl;
        Image f0 = reader.nextFrame();
        assert(!f0.empty() && f0.width == 100 && f0.height == 100 && f0.channels == 3);
        assert(f0.at(20, 10, 0) == 40);
        assert(f0.at(20, 10, 1) == 128);
        assert(f0.at(20, 10, 2) == 250);

        // 2. Verify frame_1.jpg (within JPEG compression tolerance)
        std::cout << "Testing JPG reading and pixel integrity..." << std::endl;
        Image f1 = reader.nextFrame();
        assert(!f1.empty() && f1.width == 100 && f1.height == 100 && f1.channels == 3);
        assert(std::abs(static_cast<int>(f1.at(20, 10, 0)) - 100) <= 3);
        assert(std::abs(static_cast<int>(f1.at(20, 10, 1)) - 150) <= 3);
        assert(std::abs(static_cast<int>(f1.at(20, 10, 2)) - 200) <= 3);

        // 3. Verify frame_10.PNG came AFTER frame_1.jpg (natural sort & case-insensitivity)
        std::cout << "Testing natural ordering and case-insensitive extension..." << std::endl;
        Image f2 = reader.nextFrame();
        assert(!f2.empty());
        assert(f2.at(20, 10, 0) == 210);
        assert(f2.at(20, 10, 1) == 90);
        assert(f2.at(20, 10, 2) == 30);
        assert(!reader.hasNext());

        // 4. Verify single-extension filtering and forced grayscale loading
        std::cout << "Testing extension filter and grayscale channel conversion..." << std::endl;
        ImageSequenceReader jpgOnlyReader(testDir, ".jpg");
        assert(jpgOnlyReader.getTotalFrames() == 1);
        Image grayJpg = jpgOnlyReader.nextFrame(1); // desiredChannels = 1
        assert(!grayJpg.empty() && grayJpg.channels == 1);

        // 5. Verify reset()
        reader.reset();
        assert(reader.hasNext());
        assert(reader.nextFrame().at(20, 10, 0) == 40);

        std::cout << "\nAll PNG/JPG tests passed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        cleanupDummyImages(testDir);
        return 1;
    }

    cleanupDummyImages(testDir);
    return 0;
}