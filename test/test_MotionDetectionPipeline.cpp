#include "ImageSequenceReader.h"
#include "MotionDetectionPipeline.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace {

void evalQuadraticAt(
    double x,
    double y,
    int w,
    int h,
    const std::array<double, 12>& coeffs,
    double& u,
    double& v) {
    const double cx = 0.5 * (w - 1);
    const double cy = 0.5 * (h - 1);
    const double invScale = 1.0 / (0.5 * std::max(w, h));
    const double nx = (x - cx) * invScale;
    const double ny = (y - cy) * invScale;
    const double b[6] = {1.0, nx, ny, nx * nx, nx * ny, ny * ny};

    u = 0.0;
    v = 0.0;
    for (int k = 0; k < 6; ++k) {
        u += coeffs[static_cast<size_t>(k)] * b[k];
        v += coeffs[static_cast<size_t>(k + 6)] * b[k];
    }
}

void evalBgColor(double x, double y, uint8_t& r, uint8_t& g, uint8_t& b) {
    double base = 85.0
        + 25.0 * std::sin(0.12 * x + 0.06 * y)
        + 20.0 * std::cos(0.08 * x - 0.14 * y)
        + 15.0 * std::sin(0.22 * x) * std::cos(0.18 * y);
    r = static_cast<uint8_t>(std::clamp(base * 0.85, 10.0, 220.0));
    g = static_cast<uint8_t>(std::clamp(base * 1.05, 10.0, 220.0));
    b = static_cast<uint8_t>(std::clamp(base * 0.90, 10.0, 220.0));
}

void createEndToEndSequence(const std::string& dirPath, int w, int h) {
    if (fs::exists(dirPath)) {
        fs::remove_all(dirPath);
    }
    fs::create_directory(dirPath);

    // Non-uniform 3D camera motion across (t-1, t, t+1):
    // curr -> prev shifts roughly (-1.6, +1.1) with slight rotation/zoom
    // curr -> next shifts roughly (+2.3, -1.4) with distinct rotation/zoom
    const std::array<double, 12> bgCoeffsPrev = {
       -1.6,  0.18, -0.22, -0.06,  0.05, -0.04,
        1.1,  0.22,  0.14,  0.05, -0.06,  0.05
    };
    const std::array<double, 12> bgCoeffsNext = {
        2.3, -0.24,  0.28,  0.09, -0.07,  0.05,
       -1.4, -0.28, -0.18, -0.07,  0.09, -0.07
    };

    std::vector<Image> frames(3, Image(w, h, 3));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            evalBgColor(x, y, frames[1].at(y, x, 0), frames[1].at(y, x, 1), frames[1].at(y, x, 2));

            // Next (t+1): next(x + u_next, y + v_next) = curr(x, y)
            double sxNext = static_cast<double>(x);
            double syNext = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxNext, syNext, w, h, bgCoeffsNext, u, v);
                sxNext = static_cast<double>(x) - u;
                syNext = static_cast<double>(y) - v;
            }
            evalBgColor(sxNext, syNext, frames[2].at(y, x, 0), frames[2].at(y, x, 1), frames[2].at(y, x, 2));

            // Prev (t-1): prev(x + u_prev, y + v_prev) = curr(x, y)
            double sxPrev = static_cast<double>(x);
            double syPrev = static_cast<double>(y);
            for (int it = 0; it < 10; ++it) {
                double u, v;
                evalQuadraticAt(sxPrev, syPrev, w, h, bgCoeffsPrev, u, v);
                sxPrev = static_cast<double>(x) - u;
                syPrev = static_cast<double>(y) - v;
            }
            evalBgColor(sxPrev, syPrev, frames[0].at(y, x, 0), frames[0].at(y, x, 1), frames[0].at(y, x, 2));
        }
    }

    // Stamp static 3D structure (depth ratio gamma = 0.65 -> parallax proportional to camera translation)
    const int statX0 = 18, statY0 = 22, statSize = 16;
    const double gamma = 0.65;
    const double parPrevU = gamma * bgCoeffsPrev[0];
    const double parPrevV = gamma * bgCoeffsPrev[6];
    const double parNextU = gamma * bgCoeffsNext[0];
    const double parNextV = gamma * bgCoeffsNext[6];

    for (int fIdx = 0; fIdx < 3; ++fIdx) {
        for (int dy = 0; dy < statSize; ++dy) {
            for (int dx = 0; dx < statSize; ++dx) {
                const int cx = statX0 + dx;
                const int cy = statY0 + dy;
                int px = cx;
                int py = cy;

                if (fIdx == 0) {
                    double u0, v0;
                    evalQuadraticAt(cx, cy, w, h, bgCoeffsPrev, u0, v0);
                    px = static_cast<int>(std::round(cx + u0 + parPrevU));
                    py = static_cast<int>(std::round(cy + v0 + parPrevV));
                } else if (fIdx == 2) {
                    double u0, v0;
                    evalQuadraticAt(cx, cy, w, h, bgCoeffsNext, u0, v0);
                    px = static_cast<int>(std::round(cx + u0 + parNextU));
                    py = static_cast<int>(std::round(cy + v0 + parNextV));
                }

                if (px >= 0 && px < w && py >= 0 && py < h) {
                    const double tex = 115.0 + 55.0 * std::sin(0.55 * dx) * std::cos(0.55 * dy);
                    const uint8_t pat = static_cast<uint8_t>(std::clamp(tex, 30.0, 220.0));
                    frames[fIdx].at(py, px, 0) = pat;
                    frames[fIdx].at(py, px, 1) = static_cast<uint8_t>(pat * 0.8);
                    frames[fIdx].at(py, px, 2) = 210;
                }
            }
        }
    }

    // Stamp independently moving 36x36 object:
    // Multi-scale textured outer rim (5px) + completely untextured uniform interior (26x26)
    const int movX0 = 66, movY0 = 46, movSize = 36;
    const double movPrevU =  2.1, movPrevV = 1.6;
    const double movNextU = -2.4, movNextV = 1.8;

    for (int fIdx = 0; fIdx < 3; ++fIdx) {
        double shiftX = 0.0, shiftY = 0.0;
        if (fIdx == 0) {
            shiftX = movPrevU;
            shiftY = movPrevV;
        } else if (fIdx == 2) {
            shiftX = movNextU;
            shiftY = movNextV;
        }

        const int baseX = static_cast<int>(std::round(movX0 + shiftX));
        const int baseY = static_cast<int>(std::round(movY0 + shiftY));

        for (int dy = 0; dy < movSize; ++dy) {
            for (int dx = 0; dx < movSize; ++dx) {
                const int px = baseX + dx;
                const int py = baseY + dy;
                if (px < 0 || px >= w || py < 0 || py >= h) continue;

                const bool isBorderRim = (dx < 5 || dx >= movSize - 5 || dy < 5 || dy >= movSize - 5);
                if (isBorderRim) {
                    const double mod = 225.0 + 22.0 * std::sin(0.65 * dx + 0.45 * dy);
                    frames[fIdx].at(py, px, 0) = static_cast<uint8_t>(std::clamp(mod, 195.0, 250.0));
                    frames[fIdx].at(py, px, 1) = 45;
                    frames[fIdx].at(py, px, 2) = 45;
                } else {
                    frames[fIdx].at(py, px, 0) = 225;
                    frames[fIdx].at(py, px, 1) = 48;
                    frames[fIdx].at(py, px, 2) = 48;
                }
            }
        }
    }

    assert(ImageSequenceReader::writeImage(dirPath + "/frame_00.png", frames[0]));
    assert(ImageSequenceReader::writeImage(dirPath + "/frame_01.png", frames[1]));
    assert(ImageSequenceReader::writeImage(dirPath + "/frame_02.png", frames[2]));
}

} // namespace

int main() {
    const std::string seqDir = "./test_pipeline_seq";
    const int w = 128;
    const int h = 128;

    std::cout << "Creating synthetic 3-frame sequence with 3D parallax and untextured moving object..." << std::endl;
    createEndToEndSequence(seqDir, w, h);

    try {
        ImageSequenceReader reader(seqDir, ".png");
        assert(reader.getTotalFrames() == 3);

        Image prev = reader.nextFrame();
        Image curr = reader.nextFrame();
        Image next = reader.nextFrame();

        // Pick a verified static pixel inside the static 3D parallax pillar (18..33, 22..37)
        PixelCoord verifiedStaticSeed{25, 29};

        MotionPipelineParams params;
        params.rigidityParams.rigidityThreshold = 0.25;
        params.segmentationParams.lambda = 18.0;
        params.segmentationParams.unknownStaticPrior = 0.05;

        std::cout << "Running full 5-stage MotionDetectionPipeline..." << std::endl;
        MotionPipelineResult res = MotionDetectionPipeline::run(
            prev, curr, next, verifiedStaticSeed, params
        );

        std::cout << "  Stage 2 misaligned pixels:           " << res.alignment.misalignedPixels.size() << std::endl;
        std::cout << "  Stage 3 well-conditioned flow pixels:" << res.secondaryFlow.validPixels.size() << std::endl;
        std::cout << "  Stage 4 sparse moving detections:    " << res.rigidity.movingPixels.size() << std::endl;
        std::cout << "  Stage 5 dense segmented pixels:      " << res.segmentation.movingPixels.size()
                  << " (GT object area = " << 36 * 36 << ")" << std::endl;

        assert(!res.rigidity.movingPixels.empty());
        assert(res.rigidity.movingPixels.size() < 800);

        int tp = 0;
        int fp = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const bool inMovingObj = (x >= 66 && x < 102 && y >= 46 && y < 82);
                const bool predMoving = (res.segmentation.denseMovingMask[static_cast<size_t>(y) * w + x] != 0);
                if (inMovingObj && predMoving) {
                    tp++;
                } else if (!inMovingObj && predMoving) {
                    fp++;
                }
            }
        }

        const double recall = static_cast<double>(tp) / (36.0 * 36.0);
        std::cout << "  Dense segmentation recall: " << (recall * 100.0)
                  << "%, false positives: " << fp << std::endl;

        assert(recall > 0.95);
        assert(fp < 20);

        // Verify the static 3D parallax pillar remained classified as static
        assert(res.segmentation.denseMovingMask[29 * w + 25] == 0);

        Image maskOut = MotionDetectionPipeline::renderBinaryMask(res.segmentation, w, h);
        Image overlayOut = MotionDetectionPipeline::renderOverlay(curr, res.segmentation, res.usedStaticSeeds);
        assert(!maskOut.empty() && maskOut.channels == 1);
        assert(!overlayOut.empty() && overlayOut.channels == 3);
        assert(ImageSequenceReader::writeImage(seqDir + "/out_mask.png", maskOut));
        assert(ImageSequenceReader::writeImage(seqDir + "/out_overlay.png", overlayOut));

        std::cout << "\nEnd-to-end MotionDetectionPipeline test passed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Integration test failed: " << e.what() << std::endl;
        fs::remove_all(seqDir);
        return 1;
    }

    fs::remove_all(seqDir);
    return 0;
}