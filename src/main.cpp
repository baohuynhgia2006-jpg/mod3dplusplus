#include "ImageSequenceReader.h"
#include "MotionDetectionPipeline.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace {

struct CliOptions {
    std::string inputDir;
    std::string outputDir = "./output";
    std::string extension = ".png";
    std::vector<PixelCoord> staticSeeds;
    unsigned int rngSeed = 0;
    bool hasCustomRngSeed = false;
    MotionPipelineParams pipelineParams;
    float overlayAlpha = 0.45f;
    int numThreads = 0;
    bool writeOverlay = true;
    bool writeParallax = true;
    bool showHelp = false;
};

void printUsage(const char* progName) {
    std::cout
        << "Usage: " << progName << " --input <dir> [options]\n\n"
        << "Required arguments:\n"
        << "  -i, --input <dir>           Input directory containing image sequence frames\n\n"
        << "Optional arguments:\n"
        << "  -o, --output <dir>          Output directory for masks and overlays (default: ./output)\n"
        << "  -e, --ext <ext>             Frame file extension filter (default: .png)\n"
        << "  -s, --seed <x,y>            Explicit static reference seed (repeatable; overrides auto-selection)\n"
        << "      --rng-seed <u32>        PRNG seed for automatic static seed selection (default: random_device)\n"
        << "  -j, --threads <n>           Number of OpenMP worker threads (default: auto)\n"
        << "      --planar-thresh <f>     Stage 4 planar parallax noise threshold in px (default: 0.30)\n"
        << "      --rigidity-thresh <f>   Stage 4 Eq. (8) rigidity threshold (default: 0.25)\n"
        << "      --lambda <f>            Stage 5 Graph Cut smoothness weight (default: 16.0)\n"
        << "      --moving-weight <f>     Stage 5 moving seed unary capacity (default: 18.0)\n"
        << "      --static-prior <f>      Stage 5 neutral pixel static prior (default: 0.02)\n"
        << "      --overlay-alpha <f>     Mask tint blend factor in [0, 1] (default: 0.45)\n"
        << "      --no-parallax           Skip writing pre-cut Stage 4 parallax masks and overlays\n"
        << "      --no-overlay            Write only binary masks (skip RGB overlay images)\n"
        << "  -h, --help                  Show this help message\n";
}

PixelCoord parseSeedCoord(const std::string& token) {
    const size_t commaPos = token.find(',');
    if (commaPos == std::string::npos || commaPos == 0 || commaPos + 1 >= token.size()) {
        throw std::invalid_argument("Invalid --seed format '" + token + "'. Expected <x,y> (e.g., 25,29).");
    }

    try {
        size_t idxX = 0;
        size_t idxY = 0;
        const std::string xStr = token.substr(0, commaPos);
        const std::string yStr = token.substr(commaPos + 1);
        const int x = std::stoi(xStr, &idxX);
        const int y = std::stoi(yStr, &idxY);
        if (idxX != xStr.size() || idxY != yStr.size() || x < 0 || y < 0) {
            throw std::invalid_argument("Seed coordinates must be non-negative integers.");
        }
        return PixelCoord{x, y};
    } catch (const std::exception&) {
        throw std::invalid_argument("Failed to parse --seed coordinates from '" + token + "'.");
    }
}

std::string requireNextArg(int argc, char* argv[], int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value after option '" + flag + "'.");
    }
    return std::string(argv[++i]);
}

CliOptions parseArgs(int argc, char* argv[]) {
    CliOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.showHelp = true;
            return opts;
        } else if (arg == "-i" || arg == "--input") {
            opts.inputDir = requireNextArg(argc, argv, i, arg);
        } else if (arg == "-o" || arg == "--output") {
            opts.outputDir = requireNextArg(argc, argv, i, arg);
        } else if (arg == "-e" || arg == "--ext") {
            opts.extension = requireNextArg(argc, argv, i, arg);
            if (!opts.extension.empty() && opts.extension[0] != '.') {
                opts.extension = "." + opts.extension;
            }
        } else if (arg == "-s" || arg == "--seed") {
            opts.staticSeeds.push_back(parseSeedCoord(requireNextArg(argc, argv, i, arg)));
        } else if (arg == "--rng-seed") {
            opts.rngSeed = static_cast<unsigned int>(std::stoul(requireNextArg(argc, argv, i, arg)));
            opts.hasCustomRngSeed = true;
        } else if (arg == "-j" || arg == "--threads") {
            opts.numThreads = std::stoi(requireNextArg(argc, argv, i, arg));
            if (opts.numThreads < 1) {
                throw std::invalid_argument("--threads must be >= 1.");
            }
        } else if (arg == "--planar-thresh") {
            opts.pipelineParams.rigidityParams.planarParallaxThreshold =
                std::stod(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--rigidity-thresh") {
            opts.pipelineParams.rigidityParams.rigidityThreshold =
                std::stod(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--lambda") {
            opts.pipelineParams.segmentationParams.lambda =
                std::stod(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--moving-weight") {
            opts.pipelineParams.segmentationParams.movingSeedWeight =
                std::stod(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--static-prior") {
            opts.pipelineParams.segmentationParams.unknownStaticPrior =
                std::stod(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--overlay-alpha") {
            opts.overlayAlpha = std::stof(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--no-parallax") {
            opts.writeParallax = false;
        } else if (arg == "--no-overlay") {
            opts.writeOverlay = false;
        } else {
            throw std::invalid_argument("Unknown command-line argument: '" + arg + "'.");
        }
    }

    if (opts.inputDir.empty()) {
        throw std::invalid_argument("Missing required argument: --input <dir>.");
    }

    return opts;
}

std::string formatFrameFilename(const std::string& prefix, size_t frameIndex, const std::string& ext) {
    std::ostringstream oss;
    oss << prefix << "_" << std::setw(4) << std::setfill('0') << frameIndex << ext;
    return oss.str();
}

} // namespace

int main(int argc, char* argv[]) {
    CliOptions opts;
    try {
        opts = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    if (opts.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    try {
        ImageSequenceReader reader(opts.inputDir, opts.extension);
        const size_t totalFrames = reader.getTotalFrames();

        if (totalFrames < 3) {
            std::cerr << "Error: Found " << totalFrames << " frame(s) in '" << opts.inputDir
                      << "'. At least 3 consecutive frames are required.\n";
            return 1;
        }

        std::vector<Image> frames;
        frames.reserve(totalFrames);
        for (size_t i = 0; i < totalFrames; ++i) {
            frames.push_back(reader.nextFrame());
        }

        const int frameW = frames[0].width;
        const int frameH = frames[0].height;
        const int frameC = frames[0].channels;

        for (size_t i = 1; i < totalFrames; ++i) {
            if (frames[i].width != frameW || frames[i].height != frameH || frames[i].channels != frameC) {
                throw std::runtime_error("All frames in the sequence must have identical dimensions and channels.");
            }
        }

        for (const auto& seed : opts.staticSeeds) {
            if (seed.x < 0 || seed.x >= frameW || seed.y < 0 || seed.y >= frameH) {
                std::cerr << "Error: Static seed (" << seed.x << ", " << seed.y
                          << ") is outside frame dimensions ("
                          << frameW << "x" << frameH << ").\n";
                return 1;
            }
        }

        fs::create_directories(opts.outputDir);

        const unsigned int baseRngSeed = opts.hasCustomRngSeed ? opts.rngSeed : std::random_device{}();

#ifdef _OPENMP
        if (opts.numThreads > 0) {
            omp_set_num_threads(opts.numThreads);
        }
        const int activeThreads = omp_get_max_threads();
#else
        (void)opts.numThreads;
        const int activeThreads = 1;
#endif

        const int numWindows = static_cast<int>(totalFrames) - 2;
        std::cout << "Loaded " << totalFrames << " frames ("
                  << frameW << "x" << frameH << ", " << frameC << " ch) from '" << opts.inputDir << "'\n"
                  << "Processing " << numWindows << " window(s) across " << activeThreads
                  << " thread(s) ("
                  << (opts.staticSeeds.empty()
                          ? ("auto static seeds, rng=" + std::to_string(baseRngSeed))
                          : ("explicit seeds=" + std::to_string(opts.staticSeeds.size())))
                  << ") -> '" << opts.outputDir << "'\n\n";

        const auto wallStart = std::chrono::steady_clock::now();
        std::atomic<bool> hasError{false};
        std::string firstErrorMsg;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int wIdx = 0; wIdx < numWindows; ++wIdx) {
            if (hasError.load(std::memory_order_relaxed)) {
                continue;
            }

            const size_t centerIdx = static_cast<size_t>(wIdx) + 1;

            try {
                const auto tStart = std::chrono::steady_clock::now();
                MotionPipelineResult result = MotionDetectionPipeline::run(
                    frames[centerIdx - 1],
                    frames[centerIdx],
                    frames[centerIdx + 1],
                    opts.staticSeeds,
                    opts.pipelineParams,
                    baseRngSeed + static_cast<unsigned int>(centerIdx)
                );
                const auto tEnd = std::chrono::steady_clock::now();
                const double elapsedMs =
                    std::chrono::duration<double, std::milli>(tEnd - tStart).count();

                // 1. Pre-cut Stage 4 parallax rigidity test outputs
                if (opts.writeParallax) {
                    const fs::path parallaxMaskPath =
                        fs::path(opts.outputDir) / formatFrameFilename("parallax_mask", centerIdx, ".png");
                    Image parallaxMaskImg = MotionDetectionPipeline::renderParallaxMask(
                        result.rigidity,
                        frameW,
                        frameH
                    );
                    if (!ImageSequenceReader::writeImage(parallaxMaskPath.string(), parallaxMaskImg)) {
                        throw std::runtime_error(
                            "Failed to write parallax mask to '" + parallaxMaskPath.string() + "'."
                        );
                    }

                    if (opts.writeOverlay) {
                        const fs::path parallaxOverlayPath =
                            fs::path(opts.outputDir) / formatFrameFilename("parallax_overlay", centerIdx, ".png");
                        Image parallaxOverlayImg = MotionDetectionPipeline::renderParallaxOverlay(
                            frames[centerIdx],
                            result.rigidity,
                            result.usedStaticSeeds,
                            std::max(opts.overlayAlpha, 0.65f)
                        );
                        if (!ImageSequenceReader::writeImage(parallaxOverlayPath.string(), parallaxOverlayImg)) {
                            throw std::runtime_error(
                                "Failed to write parallax overlay to '" + parallaxOverlayPath.string() + "'."
                            );
                        }
                    }
                }

                // 2. Final Stage 5 Graph Cut segmentation outputs
                const fs::path maskPath =
                    fs::path(opts.outputDir) / formatFrameFilename("mask", centerIdx, ".png");
                Image maskImg = MotionDetectionPipeline::renderBinaryMask(
                    result.segmentation,
                    frameW,
                    frameH
                );
                if (!ImageSequenceReader::writeImage(maskPath.string(), maskImg)) {
                    throw std::runtime_error("Failed to write binary mask to '" + maskPath.string() + "'.");
                }

                if (opts.writeOverlay) {
                    const fs::path overlayPath =
                        fs::path(opts.outputDir) / formatFrameFilename("overlay", centerIdx, ".png");
                    Image overlayImg = MotionDetectionPipeline::renderOverlay(
                        frames[centerIdx],
                        result.segmentation,
                        result.usedStaticSeeds,
                        opts.overlayAlpha
                    );
                    if (!ImageSequenceReader::writeImage(overlayPath.string(), overlayImg)) {
                        throw std::runtime_error("Failed to write overlay image to '" + overlayPath.string() + "'.");
                    }
                }

                #pragma omp critical
                {
                    std::cout << "[Frame " << std::setw(4) << std::setfill('0') << centerIdx << "] "
                              << "valid_flow=" << result.secondaryFlow.validPixels.size()
                              << " | rigid_moving=" << result.rigidity.movingPixels.size()
                              << " | dense_moving=" << result.segmentation.movingPixels.size()
                              << " (" << std::fixed << std::setprecision(1) << elapsedMs << " ms)\n";
                }
            } catch (const std::exception& e) {
                #pragma omp critical
                {
                    if (!hasError.exchange(true)) {
                        firstErrorMsg = "Frame " + std::to_string(centerIdx) + ": " + e.what();
                    }
                }
            }
        }

        if (hasError.load()) {
            throw std::runtime_error(firstErrorMsg);
        }

        const auto wallEnd = std::chrono::steady_clock::now();
        const double totalWallSec =
            std::chrono::duration<double>(wallEnd - wallStart).count();
        const double avgFps = static_cast<double>(numWindows) / std::max(totalWallSec, 1e-6);

        std::cout << "\nCompleted " << numWindows << " window(s) in "
                  << std::fixed << std::setprecision(2) << totalWallSec << " s ("
                  << std::setprecision(1) << avgFps << " fps effective throughput).\n";
    } catch (const std::exception& e) {
        std::cerr << "Pipeline execution error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}