#include "ImageSequenceReader.h"
#include "MotionDetectionPipeline.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CliOptions {
    std::string inputDir;
    std::string outputDir = "./output";
    std::string extension = ".png";
    std::vector<PixelCoord> staticSeeds;
    MotionPipelineParams pipelineParams;
    float overlayAlpha = 0.45f;
    bool writeOverlay = true;
    bool showHelp = false;
};

void printUsage(const char* progName) {
    std::cout
        << "Usage: " << progName << " --input <dir> --seed <x,y> [options]\n\n"
        << "Required arguments:\n"
        << "  -i, --input <dir>           Input directory containing image sequence frames\n"
        << "  -s, --seed <x,y>            Verified static reference pixel coordinate (repeatable)\n\n"
        << "Optional arguments:\n"
        << "  -o, --output <dir>          Output directory for masks and overlays (default: ./output)\n"
        << "  -e, --ext <ext>             Frame file extension filter (default: .png)\n"
        << "      --rigidity-thresh <f>   Stage 4 Eq. (8) rigidity threshold (default: 0.25)\n"
        << "      --mad-mult <f>          Stage 2 adaptive residual MAD multiplier (default: 3.5)\n"
        << "      --lambda <f>            Stage 5 Graph Cut smoothness weight (default: 18.0)\n"
        << "      --moving-weight <f>     Stage 5 moving seed unary capacity (default: 12.0)\n"
        << "      --static-prior <f>      Stage 5 unknown pixel static prior (default: 0.05)\n"
        << "      --overlay-alpha <f>     Red tint blend factor in [0, 1] (default: 0.45)\n"
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
        } else if (arg == "--rigidity-thresh") {
            opts.pipelineParams.rigidityParams.rigidityThreshold =
                std::stod(requireNextArg(argc, argv, i, arg));
        } else if (arg == "--mad-mult") {
            opts.pipelineParams.alignmentParams.madMultiplier =
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
        } else if (arg == "--no-overlay") {
            opts.writeOverlay = false;
        } else {
            throw std::invalid_argument("Unknown command-line argument: '" + arg + "'.");
        }
    }

    if (opts.inputDir.empty()) {
        throw std::invalid_argument("Missing required argument: --input <dir>.");
    }
    if (opts.staticSeeds.empty()) {
        throw std::invalid_argument("At least one verified static reference pixel (--seed <x,y>) is required.");
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

        fs::create_directories(opts.outputDir);

        Image prevFrame = reader.nextFrame();
        Image currFrame = reader.nextFrame();

        // Validate static seed coordinates against image dimensions
        for (const auto& seed : opts.staticSeeds) {
            if (seed.x < 0 || seed.x >= currFrame.width || seed.y < 0 || seed.y >= currFrame.height) {
                std::cerr << "Error: Static seed (" << seed.x << ", " << seed.y
                          << ") is outside frame dimensions ("
                          << currFrame.width << "x" << currFrame.height << ").\n";
                return 1;
            }
        }

        std::cout << "Loaded " << totalFrames << " frames ("
                  << currFrame.width << "x" << currFrame.height
                  << ", " << currFrame.channels << " ch) from '" << opts.inputDir << "'\n"
                  << "Using " << opts.staticSeeds.size() << " verified static seed(s); writing outputs to '"
                  << opts.outputDir << "'\n\n";

        for (size_t centerIdx = 1; centerIdx + 1 < totalFrames; ++centerIdx) {
            Image nextFrame = reader.nextFrame();

            const auto tStart = std::chrono::steady_clock::now();
            MotionPipelineResult result = MotionDetectionPipeline::run(
                prevFrame,
                currFrame,
                nextFrame,
                opts.staticSeeds,
                opts.pipelineParams
            );
            const auto tEnd = std::chrono::steady_clock::now();
            const double elapsedMs =
                std::chrono::duration<double, std::milli>(tEnd - tStart).count();

            const fs::path maskPath =
                fs::path(opts.outputDir) / formatFrameFilename("mask", centerIdx, ".png");
            Image maskImg = MotionDetectionPipeline::renderBinaryMask(
                result.segmentation,
                currFrame.width,
                currFrame.height
            );
            if (!ImageSequenceReader::writeImage(maskPath.string(), maskImg)) {
                throw std::runtime_error("Failed to write binary mask to '" + maskPath.string() + "'.");
            }

            if (opts.writeOverlay) {
                const fs::path overlayPath =
                    fs::path(opts.outputDir) / formatFrameFilename("overlay", centerIdx, ".png");
                Image overlayImg = MotionDetectionPipeline::renderOverlay(
                    currFrame,
                    result.segmentation,
                    result.usedStaticSeeds,
                    opts.overlayAlpha
                );
                if (!ImageSequenceReader::writeImage(overlayPath.string(), overlayImg)) {
                    throw std::runtime_error("Failed to write overlay image to '" + overlayPath.string() + "'.");
                }
            }

            std::cout << "[Frame " << std::setw(4) << std::setfill('0') << centerIdx << "] "
                      << "misaligned=" << result.alignment.misalignedPixels.size()
                      << " | rigid_moving=" << result.rigidity.movingPixels.size()
                      << " | dense_moving=" << result.segmentation.movingPixels.size()
                      << " (" << std::fixed << std::setprecision(1) << elapsedMs << " ms)\n";

            // Slide the 3-frame window forward without reloading frames
            prevFrame = std::move(currFrame);
            currFrame = std::move(nextFrame);
        }

        std::cout << "\nCompleted processing " << (totalFrames - 2) << " 3-frame window(s).\n";
    } catch (const std::exception& e) {
        std::cerr << "Pipeline execution error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}