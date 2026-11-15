#include "FrameAligner.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

inline float sampleBilinearClamped(const FloatImage& img, float x, float y) {
    const float cx = std::clamp(x, 0.0f, static_cast<float>(img.width - 1));
    const float cy = std::clamp(y, 0.0f, static_cast<float>(img.height - 1));

    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(img.width - 1, x0 + 1);
    const int y1 = std::min(img.height - 1, y0 + 1);

    const float fx = cx - static_cast<float>(x0);
    const float fy = cy - static_cast<float>(y0);

    const float v00 = img.at(y0, x0);
    const float v10 = img.at(y0, x1);
    const float v01 = img.at(y1, x0);
    const float v11 = img.at(y1, x1);

    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10)
         + fy          * ((1.0f - fx) * v01 + fx * v11);
}

} // namespace

AlignmentResult FrameAligner::warpToReference(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const AlignmentParams& params) {
    return warpToReference(
        prev,
        curr,
        next,
        principalFlow.backwardFlow,
        principalFlow.forwardFlow,
        params
    );
}

AlignmentResult FrameAligner::warpToReference(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const FlowField& backwardFlow,
    const FlowField& forwardFlow,
    const AlignmentParams& params) {
    if (prev.empty() || curr.empty() || next.empty()) {
        throw std::invalid_argument("Input frames to FrameAligner must not be empty.");
    }

    const int w = curr.width;
    const int h = curr.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    if (prev.width != w || prev.height != h || next.width != w || next.height != h ||
        backwardFlow.width != w || backwardFlow.height != h ||
        forwardFlow.width != w || forwardFlow.height != h) {
        throw std::invalid_argument("All frames and flow fields must have identical dimensions.");
    }

    AlignmentResult result;
    result.warpedPrev = FloatImage(w, h, 1);
    result.warpedNext = FloatImage(w, h, 1);
    result.inBoundsMask.assign(numPixels, 0);

    const float minX = params.borderMargin;
    const float maxX = static_cast<float>(w - 1) - params.borderMargin;
    const float minY = params.borderMargin;
    const float maxY = static_cast<float>(h - 1) - params.borderMargin;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;

            const float wxPrev = static_cast<float>(x) + backwardFlow.u[idx];
            const float wyPrev = static_cast<float>(y) + backwardFlow.v[idx];
            const float wxNext = static_cast<float>(x) + forwardFlow.u[idx];
            const float wyNext = static_cast<float>(y) + forwardFlow.v[idx];

            result.warpedPrev.at(y, x) = sampleBilinearClamped(prev, wxPrev, wyPrev);
            result.warpedNext.at(y, x) = sampleBilinearClamped(next, wxNext, wyNext);

            const bool prevIn = (wxPrev >= minX && wxPrev <= maxX && wyPrev >= minY && wyPrev <= maxY);
            const bool nextIn = (wxNext >= minX && wxNext <= maxX && wyNext >= minY && wyNext <= maxY);

            if (prevIn && nextIn) {
                result.inBoundsMask[idx] = 1;
            }
        }
    }

    result.inBoundsPixels.reserve(numPixels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (result.inBoundsMask[static_cast<size_t>(y) * w + x]) {
                result.inBoundsPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    return result;
}

AlignmentResult FrameAligner::alignAndDetectMisaligned(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const AlignmentParams& params) {
    AlignmentResult result = warpToReference(prev, curr, next, principalFlow, params);
    const int w = curr.width;
    const int h = curr.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    result.residualErrorMap = FloatImage(w, h, 1);
    result.misalignedMask.assign(numPixels, 0);

    std::vector<float> validErrors;
    validErrors.reserve(numPixels);

    for (size_t i = 0; i < numPixels; ++i) {
        if (!result.inBoundsMask[i]) continue;
        const float errPrev = std::fabs(curr.data[i] - result.warpedPrev.data[i]);
        const float errNext = std::fabs(curr.data[i] - result.warpedNext.data[i]);
        const float err = std::max(errPrev, errNext);
        result.residualErrorMap.data[i] = err;
        validErrors.push_back(err);
    }

    if (!validErrors.empty()) {
        const size_t mid = validErrors.size() / 2;
        std::nth_element(validErrors.begin(), validErrors.begin() + static_cast<ptrdiff_t>(mid), validErrors.end());
        const double med = static_cast<double>(validErrors[mid]);

        std::vector<float> absDev(validErrors.size());
        for (size_t i = 0; i < validErrors.size(); ++i) {
            absDev[i] = std::fabs(validErrors[i] - static_cast<float>(med));
        }
        std::nth_element(absDev.begin(), absDev.begin() + static_cast<ptrdiff_t>(mid), absDev.end());
        const double sigma = 1.4826 * static_cast<double>(absDev[mid]);

        result.adaptiveThreshold = std::clamp(
            med + params.madMultiplier * sigma,
            params.minResidualThreshold,
            params.maxResidualThreshold
        );

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                if (result.inBoundsMask[idx] &&
                    static_cast<double>(result.residualErrorMap.data[idx]) > result.adaptiveThreshold) {
                    result.misalignedMask[idx] = 1;
                    result.misalignedPixels.push_back(PixelCoord{x, y});
                }
            }
        }
    }

    return result;
}