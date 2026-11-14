#include "FrameAligner.h"
#include "ImageOps.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

double computeMedian(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double med = values[mid];
    if (values.size() % 2 == 0 && mid > 0) {
        auto maxLeft = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
        med = 0.5 * (med + *maxLeft);
    }
    return med;
}

double computeMAD(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    std::vector<double> copy = values;
    const double median = computeMedian(copy);

    std::vector<double> absDevs(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        absDevs[i] = std::fabs(values[i] - median);
    }
    return computeMedian(absDevs);
}

} // namespace

AlignmentResult FrameAligner::alignAndDetectMisaligned(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const QuadraticFlowResult& principalFlow,
    const AlignmentParams& params) {
    return alignAndDetectMisaligned(
        prev,
        curr,
        next,
        principalFlow.forwardFlow,
        principalFlow.backwardFlow,
        params
    );
}

AlignmentResult FrameAligner::alignAndDetectMisaligned(
    const FloatImage& prev,
    const FloatImage& curr,
    const FloatImage& next,
    const FlowField& forwardFlow,
    const FlowField& backwardFlow,
    const AlignmentParams& params) {
    if (prev.empty() || curr.empty() || next.empty()) {
        throw std::invalid_argument("Input frames must not be empty.");
    }
    if (prev.width != curr.width || prev.height != curr.height ||
        next.width != curr.width || next.height != curr.height) {
        throw std::invalid_argument("All three frames must have identical dimensions.");
    }
    if (prev.channels != 1 || curr.channels != 1 || next.channels != 1) {
        throw std::invalid_argument("FrameAligner expects single-channel FloatImage inputs.");
    }
    if (forwardFlow.width != curr.width || forwardFlow.height != curr.height ||
        backwardFlow.width != curr.width || backwardFlow.height != curr.height) {
        throw std::invalid_argument("Flow fields must match frame dimensions.");
    }

    const int w = curr.width;
    const int h = curr.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    AlignmentResult result;
    result.warpedNext = ImageOps::warpBackward(next, forwardFlow, &result.validNext);
    result.warpedPrev = ImageOps::warpBackward(prev, backwardFlow, &result.validPrev);
    result.residualMap = FloatImage(w, h, 1);
    result.misalignedMask.assign(numPixels, 0);

    FloatImage ixCurr, iyCurr;
    if (params.normalizeByGradient) {
        ImageOps::computeGradients(curr, ixCurr, iyCurr);
    }

    std::vector<uint8_t> evaluatedMask(numPixels, 0);
    std::vector<double> validResiduals;
    validResiduals.reserve(numPixels);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            const bool okPrev = result.validPrev[idx] != 0;
            const bool okNext = result.validNext[idx] != 0;

            const bool canEvaluate = params.requireBothValid ? (okPrev && okNext) : (okPrev || okNext);
            if (!canEvaluate) {
                result.residualMap.at(y, x) = 0.0f;
                continue;
            }

            const double cVal = static_cast<double>(curr.at(y, x));
            double normFactor = 1.0;
            if (params.normalizeByGradient) {
                const double gx = static_cast<double>(ixCurr.at(y, x));
                const double gy = static_cast<double>(iyCurr.at(y, x));
                normFactor = 1.0 / std::sqrt(gx * gx + gy * gy + params.gradientEps * params.gradientEps);
            }

            double errSum = 0.0;
            int count = 0;
            if (okPrev) {
                errSum += std::fabs(cVal - static_cast<double>(result.warpedPrev.at(y, x))) * normFactor;
                count++;
            }
            if (okNext) {
                errSum += std::fabs(cVal - static_cast<double>(result.warpedNext.at(y, x))) * normFactor;
                count++;
            }

            const double residual = errSum / static_cast<double>(count);
            result.residualMap.at(y, x) = static_cast<float>(residual);
            evaluatedMask[idx] = 1;
            validResiduals.push_back(residual);
        }
    }

    double threshold = params.minResidualThreshold;
    if (params.useAdaptiveThreshold && !validResiduals.empty()) {
        const double mad = computeMAD(validResiduals);
        const double sigma = 1.4826 * mad;
        threshold = std::max(params.minResidualThreshold, params.madMultiplier * sigma);
    }
    result.computedThreshold = threshold;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (evaluatedMask[idx] && static_cast<double>(result.residualMap.at(y, x)) > threshold) {
                result.misalignedMask[idx] = 1;
                result.misalignedPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    return result;
}