#include "DenseMotionSegmenter.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

class FlatDinicMaxFlow {
private:
    struct Edge {
        int to;
        int next;
        double cap;
    };

    int numNodes;
    std::vector<int> head;
    std::vector<Edge> edges;
    std::vector<int> level;
    std::vector<int> ptr;
    std::vector<int> queueBuf;

    bool bfs(int s, int t) {
        std::fill(level.begin(), level.end(), -1);
        level[static_cast<size_t>(s)] = 0;

        int qHead = 0;
        int qTail = 0;
        queueBuf[static_cast<size_t>(qTail++)] = s;

        while (qHead < qTail) {
            const int v = queueBuf[static_cast<size_t>(qHead++)];
            const int nextLevel = level[static_cast<size_t>(v)] + 1;

            for (int e = head[static_cast<size_t>(v)]; e != -1; e = edges[static_cast<size_t>(e)].next) {
                const Edge& edge = edges[static_cast<size_t>(e)];
                if (edge.cap > 1e-12 && level[static_cast<size_t>(edge.to)] == -1) {
                    level[static_cast<size_t>(edge.to)] = nextLevel;
                    queueBuf[static_cast<size_t>(qTail++)] = edge.to;
                }
            }
        }

        return level[static_cast<size_t>(t)] != -1;
    }

    double dfs(int v, int t, double pushed) {
        if (pushed <= 1e-12 || v == t) {
            return pushed;
        }

        const int targetLevel = level[static_cast<size_t>(v)] + 1;
        for (int& e = ptr[static_cast<size_t>(v)]; e != -1; e = edges[static_cast<size_t>(e)].next) {
            Edge& edge = edges[static_cast<size_t>(e)];
            const int tr = edge.to;
            if (level[static_cast<size_t>(tr)] != targetLevel || edge.cap <= 1e-12) {
                continue;
            }

            const double trPushed = dfs(tr, t, std::min(pushed, edge.cap));
            if (trPushed <= 1e-12) {
                continue;
            }

            edge.cap -= trPushed;
            edges[static_cast<size_t>(e ^ 1)].cap += trPushed;
            return trPushed;
        }

        return 0.0;
    }

public:
    FlatDinicMaxFlow(int n, size_t estimatedDirectedEdges)
        : numNodes(n),
          head(static_cast<size_t>(n), -1),
          level(static_cast<size_t>(n), -1),
          ptr(static_cast<size_t>(n), -1),
          queueBuf(static_cast<size_t>(n), 0) {
        edges.reserve(estimatedDirectedEdges);
    }

    void addDirectedEdge(int from, int to, double cap) {
        if (cap <= 0.0) return;
        const int fwdIdx = static_cast<int>(edges.size());
        edges.push_back(Edge{to, head[static_cast<size_t>(from)], cap});
        head[static_cast<size_t>(from)] = fwdIdx;

        const int bwdIdx = static_cast<int>(edges.size());
        edges.push_back(Edge{from, head[static_cast<size_t>(to)], 0.0});
        head[static_cast<size_t>(to)] = bwdIdx;
    }

    void addUndirectedEdge(int u, int v, double cap) {
        if (cap <= 0.0) return;
        const int fwdIdx = static_cast<int>(edges.size());
        edges.push_back(Edge{v, head[static_cast<size_t>(u)], cap});
        head[static_cast<size_t>(u)] = fwdIdx;

        const int bwdIdx = static_cast<int>(edges.size());
        edges.push_back(Edge{u, head[static_cast<size_t>(v)], cap});
        head[static_cast<size_t>(v)] = bwdIdx;
    }

    double maxFlow(int s, int t) {
        double flow = 0.0;
        while (bfs(s, t)) {
            ptr = head;
            while (true) {
                const double pushed = dfs(s, t, std::numeric_limits<double>::infinity());
                if (pushed <= 1e-12) {
                    break;
                }
                flow += pushed;
            }
        }
        return flow;
    }

    bool isSourceSide(int v) const {
        return level[static_cast<size_t>(v)] != -1;
    }
};

FloatImage convertToNormalizedFloat(const Image& img) {
    if (img.empty()) {
        throw std::invalid_argument("Reference image must not be empty.");
    }
    FloatImage fImg(img.width, img.height, img.channels);
    constexpr float inv255 = 1.0f / 255.0f;
    const int totalElems = static_cast<int>(img.data.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < totalElems; ++i) {
        fImg.data[static_cast<size_t>(i)] = static_cast<float>(img.data[static_cast<size_t>(i)]) * inv255;
    }
    return fImg;
}

inline double squaredColorDistance(const FloatImage& img, int y1, int x1, int y2, int x2) {
    double distSq = 0.0;
    for (int c = 0; c < img.channels; ++c) {
        const double diff = static_cast<double>(img.at(y1, x1, c)) - static_cast<double>(img.at(y2, x2, c));
        distSq += diff * diff;
    }
    return distSq;
}

DenseSegmentationResult runGraphCutCore(
    const FloatImage& referenceFrame,
    const std::vector<uint8_t>& movingSeedMask,
    const std::vector<uint8_t>& softStaticMask,
    const std::vector<uint8_t>& hardStaticMask,
    const DenseSegmentationParams& params) {
    const int w = referenceFrame.width;
    const int h = referenceFrame.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    DenseSegmentationResult result;
    result.denseMovingMask.assign(numPixels, 0);

    int minX = w, minY = h, maxX = -1, maxY = -1;
    bool hasMovingSeed = false;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (movingSeedMask[idx]) {
                hasMovingSeed = true;
                minX = std::min(minX, x);
                minY = std::min(minY, y);
                maxX = std::max(maxX, x);
                maxY = std::max(maxY, y);
            }
        }
    }

    if (!hasMovingSeed) {
        return result;
    }

    const int pad = std::max(8, params.roiPadding);
    const int roiX0 = std::max(0, minX - pad);
    const int roiY0 = std::max(0, minY - pad);
    const int roiX1 = std::min(w - 1, maxX + pad);
    const int roiY1 = std::min(h - 1, maxY + pad);

    const int roiW = roiX1 - roiX0 + 1;
    const int roiH = roiY1 - roiY0 + 1;
    const size_t roiPixels = static_cast<size_t>(roiW) * roiH;

    constexpr int dx4[2] = {1, 0};
    constexpr int dy4[2] = {0, 1};
    constexpr double dist4[2] = {1.0, 1.0};

    constexpr int dx8[4] = {1, 0, 1, -1};
    constexpr int dy8[4] = {0, 1, 1,  1};
    const double invSqrt2 = 1.0 / std::sqrt(2.0);
    const double invDist8[4] = {1.0, 1.0, invSqrt2, invSqrt2};

    const int numDirs = params.use8Connectivity ? 4 : 2;
    const int* dx = params.use8Connectivity ? dx8 : dx4;
    const int* dy = params.use8Connectivity ? dy8 : dy4;

    double sumColorDiffSq = 0.0;
    long long edgeCount = 0;
    std::vector<double> edgeDiffSq(roiPixels * static_cast<size_t>(numDirs), -1.0);

    for (int ry = 0; ry < roiH; ++ry) {
        const int y = roiY0 + ry;
        for (int rx = 0; rx < roiW; ++rx) {
            const int x = roiX0 + rx;
            const size_t baseEdgeIdx = (static_cast<size_t>(ry) * roiW + rx) * static_cast<size_t>(numDirs);

            for (int d = 0; d < numDirs; ++d) {
                const int nrx = rx + dx[d];
                const int nry = ry + dy[d];
                if (nrx >= 0 && nrx < roiW && nry >= 0 && nry < roiH) {
                    const double diffSq = squaredColorDistance(referenceFrame, y, x, roiY0 + nry, roiX0 + nrx);
                    edgeDiffSq[baseEdgeIdx + static_cast<size_t>(d)] = diffSq;
                    sumColorDiffSq += diffSq;
                    edgeCount++;
                }
            }
        }
    }

    const double meanDiffSq = (edgeCount > 0) ? (sumColorDiffSq / static_cast<double>(edgeCount)) : 0.0;
    const double effectiveMeanDiffSq = std::max(meanDiffSq, params.minMeanColorDiffSq);
    const double beta = (effectiveMeanDiffSq > 1e-12) ? (1.0 / (2.0 * effectiveMeanDiffSq)) : 0.0;
    result.computedBeta = beta;

    const int sourceNode = static_cast<int>(roiPixels);
    const int sinkNode = static_cast<int>(roiPixels + 1);
    const size_t estEdges = roiPixels * (params.use8Connectivity ? 10 : 6);
    FlatDinicMaxFlow graph(static_cast<int>(roiPixels + 2), estEdges);

    for (int ry = 0; ry < roiH; ++ry) {
        const int y = roiY0 + ry;
        const bool onCropBorderY = (ry == 0 && roiY0 > 0) || (ry == roiH - 1 && roiY1 < h - 1);

        for (int rx = 0; rx < roiW; ++rx) {
            const int x = roiX0 + rx;
            const bool onCropBorderX = (rx == 0 && roiX0 > 0) || (rx == roiW - 1 && roiX1 < w - 1);

            const int roiNode = ry * roiW + rx;
            const size_t fullIdx = static_cast<size_t>(y) * w + x;

            if (movingSeedMask[fullIdx]) {
                graph.addDirectedEdge(sourceNode, roiNode, params.movingSeedWeight);
            } else if ((!hardStaticMask.empty() && hardStaticMask[fullIdx]) || onCropBorderX || onCropBorderY) {
                graph.addDirectedEdge(roiNode, sinkNode, params.staticSeedWeight);
            } else if (!softStaticMask.empty() && softStaticMask[fullIdx]) {
                graph.addDirectedEdge(roiNode, sinkNode, params.verifiedStaticWeight);
            } else if (params.unknownStaticPrior > 0.0) {
                graph.addDirectedEdge(roiNode, sinkNode, params.unknownStaticPrior);
            }
        }
    }

    for (int ry = 0; ry < roiH; ++ry) {
        for (int rx = 0; rx < roiW; ++rx) {
            const int u = ry * roiW + rx;
            const size_t baseEdgeIdx = static_cast<size_t>(u) * static_cast<size_t>(numDirs);

            for (int d = 0; d < numDirs; ++d) {
                const double diffSq = edgeDiffSq[baseEdgeIdx + static_cast<size_t>(d)];
                if (diffSq >= 0.0) {
                    const int v = (ry + dy[d]) * roiW + (rx + dx[d]);
                    const double invDist = params.use8Connectivity ? invDist8[d] : (1.0 / dist4[d]);
                    const double weight = params.lambda * invDist * std::exp(-beta * diffSq);
                    graph.addUndirectedEdge(u, v, weight);
                }
            }
        }
    }

    result.minCutEnergy = graph.maxFlow(sourceNode, sinkNode);

    for (int ry = 0; ry < roiH; ++ry) {
        const int y = roiY0 + ry;
        for (int rx = 0; rx < roiW; ++rx) {
            const int roiNode = ry * roiW + rx;
            if (graph.isSourceSide(roiNode)) {
                const int x = roiX0 + rx;
                result.denseMovingMask[static_cast<size_t>(y) * w + x] = 1;
                result.movingPixels.push_back(PixelCoord{x, y});
            }
        }
    }

    return result;
}

} // namespace

DenseSegmentationResult DenseMotionSegmenter::segment(
    const Image& referenceFrame,
    const AlignmentResult& alignment,
    const SecondaryFlowResult& secondaryFlow,
    const ParallaxRigidityResult& rigidity,
    const std::vector<PixelCoord>& staticRefPixels,
    const DenseSegmentationParams& params) {
    FloatImage floatRef = convertToNormalizedFloat(referenceFrame);
    return segment(floatRef, alignment, secondaryFlow, rigidity, staticRefPixels, params);
}

DenseSegmentationResult DenseMotionSegmenter::segment(
    const FloatImage& referenceFrame,
    const AlignmentResult& alignment,
    const SecondaryFlowResult& secondaryFlow,
    const ParallaxRigidityResult& rigidity,
    const std::vector<PixelCoord>& staticRefPixels,
    const DenseSegmentationParams& params) {
    (void)alignment;
    (void)secondaryFlow;

    if (referenceFrame.empty()) {
        throw std::invalid_argument("referenceFrame must not be empty.");
    }

    const int w = referenceFrame.width;
    const int h = referenceFrame.height;
    const size_t numPixels = static_cast<size_t>(w) * h;

    if (rigidity.movingMask.size() != numPixels || rigidity.rigidStaticMask.size() != numPixels) {
        throw std::invalid_argument("Rigidity mask dimensions must match referenceFrame.");
    }

    std::vector<uint8_t> hardStaticMask(numPixels, 0);
    for (const auto& pt : staticRefPixels) {
        if (pt.x >= 0 && pt.x < w && pt.y >= 0 && pt.y < h) {
            const size_t idx = static_cast<size_t>(pt.y) * w + pt.x;
            if (!rigidity.movingMask[idx]) {
                hardStaticMask[idx] = 1;
            }
        }
    }

    return runGraphCutCore(
        referenceFrame,
        rigidity.movingMask,
        rigidity.rigidStaticMask,
        hardStaticMask,
        params
    );
}

DenseSegmentationResult DenseMotionSegmenter::segment(
    const FloatImage& referenceFrame,
    const std::vector<uint8_t>& movingSeedMask,
    const std::vector<uint8_t>& verifiedStaticMask,
    const DenseSegmentationParams& params,
    const std::vector<uint8_t>& misalignedMask) {
    (void)misalignedMask;
    if (referenceFrame.empty()) {
        throw std::invalid_argument("referenceFrame must not be empty.");
    }
    const size_t numPixels = static_cast<size_t>(referenceFrame.width) * referenceFrame.height;
    if (movingSeedMask.size() != numPixels || verifiedStaticMask.size() != numPixels) {
        throw std::invalid_argument("Seed mask dimensions must match referenceFrame.");
    }
    return runGraphCutCore(referenceFrame, movingSeedMask, {}, verifiedStaticMask, params);
}