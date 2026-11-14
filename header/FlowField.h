#ifndef FLOW_FIELD_H
#define FLOW_FIELD_H

#include <cstddef>
#include <vector>

struct FlowField {
    int width = 0;
    int height = 0;
    std::vector<float> u;
    std::vector<float> v;

    FlowField() = default;
    FlowField(int w, int h)
        : width(w),
          height(h),
          u(static_cast<size_t>(w) * h, 0.0f),
          v(static_cast<size_t>(w) * h, 0.0f) {}

    bool empty() const {
        return u.empty() || v.empty() || width <= 0 || height <= 0;
    }

    float& atU(int y, int x) {
        return u[static_cast<size_t>(y) * width + x];
    }

    const float& atU(int y, int x) const {
        return u[static_cast<size_t>(y) * width + x];
    }

    float& atV(int y, int x) {
        return v[static_cast<size_t>(y) * width + x];
    }

    const float& atV(int y, int x) const {
        return v[static_cast<size_t>(y) * width + x];
    }
};

#endif // FLOW_FIELD_H