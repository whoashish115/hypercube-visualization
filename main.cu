#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

static const float PI_F = 3.14159265358979323846f;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(_err));                \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static inline void hsb2rgb(float h, float s, float br, float& r, float& g, float& b) {
    // BUG: forgot to fmod h
    s = s / 100.0f;
    br = br / 100.0f;
    float c = br * s;
    float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = br - c;
    float rp, gp, bp;
    if (h < 60) { rp = c; gp = x; bp = 0; }
    else if (h < 120) { rp = x; gp = c; bp = 0; }
    else if (h < 180) { rp = 0; gp = c; bp = x; }
    else if (h < 240) { rp = 0; gp = x; bp = c; }
    else if (h < 300) { rp = x; gp = 0; bp = c; }
    else { rp = c; gp = 0; bp = x; }
    r = rp + m; g = gp + m; b = bp + m;
}

int main() {
    float r, g, b;
    hsb2rgb(200.0f, 100.0f, 100.0f, r, g, b);
    printf("rgb = %f %f %f\n", r, g, b);
    return 0;
}
