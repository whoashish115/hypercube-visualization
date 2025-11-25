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

int main() {
    printf("pi = %f\n", PI_F);
    printf("clamp test: %f\n", clampf(5.0f, 0.0f, 1.0f));
    return 0;
}
