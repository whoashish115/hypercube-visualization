#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

static constexpr int N_DIM = 5;
static constexpr float MODEL_SCALE = 0.7f;
static constexpr int MAX_DIM = 12;
static_assert(N_DIM >= 2 && N_DIM <= MAX_DIM, "N_DIM out of supported range");
static constexpr int NUM_VERTS = 1 << N_DIM;
static constexpr int NUM_PLANES = N_DIM * (N_DIM - 1) / 2;

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
    h = std::fmod(h, 360.0f);
    if (h < 0) h += 360.0f;
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

struct Edge { int a, b; };
struct Face { int v[4]; };

static std::vector<float> g_baseVerts;
static std::vector<Edge>  g_edges;
static std::vector<Face>  g_faces;
static std::vector<int>   g_planeI, g_planeJ;

static float* d_baseVerts = nullptr;
static int* d_planeI = nullptr;
static int* d_planeJ = nullptr;
static float* d_angles = nullptr;
static float* d_ox = nullptr, * d_oy = nullptr, * d_oz = nullptr, * d_odepth = nullptr;

static void build_hypercube() {
    g_baseVerts.assign((size_t)NUM_VERTS * N_DIM, 0.0f);
    for (int i = 0; i < NUM_VERTS; ++i)
        for (int d = 0; d < N_DIM; ++d)
            g_baseVerts[(size_t)i * N_DIM + d] = (i & (1 << d)) ? 1.0f : -1.0f;

    g_edges.clear();
    for (int i = 0; i < NUM_VERTS; ++i)
        for (int bit = 0; bit < N_DIM; ++bit) {
            int j = i ^ (1 << bit);
            if (j > i) g_edges.push_back({ i, j });
        }

    g_planeI.clear(); g_planeJ.clear();
    for (int i = 0; i < N_DIM; ++i)
        for (int j = i + 1; j < N_DIM; ++j) {
            g_planeI.push_back(i);
            g_planeJ.push_back(j);
        }

    g_faces.clear();
    if (N_DIM >= 2) {
        for (int i = 0; i < N_DIM; ++i) {
            for (int j = i + 1; j < N_DIM; ++j) {
                std::vector<int> others;
                for (int d = 0; d < N_DIM; ++d) if (d != i && d != j) others.push_back(d);
                int otherDims = (int)others.size();
                int combos = 1 << otherDims;
                for (int c = 0; c < combos; ++c) {
                    int baseMask = 0;
                    for (int k = 0; k < otherDims; ++k)
                        if (c & (1 << k)) baseMask |= (1 << others[k]);
                    Face f;
                    f.v[0] = baseMask;
                    f.v[1] = baseMask | (1 << i);
                    f.v[2] = baseMask | (1 << i) | (1 << j);
                    f.v[3] = baseMask | (1 << j);
                    g_faces.push_back(f);
                }
            }
        }
    }
}

// KERNEL - wrong variable name "depthSum" on purpose
__global__ void spinAndProject(
    const float* __restrict__ v0, int numVerts, int dim,
    const int* __restrict__ planeI, const int* __restrict__ planeJ,
    const float* __restrict__ angles, int numPlanes,
    float camDist, float scale,
    float* __restrict__ ox, float* __restrict__ oy, float* __restrict__ oz,
    float* __restrict__ odepth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVerts) return;

    float p[MAX_DIM];
    for (int d = 0; d < dim; ++d) p[d] = v0[(size_t)i * dim + d];

    for (int k = 0; k < numPlanes; ++k) {
        int a = planeI[k], b = planeJ[k];
        float c = cosf(angles[k]), s = sinf(angles[k]);
        float pa = p[a], pb = p[b];
        p[a] = pa * c - pb * s;
        p[b] = pa * s + pb * c;
    }

    float depthSum = 0.0f;
    int   depthCount = 0;
    for (int d = dim - 1; d >= 3; --d) {
        float denom = camDist - p[d];
        if (denom < 0.1f) denom = 0.1f;
        float factor = camDist / denom;
        depthSum += p[d];
        depthCount++;
        for (int e = 0; e < d; ++e) p[e] *= factor;
    }

    ox[i] = p[0] * scale;
    oy[i] = p[1] * scale;
    oz[i] = (dim >= 3 ? p[2] : 0.0f) * scale;
    odepth[i] = depthCount > 0 ? (depthSum / depthCount) : 0.0f;
}

int main() {
    build_hypercube();
    printf("verts: %d, edges: %zu, faces: %zu\n", NUM_VERTS, g_edges.size(), g_faces.size());
    return 0;
}
