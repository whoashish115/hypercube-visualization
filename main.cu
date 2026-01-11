#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "opengl32.lib")
#endif

#include <SDL3/SDL.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static constexpr int N_DIM = 5;
static constexpr float MODEL_SCALE = 0.7f;

static constexpr int MAX_DIM = 12;
static_assert(N_DIM >= 2 && N_DIM <= MAX_DIM, "N_DIM out of supported range");

static constexpr int NUM_VERTS = 1 << N_DIM;
static constexpr int NUM_PLANES = N_DIM * (N_DIM - 1) / 2;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static const float PI_F = 3.14159265358979323846f;

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
    for (int i = 0; i < NUM_VERTS; ++i) {
        for (int d = 0; d < N_DIM; ++d) {
            g_baseVerts[(size_t)i * N_DIM + d] = (i & (1 << d)) ? 1.0f : -1.0f;
        }
    }

    g_edges.clear();
    for (int i = 0; i < NUM_VERTS; ++i) {
        for (int bit = 0; bit < N_DIM; ++bit) {
            int j = i ^ (1 << bit);
            if (j > i) g_edges.push_back({ i, j });
        }
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

static void allocDeviceBuffers() {
    CUDA_CHECK(cudaMalloc(&d_baseVerts, sizeof(float) * g_baseVerts.size()));
    CUDA_CHECK(cudaMalloc(&d_planeI, sizeof(int) * NUM_PLANES));
    CUDA_CHECK(cudaMalloc(&d_planeJ, sizeof(int) * NUM_PLANES));
    CUDA_CHECK(cudaMalloc(&d_angles, sizeof(float) * NUM_PLANES));
    CUDA_CHECK(cudaMalloc(&d_ox, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_oy, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_oz, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_odepth, sizeof(float) * NUM_VERTS));

    CUDA_CHECK(cudaMemcpy(d_baseVerts, g_baseVerts.data(), sizeof(float) * g_baseVerts.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planeI, g_planeI.data(), sizeof(int) * NUM_PLANES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planeJ, g_planeJ.data(), sizeof(int) * NUM_PLANES, cudaMemcpyHostToDevice));
}

static void freeDeviceBuffers() {
    cudaFree(d_baseVerts); cudaFree(d_planeI); cudaFree(d_planeJ); cudaFree(d_angles);
    cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_oz); cudaFree(d_odepth);
}

static std::vector<float> g_ox, g_oy, g_oz, g_odepth;

static void stepFrame(double t) {
    std::vector<float> angles(NUM_PLANES);
    for (int k = 0; k < NUM_PLANES; ++k) {
        float speed = 0.12f + 0.05f * (float)((k * 37) % 11);
        angles[k] = (float)(t * speed);
    }
    CUDA_CHECK(cudaMemcpy(d_angles, angles.data(), sizeof(float) * NUM_PLANES, cudaMemcpyHostToDevice));

    const float camDist = 3.2f;
    const float scale = 1.0f;

    const int threads = 128;
    const int blocks = (NUM_VERTS + threads - 1) / threads;

    spinAndProject << <blocks, threads >> > (
        d_baseVerts, NUM_VERTS, N_DIM,
        d_planeI, d_planeJ, d_angles, NUM_PLANES,
        camDist, scale,
        d_ox, d_oy, d_oz, d_odepth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    g_ox.resize(NUM_VERTS); g_oy.resize(NUM_VERTS);
    g_oz.resize(NUM_VERTS); g_odepth.resize(NUM_VERTS);

    CUDA_CHECK(cudaMemcpy(g_ox.data(), d_ox, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_oy.data(), d_oy, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_oz.data(), d_oz, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_odepth.data(), d_odepth, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));
}


static const float WORLD_SCALE = 160.0f;

static const float BLUE_HUE_MIN = 200.0f;
static const float BLUE_HUE_MAX = 220.0f;

static inline float pickHue(float wobble) {
    return BLUE_HUE_MIN + clampf(wobble, 0.0f, 1.0f) * (BLUE_HUE_MAX - BLUE_HUE_MIN);
}

static inline float px(int i) { return g_ox[i] * WORLD_SCALE * MODEL_SCALE; }
static inline float py(int i) { return g_oy[i] * WORLD_SCALE * MODEL_SCALE; }
static inline float pz(int i) { return g_oz[i] * WORLD_SCALE * MODEL_SCALE; }

static void drawFaces(double t) {
    if (g_faces.empty()) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glBegin(GL_QUADS);
    for (int f = 0; f < (int)g_faces.size(); ++f) {
        const Face& fc = g_faces[f];
         float depthAvg = 0.0f;
        for (int k = 0; k < 4; ++k) depthAvg += g_odepth[fc.v[k]];
        depthAvg *= 0.25f;
        float depthFade = clampf(0.4f + depthAvg * 0.35f, 0.2f, 1.0f);
        float wobble = 0.5f + 0.5f * std::sin((float)(t * 0.15) + f * 0.11f);
        float hue = pickHue(wobble);
        float r, g, b;
        hsb2rgb(190.0f, 100.0f, 100.0f, r, g, b);

        glColor4f(r, g, b, 0.05f * depthFade);
        for (int k = 0; k < 4; ++k) {
            int vi = fc.v[k];
            glVertex3f(px(vi), py(vi), pz(vi));
        }
    }
    glEnd();

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

static void drawEdges(double t) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    struct GlowPass { float width; float alphaScale; };
    static const GlowPass passes[3] = {
        { 14.0f, 0.10f },
        { 8.0f,  0.20f },
        { 4.0f,  0.35f },
    };

    for (const GlowPass& gp : passes) {
        glLineWidth(gp.width);
        glBegin(GL_LINES);
        for (int e = 0; e < (int)g_edges.size(); ++e) {
            const Edge& ed = g_edges[e];
            float depthAvg = (g_odepth[ed.a] + g_odepth[ed.b]) * 0.5f;
           float depthFade = clampf(0.5f + depthAvg * 0.5f, 0.05f, 1.0f);
            float wobble = 0.5f + 0.5f * std::sin((float)(t * 0.2) + e * 0.3f);
            float hue = pickHue(wobble);
            float r, g, b;
            hsb2rgb(190.0f, 100.0f, 100.0f, r, g, b);
            glColor4f(0.0f, 0.8f, 0.8f, 0.8f * depthFade);
            glVertex3f(px(ed.a), py(ed.a), pz(ed.a));
            glVertex3f(px(ed.b), py(ed.b), pz(ed.b));
        }
        glEnd();
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (int e = 0; e < (int)g_edges.size(); ++e) {
        const Edge& ed = g_edges[e];
        float wobble = 0.5f + 0.5f * std::sin((float)(t * 0.2) + e * 0.3f);
        float hue = pickHue(wobble);
        float r, g, b;
        hsb2rgb(180.0f, 100.0f, 100.0f, r, g, b);
        glColor4f(r, g, b, 1.0f);
        glVertex3f(px(ed.a), py(ed.a), pz(ed.a));
        glVertex3f(px(ed.b), py(ed.b), pz(ed.b));
    }
    glEnd();

    glDisable(GL_BLEND);
}

static void drawVerticies(double t) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    struct GlowLayer { float size; float alphaScale; };
    static const GlowLayer layers[2] = {
        { 16.0f, 0.12f },
        { 8.0f,  0.28f },
    };

    for (const GlowLayer& layer : layers) {
        glBegin(GL_POINTS);
        for (int i = 0; i < NUM_VERTS; ++i) {
            float depthFade = clampf(0.55f + g_odepth[i] * 0.35f, 0.25f, 1.0f);
            float wobble = 0.5f + 0.5f * std::sin((float)(t * 0.2) + i * 0.4f);
            float hue = pickHue(wobble);
            float r, g, b;
            hsb2rgb(hue, 85.0f, 100.0f, r, g, b);
            glColor4f(r, g, b, layer.alphaScale * depthFade);
            glPointSize(layer.size);
            glVertex3f(px(i), py(i), pz(i));
        }
        glEnd();
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(3.5f);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_VERTS; ++i) {
        float wobble = 0.5f + 0.5f * std::sin((float)(t * 0.2) + i * 0.4f);
        float hue = pickHue(wobble);
        float r, g, b;
        hsb2rgb(hue, 30.0f, 100.0f, r, g, b);
        glColor4f(r, g, b, 1.0f);
        glVertex3f(px(i), py(i), pz(i));
    }
    glEnd();

    glDisable(GL_BLEND);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    int winW = 1280, winH = 800;
    char title[128];
    std::snprintf(title, sizeof(title), "Hypercube N=%d - CUDA + SDL3 + OpenGL", N_DIM);
    SDL_Window* window = SDL_CreateWindow(title, winW, winH,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GLContext glctx = SDL_GL_CreateContext(window);
    if (!glctx) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetSwapInterval(1);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    build_hypercube();
    allocDeviceBuffers();

    bool running = true;
    bool dragging = false;
    float yaw = 0.0f, pitchBase = 15.0f;
    float camDist = 700.0f;
    Uint64 lastTicks = SDL_GetTicks();
    double tsec = 0.0;
    bool autoSpin = true;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (ev.key.key == SDLK_ESCAPE) running = false;
                if (ev.key.key == SDLK_R) tsec = 0.0;
                if (ev.key.key == SDLK_SPACE) autoSpin = !autoSpin;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (ev.button.button == SDL_BUTTON_LEFT) dragging = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (ev.button.button == SDL_BUTTON_LEFT) dragging = false;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (dragging) {
                    yaw += ev.motion.xrel * 0.25f;
                    pitchBase += ev.motion.yrel * 0.25f;
                    pitchBase = clampf(pitchBase, -89.0f, 89.0f);
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                camDist -= ev.wheel.y * 30.0f;
                camDist = clampf(camDist, 150.0f, 2500.0f);
                break;
            case SDL_EVENT_WINDOW_RESIZED:
                winW = ev.window.data1;
                winH = ev.window.data2;
                break;
            default:
                break;
            }
        }

        Uint64 now = SDL_GetTicks();
        double dt = (now - lastTicks) / 1000.0;
        lastTicks = now;
        if (autoSpin) tsec += dt;

        if (!dragging) yaw += (float)(dt * 6.0);
        float pitch = pitchBase + std::sin(tsec * 0.15) * 4.0f;

        stepFrame(tsec);

        SDL_GetWindowSize(window, &winW, &winH);
        glViewport(0, 0, winW, winH);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = winH > 0 ? (float)winW / (float)winH : 1.0f;
        float nearP = 1.0f, farP = 5000.0f;
        float fovRad = 45.0f * PI_F / 180.0f;
        float top = nearP * std::tan(fovRad * 0.5f);
        float right = top * aspect;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(-right, right, -top, top, nearP, farP);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -camDist);
        glRotatef(pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(yaw, 0.0f, 1.0f, 0.0f);

        drawFaces(tsec);
        drawEdges(tsec);
        drawVerticies(tsec);

        SDL_GL_SwapWindow(window);
    }

    freeDeviceBuffers();
    SDL_GL_DestroyContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
