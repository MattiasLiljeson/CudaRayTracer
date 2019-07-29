#include <cstdint>
#include "Vec.cuh"
#ifndef OPTIONS_CUH
#define OPTIONS_CUH

#define PI 3.1415926536f

struct Options {
    uint32_t width;
    uint32_t height;
    float fov;
    uint8_t maxDepth;
    uint8_t samples;
    Vec3f backgroundColor;
    float bias;
    float scale;
    float imageAspectRatio;
};

inline __host__ __device__ float deg2rad(const float &deg) {
    return deg * PI / 180;
}

#endif