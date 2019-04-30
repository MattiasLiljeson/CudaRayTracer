#include <cstdint>
#include "Vec3f.cuh"
#ifndef OPTIONS_CUH
#define OPTIONS_CUH

#define PI 3.1415926536f

struct Options {
    uint32_t width;
    uint32_t height;
    float fov;
    float imageAspectRatio;
    uint8_t maxDepth;
    Vec3f backgroundColor;
    float bias;
    float scale;
};

inline __host__ __device__ float deg2rad(const float &deg) {
    return deg * PI / 180;
}

#endif