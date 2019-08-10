#ifndef OPTIONS_CUH
#define OPTIONS_CUH

#include <cstdint>
#include <string>
#include "Vec.cuh"

#define PI 3.1415926536f

struct ModelDesc {
    std::string folder;
    std::string fname;
    Mat44f transform;
};

struct HostOpts {
    ModelDesc model;
    int lightCnt;
    float fov;
};

struct DeviceOpts {
    float imageAspectRatio;
    float scale;
    int width;
    int height;
    Vec3f backgroundColor;
    float shadowBias;
    int blockSize;
    int maxDepth;
    int samples;
    bool gammaCorrection;
};

struct Options {
    HostOpts host;
    DeviceOpts device;
};
inline __host__ __device__ float deg2rad(const float &deg) {
    return deg * PI / 180;
}

#endif