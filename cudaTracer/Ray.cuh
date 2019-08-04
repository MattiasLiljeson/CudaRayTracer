#ifndef RAY_CUH
#define RAY_CUH
#include <limits>

#include "Vec.cuh"
struct Ray {
    Vec3f origin;
    Vec3f dir;
    float tMax;

    //__host__ __device__ Ray() : tMax(FLT_MAX) {}
    __host__ __device__ Ray(const Vec3f& o, const Vec3f& d,
                            float tMax = FLT_MAX)
        : origin(o), dir(d), tMax(tMax) {}
    __host__ __device__ Vec3f operator()() const { return origin + dir * tMax; }
    __host__ __device__ Vec3f operator()(float t) const {
        return origin + dir * t;
    }
};
#endif RAY_CUH