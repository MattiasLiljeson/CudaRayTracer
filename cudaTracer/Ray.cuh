#ifndef RAY_CUH

#include <limits>

#include "Vec.cuh"
struct Ray {
    Vec3f origin;
    Vec3f dir;
    float tMax;

    Ray() : tMax(FLT_MAX) {}
    Ray(const Vec3f& o, const Vec3f& d, float tMax = FLT_MAX)
        : origin(o), dir(d), tMax(tMax) {}
    Vec3f operator()(float t) const { return origin + dir * t; }
};
#endif RAY_CUH