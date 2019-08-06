#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "Vec.cuh"

struct Light {
    __host__ constexpr Light()
        : position(Vec3f(0.0f, 0.0f, 0.0f)),
          intensity(Vec3f(0.0f, 0.0f, 0.0f)),
          power(1.0f) {}
    __host__ constexpr Light(const Vec3f &p, const Vec3f &i, float power)
        : position(p), intensity(i), power(power) {}

    Vec3f position;
    Vec3f intensity;
    float power;
};

#endif  // !LIGHT_CUH
