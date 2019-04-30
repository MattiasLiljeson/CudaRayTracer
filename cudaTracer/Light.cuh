#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "Vec3f.cuh"

struct Light {
    __host__ constexpr Light()
        : Light(Vec3f(0.0f,0.0f,0.0f),Vec3f(0.0f,0.0f,0.0f)){}
    __host__ constexpr Light(const Vec3f &p, const Vec3f &i) : position(p), intensity(i){
    }

    Vec3f position;
    Vec3f intensity;
};

#endif  // !LIGHT_CUH
