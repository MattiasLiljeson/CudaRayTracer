#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "Vec.cuh"

struct Material {
    enum Type {
        DIFFUSE_AND_GLOSSY,
        REFLECTION_AND_REFRACTION,
        REFLECTION
    };

    Type materialType;
    float ior;
    float Kd, Ks;
    Vec3f diffuseColor;
    float specularExponent;

    __host__ __device__ Material() {
        materialType = DIFFUSE_AND_GLOSSY;
        ior = 1.3f;
        Kd = 0.8f;
        Ks = 0.2f;
        specularExponent = 25.0f;
        diffuseColor = Vec<float, 3>(0.2f, 0.2f, 0.2f);
    }

    __device__ Vec3f evalDiffuseColor(const Vec2f & vec) const {
        return diffuseColor;
    }
};

#endif