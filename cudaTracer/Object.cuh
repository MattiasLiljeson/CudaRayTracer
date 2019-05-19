#ifndef OBJECT_CUH
#define OBJECT_CUH

#include "Vec.cuh"

struct Object {
    enum MaterialType {
        DIFFUSE_AND_GLOSSY,
        REFLECTION_AND_REFRACTION,
        REFLECTION
    };

    // material properties
    MaterialType materialType;
    float ior;
    float Kd, Ks;
    Vec3f diffuseColor;
    float specularExponent;

    __host__ __device__ static Object *init(Object *o) {
        o->materialType = DIFFUSE_AND_GLOSSY;
        o->ior = 1.3f;
        o->Kd = 0.8f;
        o->Ks = 0.2f;
        o->specularExponent = 25.0f;
        o->diffuseColor = Vec<float, 3>(0.2f, 0.2f, 0.2f);
        return o;
    }

    __host__ Object() {
        materialType = DIFFUSE_AND_GLOSSY;
        ior = 1.3f;
        Kd = 0.8f;
        Ks = 0.2f;
        specularExponent = 25.0f;
        diffuseColor = Vec<float, 3>(0.2f, 0.2f, 0.2f);
    }

    //__device__ virtual ~Object() {}
    //__device__ virtual bool intersect(const Vec3f &orig, const Vec3f &dir,
    //                                  float &tnear, uint32_t &index,
    //                                  Vec2f &uv) const = 0;
    //__device__ virtual void getSurfaceProperties(const Vec3f &, const Vec3f &,
    //                                             const uint32_t &, const Vec2f
    //                                             &, Vec3f &, Vec2f &) const =
    //                                             0;
    __device__ Vec3f evalDiffuseColor(const Vec2f &) const {
        return diffuseColor;
    }
};

#endif