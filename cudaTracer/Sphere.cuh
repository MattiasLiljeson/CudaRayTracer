#ifndef SPHERE_H
#define SPHERE_H

#include "Vec.cuh"

enum MaterialType { DIFFUSE_AND_GLOSSY, REFLECTION_AND_REFRACTION, REFLECTION };

inline __device__ bool solveQuadratic(const float &a, const float &b,
                                      const float &c, float &x0, float &x1) {
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr == 0)
        x0 = x1 = -0.5 * b / a;
    else {
        float q = (b > 0) ? -0.5 * (b + sqrt(discr)) : -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) {
        float tmp = x0;
        x0 = x1;
        x1 = tmp;
    }
    return true;
}

template <class T>
class Object {
   public:
    // material properties
    MaterialType materialType;
    float ior;
    float Kd, Ks;
    Vec3f diffuseColor;
    float specularExponent;

    __device__ Object() {
        materialType = DIFFUSE_AND_GLOSSY;
        ior = 1.3f;
        Kd = 0.8f;
        Ks = 0.2f;
        specularExponent = 25.0f;
        diffuseColor = Vec<float, 3>(0.2f, 0.2f, 0.2f);
    }

    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              uint32_t &index, Vec2f &uv) const {
        static_cast<T *>(this)->intersect(orig, dir, tnear, index, uv);
    }
    __device__ void getSurfaceProperties(const Vec3f &P, const Vec3f &I,
                                         const uint32_t &index, const Vec2f &uv,
                                         Vec3f &N, Vec2f &st) const {
        static_cast<T *>(this)->getSurfaceProperties(P, I, index, uv, N, st);
    }
    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        static_cast<T *>(this)->evalDiffuseColor(st);
    }
};

class Sphere : public Object<Sphere> {
   public:
    // Object object;
    Vec3f center;
    float radius, radius2;

    __host__ __device__ Sphere(const Vec3f &c, const float &r) {
        center = c;
        radius = r;
        radius2 = r * r;
    }
    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              uint32_t &index, Vec2f &uv) const {
        // analytic solution
        Vec3f L = orig - center;
        float a = dir.dot(dir);
        float b = 2 * dir.dot(L);
        float c = L.dot(L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        tnear = t0;

        return true;
    }
    __device__ void getSurfaceProperties(const Vec3f &P, const Vec3f &I,
                                         const uint32_t &index, const Vec2f &uv,
                                         Vec3f &N, Vec2f &st) const {
        N = (P - center).normalized();
    }

    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        return diffuseColor;
    }
};

#endif