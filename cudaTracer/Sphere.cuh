#ifndef SPHERE_H
#define SPHERE_H

#include "Vec3f.cuh"


enum MaterialType { DIFFUSE_AND_GLOSSY, REFLECTION_AND_REFRACTION, REFLECTION };

struct Vec2f {
    float x, y;
};

inline __host__ __device__ Vec2f vec2f(float x, float y) {
    Vec2f ret;
    ret.x = x;
    ret.y = y;
    return ret;
}
inline __host__ __device__ Vec2f vec2f(float x) { return vec2f(x); }
inline __host__ __device__ Vec2f vec2f() { return vec2f(0.0f); }
inline __device__ Vec2f times(const Vec2f v, const float &r) {
    return vec2f(v.x * r, v.y * r);
}
inline __device__ Vec2f plus(const Vec2f &v1, const Vec2f &v2) {
    return vec2f(v1.x + v2.x, v1.y + v2.y);
}

struct Object {
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
        o->diffuseColor = Vec<float,3>(0.2f, 0.2f, 0.2f);
        return o;
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

inline __device__ bool solveQuadratic(const float &a, const float &b,
                                      const float &c,
                               float &x0, float &x1) {
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

struct Sphere : public Object {
    Vec3f center;
    float radius, radius2;

    __host__ __device__ static Sphere sphere(const Vec3f &c, const float &r) {
        Sphere s;
        Object::init(&s);
        s.center = c;
        s.radius = r;
        s.radius2 = r * r;
        return s;
    }
    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              uint32_t &index, Vec2f &uv) const {
        // analytic solution
        Vec3f L = orig - center;
        float a = dir.dotProduct(dir);
        float b = 2 * dir.dotProduct(L);
        float c = L.dotProduct(L) - radius2;
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
};

#endif