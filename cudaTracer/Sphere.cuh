#ifndef SPHERE_H
#define SPHERE_H

#include "Vec.cuh"
#include "Object.cuh"

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

struct Sphere {
    Object object;
    Vec3f center;
    float radius, radius2;

    __host__ __device__ static Sphere sphere(const Vec3f &c, const float &r) {
        Sphere s;
        Object::init(&s.object);
        s.center = c;
        s.radius = r;
        s.radius2 = r * r;
        return s;
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
};

#endif