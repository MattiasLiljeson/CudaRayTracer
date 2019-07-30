#ifndef SPHERE_H
#define SPHERE_H

#include "Object.cuh"
#include "Vec.cuh"

inline __device__ bool solveQuadratic(const float &a, const float &b,
                                      const float &c, float &x0, float &x1) {
    float discr = b * b - 4 * a * c;
    if (discr < 0) {
        return false;
    } else if (discr == 0) {
        x0 = x1 = -0.5f * b / a;
    } else {
        float q =
            (b > 0) ? -0.5f * (b + sqrt(discr)) : -0.5f * (b - sqrt(discr));
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
    Vec3f center;
    float radius, radius2;

    __host__ __device__ Sphere() {}

    __host__ __device__ Sphere(const Vec3f &c, const float &r) {
        center = c;
        radius = r;
        radius2 = r * r;
    }
    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              int &index, Vec2f &uv) const {
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
    __device__ Vec3f getNormal(const Vec3f &P) const {
        return (P - center).normalized();
    }
};

#endif