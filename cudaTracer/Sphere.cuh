#ifndef SPHERE_H
#define SPHERE_H

#include "MaterialType.cuh"
#include "Math.cuh"
#include "Vec.cuh"

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I
//
// If the ray is inside, you need to invert the refractive indices and
// negate the normal N
// [/comment]
__device__ Vec3f refract(const Vec3f &I, const Vec3f &N, const float &ior) {
    float cosi = clamp(-1, 1, I.dot(N));
    float etai = 1, etat = ior;
    Vec3f n = N;
    if (cosi < 0) {
        cosi = -cosi;
    } else {
        float tmp = etai;
        etai = etat;
        etat = tmp;
        // std::swap(etai, etat);
        n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the mateural refractive index
//
// \param[out] kr is the amount of light reflected
// [/comment]
__device__ void fresnel(const Vec3f &I, const Vec3f &N, const float &ior,
                        float &kr) {
    float cosi = clamp(-1, 1, I.dot(N));
    float etai = 1, etat = ior;
    if (cosi > 0) {
        float tmp = etat;
        etat = etai;
        etai = tmp;
    }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(fmax(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        kr = 1;
    } else {
        float cost = sqrtf(fmax(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs =
            ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp =
            ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is
    // given by: kt = 1 - kr;
}

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

template <class Shape, class Material>
class Object {
   public:
    // material properties
    MaterialType<Material, Shape> materialType;
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
        static_cast<Shape *>(this)->intersect(orig, dir, tnear, index, uv);
    }
    __device__ void getSurfaceProperties(const Vec3f &P, const Vec3f &I,
                                         const uint32_t &index, const Vec2f &uv,
                                         Vec3f &N, Vec2f &st) const {
        static_cast<Shape *>(this)->getSurfaceProperties(P, I, index, uv, N,
                                                         st);
    }
    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        static_cast<Shape *>(this)->evalDiffuseColor(st);
    }
};

template <class Material>
class Sphere : public Object<Sphere<Material>, Material> {
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