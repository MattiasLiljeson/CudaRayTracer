#ifndef MATH_CUH
#define MATH_CUH

#include <cuda_runtime.h>
#include "Vec.cuh"
#include "cuda.h"

class Math {
    public:
    __device__ static float clamp(const float &lo, const float &hi,
                                  const float &v) {
        return fmax(lo, fmin(hi, v));
    }

    __device__ static float saturate(const float &v) {
        return fmax(0.0f, fmin(1.0f, v));
    }

    template <typename T>
    __device__ static void tswap(T &a, T &b) {
        T c(a);
        a = b;
        b = c;
    }

    /**
     * Fresnel computation
     * \param i incident view direction
     * \param n normal at the hit point
     * \param ior refractactive index
     * \return the amount of reflected light
     *
     *  Due to the conservation of energy, transmittance is 1 - kr;
     */
    __device__ static float fresnel(const Vec3f &I, const Vec3f &N,
                                    const float &ior) {
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
            return 1;
        } else {
            float cost = sqrtf(fmax(0.f, 1 - sint * sint));
            cosi = fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) /
                       ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) /
                       ((etai * cosi) + (etat * cost));
            return (Rs * Rs + Rp * Rp) * 0.5f;
        }
    }

    /**
     * Refraction using Snell's law
     * Two cases: inside or outside object
     */
    __device__ static Vec3f refract(const Vec3f &I, const Vec3f &N,
                                    const float &ior) {
        float cosi = clamp(-1, 1, I.dot(N));
        float etai = 1, etat = ior;
        Vec3f n = N;
        if (cosi < 0) {  // outside
            cosi = -cosi;
        } else {  // inside
            tswap<float>(etai, etat);
            n = -N;
        }
        float eta = etai / etat;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
    }
};

#endif