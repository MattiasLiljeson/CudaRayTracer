#ifndef TRACER_CUH
#define TRACER_CUH
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Scene.cuh"
#include "Vec.cuh"

#include "Ray.cuh"
#include "Scene.cuh"

enum colors { RED, GREEN, BLUE, ALPHA, COLOR_CNT };

#define INF 2e10f

__device__ __constant__ extern size_t C_PITCH;
__device__ extern Options g_options;
__device__ extern Scene g_scene;

class Tracer {
    unsigned char *surface;

   public:
    __device__ Tracer(unsigned char *surface);

    __device__ Vec3f castRay(Ray &ray, uint32_t depth);
    __device__ bool trace(Ray &ray, uint32_t &index, Vec2f &uv,
                          const Shape **hit);
    __device__ Vec3f reflectionAndRefraction(const Vec3f &dir, uint32_t &index,
                                             Vec2f &uv, Vec2f &st,
                                             const Shape *hitObject,
                                             const Vec3f &hitPoint,
                                             const Vec3f &N, const int depth);
    __device__ Vec3f reflection(const Vec3f &dir, uint32_t &index, Vec2f &uv,
                                Vec2f &st, const Shape *hitObject,
                                const Vec3f &hitPoint, const Vec3f &N,
                                const int depth);
    __device__ Vec3f diffuseAndGlossy(const Vec3f &dir, uint32_t &index,
                                      Vec2f &uv, Vec2f &st,
                                      const Shape *hitObject,
                                      const Vec3f &hitPoint, const Vec3f &N,
                                      const int depth);
};
#endif