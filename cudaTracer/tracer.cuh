#ifndef TRACER_CUH
#define TRACER_CUH
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Scene.cuh"
#include "Vec.cuh"

#include "Ray.cuh"

enum colors { RED, GREEN, BLUE, ALPHA, COLOR_CNT };

#define INF 2e10f

__device__ __constant__ extern size_t C_PITCH;
__device__ extern Options g_options;
__device__ extern Scene g_scene;

class Tracer {
    unsigned char *surface;

   public:
    __device__ Tracer(unsigned char *surface);

    __device__ Vec3f castRay(Ray &ray, int depth);
    __device__ bool trace(Ray &ray, SurfaceData &data);
    __device__ Vec3f reflectionAndRefraction(const Vec3f &dir,
                                             SurfaceData &data,
                                             const int depth);
    __device__ Vec3f reflection(const Vec3f &dir, SurfaceData &data,
                                const int depth);
    __device__ Vec3f diffuseAndGlossy(const Vec3f &dir, SurfaceData &data,
                                      const int depth);
};
#endif