#ifndef TRACER_CUH
#define TRACER_CUH
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Scene.cuh"
#include "Vec.cuh"

#include "Scene.cuh"
void cudamain(const Options &options, const Scene &scene, const void *surface,
              size_t pitch, int blockDim);

class Trace {
    unsigned char *surface;

   public:
    __device__ Trace(unsigned char *surface);

    __device__ Vec3f castRay(const Vec3f &orig, const Vec3f &dir,
                             uint32_t depth);
    __device__ bool trace(const Vec3f &orig, const Vec3f &dir, float &tNear,
                          uint32_t &index, Vec2f &uv, const Shape **hit);
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