#ifndef TRACER_CUH
#define TRACER_CUH
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "Vec.cuh"

void cudamain(const Options *options, const Light *lights, const int lightCnt,
              const Sphere *spheres, const int sphereCnt, const void *surface,
              const int width, const int height, size_t pitch,
              const Vec3f &orig, const Mat44f &camera);

//Sphere *createSpheres(const SphereData *sphereDatas,
//                      const MaterialProperties *properties,
//                      const int sphereCnt);

class Trace {
    const Options *options;
    const Light *lights;
    const int lightCnt;
    const Sphere *spheres;
    const int sphereCnt;
    unsigned char *surface;

   public:
    __device__ Trace(const Options *options, const Light *lights,
                     const int lightCnt, const Sphere *spheres,
                     const int sphereCnt, unsigned char *surface);

    __device__ Vec3f castRay(const Vec3f &orig, const Vec3f &dir,
                             uint32_t depth);
    template <typename T>
    __device__ bool trace(const Vec3f &orig, const Vec3f &dir, float &tNear,
                          uint32_t &index, Vec2f &uv, const T **hitObject);
    __device__ Vec3f reflectionAndRefraction(const Vec3f &dir, uint32_t &index,
                                             Vec2f &uv, Vec2f &st,
                                             const Sphere *hitObject,
                                             const Vec3f &hitPoint,
                                             const Vec3f &N, const int depth);
    __device__ Vec3f reflection(const Vec3f &dir, uint32_t &index, Vec2f &uv,
                                Vec2f &st, const Sphere *hitObject,
                                const Vec3f &hitPoint, const Vec3f &N,
                                const int depth);
    __device__ Vec3f diffuseAndGlossy(const Vec3f &dir, uint32_t &index,
                                      Vec2f &uv, Vec2f &st,
                                      const Sphere *hitObject,
                                      const Vec3f &hitPoint, const Vec3f &N,
                                      const int depth);
};
#endif