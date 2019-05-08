#ifndef TRACER_CUH
#define TRACER_CUH
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "Vec.cuh"

template <class T>
void cudamain(const Options *options, const Light *lights, const int lightCnt,
              const T *spheres, const int sphereCnt, const void *surface,
              const int width, const int height, size_t pitch,
              const Vec3f &orig, const Mat44f &camera);

// Sphere *createSpheres(const SphereData *sphereDatas,
//                      const MaterialProperties *properties,
//                      const int sphereCnt);

template <class Shape>
class Trace {
   public:
    const Options *options;
    const Light *lights;
    const int lightCnt;
    const Shape *shapes;
    const int sphereCnt;
    unsigned char *surface;

    __device__ Trace(const Options *options, const Light *lights,
                     const int lightCnt, const Shape *shapes,
                     const int sphereCnt, unsigned char *surface)
        : options(options),
          lights(lights),
          lightCnt(lightCnt),
          shapes(shapes),
          sphereCnt(sphereCnt),
          surface(surface) {}

    __device__ Vec3f castRay(const Vec3f &orig, const Vec3f &dir,
                             uint32_t depth) const {
        if (depth > options->maxDepth) {
            return options->backgroundColor;
        }
        float tnear = INF;
        Vec2f uv;
        uint32_t index = 0;
        const Shape *hitObject = nullptr;
        if (trace(orig, dir, tnear, index, uv, &hitObject)) {
            Vec3f hitPoint = orig + dir * tnear;
            Vec3f N;   // normal
            Vec2f st;  // st coordinates
            hitObject->getSurfaceProperties(hitPoint, dir, index, uv, N, st);
            return hitObject->materialType.eval(dir, index, uv, st, hitObject,
                                                hitPoint, N, depth);
        }
    }

    __device__ bool trace(const Vec3f &orig, const Vec3f &dir, float &tNear,
                          uint32_t &index, Vec2f &uv,
                          const Shape **hitObject) const {
        *hitObject = nullptr;
        for (uint32_t k = 0; k < sphereCnt; ++k) {
            float tNearK = INF;
            uint32_t indexK;
            Vec2f uvK;
            if (shapes[k].intersect(orig, dir, tNearK, indexK, uvK) &&
                tNearK < tNear) {
                *hitObject = &shapes[k];
                tNear = tNearK;
                index = indexK;
                uv = uvK;
            }
        }

        return (*hitObject != nullptr);
    }
};
#endif