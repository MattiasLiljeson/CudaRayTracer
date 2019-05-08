#ifndef MATERIAL_TYPE_H
#define MATERIAL_TYPE_H

#include "Vec.cuh"

#include <cuda_runtime.h>
#include "cuda.h"

template <class Shape>
class Trace;
//template <class Material>
template <class Material>
class Sphere;

template <class T, class Shape>
class MaterialType {
    Vec3f eval(const Trace<Shape> *traceInst, const Vec3f &dir, uint32_t &index,
               Vec2f &uv, Vec2f &st, const Shape *hitObject,
               const Vec3f &hitPoint, const Vec3f &N, const int depth) {
        static_cast<T *>(this)->eval(dir, index, uv, st, hitObject, hitPoint, N,
                                     depth);
    }
};

#endif