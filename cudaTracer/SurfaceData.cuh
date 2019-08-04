#ifndef SURFACE_DATA_CUH
#define SURFACE_DATA_CUH

#include "Vec.cuh"
#include "Triangle.cuh"

struct Shape;

struct SurfaceData {
    Shape *hit;
    Vec3f hitPoint;
    Triangle triangle;
    Vec2f uv;
    Vec2f st;
    Vec3f n;
};

#endif