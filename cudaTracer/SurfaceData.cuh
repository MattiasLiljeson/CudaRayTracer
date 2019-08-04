#ifndef SURFACE_DATA_CUH
#define SURFACE_DATA_CUH

#include "Triangle.cuh"
#include "Vec.cuh"

struct LinearNode;
struct Shape;

struct SurfaceData {
    Shape* hit;
    Vec3f hitPoint;     // sphere
    Triangle triangle;  // triangle
    Vec2f uv;
    Vec2f st;
    Vec3f n;
};

#endif