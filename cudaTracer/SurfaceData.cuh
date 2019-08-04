#ifndef SURFACE_DATA_CUH
#define SURFACE_DATA_CUH

#include "Triangle.cuh"
#include "Vec.cuh"

struct LinearNode;
struct Shape;

struct SurfaceData {
    enum HitWhat { OTHER, BOUNDING_BOX };
    HitWhat hitWhat;
    Shape* hit;
    Vec3f hitPoint;     // sphere
    Triangle triangle;  // triangle
    LinearNode* node; //boundingbox
    int nodesHit;
    Vec2f uv;
    Vec2f st;
    Vec3f n;

    SurfaceData() : hitWhat(OTHER), nodesHit(0){}
};

#endif