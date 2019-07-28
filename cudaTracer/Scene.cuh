#ifndef SCENE_CUH
#define SCENE_CUH

#include "Mat.cuh"
#include "Vec.cuh"
#include "Shape.cuh"

class Light;

struct Scene {
    Light *lights;
    int lightCnt;
    Shape *shapes;
    int shapeCnt;
    Vec3f orig;
    Mat44f camera;
};

#endif  // !SCENE_CUH