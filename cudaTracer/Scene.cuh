#ifndef SCENE_CUH
#define SCENE_CUH

#include "Vec.cuh"
#include "Mat.cuh"

class Light;
class Sphere;

struct Scene {
    Light *lights;
    int lightCnt;
    Sphere *spheres;
    int sphereCnt;
    Vec3f orig;
    Mat44f camera;
};

#endif  // !SCENE_CUH
