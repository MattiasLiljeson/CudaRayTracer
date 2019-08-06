#ifndef SCENE_CUH
#define SCENE_CUH

#include "Mat.cuh"
#include "Shape.cuh"
#include "Vec.cuh"

class Light;

struct Scene {
    Light *lights;
    int lightCnt;
    Shape *shapes;
    int shapeCnt;
    Vec3f orig;
    Mat44f camera;
    LinearNode *nodes;

    __device__ bool intersect(Ray &ray, SurfaceData &surface) const {
        bool hit = false;
        Vec3f invDir(1.0f / ray.dir[X], 1.0f / ray.dir[Y], 1.0f / ray.dir[Z]);
        int dirIsNeg[3] = {invDir[X] < 0, invDir[Y] < 0, invDir[Z] < 0};

        int toVisitOffset = 0;
        int currentNodeIdx = 0;
        int nodesToVisit[64];
        while (true) {
            /*const*/ LinearNode *node = &nodes[currentNodeIdx];
            if (node->bb.intersect(ray, invDir, dirIsNeg)) {
                        //hit = true;
                        surface.hitBb = node;
                        surface.hitBbCnt++;
                if (node->primtiveCnt > 0) {
                    // leaf
                    for (int i = 0; i < node->primtiveCnt; ++i) {
                        Shape* prim = &shapes[node->primitivesOffset + i];
                        if (prim->intersect(ray, surface)) {
                            surface.hit = prim;
                            hit = true;
                        }
                    }
                    if (toVisitOffset == 0) {
                        break;
                    }
                    currentNodeIdx = nodesToVisit[--toVisitOffset];
                } else {
                    // inside node, put far node on stack, advance to near
                    // check dir of ray so that nearest child is tested
                    // first
                    if (dirIsNeg[node->axis]) {
                        nodesToVisit[toVisitOffset++] = currentNodeIdx + 1;
                        currentNodeIdx = node->secondChildOffset;
                    } else {
                        nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                        currentNodeIdx = currentNodeIdx + 1;
                    }
                }
            } else {
                if (toVisitOffset == 0) {
                    break;
                }
                currentNodeIdx = nodesToVisit[--toVisitOffset];
            }
        }
        return hit;
    }
};

#endif  // !SCENE_CUH