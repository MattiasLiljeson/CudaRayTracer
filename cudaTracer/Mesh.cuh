#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include "Object.cuh"
#include "Vec.cuh"

using namespace axis;

inline __device__ bool rayTriangleIntersect(const Vec3f &v0, const Vec3f &v1,
                                            const Vec3f &v2, const Vec3f &orig,
                                            const Vec3f &dir, float &tnear,
                                            float &u, float &v) {
    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;
    Vec3f pvec = dir.cross(edge2);
    float det = edge1.dot(pvec);
    if (det == 0 || det < 0) return false;

    Vec3f tvec = orig - v0;
    u = tvec.dot(pvec);
    if (u < 0 || u > det) return false;

    Vec3f qvec = tvec.cross(edge1);
    v = dir.dot(qvec);
    if (v < 0 || u + v > det) return false;

    float invDet = 1 / det;

    tnear = edge2.dot(qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}

inline __device__ Vec3f mix(const Vec3f &a, const Vec3f &b,
                            const float &mixValue) {
    return a * (1 - mixValue) + b * mixValue;
}

class Mesh {
   public:
    int triangleCnt;
    Vec3f *vertices;
    int *vertexIndex;
    Vec2f *stCoordinates;

    __host__ __device__ Mesh() {}

    __host__ __device__ Mesh(Vec3f *vertices, int *vertexIndex, int triangleCnt,
                             Vec2f *st) {
        this->vertices = vertices;
        this->vertexIndex = vertexIndex;
        this->triangleCnt = triangleCnt;
        this->stCoordinates = st;
    }

    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              int &index, Vec2f &uv) const {
        bool intersect = false;
        for (int k = 0; k < triangleCnt; ++k) {
            const Vec3f &v0 = vertices[vertexIndex[k * 3]];
            const Vec3f &v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vec3f &v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, t, u, v) &&
                t < tnear) {
                tnear = t;
                uv[X] = u;
                uv[Y] = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    __device__ void getSurfaceProperties(const Vec3f &P, const Vec3f &I,
                                         const int &index, const Vec2f &uv,
                                         Vec3f &N, Vec2f &st) const {
        const Vec3f &v0 = vertices[vertexIndex[index * 3]];
        const Vec3f &v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vec3f &v2 = vertices[vertexIndex[index * 3 + 2]];
        Vec3f e0 = (v1 - v0).normalized();
        Vec3f e1 = (v2 - v1).normalized();
        N = (e0.cross(e1)).normalized();
        const Vec2f &st0 = stCoordinates[vertexIndex[index * 3]];
        const Vec2f &st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vec2f &st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv[Vec2f::X] - uv[Vec2f::Y]) + st1 * uv[Vec2f::X] +
             st2 * uv[Vec2f::Y];
    }

    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        float scale = 5.0f;
        float pattern = (fmodf(st[X] * scale, 1.0f) > 0.5f) ^
                        (fmodf(st[Y] * scale, 1.0f) > 0.5f);
        return mix(Vec3f(0.815f, 0.235f, 0.031f), Vec3f(0.937f, 0.937f, 0.231f),
                   pattern);
    }
};

#endif