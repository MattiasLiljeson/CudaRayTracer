#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include "Object.cuh"
#include "Vertex.cuh"
#include "Texture.cuh"

using namespace axis;

class Mesh {
   public:
    int triangleCnt;
    int *indices;
    Vertex *vertices;
    Texture texture;

    __host__ __device__ Mesh() {
        triangleCnt = -1;
        indices = NULL;
        vertices = NULL;
    }

    __host__ __device__ Mesh(int *indices, int triangleCnt, Vertex *vertices,
                             Texture texture) {
        this->indices = indices;
        this->triangleCnt = triangleCnt;
        this->vertices = vertices;
        this->texture = texture;
    }

    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              int &index, Vec2f &uv) const {
        bool intersect = false;
        for (int k = 0; k < triangleCnt; ++k) {
            const Vec3f &v0 = vertices[indices[k * 3]].position;
            const Vec3f &v1 = vertices[indices[k * 3 + 1]].position;
            const Vec3f &v2 = vertices[indices[k * 3 + 2]].position;
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
        const Vec2f &st0 = vertices[indices[index * 3]].texCoord;
        const Vec2f &st1 = vertices[indices[index * 3 + 1]].texCoord;
        const Vec2f &st2 = vertices[indices[index * 3 + 2]].texCoord;
        st = st0 * (1 - uv[X] - uv[Y]) +  //
             st1 * uv[X] +                //
             st2 * uv[Y];
        const Vec3f &n0 = vertices[indices[index * 3]].normal;
        const Vec3f &n1 = vertices[indices[index * 3 + 1]].normal;
        const Vec3f &n2 = vertices[indices[index * 3 + 2]].normal;
        N = n0 * (1 - uv[X] - uv[Y]) +  //
            n1 * uv[X] +                //
            n2 * uv[Y];
    }

    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        return texture.sample(st[0], st[1]);
    }

    __device__ bool static rayTriangleIntersect(
        const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Vec3f &orig,
        const Vec3f &dir, float &tnear, float &u, float &v) {
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
};

#endif