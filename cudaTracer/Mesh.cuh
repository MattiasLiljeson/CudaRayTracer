#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include "Object.cuh"
#include "Texture.cuh"
#include "Vertex.cuh"
#include "Ray.cuh"


using namespace vectorAxes;

class Mesh {
   public:
    int triangleCnt;
    int *indices;
    Vertex *vertices;
    Texture diffuse;
    Texture normals;

    __host__ __device__ Mesh() {
        triangleCnt = -1;
        indices = NULL;
        vertices = NULL;
    }

    __host__ __device__ Mesh(int *indices, int triangleCnt, Vertex *vertices,
                             Texture diffuse, Texture normals) {
        this->indices = indices;
        this->triangleCnt = triangleCnt;
        this->vertices = vertices;
        this->diffuse = diffuse;
        this->normals = normals;
    }

    __device__ bool intersect(const Ray &ray, float &tnear, int &index,
                              Vec2f &uv) const {
        bool intersect = false;
        for (int k = 0; k < triangleCnt; ++k) {
            const Vec3f &v0 = vertices[indices[k * 3]].position;
            const Vec3f &v1 = vertices[indices[k * 3 + 1]].position;
            const Vec3f &v2 = vertices[indices[k * 3 + 2]].position;
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, ray, t, u, v) && t < tnear) {
                tnear = t;
                uv[X] = u;
                uv[Y] = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    __device__ Vec2f getStCoords(const int &index, const Vec2f &uv) const {
        const Vertex &v0 = vertices[indices[index * 3]];
        const Vertex &v1 = vertices[indices[index * 3 + 1]];
        const Vertex &v2 = vertices[indices[index * 3 + 2]];
        return interpolate<2>(uv, v0.texCoord, v1.texCoord, v2.texCoord);
    }

    __device__ Vec3f getNormal(const int &index, const Vec2f &uv,
                               const Vec2f &st) const {
        const Vertex &v0 = vertices[indices[index * 3]];
        const Vertex &v1 = vertices[indices[index * 3 + 1]];
        const Vertex &v2 = vertices[indices[index * 3 + 2]];

        Vec3f t = interpolate<3>(uv, v0.tangent, v1.tangent, v2.tangent);
        Vec3f b = interpolate<3>(uv, v0.bitangent, v1.bitangent, v2.bitangent);
        Vec3f n = interpolate<3>(uv, v0.normal, v1.normal, v2.normal);
        Vec3f normSamp = normals.sample(st);
        normSamp = ((normSamp * 2.0f) - 1.0f);
        return (n + normSamp[X] * t + normSamp[Y] * b).normalized();
    }

    template <int Size>
    __device__ Vec<float, Size> interpolate(const Vec2f &uv,
                                            const Vec<float, Size> &v0,
                                            const Vec<float, Size> &v1,
                                            const Vec<float, Size> &v2) const {
        return v0 * (1 - uv[X] - uv[Y]) +  //
               v1 * uv[X] +                //
               v2 * uv[Y];
    }

    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        return diffuse.sample(st);
    }

    __device__ bool static rayTriangleIntersect(const Vec3f &v0,
                                                const Vec3f &v1,
                                                const Vec3f &v2, const Ray &ray,
                                                float &tnear, float &u,
                                                float &v) {
        Vec3f edge1 = v1 - v0;
        Vec3f edge2 = v2 - v0;
        Vec3f pvec = ray.dir.cross(edge2);
        float det = edge1.dot(pvec);
        if (det == 0 || det < 0) return false;

        Vec3f tvec = ray.origin - v0;
        u = tvec.dot(pvec);
        if (u < 0 || u > det) return false;

        Vec3f qvec = tvec.cross(edge1);
        v = ray.dir.dot(qvec);
        if (v < 0 || u + v > det) return false;

        float invDet = 1 / det;

        tnear = edge2.dot(qvec) * invDet;
        u *= invDet;
        v *= invDet;

        return true;
    }
};

#endif