#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include "LinearNode.cuh"
#include "Material.cuh"
#include "Ray.cuh"
#include "SurfaceData.cuh"
#include "Texture.cuh"
#include "Triangle.cuh"
#include "Vertex.cuh"

using namespace vectorAxes;

class Mesh {
   public:
    int triangleCnt;
    Triangle *triangles;
    Vertex *vertices;
    Texture diffuse;
    Texture normals;
    Texture specular;
    LinearNode *nodes;
    Mat44f transformInverse;

    __host__ __device__ Mesh() {
        triangleCnt = -1;
        triangles = NULL;
        vertices = NULL;
    }

    __host__ __device__ Mesh(Triangle *triangles, int triangleCnt,
                             Vertex *vertices, Texture diffuse, Texture normals,
                             Texture specular, LinearNode *nodes,
                             Mat44f transform) {
        this->triangles = triangles;
        this->triangleCnt = triangleCnt;
        this->vertices = vertices;
        this->diffuse = diffuse;
        this->normals = normals;
        this->specular = specular;
        this->nodes = nodes;
        this->transformInverse = transform.inversed();
    }

    __device__ void getStAndNormal(SurfaceData &data) const {
        const Vertex &v0 = vertices[data.triangle[0]];
        const Vertex &v1 = vertices[data.triangle[1]];
        const Vertex &v2 = vertices[data.triangle[2]];

        data.st =
            interpolate<2>(data.uv, v0.texCoord, v1.texCoord, v2.texCoord);
        Vec3f t = interpolate<3>(data.uv, v0.tangent, v1.tangent, v2.tangent);
        Vec3f b =
            interpolate<3>(data.uv, v0.bitangent, v1.bitangent, v2.bitangent);
        Vec3f n = interpolate<3>(data.uv, v0.normal, v1.normal, v2.normal);

        Vec3f normSamp = normals.sample(data.st);
        normSamp = ((normSamp * 2.0f) - 1.0f);
        data.n = (n + normSamp[X] * t + normSamp[Y] * b).normalized();
    }

    __device__ Vec2f getStCoords(const Triangle &triangle,
                                 const Vec2f &uv) const {
        const Vertex &v0 = vertices[triangle[0]];
        const Vertex &v1 = vertices[triangle[1]];
        const Vertex &v2 = vertices[triangle[2]];
        return interpolate<2>(uv, v0.texCoord, v1.texCoord, v2.texCoord);
    }

    __device__ Vec3f getNormal(const Triangle &triangle, const Vec2f &uv,
                               const Vec2f &st) const {
        if (false) {
            const Vec3f &v0 = vertices[triangle[0]].position;
            const Vec3f &v1 = vertices[triangle[1]].position;
            const Vec3f &v2 = vertices[triangle[2]].position;
            Vec3f e0 = (v1 - v0).normalized();
            Vec3f e1 = (v2 - v1).normalized();
            return (e0.cross(e1)).normalized();
        }

        const Vertex &v0 = vertices[triangle[0]];
        const Vertex &v1 = vertices[triangle[1]];
        const Vertex &v2 = vertices[triangle[2]];

        Vec3f t = interpolate<3>(uv, v0.tangent, v1.tangent, v2.tangent);
        Vec3f b = interpolate<3>(uv, v0.bitangent, v1.bitangent, v2.bitangent);
        Vec3f n = interpolate<3>(uv, v0.normal, v1.normal, v2.normal);

        Vec3f normSamp = normals.sample(st);
        normSamp = ((normSamp * 2.0f) - 1.0f);
        return (n + normSamp[X] * t + normSamp[Y] * b).normalized();
    }

    __device__ float getSpecularMask(const Vec2f &st) const {
        return specular.sample(st)[X];
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

    __device__ bool intersect(Ray &ray, SurfaceData &surface) const {
        Ray tRay(transformInverse.multPoint(ray.origin),
                 transformInverse.multVec(ray.dir), ray.tMax);

        bool hit = false;
        Vec3f invDir(1.0f / tRay.dir[X], 1.0f / tRay.dir[Y],
                     1.0f / tRay.dir[Z]);
        int dirIsNeg[3] = {invDir[X] < 0, invDir[Y] < 0, invDir[Z] < 0};

        int toVisitOffset = 0;
        int currentNodeIdx = 0;
        int nodesToVisit[64];
        while (true) {
            const LinearNode *node = &nodes[currentNodeIdx];
            if (node->bb.intersect(tRay, invDir, dirIsNeg)) {
                if (node->primtiveCnt > 0) {
                    // leaf
                    for (int i = 0; i < node->primtiveCnt; ++i) {
                        Triangle tri = triangles[node->primitivesOffset + i];
                        if (rayTriangleIntersect(tri, tRay, surface)) {
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
        ray.tMax = tRay.tMax;
        return hit;
    }

    // Moller-Trumbore ray-triangle intersection algorithm
    __device__ bool rayTriangleIntersect(const Triangle &triangle, Ray &ray,
                                         SurfaceData &data) const {
        const Vec3f &v0 = vertices[triangle[0]].position;
        const Vec3f &v1 = vertices[triangle[1]].position;
        const Vec3f &v2 = vertices[triangle[2]].position;
        Vec3f edge1 = v1 - v0;
        Vec3f edge2 = v2 - v0;
        Vec3f pvec = ray.dir.cross(edge2);
        float det = edge1.dot(pvec);
        if (det == 0 || det < 0) return false;

        Vec3f tvec = ray.origin - v0;
        float u = tvec.dot(pvec);
        if (u < 0 || u > det) return false;

        Vec3f qvec = tvec.cross(edge1);
        float v = ray.dir.dot(qvec);
        if (v < 0 || u + v > det) return false;

        float invDet = 1 / det;
        float t = edge2.dot(qvec) * invDet;
        if (t < 0.0f || ray.tMax < t) {
            return false;
        }

        ray.tMax = t;
        data.uv[U] = u * invDet;
        data.uv[V] = v * invDet;
        data.triangle = triangle;
        return true;
    }
};

#endif