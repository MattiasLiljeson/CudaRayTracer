#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include "LinearNode.cuh"
#include "Object.cuh"
#include "Ray.cuh"
#include "SurfaceData.cuh"
#include "Texture.cuh"
#include "Triangle.cuh"
#include "Vertex.cuh"

using namespace vectorAxes;

class Mesh {
   public:
    int triangleCnt;
    // int *indices;
    Triangle *triangles;
    Vertex *vertices;
    Texture diffuse;
    Texture normals;
    LinearNode *nodes;

    __host__ __device__ Mesh() {
        triangleCnt = -1;
        triangles = NULL;
        vertices = NULL;
    }

    __host__ __device__ Mesh(Triangle *triangles, int triangleCnt,
                             Vertex *vertices, Texture diffuse, Texture normals,
                             LinearNode *nodes /*, int *indices*/) {
        this->triangles = triangles;
        this->triangleCnt = triangleCnt;
        this->vertices = vertices;
        this->diffuse = diffuse;
        this->normals = normals;
        this->nodes = nodes;
        // this->indices = indices;
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

    //__device__ bool intersect(Ray &ray, SurfaceData &data) const {
    //    bool intersect = false;
    //    for (int k = 0; k < triangleCnt; ++k) {
    //        //const Vec3f &v0 = vertices[indices[k * 3]].position;
    //        //const Vec3f &v1 = vertices[indices[k * 3 + 1]].position;
    //        //const Vec3f &v2 = vertices[indices[k * 3 + 2]].position;
    //        const Vec3f &v0 = vertices[triangles[k][0]].position;
    //        const Vec3f &v1 = vertices[triangles[k][1]].position;
    //        const Vec3f &v2 = vertices[triangles[k][2]].position;
    //        float t, u, v;
    //        if (rayTriangleIntersect(v0, v1, v2, ray, t, u, v) &&
    //            t < ray.tMax) {
    //            ray.tMax = t;
    //            data.uv[X] = u;
    //            data.uv[Y] = v;
    //            // data.index = k;
    //            data.triangle = triangles[k];
    //            intersect |= true;
    //        }
    //    }
    //    return intersect;
    //}
    //__device__ bool intersect(Ray &ray, SurfaceData &data) const {
    //    bool intersect = false;
    //    for (int k = 0; k < triangleCnt; ++k) {
    //        intersect |= rayTriangleIntersect(triangles[k], ray, data);
    //    }
    //    return intersect;
    //}

    __device__ bool intersect(Ray &ray, SurfaceData &surface) const {
        bool hit = false;

        //for (int k = 0; k < triangleCnt; ++k) {
        //    hit |= rayTriangleIntersect(triangles[k], ray, surface);
        //}

        Vec3f invDir(1.0f / ray.dir[X], 1.0f / ray.dir[Y], 1.0f / ray.dir[Z]);
        int dirIsNeg[3] = {invDir[X] < 0, invDir[Y] < 0, invDir[Z] < 0};

        int toVisitOffset = 0;
        int currentNodeIdx = 0;
        int nodesToVisit[64];
        while (true) {
            const LinearNode *node = &nodes[currentNodeIdx];
            if (node->bb.intersect(ray, invDir, dirIsNeg)) {
                if (node->primtiveCnt > 0) {
                    hit = true;
                    // leaf
                    for (int i = 0; i < node->primtiveCnt; ++i) {
                        Triangle tri = triangles[node->primitivesOffset + i];
                        if (rayTriangleIntersect(tri, ray, surface)) {
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

    __device__ bool rayTriangleIntersect(const Triangle &triangle, Ray &ray,
                                         SurfaceData &data) const {
        const Vec3f &v0 = vertices[triangle[0]].position;
        const Vec3f &v1 = vertices[triangle[1]].position;
        const Vec3f &v2 = vertices[triangle[2]].position;
        // return rayTriangleIntersect(v0, v1, v2, ray, data);
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
        if (ray.tMax < t) {
            return false;
        }

        ray.tMax = t;
        data.uv[U] = u * invDet;
        data.uv[V] = v * invDet;
        data.triangle = triangle;
        return true;
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