#ifndef BOUNDINGBOX_CUH
#define BOUNDINGBOX_CUH

#include <algorithm>
#include <limits>
#include <vector>
#include "Ray.cuh"
#include "Vec.cuh"
#include "Vertex.cuh"

using namespace vectorAxes;


struct Triangle {
    int i[3];
    __host__ __device__ Triangle() : i{-1, -1, -1} {}
    __host__ __device__ Triangle(int v0, int v1, int v2) : i{v0, v1, v2} {}
    __host__ __device__ Triangle(int i, int v[])
        : i{v[i], v[i + 1], v[i + 2]} {}
};

static constexpr float MachineEpsilon =
    std::numeric_limits<float>::epsilon() * 0.5;

__device__ inline constexpr float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

struct BoundingBox {
    Vec3f bbmin;
    Vec3f bbmax;

    __host__ BoundingBox()
        : bbmin(Vec3f(0.0f, 0.0f, 0.0f)), bbmax(Vec3f(0.0f, 0.0f, 0.0f)) {}

    __host__ BoundingBox(const Triangle& t,
                                    const std::vector<Vertex>& v) {
        bbmin = FLT_MAX;
        bbmax = FLT_MIN;
        for (int vertexIdx = 0; vertexIdx < 3; ++vertexIdx) {
            Vec3f pos = v[t.i[vertexIdx]].position;
            for (int axisIdx = 0; axisIdx < 3; ++axisIdx) {
                if (pos[axisIdx] > bbmax[axisIdx]) {
                    bbmax[axisIdx] = pos[axisIdx];
                } else if (pos[axisIdx] < bbmin[axisIdx]) {
                    bbmin[axisIdx] = pos[axisIdx];
                }
            }
        }
    }

    __host__ BoundingBox(const Vec3f& bbmin, const Vec3f& bbmax)
        : bbmin(bbmin), bbmax(bbmax) {}

    __device__ const Vec3f& operator[](int i) const;
    __device__ Vec3f& operator[](int i);

    __host__ Vec3f centroid() const { return (bbmin + bbmax) / 2.0f; }

    __host__ Vec3f diagonal() const { return bbmax - bbmin; }

    __device__ int maxExtent() const {
        Vec3f d = diagonal();
        if (d[X] > d[Y] && d[X] > d[Z]) {
            return 0;  // X
        } else if (d[Y] > d[Z]) {
            return 1;  // Y
        } else {
            return 2;  // Z
        }
    }

    __host__ static BoundingBox unionn(BoundingBox b1, BoundingBox b2) {
        return BoundingBox(Vec3f((std::min)(b1.bbmin[X], b2.bbmin[X]),
                                 (std::min)(b1.bbmin[Y], b2.bbmin[Y]),
                                 (std::min)(b1.bbmin[Z], b2.bbmin[Z])),
                           Vec3f((std::max)(b1.bbmax[X], b2.bbmax[X]),
                                 (std::max)(b1.bbmax[Y], b2.bbmax[Y]),
                                 (std::max)(b1.bbmax[Z], b2.bbmax[Z])));
    }

    __host__ static BoundingBox unionn(BoundingBox b, Vec3f p) {
        return BoundingBox(Vec3f((std::min)(b.bbmin[X], p[X]),  //
                                 (std::min)(b.bbmin[Y], p[Y]),  //
                                 (std::min)(b.bbmin[Z], p[Z])),
                           Vec3f((std::max)(b.bbmax[X], p[X]),  //
                                 (std::max)(b.bbmax[Y], p[Y]),  //
                                 (std::max)(b.bbmax[Z], p[Z])));
    }

    __device__ bool intersect(const Ray& ray, float* hitt0, float* hitt1) const {
        float t0 = 0;
        float t1 = ray.tMax;
        for (int i = 0; i < 3; ++i) {
            // Update interval for ith bounding box slab
            float invRayDir = 1 / ray.dir[i];
            float tNear = (bbmin[i] - ray.origin[i]) * invRayDir;
            float tFar = (bbmax[i] - ray.origin[i]) * invRayDir;
            // Update parametric interval from slab intersection values
            if (tNear > tFar) std::swap(tNear, tFar);
            // Update tFar to ensure robust ray–bounds intersection
            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if (t0 > t1) {
                return false;
            }
        }
        if (hitt0) *hitt0 = t0;
        if (hitt1) *hitt1 = t1;
        return true;
    }

    __device__ inline bool intersect(const Ray& ray, const Vec3f& invDir,
                          const int dirIsNeg[3]) const {
        const BoundingBox& bounds = *this;
        // Check for ray intersection against  and  slabs
        float tMin = (bounds[dirIsNeg[0]][X] - ray.origin[X]) * invDir[X];
        float tMax = (bounds[1 - dirIsNeg[0]][X] - ray.origin[X]) * invDir[X];
        float tyMin = (bounds[dirIsNeg[1]][Y] - ray.origin[Y]) * invDir[Y];
        float tyMax = (bounds[1 - dirIsNeg[1]][Y] - ray.origin[Y]) * invDir[Y];
        // Update tMax and tyMax to ensure robust bounds intersection
        tMax *= 1 + 2 * gamma(3.0f);
        tyMax *= 1 + 2 * gamma(3.0f);
        if (tMin > tyMax || tyMin > tMax) return false;
        if (tyMin > tMin) tMin = tyMin;
        if (tyMax < tMax) tMax = tyMax;

        // Check for ray intersection against  slab
        float tzMin = (bounds[dirIsNeg[2]][Z] - ray.origin[Z]) * invDir[Z];
        float tzMax = (bounds[1 - dirIsNeg[2]][Z] - ray.origin[Z]) * invDir[Z];
        // Update tzMax to ensure robust bounds intersection
        if (tMin > tzMax || tzMin > tMax) return false;
        if (tzMin > tMin) tMin = tzMin;
        if (tzMax < tMax) tMax = tzMax;

        return (tMin < ray.tMax) && (tMax > 0);
    }
};

#endif