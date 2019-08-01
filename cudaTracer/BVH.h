#ifndef BVH_CUH
#define BVH_CUH

#include <algorithm>
#include <vector>
#include "Vec.cuh"
#include "Vertex.cuh"
#include "Ray.cuh"

using namespace vectorAxes;

struct Triangle {
    int i[3];
    Triangle() : i{-1, -1, -1} {}
    Triangle(int v0, int v1, int v2) : i{v0, v1, v2} {}
    Triangle(int i, int v[]) : i{v[i], v[i + 1], v[i + 2]} {}
};

static constexpr float MachineEpsilon =
    std::numeric_limits<float>::epsilon() * 0.5;

inline constexpr float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

struct BoundingBox {
    Vec3f min;
    Vec3f max;

    BoundingBox() : min(0.0f), max(0.0f) {}

    BoundingBox(const Triangle& t, const std::vector<Vertex>& v) {
        min = FLT_MAX;
        max = FLT_MIN;
        for (int vertexIdx = 0; vertexIdx < 3; ++vertexIdx) {
            Vec3f pos = v[t.i[vertexIdx]].position;
            for (int axisIdx = 0; axisIdx < 3; ++axisIdx) {
                if (pos[axisIdx] > max[axisIdx]) {
                    max[axisIdx] = pos[axisIdx];
                } else if (pos[axisIdx] < min[axisIdx]) {
                    min[axisIdx] = pos[axisIdx];
                }
            }
        }
    }

    BoundingBox(const Vec3f& min, const Vec3f& max) : min(min), max(max) {}

    const Vec3f& operator[](int i) const;
    Vec3f& operator[](int i);

    Vec3f centroid() const {
        return (min + max) / 2.0f;
    }

    Vec3f diagonal() const {
        return max - min;
    }

    int MaximumExtent() const {
        Vec3f d = diagonal();
        if (d[X] > d[Y] && d[X] > d[Z]) {
            return 0;  // X
        } else if (d[Y] > d[Z]) {
            return 1;  // Y
        } else {
            return 2;  // Z
        }
    }

    static BoundingBox unionn(BoundingBox b1, BoundingBox b2) {
        return BoundingBox(Vec3f(std::min(b1.min[X], b2.min[X]),
                                 std::min(b1.min[Y], b2.min[Y]),
                                 std::min(b1.min[Z], b2.min[Z])),
                           Vec3f(std::max(b1.max[X], b2.max[X]),
                                 std::max(b1.max[Y], b2.max[Y]),
                                 std::max(b1.max[Z], b2.max[Z])));
    }

    static BoundingBox unionn(BoundingBox b, Vec3f p) {
        return BoundingBox(Vec3f(std::min(b.min[X], p[X]),  //
                                 std::min(b.min[Y], p[Y]),  //
                                 std::min(b.min[Z], p[Z])),
                           Vec3f(std::max(b.max[X], p[X]),  //
                                 std::max(b.max[Y], p[Y]),  //
                                 std::max(b.max[Z], p[Z])));
    }

    bool intersect(const Ray& ray, float* hitt0, float* hitt1) const {
        float t0 = 0;
        float t1 = ray.tMax;
        for (int i = 0; i < 3; ++i) {
            // Update interval for ith bounding box slab
            float invRayDir = 1 / ray.dir[i];
            float tNear = (min[i] - ray.origin[i]) * invRayDir;
            float tFar = (max[i] - ray.origin[i]) * invRayDir;
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

    inline bool intersect(const Ray& ray, const Vec3f& invDir,
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

struct ConstructionData {
    int triIdx;
    BoundingBox bb;
    Vec3f centroid;
    ConstructionData() : triIdx(-1) {}
    ConstructionData(int i, BoundingBox bb)
        : triIdx(i), bb(bb), centroid(bb.centroid()) {}
};

struct InterBvh {
    BoundingBox bb;
    InterBvh* children[2];
    int splitAxis;
    int firstPrimOffset;
    int primitiveCnt;

    InterBvh() : bb(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f)) {
        children[0] = nullptr;
        children[1] = nullptr;
        splitAxis = -1;
        firstPrimOffset = -1;
        primitiveCnt = -1;
    }

    void leaf(int firstPrim, int primCnt, BoundingBox bb) {
        firstPrimOffset = firstPrim;
        primitiveCnt = primCnt;
        bb = bb;
    }

    void interior(int axis, InterBvh* c0, InterBvh* c1) {
        children[0] = c0;
        children[1] = c1;
        bb = BoundingBox::unionn(c0->bb, c1->bb);
        splitAxis = axis;
        primitiveCnt = 0;
    }
};

struct LinearBVHNode {
    BoundingBox bb;
    union {
        int primitivesOffset;   // leaf
        int secondChildOffset;  // interior
    };
    uint16_t primtiveCnt;  // 0 -> interior node
    uint8_t axis;          // interior node: xyz
    uint8_t pad[1];        // ensure 32 byte total size
};

class BvhFactory {
    std::vector<Triangle> orderedPrims;
    std::vector<Triangle> primitives;
    std::vector<ConstructionData> primitiveInfo;
    std::vector<LinearBVHNode> nodes;

    BvhFactory(const std::vector<Vertex>& vertices,
               std::vector<Triangle>& primitives) {
        primitiveInfo.resize(primitives.size());
        for (int i = 0; i < primitives.size(); ++i) {
            primitiveInfo[i] = {i, BoundingBox(primitives[i], vertices)};
        }

        int totalNodes = 0;
        InterBvh* root = recursive(0, primitives.size(), totalNodes);
        // fetch new vector and replace yourself instead
        // primitives.swap(orderedPrims);

        nodes = std::vector<LinearBVHNode>(totalNodes);
        int offset = 0;
        flattenBvhTree(root, &offset);
    }

    InterBvh* recursive(const int start, const int end, int& totalNodes) {
        InterBvh* node = new InterBvh();  // TODO: new....
        totalNodes++;
        BoundingBox bb;
        for (int i = start; i < end; ++i) {
            bb = BoundingBox::unionn(bb, primitiveInfo[i].bb);
        }

        int primCnt = end - start;
        if (primCnt == 1) {
            leaf(bb, primCnt, start, end, node);
            return node;
        } else {
            // Compute bound of primitive centroids, choose split dimension dim>
            BoundingBox centroidBounds;
            for (int i = start; i < end; ++i) {
                bb = BoundingBox::unionn(centroidBounds,
                                         primitiveInfo[i].bb.centroid());
            }
            int dim = centroidBounds.MaximumExtent();

            // Partition primitives into two sets and build children
            if (centroidBounds.max[dim] == centroidBounds.min[dim]) {
                leaf(bb, primCnt, start, end, node);
                return node;
            } else {
                interior(primCnt, start, end, dim, node, totalNodes);
            }
        }
        return node;
    }

    // partitition primitives into equally sized subsets
    void interior(const int primCnt, const int start, const int end,
                  const int dim, InterBvh* node, int& totalNodes) {
        int mid = (start + end) / 2;
        std::nth_element(
            &primitiveInfo[start], &primitiveInfo[mid],
            &primitiveInfo[end - 1] + 1,
            [dim](const ConstructionData& a, const ConstructionData& b) {
                return a.bb.centroid()[dim] < b.bb.centroid()[dim];
            });

        node->interior(dim, recursive(start, mid, totalNodes),
                       recursive(mid, end, totalNodes));
    }

    void leaf(const BoundingBox& bb, const int primCnt, const int start,
              const int end, InterBvh* node) {
        int firstPrimOffset = orderedPrims.size();
        for (int i = start; i < end; ++i) {
            int idx = primitiveInfo[i].triIdx;
            orderedPrims.push_back(primitives[idx]);
        }
        node->leaf(firstPrimOffset, primCnt, bb);
    }

    int flattenBvhTree(InterBvh* node, int* offset) {
        LinearBVHNode* linearNode = &nodes[*offset];
        linearNode->bb = node->bb;
        int myOffset = (*offset)++;
        if (node->primitiveCnt > 0) {
            linearNode->primitivesOffset = node->firstPrimOffset;
            linearNode->primtiveCnt = node->primitiveCnt;
        } else {
            linearNode->axis = node->splitAxis;
            linearNode->primtiveCnt = 0;
            flattenBvhTree(node->children[0], offset);
            linearNode->secondChildOffset =
                flattenBvhTree(node->children[1], offset);
        }
        return myOffset;
    }
};

struct SurfaceData {
    // Todo: populate
};

struct inraytracer {
    // TODO: move to cuda...
    std::vector<LinearBVHNode> nodes;
    std::vector<Triangle> primitives;

    bool intersect(const Ray& ray, SurfaceData* surface) {
        bool hit = false;
        Vec3f invDir(1 / ray.dir[X], 1 / ray.dir[Y], 1 / ray.dir[Z]);
        int dirIsNeg[3] = {invDir[X] < 0, invDir[Y] < 0, invDir[Z] < 0};

        int toVisitOffset = 0;
        int currentNodeIdx = 0;
        int nodesToVisit[64];
        while (true) {
            const LinearBVHNode* node = &nodes[currentNodeIdx];
            if (node->bb.intersect(ray, invDir, dirIsNeg)) {
                if (node->primtiveCnt > 0) {
                    // leaf
                    for (int i = 0; i < node->primtiveCnt; ++i) {
                        Triangle tri = primitives[node->primitivesOffset + i];
                        if (intersect(tri, ray, surface)) {
                            hit = true;
                        }
                    }
                    if (toVisitOffset == 0) {
                        break;
                    }
                    currentNodeIdx = nodesToVisit[--toVisitOffset];
                } else {
                    // inside node, put far node on stack, advance to near
                    // check dir of ray so that nearest child is tested first
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
    bool intersect(const Triangle& t, const Ray& ray, SurfaceData* sd) {
        return true;
        // TODO: move to cuda and implement...
    }
};

void createBVH() {
    // for each tri
    // find min, max,
    // collect centroids
    // find median
    // split-plane = longest min<->max-length among X,Y,Z
    // bvh.min = min
    // bvh.max = max
    // left = min <-> median.max
    // right = median.min <-> max
}
#endif