#ifndef BVH_CUH
#define BVH_CUH

#include <algorithm>
#include <vector>
#include "BoundingBox.cuh"
#include "Ray.cuh"
#include "Vec.cuh"
#include "Vertex.cuh"

using namespace vectorAxes;

namespace BVH {

struct PreNode {
    int triIdx;
    BoundingBox bb;
    Vec3f centroid;
    PreNode() : triIdx(-1) {}
    PreNode(int i, BoundingBox bb)
        : triIdx(i), bb(bb), centroid(bb.centroid()) {}
};

struct InterNode {
    BoundingBox bb;
    InterNode* children[2];
    int splitAxis;
    int firstPrimOffset;
    int primitiveCnt;

    InterNode() : bb(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f)) {
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

    void interior(int axis, InterNode* c0, InterNode* c1) {
        children[0] = c0;
        children[1] = c1;
        bb = BoundingBox::unionn(c0->bb, c1->bb);
        splitAxis = axis;
        primitiveCnt = 0;
    }
};

struct LinearNode {
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
   public:
    std::vector<Triangle> orderedPrims;
    std::vector<Triangle> primitives;
    std::vector<PreNode> primitiveInfo;
    std::vector<LinearNode> nodes;

    BvhFactory(const std::vector<Vertex>& vertices,
               const std::vector<Triangle>& primitives) {
        this->primitives = vector<Triangle>(primitives);
        primitiveInfo.resize(primitives.size());
        for (int i = 0; i < primitives.size(); ++i) {
            primitiveInfo[i] = {i, BoundingBox(primitives[i], vertices)};
        }

        int totalNodes = 0;
        InterNode* root = recursive(0, primitives.size(), totalNodes);
        // fetch new vector and replace yourself instead
        // primitives.swap(orderedPrims);

        nodes = std::vector<LinearNode>(totalNodes);
        int offset = 0;
        flattenBvhTree(root, &offset);
    }

    InterNode* recursive(const int start, const int end, int& totalNodes) {
        InterNode* node = new InterNode();  // TODO: new....
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
            int dim = centroidBounds.maxExtent();

            // Partition primitives into two sets and build children
            if (centroidBounds.bbmax[dim] == centroidBounds.bbmin[dim]) {
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
                  const int dim, InterNode* node, int& totalNodes) {
        int mid = (start + end) / 2;
        std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
                         &primitiveInfo[end - 1] + 1,
                         [dim](const PreNode& a, const PreNode& b) {
                             return a.bb.centroid()[dim] < b.bb.centroid()[dim];
                         });

        node->interior(dim, recursive(start, mid, totalNodes),
                       recursive(mid, end, totalNodes));
    }

    void leaf(const BoundingBox& bb, const int primCnt, const int start,
              const int end, InterNode* node) {
        int firstPrimOffset = orderedPrims.size();
        for (int i = start; i < end; ++i) {
            int idx = primitiveInfo[i].triIdx;
            orderedPrims.push_back(primitives[idx]);
        }
        node->leaf(firstPrimOffset, primCnt, bb);
    }

    int flattenBvhTree(InterNode* node, int* offset) {
        LinearNode* linearNode = &nodes[*offset];
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
    std::vector<LinearNode> nodes;
    std::vector<Triangle> primitives;

    bool intersect(const Ray& ray, SurfaceData* surface) {
        bool hit = false;
        Vec3f invDir(1 / ray.dir[X], 1 / ray.dir[Y], 1 / ray.dir[Z]);
        int dirIsNeg[3] = {invDir[X] < 0, invDir[Y] < 0, invDir[Z] < 0};

        int toVisitOffset = 0;
        int currentNodeIdx = 0;
        int nodesToVisit[64];
        while (true) {
            const LinearNode* node = &nodes[currentNodeIdx];
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
}  // namespace BVH
#endif