#ifndef BVH_CUH
#define BVH_CUH

#include <algorithm>
#include <vector>
#include "BoundingBox.cuh"
#include "LinearNode.cuh"
#include "Ray.cuh"
#include "Vec.cuh"
#include "Vertex.cuh"

using namespace vectorAxes;

namespace BVH {

struct PreNode {
    int triIdx;
    BoundingBox bb;
    Vec3f centroid;
    PreNode()
        : triIdx(-1),
          centroid(Vec3f(-1.0f, -2.0f, -3.0f)),
          bb(Vec3f(-10.0f, -20.0f, -30.0f), Vec3f(-1.0f, -2.0f, -3.0f)) {}
    PreNode(int i, BoundingBox bb)
        : triIdx(i), bb(bb), centroid(bb.centroid()) {}
};

struct InterNode {
    BoundingBox bb;
    InterNode* children[2];
    int splitAxis;
    int firstPrimOffset;
    int primitiveCnt;
    enum NodeTypes { NOT_SET = -1, LEAF, INTERIOR };
    NodeTypes nodeType;

    InterNode() {
        children[0] = nullptr;
        children[1] = nullptr;
        splitAxis = -1;
        firstPrimOffset = -1;
        primitiveCnt = -1;
        nodeType = NOT_SET;
    }

    void leaf(int firstPrim, int primCnt, BoundingBox bb) {
        firstPrimOffset = firstPrim;
        primitiveCnt = primCnt;
        this->bb = bb;
        nodeType = LEAF;
    }

    void interior(int axis, int primCnt, InterNode* c0, InterNode* c1) {
        children[0] = c0;
        children[1] = c1;
        bb = BoundingBox::unionn(c0->bb, c1->bb);
        splitAxis = axis;
        primitiveCnt = primCnt;
        nodeType = INTERIOR;
    }
};

class BvhFactory {
   public:
    std::vector<Triangle> orderedPrims;
    std::vector<Triangle> primitives;
    std::vector<PreNode> primitiveInfo;
    std::vector<LinearNode> nodes;

    BvhFactory(const std::vector<Vertex>& vertices,
               const std::vector<Triangle>& primitives) {
        this->primitives = std::vector<Triangle>(primitives);
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
        if (primCnt < 200) {
            leaf(bb, primCnt, start, end, node);
            return node;
        } else {
            // Compute bound of primitive centroids, choose split dimension dim>
            BoundingBox centroidBounds;
            for (int i = start; i < end; ++i) {
                centroidBounds = BoundingBox::unionn(
                    centroidBounds, primitiveInfo[i].bb.centroid());
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

        node->interior(dim, primCnt,                       //
                       recursive(start, mid, totalNodes),  //
                       recursive(mid, end, totalNodes)     //
        );
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
        if (node->nodeType == InterNode::LEAF) {
            linearNode->primitivesOffset = node->firstPrimOffset;
            linearNode->primtiveCnt = node->primitiveCnt;
        } else if (node->nodeType == InterNode::INTERIOR) {
            linearNode->axis = node->splitAxis;
            linearNode->primtiveCnt = 0;
            flattenBvhTree(node->children[0], offset);
            linearNode->secondChildOffset =
                flattenBvhTree(node->children[1], offset);
        } else {
            // TODO:  replace with assert?
            throw "THIS SHOULD NEVER HAPPEN";
        }
        return myOffset;
    }
};
}  // namespace BVH
#endif