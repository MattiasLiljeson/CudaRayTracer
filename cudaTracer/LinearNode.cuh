#ifndef LINEAR_NODE_CUH
#define LINEAR_NODE_CUH
#include "BoundingBox.cuh"
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
#endif