#ifdef _TEST

#include "../Catch2/single_include/catch2/catch.hpp"
#include "BoundingBox.cuh"
#include "BVH.h"

TEST_CASE("PreNode") {
    BoundingBox bb(Vec3f(-3.0f, -2.0f, -1.0f), Vec3f(0.0f, 1.0f, 2.0f));
    BVH::PreNode n(0, bb);
    REQUIRE(n.bb.bbmin[X] == Approx(-3.0));
    REQUIRE(n.bb.bbmin[Y] == Approx(-2.0));
    REQUIRE(n.bb.bbmin[Z] == Approx(-1.0));
    REQUIRE(n.bb.bbmax[X] == Approx(0.0));
    REQUIRE(n.bb.bbmax[Y] == Approx(1.0));
    REQUIRE(n.bb.bbmax[Z] == Approx(2.0));
    REQUIRE(n.centroid[X] == Approx(-1.5));
    REQUIRE(n.centroid[Y] == Approx(-0.5));
    REQUIRE(n.centroid[Z] == Approx(0.5));
}

#endif