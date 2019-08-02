#ifdef _TEST

#include "../Catch2/single_include/catch2/catch.hpp"
#include "BoundingBox.cuh"

TEST_CASE("Triangle") {
    SECTION("ctor, empty") { REQUIRE(Triangle().i[0] == -1); }
    SECTION("ctor, three ints") {
        Triangle t(1, 2, 3);
        REQUIRE(t.i[0] == 1);
        REQUIRE(t.i[1] == 2);
        REQUIRE(t.i[2] == 3);
    }
    SECTION("ctor, three ints") {
        int values[5] = {0, 1, 2, 3, 4};
        Triangle t(1, values);
        REQUIRE(t.i[0] == 1);
        REQUIRE(t.i[1] == 2);
        REQUIRE(t.i[2] == 3);
    }
}

TEST_CASE("BoundingBox") {
    SECTION("Create BoundingBox from Vectors") {
        Vec3f a(0.0f, 1.0f, 2.0f);
        Vec3f b(3.0f, 4.0f, 5.0f);
        BoundingBox bb(a, b);
        REQUIRE(bb.bbmin[X] == Approx(0.0));
        REQUIRE(bb.bbmin[Y] == Approx(1.0));
        REQUIRE(bb.bbmin[Z] == Approx(2.0));
        REQUIRE(bb.bbmax[X] == Approx(3.0));
        REQUIRE(bb.bbmax[Y] == Approx(4.0));
        REQUIRE(bb.bbmax[Z] == Approx(5.0));
    }

    SECTION("Create BoundingBox from Triangle") {
        std::vector<Vertex> vertices = {
            Vertex(-2.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.1f, 0.2f),  //
            Vertex(-1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.3f, 0.4f),   //
            Vertex(1.0f, 2.0f, 3.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.6f)};
        Triangle tri = {0, 1, 2};
        BoundingBox bb(tri, vertices);
        REQUIRE(bb.bbmin[X] == Approx(-2.0));
        REQUIRE(bb.bbmin[Y] == Approx(-1.0));
        REQUIRE(bb.bbmin[Z] == Approx(0.0));
        REQUIRE(bb.bbmax[X] == Approx(1.0));
        REQUIRE(bb.bbmax[Y] == Approx(2.0));
        REQUIRE(bb.bbmax[Z] == Approx(3.0));
    }

    SECTION("centroid()") {
        Vec3f a(0.0f, 1.0f, 2.0f);
        Vec3f b(1.0f, 4.0f, 5.0f);
        BoundingBox bb(a, b);
        Vec3f c = bb.centroid();

        REQUIRE(c[X] == Approx(0.5));
        REQUIRE(c[Y] == Approx(2.5));
        REQUIRE(c[Z] == Approx(3.5));
    }

    SECTION("diagonal()") {
        SECTION("X") {
            Vec3f a(1.0f, 0.0f, 0.0f);
            Vec3f b(5.0f, 2.0f, 1.0f);
            Vec3f c = BoundingBox(a, b).diagonal();

            REQUIRE(c[X] == Approx(4.0));
            REQUIRE(c[Y] == Approx(2.0));
            REQUIRE(c[Z] == Approx(1.0));
        }

        SECTION("Y==Z") {
            Vec3f a(0.0f, 1.0f, 2.0f);
            Vec3f b(1.0f, 4.0f, 5.0f);
            Vec3f c = BoundingBox(a, b).diagonal();

            REQUIRE(c[X] == Approx(1.0));
            REQUIRE(c[Y] == Approx(3.0));
            REQUIRE(c[Z] == Approx(3.0));
        }

        SECTION("X==Y==Z") {
            Vec3f a(1.0f, 0.0f, 0.0f);
            Vec3f b(2.0f, 1.0f, 1.0f);
            Vec3f c = BoundingBox(a, b).diagonal();

            REQUIRE(c[X] == Approx(1.0));
            REQUIRE(c[Y] == Approx(1.0));
            REQUIRE(c[Z] == Approx(1.0));
        }
    }

    SECTION("extent()") {
        SECTION("X") {
            Vec3f a(1.0f, 0.0f, 0.0f);
            Vec3f b(5.0f, 2.0f, 1.0f);
            REQUIRE(BoundingBox(a, b).maxExtent() == 0);
        }
        SECTION("Y==Z -> Z") {
            Vec3f a(0.0f, 1.0f, 2.0f);
            Vec3f b(1.0f, 4.0f, 5.0f);
            REQUIRE(BoundingBox(a, b).maxExtent() == 2);
        }
        SECTION("X==Y==Z -> Z") {
            Vec3f a(1.0f, 0.0f, 0.0f);
            Vec3f b(2.0f, 1.0f, 1.0f);
            REQUIRE(BoundingBox(a, b).maxExtent() == 2);
        }
    }

    SECTION("unionn with two BBs") {
        Vec3f a(0.0f, 1.0f, 2.0f);
        Vec3f b(-1.0f, 4.0f, 5.0f);
        BoundingBox bb1(a, b);
        BoundingBox bb2(a + 2, b + 2);
        BoundingBox bb3 = BoundingBox::unionn(bb1, bb2);
        SECTION("min and max are not swapped if they represent the opposite") {
            REQUIRE(bb3.bbmin[X] != Approx(-1.0));
            REQUIRE(bb3.bbmin[X] == Approx(-0.0));
        }
        REQUIRE(bb3.bbmin[Y] == Approx(1.0));
        REQUIRE(bb3.bbmin[Z] == Approx(2.0));
        REQUIRE(bb3.bbmax[X] == Approx(1.0));
        REQUIRE(bb3.bbmax[Y] == Approx(6.0));
        REQUIRE(bb3.bbmax[Z] == Approx(7.0));
    }

    SECTION("unionn with one BB and one vector") {
        Vec3f a(0.0f, 1.0f, 2.0f);
        Vec3f b(1.0f, 4.0f, 5.0f);
        BoundingBox bb1(a, b);
        Vec3f c = b * 3;
        BoundingBox bb3 = BoundingBox::unionn(bb1, c);
        REQUIRE(bb3.bbmin[X] == Approx(0.0));
        REQUIRE(bb3.bbmin[Y] == Approx(1.0));
        REQUIRE(bb3.bbmin[Z] == Approx(2.0));
        REQUIRE(bb3.bbmax[X] == Approx(3.0));
        REQUIRE(bb3.bbmax[Y] == Approx(12.0));
        REQUIRE(bb3.bbmax[Z] == Approx(15.0));
    }
}

#endif