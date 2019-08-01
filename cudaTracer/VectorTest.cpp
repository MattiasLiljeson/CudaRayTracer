#ifdef _TEST

#include "../Catch2/single_include/catch2/catch.hpp"
#include "Vec.cuh"

TEST_CASE("Vectors") {
    typedef Vec<int, 3> Vec3i;
    Vec3i x(1, 0, 0);
    Vec3i y(0, 1, 0);
    Vec3i z(0, 0, 1);
    Vec3i v(1, 2, 3);

    SECTION("ctor") {
        Vec3i a(1, 2, 3);
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 2);
        REQUIRE(a[Vec3i::Z] == 3);
    }

    SECTION("copy ctor") {
        Vec3i a(v);
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 2);
        REQUIRE(a[Vec3i::Z] == 3);
    }

    SECTION("operator * with scalar") {
        Vec3i a = x * 3;
        REQUIRE(a[Vec3i::X] == 3);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = v * 3;
        REQUIRE(b[Vec3i::X] == 3);
        REQUIRE(b[Vec3i::Y] == 6);
        REQUIRE(b[Vec3i::Z] == 9);
    }

    SECTION("operator * with vector") {
        Vec3i a = x * y;
        REQUIRE(a[Vec3i::X] == 0);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = v * z;
        REQUIRE(b[Vec3i::X] == 0);
        REQUIRE(b[Vec3i::Y] == 0);
        REQUIRE(b[Vec3i::Z] == 3);
    }

    SECTION("operator / with scalar") {
        Vec3i a = x / 1;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = v * 6 / 3;
        REQUIRE(b[Vec3i::X] == 2);
        REQUIRE(b[Vec3i::Y] == 4);
        REQUIRE(b[Vec3i::Z] == 6);
    }

    SECTION("operator / with vector") {
        Vec3i a = x / v;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = z / v;
        REQUIRE(b[Vec3i::X] == 0);
        REQUIRE(b[Vec3i::Y] == 0);
        REQUIRE(b[Vec3i::Z] == 0);

        Vec3i c = Vec3i(10, 20, 30) / v;
        REQUIRE(c[Vec3i::X] == 10);
        REQUIRE(c[Vec3i::Y] == 10);
        REQUIRE(c[Vec3i::Z] == 10);
    }

    SECTION("operator - with scalar") {
        Vec3i a = x - 3;
        REQUIRE(a[Vec3i::X] == -2);
        REQUIRE(a[Vec3i::Y] == -3);
        REQUIRE(a[Vec3i::Z] == -3);

        Vec3i b = v - 3;
        REQUIRE(b[Vec3i::X] == -2);
        REQUIRE(b[Vec3i::Y] == -1);
        REQUIRE(b[Vec3i::Z] == 0);
    }

    SECTION("operator - with vector") {
        Vec3i a = x - y;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == -1);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = v - z;
        REQUIRE(b[Vec3i::X] == 1);
        REQUIRE(b[Vec3i::Y] == 2);
        REQUIRE(b[Vec3i::Z] == 2);
    }

    SECTION("operator + with scalar") {
        Vec3i a = x + 3;
        REQUIRE(a[Vec3i::X] == 4);
        REQUIRE(a[Vec3i::Y] == 3);
        REQUIRE(a[Vec3i::Z] == 3);

        Vec3i b = v + 3;
        REQUIRE(b[Vec3i::X] == 4);
        REQUIRE(b[Vec3i::Y] == 5);
        REQUIRE(b[Vec3i::Z] == 6);
    }

    SECTION("operator + with vector") {
        Vec3i a = x + y;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 1);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = v + z;
        REQUIRE(b[Vec3i::X] == 1);
        REQUIRE(b[Vec3i::Y] == 2);
        REQUIRE(b[Vec3i::Z] == 4);
    }

    SECTION("operator - (negation)") {
        Vec3i a = -x;
        REQUIRE(a[Vec3i::X] == -1);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b = -v;
        REQUIRE(b[Vec3i::X] == -1);
        REQUIRE(b[Vec3i::Y] == -2);
        REQUIRE(b[Vec3i::Z] == -3);
    }

    SECTION("operator += with scalar") {
        Vec3i a(x);
        a += 3;
        REQUIRE(a[Vec3i::X] == 4);
        REQUIRE(a[Vec3i::Y] == 3);
        REQUIRE(a[Vec3i::Z] == 3);

        Vec3i b(v);
        b += 3;
        REQUIRE(b[Vec3i::X] == 4);
        REQUIRE(b[Vec3i::Y] == 5);
        REQUIRE(b[Vec3i::Z] == 6);
    }

    SECTION("operator += with vector") {
        Vec3i a(x);
        a += y;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 1);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b(v);
        b += z;
        REQUIRE(b[Vec3i::X] == 1);
        REQUIRE(b[Vec3i::Y] == 2);
        REQUIRE(b[Vec3i::Z] == 4);
    }

    SECTION("operator -= with scalar") {
        Vec3i a(x);
        a -= 3;
        REQUIRE(a[Vec3i::X] == -2);
        REQUIRE(a[Vec3i::Y] == -3);
        REQUIRE(a[Vec3i::Z] == -3);

        Vec3i b(v);
        b -= 3;
        REQUIRE(b[Vec3i::X] == -2);
        REQUIRE(b[Vec3i::Y] == -1);
        REQUIRE(b[Vec3i::Z] == 0);
    }

    SECTION("operator -= with vector") {
        Vec3i a(x);
        a -= y;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == -1);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b(v);
        b -= z;
        REQUIRE(b[Vec3i::X] == 1);
        REQUIRE(b[Vec3i::Y] == 2);
        REQUIRE(b[Vec3i::Z] == 2);
    }

    SECTION("operator *= with scalar") {
        Vec3i a(x);
        a *= 3;
        REQUIRE(a[Vec3i::X] == 3);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b(v);
        b *= 3;
        REQUIRE(b[Vec3i::X] == 3);
        REQUIRE(b[Vec3i::Y] == 6);
        REQUIRE(b[Vec3i::Z] == 9);
    }

    SECTION("operator *= with vector") {
        Vec3i a(x);
        a *= y;
        REQUIRE(a[Vec3i::X] == 0);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b(v);
        b *= z;
        REQUIRE(b[Vec3i::X] == 0);
        REQUIRE(b[Vec3i::Y] == 0);
        REQUIRE(b[Vec3i::Z] == 3);
    }

    SECTION("operator /= with scalar") {
        Vec3i a(x);
        a /= 3;
        REQUIRE(a[Vec3i::X] == 0);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b(v);
        b /= 3;
        REQUIRE(b[Vec3i::X] == 0);
        REQUIRE(b[Vec3i::Y] == 0);
        REQUIRE(b[Vec3i::Z] == 1);

        Vec3i c = Vec3i(10, 20, 30) / 3;
        REQUIRE(c[Vec3i::X] == 3);
        REQUIRE(c[Vec3i::Y] == 6);
        REQUIRE(c[Vec3i::Z] == 10);
    }

    SECTION("operator /= with vector") {
        Vec3i a(x);
        a *= y;
        REQUIRE(a[Vec3i::X] == 0);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 0);

        Vec3i b(v);
        b *= z;
        REQUIRE(b[Vec3i::X] == 0);
        REQUIRE(b[Vec3i::Y] == 0);
        REQUIRE(b[Vec3i::Z] == 3);

        Vec3i c = Vec3i(10, 20, 30) / v;
        REQUIRE(c[Vec3i::X] == 10);
        REQUIRE(c[Vec3i::Y] == 10);
        REQUIRE(c[Vec3i::Z] == 10);
    }

    SECTION("[] write") {
        x[Vec3i::X] = 2;
        REQUIRE(x[Vec3i::X] == 2);
    }

    SECTION("[] read") {
        REQUIRE(x[Vec3i::X] == 1);
        REQUIRE(x[Vec3i::Y] == 0);
        REQUIRE(x[Vec3i::Z] == 0);
    }

    SECTION("friend operator + with scalar") {
        Vec3i a = z + v;
        REQUIRE(a[Vec3i::X] == 1);
        REQUIRE(a[Vec3i::Y] == 2);
        REQUIRE(a[Vec3i::Z] == 4);
    }

    SECTION("friend operator - with scalar") {
        Vec3i a = z - v;
        REQUIRE(a[Vec3i::X] == -1);
        REQUIRE(a[Vec3i::Y] == -2);
        REQUIRE(a[Vec3i::Z] == -2);
    }

    SECTION("friend operator * with scalar") {
        Vec3i a = z * v;
        REQUIRE(a[Vec3i::X] == 0);
        REQUIRE(a[Vec3i::Y] == 0);
        REQUIRE(a[Vec3i::Z] == 3);
    }

    SECTION("friend operator / with scalar") {
        Vec3i a = Vec3i(10, 4, 90) / v;
        REQUIRE(a[Vec3i::X] == 10);
        REQUIRE(a[Vec3i::Y] == 2);
        REQUIRE(a[Vec3i::Z] == 30);
    }

    SECTION("friend operator ==") { REQUIRE(v == Vec3i(1, 2, 3)); }

    SECTION("friend operator !=") { REQUIRE(x != y); }

    SECTION("magnitude()") {
        REQUIRE(x.magnitude() == Approx(1));
        REQUIRE(v.magnitude() == Approx(3.75).margin(0.01));
    }

    SECTION("normalized()") {
        REQUIRE(x.normalized() == Vec3i(1, 0, 0));
        REQUIRE(v.normalized() == Vec3i(0, 0, 0));
        Vec3f a = Vec3f(1.0f, 2.0f, 3.0f).normalized();
        REQUIRE(a[Vec3f::X] == Approx(0.267).margin(0.01));
        REQUIRE(a[Vec3f::Y] == Approx(0.535).margin(0.01));
        REQUIRE(a[Vec3f::Z] == Approx(0.802).margin(0.01));
    }

    SECTION("dot product") {
        REQUIRE(x.dot(y) == 0);
        REQUIRE(x.dot(v) == 1);
        REQUIRE(y.dot(v) == 2);
        REQUIRE(v.dot(v) == 14);
    }

    SECTION("cross product") {
        REQUIRE(x.cross(y) == z);
        Vec3i tmp1 = x.cross(y);
        Vec3i tmp = y.cross(x);
        REQUIRE(y.cross(x) == -z);
    }

    SECTION("Previous bugs") {
        SECTION("Only the X axis was modified") {}
        Vec3f normSamp(0.5f, 1.0f, 0.1f);
        normSamp = ((normSamp * 2.0f) - 1.0f);
        REQUIRE(normSamp[Vec3f::X] == Approx(0.0));
        REQUIRE(normSamp[Vec3f::Y] == Approx(1.0));
        REQUIRE(normSamp[Vec3f::Z] == Approx(-0.8));
    }
}

#endif