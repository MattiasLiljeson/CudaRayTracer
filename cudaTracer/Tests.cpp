#ifdef _TEST

#include <iostream>
#include "../Catch2/single_include/catch2/catch.hpp"
#include "GlobalCudaArray.h"
#include "Sphere.cuh"
#include "Mat.cuh"
#include "Vec.cuh"

TEST_CASE("GlobalCudaArray") {
    {
        GlobalCudaArray<Sphere, 1> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 2> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 4> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 8> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 16> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 25> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 32> arr;
        arr.copyToDevice();
    }
    {
        GlobalCudaArray<Sphere, 64> arr;
        arr.copyToDevice();
    }
}

TEST_CASE("Matrix multiplication") {
    Mat44<int> I = Mat44<int>::identity();
    SECTION("I * I = I") {
        Mat44<int> I = Mat44<int>::identity();
        REQUIRE(I * I == I);
    }

    SECTION("R*I=R, R*I=R") {
        Mat44<int> R = Mat44<int>::rotationZ(90.0f);
        REQUIRE(I * R == R);
        REQUIRE(R * I == R);
    }

    SECTION("M*N != N*M") {
        Mat44<int> M =
            Mat44<int>(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6);
        Mat44<int> N =
            Mat44<int>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5);
        REQUIRE(N * M != M * N);
    }
}

TEST_CASE("Matrix - vector multiplication") {
    SECTION("Points") {
        Vec<int, 3> origo = Vec<int, 3>(0, 0, 0);
        Vec<int, 3> fours = Vec<int, 3>(4, 4, 4);
        SECTION("Translation works only on points") {
            Mat44<int> translation(1, 0, 0, 0,  //
                                   0, 1, 0, 0,  //
                                   0, 0, 1, 0,  //
                                   4, 16, 44, 4);
            SECTION("Origo") {
                Vec<int, 3> translated = translation.multPoint(origo);
                REQUIRE(translated[Vec3f::X] == 1);
                REQUIRE(translated[Vec3f::Y] == 4);
                REQUIRE(translated[Vec3f::Z] == 11);
            }
            SECTION("Fours") {
                Vec<int, 3> translated = translation.multPoint(fours);
                REQUIRE(translated[Vec3f::X] == 2);
                REQUIRE(translated[Vec3f::Y] == 5);
                REQUIRE(translated[Vec3f::Z] == 12);
            }
        }

        SECTION("Scaling with XYZ works with points") {
            Mat44<int> scalingXYZ(1, 0, 0, 0,  //
                                  0, 2, 0, 0,  //
                                  0, 0, 4, 0,  //
                                  0, 0, 0, 1);
            SECTION("Origo") {
                Vec<int, 3> translated = scalingXYZ.multPoint(origo);
                REQUIRE(translated[Vec3f::X] == 0);
                REQUIRE(translated[Vec3f::Y] == 0);
                REQUIRE(translated[Vec3f::Z] == 0);
            }
            SECTION("Fours") {
                Vec<int, 3> translated = scalingXYZ.multPoint(fours);
                REQUIRE(translated[Vec3f::X] == 4);
                REQUIRE(translated[Vec3f::Y] == 8);
                REQUIRE(translated[Vec3f::Z] == 16);
            }
        }

        SECTION("Scaling with W works only with points") {
            Mat44<int> scalingW(1, 0, 0, 0,  //
                                0, 1, 0, 0,  //
                                0, 0, 1, 0,  //
                                0, 0, 0, 4);
            SECTION("Origo") {
                Vec<int, 3> translated = scalingW.multPoint(origo);
                REQUIRE(translated[Vec3f::X] == 0);
                REQUIRE(translated[Vec3f::Y] == 0);
                REQUIRE(translated[Vec3f::Z] == 0);
            }
            SECTION("Fours") {
                Vec<int, 3> translated = scalingW.multPoint(fours);
                REQUIRE(translated[Vec3f::X] == 1);
                REQUIRE(translated[Vec3f::Y] == 1);
                REQUIRE(translated[Vec3f::Z] == 1);
            }
        }
    }

    SECTION("Vectors") {
        Vec<int, 3> zeros = Vec<int, 3>(0, 0, 0);
        Vec<int, 3> fours = Vec<int, 3>(4, 4, 4);
        SECTION("Translation on vectors does not work") {
            Mat44<int> translation(1, 0, 0, 0,  //
                                   0, 1, 0, 0,  //
                                   0, 0, 1, 0,  //
                                   4, 16, 44, 4);
            SECTION("Zeros") {
                Vec<int, 3> translated = translation.multVec(zeros);
                REQUIRE(translated[Vec3f::X] == 0);
                REQUIRE(translated[Vec3f::Y] == 0);
                REQUIRE(translated[Vec3f::Z] == 0);
            }
            SECTION("Fours") {
                Vec<int, 3> translated = translation.multVec(fours);
                REQUIRE(translated[Vec3f::X] == 4);
                REQUIRE(translated[Vec3f::Y] == 4);
                REQUIRE(translated[Vec3f::Z] == 4);
            }
        }

        SECTION("Scaling with XYZ works with Vectors") {
            Mat44<int> scalingXYZ(1, 0, 0, 0,  //
                                  0, 2, 0, 0,  //
                                  0, 0, 4, 0,  //
                                  0, 0, 0, 1);
            SECTION("Zeros") {
                Vec<int, 3> translated = scalingXYZ.multVec(zeros);
                REQUIRE(translated[Vec3f::X] == 0);
                REQUIRE(translated[Vec3f::Y] == 0);
                REQUIRE(translated[Vec3f::Z] == 0);
            }
            SECTION("Fours") {
                Vec<int, 3> translated = scalingXYZ.multVec(fours);
                REQUIRE(translated[Vec3f::X] == 4);
                REQUIRE(translated[Vec3f::Y] == 8);
                REQUIRE(translated[Vec3f::Z] == 16);
            }
        }

        SECTION("Can't scale Vectors with W") {
            Mat44<int> scalingW(1, 0, 0, 0,  //
                                0, 1, 0, 0,  //
                                0, 0, 1, 0,  //
                                0, 0, 0, 4);
            SECTION("Zeros") {
                Vec<int, 3> translated = scalingW.multVec(zeros);
                REQUIRE(translated[Vec3f::X] == 0);
                REQUIRE(translated[Vec3f::Y] == 0);
                REQUIRE(translated[Vec3f::Z] == 0);
            }
            SECTION("Fours") {
                Vec<int, 3> translated = scalingW.multVec(fours);
                REQUIRE(translated[Vec3f::X] == 4);
                REQUIRE(translated[Vec3f::Y] == 4);
                REQUIRE(translated[Vec3f::Z] == 4);
            }
        }
    }
}

TEST_CASE("Rotation matrix") {
    Vec3f right = Vec3f(1.0f, 0.0f, 0.0f);
    Vec3f left = Vec3f(-1.0f, 0.0f, 0.0f);
    Vec3f up = Vec3f(0.0f, 1.0f, 0.0f);
    Vec3f down = Vec3f(0.0f, -1.0f, 0.0f);
    Vec3f forward = Vec3f(0.0f, 0.0f, 1.0f);
    Vec3f backward = Vec3f(0.0f, 0.0f, -1.0f);

    SECTION("X") {
        Mat44f rotMat = Mat44f::rotation(right, 90.0f);
        Mat44f rotMatX = Mat44f::rotationX(90.0f);
        Vec3f rotated = rotMat.multVec(up);
        Vec3f rotatedX = rotMatX.multVec(up);

        REQUIRE(rotated[Vec3f::X] == Approx(0.0).margin(0.1));
        REQUIRE(rotated[Vec3f::Y] == Approx(0.0).margin(0.1));
        REQUIRE(rotated[Vec3f::Z] == Approx(-1.0).margin(0.1));

        REQUIRE(rotated[Vec3f::X] == Approx(rotatedX[Vec3f::X]).margin(0.1));
        REQUIRE(rotated[Vec3f::Y] == Approx(rotatedX[Vec3f::Y]).margin(0.1));
        REQUIRE(rotated[Vec3f::Z] == Approx(rotatedX[Vec3f::Z]).margin(0.1));
    }

    SECTION("Y") {
        Mat44f rotMat = Mat44f::rotation(up, 90.0f);
        Mat44f rotMatY = Mat44f::rotationY(90.0f);
        Vec3f rotated = rotMat.multVec(right);
        Vec3f rotatedY = rotMatY.multVec(right);

        REQUIRE(rotated[Vec3f::X] == Approx(0.0).margin(0.1));
        REQUIRE(rotated[Vec3f::Y] == Approx(0.0).margin(0.1));
        REQUIRE(rotated[Vec3f::Z] == Approx(1.0).margin(0.1));

        REQUIRE(rotated[Vec3f::X] == Approx(rotatedY[Vec3f::X]).margin(0.1));
        REQUIRE(rotated[Vec3f::Y] == Approx(rotatedY[Vec3f::Y]).margin(0.1));
        REQUIRE(rotated[Vec3f::Z] == Approx(rotatedY[Vec3f::Z]).margin(0.1));
    }

    SECTION("Z") {
        Mat44f rotMat = Mat44f::rotation(forward, 90.0f);
        Mat44f rotMatZ = Mat44f::rotationZ(90.0f);
        Vec3f rotated = rotMat.multVec(right);
        Vec3f rotatedZ = rotMatZ.multVec(right);

        REQUIRE(rotated[Vec3f::X] == Approx(0.0).margin(0.1));
        REQUIRE(rotated[Vec3f::Y] == Approx(-1.0).margin(0.1));
        REQUIRE(rotated[Vec3f::Z] == Approx(0.0).margin(0.1));

        REQUIRE(rotated[Vec3f::X] == Approx(rotatedZ[Vec3f::X]).margin(0.1));
        REQUIRE(rotated[Vec3f::Y] == Approx(rotatedZ[Vec3f::Y]).margin(0.1));
        REQUIRE(rotated[Vec3f::Z] == Approx(rotatedZ[Vec3f::Z]).margin(0.1));
    }
}
#endif