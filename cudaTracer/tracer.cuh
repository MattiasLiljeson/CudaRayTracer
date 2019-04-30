#pragma once

extern "C" {
void main(const Options *options, const Light *lights, const int lightCnt,
          const Sphere *spheres, const int sphereCnt, const void *surface,
          const int width, const int height, size_t pitch, const Vec3f &orig);
}