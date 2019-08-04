#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include <cuda_runtime.h>

struct Triangle {
    int i[3];
    __host__ __device__ Triangle() : i{-1, -1, -1} {}
    __host__ __device__ Triangle(int v0, int v1, int v2) : i{v0, v1, v2} {}
    __host__ __device__ Triangle(int i, int v[])
        : i{v[i], v[i + 1], v[i + 2]} {}

    __device__ int& operator[](int idx) { return i[idx]; }
    __device__ const int& operator[](int idx) const { return i[idx]; }
};

#endif