#ifndef MATH_CUH
#define MATH_CUH

#include <cuda_runtime.h>

#define INF 2e10f

__device__ float clamp(const float &lo, const float &hi, const float &v) {
    return fmax(lo, fmin(hi, v));
}

#endif