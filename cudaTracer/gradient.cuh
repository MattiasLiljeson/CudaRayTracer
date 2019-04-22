#ifndef GRADIENT_CUH
#define GRADIENT_CUH
#include <cuda_runtime.h>

extern "C" {
	void cu_gradient(void *surface, int width, int height, size_t pitch, float t);
}

#endif