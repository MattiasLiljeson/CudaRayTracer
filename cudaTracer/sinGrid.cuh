#ifndef SINGRID_CUH
#define SINGRID_CUH
#include <cuda_runtime.h>

extern "C" {
	void cu_sinGrid(void *surface, int width, int height, size_t pitch, float t);
}

#endif