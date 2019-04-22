#include "gradient.cuh"

#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h>
#include <string.h>

#define PI 3.1415926536f

texture<float, 2, cudaReadModeElementType> texRef;
 /*
 * Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard.
 */

//=================================
// write to texture; 
//=================================
enum colors
{
	RED, GREEN, BLUE, ALPHA
};

__global__ void cuke_gradient(unsigned char *surface, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	float* pixel = (float *)(surface + y*pitch) + 4*x;

	pixel[RED]		= x/640.0f;
	pixel[GREEN]	= y/480.0f;
	pixel[BLUE]		= 0.0f;
	pixel[ALPHA]	= 1.0f;
}

void cu_gradient(void *surface, int width, int height, size_t pitch, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

	cuke_gradient<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch, t);

	error = cudaGetLastError();

	if( error != cudaSuccess ){
		printf( "cuda_kernel_texture_2d() failed to launch error = %d\n",
			error );
	}
}