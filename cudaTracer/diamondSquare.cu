#include "diamondSquare.cuh"

#include "device_launch_parameters.h"

#include <curand_kernel.h>
#include <curand_normal.h>

#include "globals.h"
#include "cudaUtils.h"

__constant__ size_t C_PITCH;
__constant__ int C_WIDTH;
__constant__ int C_HEIGHT;

enum colors
{
	RED, GREEN, BLUE, ALPHA
};

__device__ float rand( curandState* const localState ){
	return curand_uniform( localState );
	//return 0.5f;
}

__device__ float* getPixel( const unsigned char* surface, int x, int y )
{
	x = x<0				? x+(C_WIDTH-1)  : x;
	x = x>(C_WIDTH-1)	? x-(C_WIDTH-1)  : x;
	y = y<0				? y+(C_HEIGHT-1) : y;
	y = y>(C_HEIGHT-1)	? y-(C_HEIGHT-1) : y;

	// get a pointer to the pixel at (x,y)
	return (float *)(surface + y*C_PITCH) + 4*x;
}


__device__ float getColOrRand( const unsigned char* surface,
	curandState* const localState, int2 pixel, int idx ){

	float* pixPtr = getPixel( surface, pixel.x, pixel.y );
	return pixPtr[idx];
}

__device__ float createColor( const unsigned char* surface,
	curandState* const localState, int side, int2 a, int2 b, int2 c,
	int2 d, int idx ){
	float aCol = getColOrRand( surface, localState, a, idx );
	float bCol = getColOrRand( surface, localState, b, idx );
	float cCol = getColOrRand( surface, localState, c, idx );
	float dCol = getColOrRand( surface, localState, d, idx );
	float finalCol = (aCol + bCol + cCol + dCol) * 0.25f;
	return finalCol + ((rand(localState)*2 - 1)) * sqrt((float)side) * RAND_FAC;
}

__device__ void diamond( const unsigned char* surface,
	curandState* const rngStates, int x, int y, int side ){

	float* pixel = getPixel( surface, x, y );
	int2 lt; lt.x = x-side; lt.y = y-side;
	int2 lb; lb.x = x-side; lb.y = y+side;
	int2 rt; rt.x = x+side; rt.y = y-side;
	int2 rb; rb.x = x+side; rb.y = y+side;
	curandState* localState = rngStates + threadIdx.y*blockDim.x + threadIdx.x;

	pixel[RED]		= createColor( surface, localState, side, lt, lb, rt, rb, RED );
	pixel[GREEN]	= createColor( surface, localState, side, lt, lb, rt, rb, GREEN );
	pixel[BLUE]		= createColor( surface, localState, side, lt, lb, rt, rb, BLUE );
	pixel[ALPHA]	= 1.0f;
}

__device__ void square( const unsigned char* surface,
	curandState* const rngStates, int x, int y, int side ){

	float* pixel = getPixel( surface, x, y);
	int2 t; t.x = x; t.y = y-side;
	int2 b; b.x = x; b.y = y+side;
	int2 l; l.x = x-side; l.y = y;
	int2 r; r.x = x+side; r.y = y;
	curandState* localState = rngStates + threadIdx.y*blockDim.x + threadIdx.x;

	pixel[RED]		= createColor( surface, localState, side, t, b, l, r, RED );
	pixel[GREEN]	= createColor( surface, localState, side, t, b, l, r, GREEN );
	pixel[BLUE]		= createColor( surface, localState, side, t, b, l, r, BLUE );
	pixel[ALPHA]	= 1.0f;
}

__global__ void cuke_createSquares( const unsigned char* surface, int side,
								   curandState* const rngStates ){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= C_WIDTH || y >= C_HEIGHT) return;

	if( (x%(side) == 0) && (y%(side) == 0) ){
		bool xMid = (x+side) % (side*2) == 0;
		bool yMid = (y+side) % (side*2) == 0;
		if( xMid && yMid ){
			diamond( surface, rngStates, x, y, side );
		}		
	}
}

__global__ void cuke_createDiamonds( const unsigned char* surface, int side,
									curandState* const rngStates ){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= C_WIDTH || y >= C_HEIGHT) return;

	if( (x%(side) == 0) && (y%(side) == 0) ){
		bool xMid = (x+side) % (side*2) == 0;
		bool yMid = (y+side) % (side*2) == 0;
		if( xMid != yMid ){
			square( surface, rngStates, x, y, side );
		}
	}
}

__global__ void cuke_clear( const unsigned char* surface ){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= C_WIDTH || y >= C_HEIGHT) return;

	// get a pointer to the pixel at (x,y)
	float* pixel = (float *)(surface + y*C_PITCH) + 4*x;

	pixel[RED]		= 0.0f;
	pixel[GREEN]	= 0.0f;
	pixel[BLUE]		= 0.0f;
	pixel[ALPHA]	= 1.0f;

	float val = 0.5f;

	if( x==0 && y==0 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	} else if( x==C_WIDTH-1 && y==0 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	} else if( x==0 && y==C_HEIGHT-1 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	} else if( x==C_WIDTH-1 && y==C_HEIGHT-1 ){
		pixel[RED]		= val;
		pixel[GREEN]	= val;
		pixel[BLUE]		= val;
	}
}

void cu_diamondSquare( const void* surface, int width, int height, size_t pitch, 
					  unsigned char* p_rngStates ){
	dim3 block = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 grid = dim3((width+block.x-1)/block.x, (height+block.y-1)/block.y);

	cudaMemcpyToSymbol( C_PITCH, &pitch, sizeof(size_t) );
	cudaMemcpyToSymbol( C_WIDTH, &width, sizeof(int) );
	cudaMemcpyToSymbol( C_HEIGHT, &height, sizeof(int) );
	gpuErrchk( cudaPeekAtLastError() );

	// Do the magic
	cuke_clear<<< grid, block >>>( (unsigned char *)surface );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	for( int i=width; i>0; i/=2 ){
		cuke_createSquares<<< grid, block >>>( (unsigned char *)surface,
			i, (curandState*)p_rngStates );
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		cuke_createDiamonds<<< grid, block >>>( (unsigned char *)surface,
			i, (curandState*)p_rngStates );
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
}

// RNG init kernel
__global__ void cuke_initRNG(curandState *const rngStates,
						const unsigned int seed, int blkXIdx )
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int x = blkXIdx*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	tid = x*gridDim.x + y;

	// Initialise the RNG
	curand_init( seed, tid, 0, &rngStates[tid] );
}

unsigned char* cu_initCurand( int width, int height ){
	cudaError_t error = cudaSuccess;

	dim3 block = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 grid = dim3((width+block.x-1)/block.x, (height+block.y-1)/block.y);
	
	// init curand
	curandState *rngStates = NULL;
	cudaError_t cudaResult = cudaMalloc((void **)&rngStates, grid.x *
		block.x * /*grid.y * block.y **/ sizeof(curandState));
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	unsigned int seed = 1234;

	for( int blkXIdx=0; blkXIdx<grid.x; blkXIdx++ ){
		cuke_initRNG<<< dim3(1,grid.y), block >>>( rngStates, seed, blkXIdx );
	}

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	return (unsigned char*)rngStates;
}

void cu_cleanCurand( unsigned char* p_rngStates ){
	// cleanup
	if( p_rngStates ){
		cudaFree( p_rngStates );
	}
}