#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_normal.h>

#include "device_launch_parameters.h"

#include "cudaUtils.h"
#include "Tracer.cuh"

// Global variables
__device__ __constant__ size_t C_PITCH;
__device__ Options g_options;
__device__ Scene g_scene;

// Function declarations
__global__ void kernel(unsigned char *surface, curandState *const rngStates);
__device__ float randk(curandState *const localState);
__global__ void cuke_initRNG(curandState *const rngStates,
                             const unsigned int seed, int blkXIdx);

void cudamain(const Options &options, const Scene &scene, const void *surface,
              size_t pitch, int blockDim, unsigned char *rngStates) {
    gpuErrchk(cudaMemcpyToSymbol(C_PITCH, &pitch, sizeof(size_t)));

    gpuErrchk(cudaMemcpyToSymbol(g_options, &options, sizeof(Options)));
    gpuErrchk(cudaMemcpyToSymbol(g_scene, &scene, sizeof(Scene)));
    gpuErrchk(cudaPeekAtLastError());

    dim3 threads = dim3(blockDim, blockDim);
    dim3 grids = dim3((options.width + threads.x - 1) / threads.x,
                      (options.height + threads.y - 1) / threads.y);

    // fprintf(stderr, "width: %d, height: %d, threads: %d, %d grids: %d, %d\n",
    // width, height, threads.x, threads.y,
    //        grids.x, grids.y);
    kernel<<<grids, threads>>>((unsigned char *)surface,
                               (curandState *)rngStates);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void kernel(unsigned char *surface, curandState *const rngStates) {
    unsigned int start_time = 0, stop_time = 0;

    start_time = clock();

    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= g_options.width || y >= g_options.height) return;

    // get a pointer to the pixel at (x,y)
    float *pixel = (float *)(surface + y * C_PITCH) + 4 * x;
    pixel[RED] = 0.0f;
    pixel[GREEN] = 0.0f;
    pixel[BLUE] = 0.0f;
    pixel[ALPHA] = 1.0f;

    float widthScale = 1 / (float)g_options.width;
    float heightScale = 1 / (float)g_options.height;

    curandState *localState =
        rngStates + threadIdx.y * blockDim.x + threadIdx.x;
    float ndcX = (2.0f * (x + 0.5f) * widthScale - 1.0f) *
                 g_options.imageAspectRatio * g_options.scale;
    float ndcY = (1.0f - 2.0f * (y + 0.5f) * heightScale) * g_options.scale;

    for (int i = 0; i < g_options.samples; ++i) {
        float xJitter = g_options.samples > 1 ? randk(localState) - 0.5 : 0.0f;
        float yJitter = g_options.samples > 1 ? randk(localState) - 0.5 : 0.0f;

        Vec3f dir =
            Vec3f(ndcX + xJitter * widthScale, ndcY + yJitter * heightScale, 1)
                .normalized();
        dir = g_scene.camera.multVec(dir);
        dir = dir.normalized();
        Tracer trace(surface);
        Vec3f result = trace.castRay(g_scene.orig, dir, 0);

        pixel[RED] += result[Vec3f::X];
        pixel[GREEN] += result[Vec3f::Y];
        pixel[BLUE] += result[Vec3f::Z];
    }

    for (int i = 0; i < ALPHA; ++i) {
        pixel[i] /= g_options.samples;
    }

    // stop_time = clock();

    // float time = (stop_time - start_time);
    // pixel[RED] = time*0.000001f;
    // pixel[GREEN] = time * 0.0000001f;
    // pixel[BLUE] = time *  0.00000001f;
}

__device__ float randk(curandState *const localState) {
    return curand(localState) / INT32_MAX;
    // return 0.5f;
}

// RNG init kernel
unsigned char *cu_initCurand(int width, int height) {
    cudaError_t error = cudaSuccess;

    dim3 block = dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 grid =
        dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // init curand
    curandState *rngStates = NULL;
    cudaError_t cudaResult = cudaMalloc(
        (void **)&rngStates,
        grid.x * block.x * /*grid.y * block.y **/ sizeof(curandState));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    unsigned int seed = 1234;

    for (int blkXIdx = 0; blkXIdx < grid.x; blkXIdx++) {
        cuke_initRNG<<<dim3(1, grid.y), block>>>(rngStates, seed, blkXIdx);
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return (unsigned char *)rngStates;
}

__global__ void cuke_initRNG(curandState *const rngStates,
                             const unsigned int seed, int blkXIdx) {
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int x = blkXIdx * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    tid = x * gridDim.x + y;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

void cu_cleanCurand(unsigned char *p_rngStates) {
    // cleanup
    if (p_rngStates) {
        cudaFree(p_rngStates);
    }
}