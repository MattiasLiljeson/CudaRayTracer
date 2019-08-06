#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_normal.h>

#include "device_launch_parameters.h"

#include "Tracer.cuh"
#include "cudaUtils.h"

// Global variables
__device__ __constant__ size_t C_PITCH;
__device__ Options g_options;
__device__ Scene g_scene;

// Function declarations
__global__ void kernel(unsigned char *surface, curandState *const rngStates);
__device__ float randk(curandState *const localState);
__global__ void cuke_initRNG(curandState *const rngStates,
                             const unsigned int seed /*, int blkXIdx*/);

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

    float widthScale = 1 / (float)g_options.width;
    float heightScale = 1 / (float)g_options.height;

    curandState *localState =
        rngStates + threadIdx.y * blockDim.x + threadIdx.x;
    float ndcX = (2.0f * (x + 0.5f) * widthScale - 1.0f) *
                 g_options.imageAspectRatio * g_options.scale;
    float ndcY = (1.0f - 2.0f * (y + 0.5f) * heightScale) * g_options.scale;

    Vec3f pixCol(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < g_options.samples; ++i) {
        float xJitter = g_options.samples > 1 ? randk(localState) - 0.5 : 0.0f;
        float yJitter = g_options.samples > 1 ? randk(localState) - 0.5 : 0.0f;

        Vec3f dir =
            Vec3f(ndcX + xJitter * widthScale, ndcY + yJitter * heightScale, 1)
                .normalized();
        dir = g_scene.camera.multVec(dir);
        dir = dir.normalized();
        Tracer trace(surface);
        pixCol += trace.castRay(Ray(g_scene.orig, dir), 0);
    }

    for (int i = 0; i < ALPHA; ++i) {
        pixCol[i] /= g_options.samples;
    }

    float *pixel = (float *)(surface + y * C_PITCH) + 4 * x;
    if (g_options.gammaCorrection) {
        const float screenGamma = 2.2;
        const float sgInv = 1.0f / screenGamma;
        // get a pointer to the pixel at (x,y)
        pixel[RED] = powf(pixCol[RED], sgInv);
        pixel[GREEN] = powf(pixCol[GREEN], sgInv);
        pixel[BLUE] = powf(pixCol[BLUE], sgInv);
        pixel[ALPHA] = 1.0f;
    } else {
        pixel[RED] = pixCol[RED];
        pixel[GREEN] = pixCol[GREEN];
        pixel[BLUE] = pixCol[BLUE];
        pixel[ALPHA] = 1.0f;
    }

    // stop_time = clock();

    // float time = (stop_time - start_time);
    // pixel[RED] = time*0.000001f;
    // pixel[GREEN] = time * 0.0000001f;
    // pixel[BLUE] = time *  0.00000001f;
}

__device__ float randk(curandState *const localState) {
    return curand(localState) / INT32_MAX;
}

// RNG init kernel
unsigned char *cu_initCurand(int width, int height) {
    cudaError_t error = cudaSuccess;
    int threadCnt = 16;
    dim3 threads = dim3(threadCnt, threadCnt);
    dim3 grids = dim3((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    // init curand
    curandState *rngStates = NULL;
    cudaMalloc(&rngStates,
               grids.x * threads.x * grids.y * threads.y * sizeof(curandState));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    unsigned int seed = 1234;

    cuke_initRNG<<<grids, threads>>>(rngStates, seed);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return (unsigned char *)rngStates;
}

__global__ void cuke_initRNG(curandState *const rngStates,
                             const unsigned int seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    curandState *localState = rngStates + offset;
    curand_init(seed, 0, 0, localState);
}

void cu_cleanCurand(unsigned char *p_rngStates) {
    if (p_rngStates) {
        cudaFree(p_rngStates);
    }
}