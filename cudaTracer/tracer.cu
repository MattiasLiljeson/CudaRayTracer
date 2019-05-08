//#include <curand_kernel.h>
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "Vec.cuh"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include "cuda.h"
#include "cudaUtils.h"
#include "tracer.cuh"
#include "Math.cuh"

enum colors { RED, GREEN, BLUE, ALPHA };

__device__ __constant__ Vec3f C_ORIG;
__device__ __constant__ size_t C_PITCH;
__device__ __constant__ Mat44f C_CAMERA;    

template <class Shape>
__global__ void kernel(const Options *options, const Light *lights,
                       const int lightCnt, const Shape *spheres,
                       const int sphereCnt, unsigned char *surface) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= options->width || y >= options->height) return;

    // generate primary ray direction
    float ndc_x = (2.0f * (x + 0.5f) / (float)options->width - 1.0f) *
                  options->imageAspectRatio * options->scale;
    float ndc_y =
        (1.0f - 2.0f * (y + 0.5f) / (float)options->height) * options->scale;
    Vec3f dir = Vec3f(ndc_x, ndc_y, 1).normalized();
    dir = C_CAMERA.multVec(dir);
    dir = dir.normalized();
    // Vec3f result = castRay(options, lights, lightCnt, spheres, sphereCnt,
    // C_ORIG, dir, 0);
    Trace<Shape> trace(options, lights, lightCnt, spheres, sphereCnt,
                                 surface);
    Vec3f result = trace.castRay(C_ORIG, dir, 0);
    // get a pointer to the pixel at (x,y)
    float *pixel = (float *)(surface + y * C_PITCH) + 4 * x;

    pixel[RED] = result.data[Vec3f::X];
    pixel[GREEN] = result.data[Vec3f::Y];
    pixel[BLUE] = result.data[Vec3f::Z];
    pixel[ALPHA] = 1.0f;

    // pixel[RED]   = 0; //dir[Vec3f::X];
    // pixel[GREEN] = 0; //dir[Vec3f::Y];
    // pixel[BLUE] = -dir[Vec3f::Z];
    // pixel[ALPHA] = 1.0f;
}

template <typename T>
void cudamain(const Options *options, const Light *lights, const int lightCnt,
              const T *spheres, const int sphereCnt, const void *surface,
              const int width, const int height, size_t pitch,
              const Vec3f &orig, const Mat44f &camera) {
    gpuErrchk(cudaMemcpyToSymbol(C_ORIG, &orig, sizeof(Vec3f)));
    gpuErrchk(cudaMemcpyToSymbol(C_PITCH, &pitch, sizeof(size_t)));
    gpuErrchk(cudaMemcpyToSymbol(C_CAMERA, &camera.inversed(), sizeof(Mat44f)));
    gpuErrchk(cudaPeekAtLastError());

    dim3 threads =
        dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 grids = dim3((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    // fprintf(stderr, "width: %d, height: %d, threads: %d, %d grids: %d, %d\n",
    // width, height, threads.x, threads.y,
    //        grids.x, grids.y);
    kernel<<<grids, threads>>>(options, lights, lightCnt, spheres, sphereCnt,
                               (unsigned char *)surface);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

//__global__ void createSpheresKernel(Sphere* &out_spheres, const SphereData *data,
//                                    const int sphereCnt) {
//    int bytes = sizeof(Sphere) * sphereCnt;
//    out_spheres = (Sphere *)malloc(bytes);
//    for (int i = 0; i < sphereCnt; ++i) {
//        out_spheres[i] = Sphere(data[i]);
//    }
//}
//
//Sphere *createSpheres(const SphereData *sphereDatas, const MaterialProperties* properties, const int sphereCnt) {
//    dim3 threads = dim3(1, 1);  // block dimensions are fixed to be 256 threads
//    dim3 grids = dim3(1, 1, 1);
//    Sphere *spheres;
//    createSpheresKernel<<<grids, threads>>>(spheres, sphereDatas, sphereCnt);
//    return spheres;
//}