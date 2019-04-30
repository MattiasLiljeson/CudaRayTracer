#include <curand_kernel.h>
#include "Vec3f.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "Light.cuh"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include "cuda.h"
#include "cudaUtils.h"

enum colors { RED, GREEN, BLUE, ALPHA };

#define INF 2e10f

///////////////scratchApixel

__constant__ Vec3f C_ORIG;
__constant__ size_t C_PITCH;

__device__ bool trace(const Sphere *spheres, const int sphereCnt, const Vec3f &orig,
                      const Vec3f &dir, float &tNear,
                      uint32_t &index, Vec2f &uv, const Sphere **hitObject) {
    *hitObject = nullptr;
    for (uint32_t k = 0; k < sphereCnt; ++k) {
        float tNearK = INF;
        uint32_t indexK;
        Vec2f uvK;
        if (spheres[k].intersect(orig, dir, tNearK, indexK, uvK) &&
            tNearK < tNear) {
            *hitObject = &spheres[k];
            tNear = tNearK;
            index = indexK;
            uv = uvK;
        }
    }

    return (*hitObject != nullptr);
}

__device__ Vec3f castRay(const Options *options, const Light *lights,
                         const int lightCnt,
                         const Sphere *spheres,
                         const int sphereCnt, const Vec3f &orig,
                         const Vec3f &dir, uint32_t depth,
                         bool test = false) {
    if (depth > options->maxDepth) {
        return options->backgroundColor;
    }
    Vec3f hitColor = options->backgroundColor;
    float tnear = INF;
    Vec2f uv;
    uint32_t index = 0;
    const Sphere *hitObject = nullptr;
    if (trace(spheres, sphereCnt, orig, dir, tnear, index, uv, &hitObject)) {
        Vec3f tmp2 = hitObject->center;
        Vec3f hitPoint = orig + dir * tnear;
        Vec3f N;   // normal
        Vec2f st;  // st coordinates
        hitObject->getSurfaceProperties(hitPoint, dir, index, uv, N, st);
        Vec3f tmp = hitPoint;
        switch (hitObject->materialType) {
            // case REFLECTION_AND_REFRACTION: {
            //    Vec3f reflectionDirection = normalize(reflect(dir, N));
            //    Vec3f refractionDirection =
            //        normalize(refract(dir, N, hitObject->ior));
            //    Vec3f reflectionRayOrig =
            //        (dotProduct(reflectionDirection, N) < 0)
            //            ? hitPoint - N * options.bias
            //            : hitPoint + N * options.bias;
            //    Vec3f refractionRayOrig =
            //        (dotProduct(refractionDirection, N) < 0)
            //            ? hitPoint - N * options.bias
            //            : hitPoint + N * options.bias;
            //    Vec3f reflectionColor =
            //        castRay(reflectionRayOrig, reflectionDirection, objects,
            //                lights, options, depth + 1, 1);
            //    Vec3f refractionColor =
            //        castRay(refractionRayOrig, refractionDirection, objects,
            //                lights, options, depth + 1, 1);
            //    float kr;
            //    fresnel(dir, N, hitObject->ior, kr);
            //    hitColor = reflectionColor * kr + refractionColor * (1 - kr);
            //    break;
            //}
            // case REFLECTION: {
            //    float kr;
            //    fresnel(dir, N, hitObject->ior, kr);
            //    Vec3f reflectionDirection = reflect(dir, N);
            //    Vec3f reflectionRayOrig =
            //        (dotProduct(reflectionDirection, N) < 0)
            //            ? hitPoint + N * options.bias
            //            : hitPoint - N * options.bias;
            //    hitColor = castRay(reflectionRayOrig, reflectionDirection,
            //                       objects, lights, options, depth + 1) *
            //               kr;
            //    break;
            //}
            default: {
                // [comment]
                // We use the Phong illumation model int the default case. The
                // phong model is composed of a diffuse and a specular
                // reflection component.
                // [/comment]
                Vec3f lightAmt = Vec<float, 3>(0.0f, 0.0f, 0.0f);
                Vec3f specularColor = Vec<float, 3>(0.f, 0.0f, 0.0f);
                Vec3f shadowPointOrig = (dir.dotProduct(N) < 0)
                                            ? N * options->bias
                                            : hitPoint - N * options->bias;
                // [comment]
                // Loop over all lights in the scene and sum their contribution
                // up We also apply the lambert cosine law here though we
                // haven't explained yet what this means.
                // [/comment]
                for (uint32_t i = 0; i < lightCnt; ++i) {
                    Vec3f lightDir = lights[i].position - hitPoint;
                    // square of the distance between hitPoint and the light
                    float lightDistance2 =
                        lightDir.dotProduct(lightDir);
                    lightDir = lightDir.normalized();
                    float LdotN = fmaxf(0.f, lightDir.dotProduct(N));
                    Sphere *shadowHitObject = nullptr;
                    float tNearShadow = INF;
                    // is the point in shadow, and is the nearest occluding
                    // object closer to the object than the light itself?
                    bool inShadow =
                        trace(spheres, sphereCnt, shadowPointOrig, lightDir, tNearShadow, index, uv,
                              &shadowHitObject) &&
                        tNearShadow * tNearShadow < lightDistance2;
                    lightAmt += lights[i].intensity * LdotN * (1 - inShadow);
                    Vec3f reflectionDirection = (-lightDir).reflect(N);
                    float dotp = fmaxf(0.f, -reflectionDirection.dotProduct(dir));
					specularColor +=
                        powf(dotp,hitObject->specularExponent) *
                        lights[i].intensity; 
                
                }
                hitColor = lightAmt * hitObject->evalDiffuseColor(st) * hitObject->Kd;
                hitColor += specularColor * hitObject->Ks;
                break;
            }
        }
    }

    return hitColor;
}

__global__ void kernel(const Options *options, const Light *lights, const int lightCnt,
                       const Sphere *spheres, const int sphereCnt, unsigned char *surface) {
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
    Vec3f dir = Vec3f(ndc_x, ndc_y, -1).normalized();
    Vec3f result = castRay(options, lights, lightCnt, spheres, sphereCnt, C_ORIG, dir, 0);

    // get a pointer to the pixel at (x,y)
    float *pixel = (float *)(surface + y * C_PITCH) + 4 * x;

    pixel[RED] = result.data[Vec3f::X];
    pixel[GREEN] = result.data[Vec3f::Y];
    pixel[BLUE] = result.data[Vec3f::Z];
    pixel[ALPHA] = 1.0f;

	//pixel[RED] = ndc_x+1.0f;
    //pixel[GREEN] = ndc_y + 1.0f;
    //pixel[BLUE] = 1.0f;
    //pixel[ALPHA] = 1.0f;
}

void pre(const void *surface, int width, int height, size_t pitch) {}

void main(const Options *options, const Light *lights, const int lightCnt,
          const Sphere *spheres, const int sphereCnt, const void *surface,
          const int width, const int height, size_t pitch, const Vec3f& orig) {
    gpuErrchk(cudaMemcpyToSymbol(C_ORIG, &orig, sizeof(Vec3f)));
    gpuErrchk(cudaMemcpyToSymbol(C_PITCH, &pitch, sizeof(size_t)));
    gpuErrchk(cudaPeekAtLastError());

    dim3 threads = dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 grids = dim3((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    //fprintf(stderr, "width: %d, height: %d, threads: %d, %d grids: %d, %d\n", width, height, threads.x, threads.y,
    //        grids.x, grids.y);
    kernel<<<grids, threads>>>(options, lights, lightCnt, spheres, sphereCnt, (unsigned char *) surface);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

////////// cuda by example
/*
#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)

struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};
#define SPHERES 20

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *surface) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= C_WIDTH || y >= C_HEIGHT) return;

    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    




//surface[offset * 4 + 0] = (int)(r * 255);
//surface[offset * 4 + 1] = (int)(g * 255);
//surface[offset * 4 + 2] = (int)(b * 255);
//surface[offset * 4 + 3] = 255;


    // get a pointer to the pixel at (x,y)
    float *pixel = (float *)(surface + y * C_PITCH) + 4 * x;

    pixel[RED] = r;
    pixel[GREEN] = g;
    pixel[BLUE] = b;
    pixel[ALPHA] = 1.0f;
}

void main(const void *surface, int width, int height, size_t pitch,
          unsigned char *p_rngStates) {
    //
    //DataBlock data;
    //// capture the start time
    //cudaEvent_t start, stop;
    //HANDLE_ERROR(cudaEventCreate(&start));
    //HANDLE_ERROR(cudaEventCreate(&stop));
    //HANDLE_ERROR(cudaEventRecord(start, 0));
        //
    //CPUBitmap bitmap(DIM, DIM, &data);
    //unsigned char *dev_bitmap;
        //
    //// allocate memory on the GPU for the output bitmap
    //HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));
    //
    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }
    gpuErrchk(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    cudaMemcpyToSymbol(C_PITCH, &pitch, sizeof(size_t));
    cudaMemcpyToSymbol(C_WIDTH, &width, sizeof(int));
    cudaMemcpyToSymbol(C_HEIGHT, &height, sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    // generate a bitmap from our sphere data
    //dim3 grids(DIM / 16, DIM / 16);
    //dim3 threads(16, 16);
    dim3 threads =
        dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 grids = dim3((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    // Do the magic
    //cuke_clear<<<grids, threads>>>((unsigned char *)surface);
    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());
        





    kernel<<<grids, threads>>>((unsigned char *)surface);

//
//// copy our bitmap back from the GPU for display
//HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
//                        cudaMemcpyDeviceToHost));
//
//// get stop time, and display the timing results
//HANDLE_ERROR(cudaEventRecord(stop, 0));
//HANDLE_ERROR(cudaEventSynchronize(stop));
//float elapsedTime;
//HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
//printf("Time to generate:  %3.1f ms\n", elapsedTime);
//
//HANDLE_ERROR(cudaEventDestroy(start));
//HANDLE_ERROR(cudaEventDestroy(stop));
//
//HANDLE_ERROR(cudaFree(dev_bitmap));
//
//// display
//bitmap.display_and_exit();
//
}
*/
