//#include <curand_kernel.h>
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "Vec.cuh"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include "Scene.cuh"
#include "cuda.h"
#include "cudaUtils.h"
#include "tracer.cuh"

enum colors { RED, GREEN, BLUE, ALPHA };

__device__ __constant__ size_t C_PITCH;

__device__ Options g_options;
__device__ Scene g_scene;

#define INF 2e10f

///////////////scratchApixel

__device__ float clamp(const float &lo, const float &hi, const float &v) {
    return fmax(lo, fmin(hi, v));
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the mateural refractive index
//
// \param[out] kr is the amount of light reflected
// [/comment]
__device__ void fresnel(const Vec3f &I, const Vec3f &N, const float &ior,
                        float &kr) {
    float cosi = clamp(-1, 1, I.dot(N));
    float etai = 1, etat = ior;
    if (cosi > 0) {
        float tmp = etat;
        etat = etai;
        etai = tmp;
    }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(fmax(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        kr = 1;
    } else {
        float cost = sqrtf(fmax(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs =
            ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp =
            ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given
    // by: kt = 1 - kr;
}

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I
//
// If the ray is inside, you need to invert the refractive indices and negate
// the normal N
// [/comment]
__device__ Vec3f refract(const Vec3f &I, const Vec3f &N, const float &ior) {
    float cosi = clamp(-1, 1, I.dot(N));
    float etai = 1, etat = ior;
    Vec3f n = N;
    if (cosi < 0) {
        cosi = -cosi;
    } else {
        float tmp = etai;
        etai = etat;
        etat = tmp;
        n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

bool Trace::trace(const Vec3f &orig, const Vec3f &dir, float &tNear,
                  uint32_t &index, Vec2f &uv, const Sphere **hitObject) {
    *hitObject = nullptr;
    for (uint32_t k = 0; k < g_scene.sphereCnt; ++k) {
        float tNearK = INF;
        uint32_t indexK;
        Vec2f uvK;
        if (g_scene.spheres[k].intersect(orig, dir, tNearK, indexK, uvK) &&
            tNearK < tNear) {
            *hitObject = &g_scene.spheres[k];
            tNear = tNearK;
            index = indexK;
            uv = uvK;
        }
    }

    return (*hitObject != nullptr);
}

Trace::Trace(unsigned char *surface) : surface(surface) {}

Vec3f Trace::reflectionAndRefraction(const Vec3f &dir, uint32_t &index,
                                     Vec2f &uv, Vec2f &st,
                                     const Sphere *hitObject,
                                     const Vec3f &hitPoint, const Vec3f &N,
                                     const int depth) {
    Vec3f hitColor = g_options.backgroundColor;
    Vec3f reflectionDirection = N.reflect(dir).normalized();
    Vec3f refractionDirection =
        refract(dir, N, hitObject->object.ior).normalized();
    Vec3f reflectionRayOrig = (reflectionDirection.dot(N) < 0)
                                  ? hitPoint - N * g_options.bias
                                  : hitPoint + N * g_options.bias;
    Vec3f refractionRayOrig = (refractionDirection.dot(N) < 0)
                                  ? hitPoint - N * g_options.bias
                                  : hitPoint + N * g_options.bias;
    Vec3f reflectionColor =
        castRay(reflectionRayOrig, reflectionDirection, depth + 1);
    Vec3f refractionColor =
        castRay(refractionRayOrig, refractionDirection, depth + 1);
    float kr;
    fresnel(dir, N, hitObject->object.ior, kr);
    hitColor = reflectionColor * kr + refractionColor * (1 - kr);
    return hitColor;
}

Vec3f Trace::reflection(const Vec3f &dir, uint32_t &index, Vec2f &uv, Vec2f &st,
                        const Sphere *hitObject, const Vec3f &hitPoint,
                        const Vec3f &N, const int depth) {
    Vec3f hitColor = g_options.backgroundColor;
    float kr = 0.5f;
    // fresnel(dir, N, hitObject->object.ior, kr);
    Vec3f reflectionDirection = dir.reflect(N);
    Vec3f reflectionRayOrig = (reflectionDirection.dot(N) < 0)
                                  ? hitPoint + N * g_options.bias
                                  : hitPoint - N * g_options.bias;
    hitColor = castRay(reflectionRayOrig.normalized(),
                       reflectionDirection.normalized(), depth + 1) *
               kr;
    return hitColor;
}

Vec3f Trace::diffuseAndGlossy(const Vec3f &dir, uint32_t &index, Vec2f &uv,
                              Vec2f &st, const Sphere *hitObject,
                              const Vec3f &hitPoint, const Vec3f &N,
                              const int depth) {
    Vec3f hitColor = g_options.backgroundColor;
    // [comment]
    // We use the Phong illumation model int the default case. The
    // phong model is composed of a diffuse and a specular
    // reflection component.
    // [/comment]
    Vec3f lightAmt = Vec<float, 3>(0.0f, 0.0f, 0.0f);
    Vec3f specularColor = Vec<float, 3>(0.f, 0.0f, 0.0f);
    Vec3f shadowPointOrig =
        (dir.dot(N) < 0) ? N * g_options.bias : hitPoint - N * g_options.bias;
    // [comment]
    // Loop over all lights in the scene and sum their contribution
    // up We also apply the lambert cosine law here though we
    // haven't explained yet what this means.
    // [/comment]
    for (uint32_t i = 0; i < g_scene.lightCnt; ++i) {
        Vec3f lightDir = g_scene.lights[i].position - hitPoint;
        // square of the distance between hitPoint and the light
        float lightDistance2 = lightDir.dot(lightDir);
        lightDir = lightDir.normalized();
        float LdotN = fmaxf(0.f, lightDir.dot(N));
        Sphere *shadowHitObject = nullptr;
        float tNearShadow = INF;
        // is the point in shadow, and is the nearest occluding
        // object closer to the object than the light itself?
        bool inShadow = trace(shadowPointOrig, lightDir, tNearShadow, index, uv,
                              &shadowHitObject) &&
                        tNearShadow * tNearShadow < lightDistance2;
        lightAmt += g_scene.lights[i].intensity * LdotN * (1 - inShadow);
        Vec3f reflectionDirection = (-lightDir).reflect(N);
        float dotp = fmaxf(0.f, -reflectionDirection.dot(dir));
        specularColor += powf(dotp, hitObject->object.specularExponent) *
                         g_scene.lights[i].intensity;
    }
    hitColor = lightAmt * hitObject->object.evalDiffuseColor(st) *
               hitObject->object.Kd;
    hitColor += specularColor * hitObject->object.Ks;
    return hitColor;
}

Vec3f Trace::castRay(const Vec3f &orig, const Vec3f &dir, uint32_t depth) {
    if (depth > g_options.maxDepth) {
        return g_options.backgroundColor;
    }
    Vec3f hitColor = g_options.backgroundColor;
    float tnear = INF;
    Vec2f uv;
    uint32_t index = 0;
    Sphere *hitObject = nullptr;
    if (trace(orig, dir, tnear, index, uv, &hitObject)) {
        Vec3f hitPoint = orig + dir * tnear;
        Vec3f N;   // normal
        Vec2f st;  // st coordinates
        hitObject->getSurfaceProperties(hitPoint, dir, index, uv, N, st);
        switch (hitObject->object.materialType) {
            case Object::REFLECTION_AND_REFRACTION:
                return reflectionAndRefraction(dir, index, uv, st, hitObject,
                                               hitPoint, N, depth);
            case Object::REFLECTION:
                return reflection(dir, index, uv, st, hitObject, hitPoint, N,
                                  depth);
            case Object::DIFFUSE_AND_GLOSSY:
            default:
                return diffuseAndGlossy(dir, index, uv, st, hitObject, hitPoint,
                                        N, depth);
        }
    }

    return hitColor;
}

__global__ void kernel(unsigned char *surface) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= g_options.width || y >= g_options.height) return;

    // generate primary ray direction
    float ndc_x = (2.0f * (x + 0.5f) / (float)g_options.width - 1.0f) *
                  g_options.imageAspectRatio * g_options.scale;
    float ndc_y =
        (1.0f - 2.0f * (y + 0.5f) / (float)g_options.height) * g_options.scale;
    Vec3f dir = Vec3f(ndc_x, ndc_y, 1).normalized();
    dir = g_scene.camera.multVec(dir);
    dir = dir.normalized();
    Trace trace(surface);
    Vec3f result = trace.castRay(g_scene.orig, dir, 0);
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

void cudamain(const Options& options, const Scene &scene, const void *surface,
              size_t pitch) {
    gpuErrchk(cudaMemcpyToSymbol(C_PITCH, &pitch, sizeof(size_t)));

    gpuErrchk(cudaMemcpyToSymbol(g_options, &options, sizeof(Options)));
    gpuErrchk(cudaMemcpyToSymbol(g_scene, &scene, sizeof(Scene)));
    gpuErrchk(cudaPeekAtLastError());

    dim3 threads =
        dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 grids = dim3((options.width + threads.x - 1) / threads.x,
                      (options.height + threads.y - 1) / threads.y);

    // fprintf(stderr, "width: %d, height: %d, threads: %d, %d grids: %d, %d\n",
    // width, height, threads.x, threads.y,
    //        grids.x, grids.y);
    kernel<<<grids, threads>>>((unsigned char *)surface);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}