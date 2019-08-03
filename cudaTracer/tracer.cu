#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "Vec.cuh"

#include <cuda_runtime.h>
#include "Scene.cuh"
#include "Tracer.cuh"
#include "cuda.h"
#include "cudaUtils.h"

Tracer::Tracer(unsigned char *surface) : surface(surface) {}

Vec3f Tracer::castRay(Ray &ray, uint32_t depth) {
    if (depth > g_options.maxDepth) {
        return g_options.backgroundColor;
    }
    Vec3f hitColor = g_options.backgroundColor;
    float tnear = INF;
    Vec2f uv;
    uint32_t index = 0;
    Shape *hitObject = nullptr;
    if (trace(ray, index, uv, &hitObject)) {
        Vec3f hitPoint = ray();
        Vec2f st = hitObject->getStCoords(index, uv);
        Vec3f N = hitObject->getNormal(hitPoint, index, uv, st);

        Object::MaterialType materialType =
            hitObject->getObject()->materialType;
        switch (materialType) {
            case Object::REFLECTION_AND_REFRACTION:
                return reflectionAndRefraction(ray.dir, index, uv, st,
                                               hitObject, hitPoint, N, depth);
            case Object::REFLECTION:
                return reflection(ray.dir, index, uv, st, hitObject, hitPoint,
                                  N, depth);
            case Object::DIFFUSE_AND_GLOSSY:
                return diffuseAndGlossy(ray.dir, index, uv, st, hitObject,
                                        hitPoint, N, depth);
            default:
                return Vec3f(1.0f, 0.5f, 0.25f);
        }
    }

    return hitColor;
}

bool Tracer::trace(Ray &ray, uint32_t &index, Vec2f &uv, const Shape **hit) {
    *hit = nullptr;
    for (int k = 0; k < g_scene.shapeCnt; ++k) {
        int indexK;
        Vec2f uvK;
        bool intersected =
            g_scene.shapes[k].intersect(ray, indexK, uvK);
        if (intersected) {
            *hit = &g_scene.shapes[k];
            index = indexK;
            uv = uvK;
        }
    }

    return (*hit != nullptr);
}

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

Vec3f Tracer::reflectionAndRefraction(const Vec3f &dir, uint32_t &index,
                                      Vec2f &uv, Vec2f &st,
                                      const Shape *hitObject,
                                      const Vec3f &hitPoint, const Vec3f &N,
                                      const int depth) {
    const Object *object = hitObject->getObject();
    Vec3f hitColor = g_options.backgroundColor;
    Vec3f reflectionDirection = N.reflect(dir).normalized();
    Vec3f refractionDirection = refract(dir, N, object->ior).normalized();
    Vec3f reflectionRayOrig = (reflectionDirection.dot(N) < 0)
                                  ? hitPoint - N * g_options.bias
                                  : hitPoint + N * g_options.bias;
    Vec3f refractionRayOrig = (refractionDirection.dot(N) < 0)
                                  ? hitPoint - N * g_options.bias
                                  : hitPoint + N * g_options.bias;
    Vec3f reflectionColor =
        castRay(Ray(reflectionRayOrig, reflectionDirection), depth + 1);
    Vec3f refractionColor =
        castRay(Ray(refractionRayOrig, refractionDirection), depth + 1);
    float kr;
    fresnel(dir, N, object->ior, kr);
    hitColor = reflectionColor * kr + refractionColor * (1 - kr);
    return hitColor;
}

Vec3f Tracer::reflection(const Vec3f &dir, uint32_t &index, Vec2f &uv,
                         Vec2f &st, const Shape *hitObject,
                         const Vec3f &hitPoint, const Vec3f &N,
                         const int depth) {
    Vec3f hitColor = g_options.backgroundColor;
    float kr = 0.5f;
    // fresnel(dir, N, hitObject->material.ior, kr);
    Vec3f reflectionDirection = dir.reflect(N);
    Vec3f reflectionRayOrig = (reflectionDirection.dot(N) < 0)
                                  ? hitPoint + N * g_options.bias
                                  : hitPoint - N * g_options.bias;
    hitColor = castRay(Ray(reflectionRayOrig.normalized(),
                           reflectionDirection.normalized()),
                       depth + 1) *
               kr;
    return hitColor;
}

Vec3f Tracer::diffuseAndGlossy(const Vec3f &dir, uint32_t &index, Vec2f &uv,
                               Vec2f &st, const Shape *hitObject,
                               const Vec3f &hitPoint, const Vec3f &N,
                               const int depth) {
    const Object *object = hitObject->getObject();

    Vec3f hitColor = g_options.backgroundColor;
    // [comment]
    // We use the Phong illumation model int the default case. The
    // phong model is composed of a diffuse and a specular
    // reflection component.
    // [/comment]
    Vec3f lightAmt = Vec3f(0.0f, 0.0f, 0.0f);
    Vec3f specularColor = Vec3f(0.f, 0.0f, 0.0f);
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
        Shape *shadowHitObject = nullptr;
        // is the point in shadow, and is the nearest occluding
        // object closer to the object than the light itself?
        Ray shadowRay(shadowPointOrig, lightDir);
        bool inShadow = trace(shadowRay, index, uv, &shadowHitObject) &&
                        shadowRay.tMax * shadowRay.tMax < lightDistance2;
        lightAmt += g_scene.lights[i].intensity * LdotN * (1 - inShadow);
        Vec3f reflectionDirection = (-lightDir).reflect(N);
        float dotp = fmaxf(0.f, -reflectionDirection.dot(dir));

        specularColor +=
            powf(dotp, object->specularExponent) * g_scene.lights[i].intensity;
    }

    //return Vec3f(st[Vec3f::X], st[Vec3f::Y], 255.0f);
    Vec3f diffuse = hitObject->evalDiffuseColor(st);
    hitColor = lightAmt * diffuse * object->Kd + specularColor * object->Ks;
    return hitColor;
}