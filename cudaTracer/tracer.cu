#include <cuda_runtime.h>
#include "BoundingBox.cuh"
#include "Light.cuh"
#include "Mat.cuh"
#include "Options.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Tracer.cuh"
#include "Vec.cuh"
#include "cuda.h"
#include "cudaUtils.h"

Tracer::Tracer(unsigned char *surface) : surface(surface) {}

Vec3f Tracer::castRay(Ray &ray, int depth) {
    if (depth > g_options.maxDepth) {
        return g_options.backgroundColor;
    }
    Vec3f hitColor = g_options.backgroundColor;
    float tnear = INF;
    SurfaceData sd;
    if (trace(ray, sd)) {
        sd.hitPoint = ray();
        sd.st = sd.hit->getStCoords(sd.triangle, sd.uv);
        sd.n = sd.hit->getNormal(sd.hitPoint, sd.triangle, sd.uv, sd.st);

        Material::Type materialType = sd.hit->getObject()->materialType;
        switch (materialType) {
            case Material::REFLECTION_AND_REFRACTION:
                return reflectionAndRefraction(ray.dir, sd, depth);
            case Material::REFLECTION:
                return reflection(ray.dir, sd, depth);
            case Material::DIFFUSE_AND_GLOSSY:
                return diffuseAndGlossy(ray.dir, sd, depth);
            default:
                return Vec3f(1.0f, 0.5f, 0.25f);
        }
    }
    return hitColor;
}

bool Tracer::trace(Ray &ray, SurfaceData &data) {
    data.hit = nullptr;
    for (int k = 0; k < g_scene.shapeCnt; ++k) {
        bool intersected = g_scene.shapes[k].intersect(ray, data);
        if (intersected) {
            data.hit = &g_scene.shapes[k];
        }
    }
    return data.hit != nullptr;
}

__device__ float clamp(const float &lo, const float &hi, const float &v) {
    return fmax(lo, fmin(hi, v));
}

/**
 * Fresnel computation
 * \param i incident view direction
 * \param n normal at the hit point
 * \param ior refractactive index
 * \return the amount of reflected light
 *
 *  Due to the conservation of energy, transmittance is 1 - kr;
 */
__device__ float fresnel(const Vec3f &I, const Vec3f &N, const float &ior) {
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
        return 1;
    } else {
        float cost = sqrtf(fmax(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs =
            ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp =
            ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rs * Rs + Rp * Rp) * 0.5f;
    }
}

template <typename T>
__device__ void swap(T &a, T &b) {
    T c(a);
    a = b;
    b = c;
}

/**
 * Refraction using Snell's law
 * Two cases: inside or outside object
 */
__device__ Vec3f refract(const Vec3f &I, const Vec3f &N, const float &ior) {
    float cosi = clamp(-1, 1, I.dot(N));
    float etai = 1, etat = ior;
    Vec3f n = N;
    if (cosi < 0) {  // outside
        cosi = -cosi;
    } else {  // inside
        swap(etai, etat);
        n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

Vec3f Tracer::reflectionAndRefraction(const Vec3f &dir, SurfaceData &data,
                                      const int depth) {
    Vec3f reflectionDirection = data.n.reflect(dir).normalized();
    Vec3f reflectionRayOrig = invBiasedOrigin(reflectionDirection, data);
    Ray reflectionRay(reflectionRayOrig, reflectionDirection);
    Vec3f reflectionColor = castRay(reflectionRay, depth + 1);

    const float ior = data.hit->getObject()->ior;
    Vec3f refractionDirection = refract(dir, data.n, ior).normalized();
    Vec3f refractionRayOrig = invBiasedOrigin(refractionDirection, data);
    Ray refractionRay(refractionRayOrig, refractionDirection);
    Vec3f refractionColor = castRay(refractionRay, depth + 1);

    float kr = fresnel(dir, data.n, ior);
    return reflectionColor * kr + refractionColor * (1 - kr);
}

Vec3f Tracer::reflection(const Vec3f &dir, SurfaceData &data, const int depth) {
    Vec3f reflectionDirection = dir.reflect(data.n).normalized();
    Vec3f reflectionRayOrig = invBiasedOrigin(reflectionDirection, data);
    Ray reflectionRay(reflectionRayOrig, reflectionDirection);
    Vec3f reflectionColor = castRay(reflectionRay, depth + 1);

    float kr = fresnel(dir, data.n, data.hit->material.ior);
    return reflectionColor * kr;
}


__device__ float blinnPhong(const Vec3f &dir, const SurfaceData &surface,
                            const Vec3f &lightDir, const float &shininess) {
    Vec3f halfDir = (lightDir + dir).normalized();
    float specAngle = fmax(halfDir.dot(surface.n), 0.0f);
    return pow(specAngle, shininess);
}

__device__ float phong(const Vec3f &dir, const SurfaceData &surface,
                       const Vec3f &lightDir, const float &shininess) {
    Vec3f reflectDir = -lightDir.reflect(surface.n);
    float specAngle = fmax(reflectDir.dot(dir), 0.0f);
    // note that the exponent is different here
    return powf(specAngle, shininess / 4.0f);
}

__device__ Vec3f shade(const Vec3f &dir, const SurfaceData &surface,
                       const Light &light, const Vec3f &diffuseColor,
                       const Vec3f &specColor, const Vec3f &lightDir,
                       const float &distance) {
     float lambertian = fmax(lightDir.dot(surface.n), 0.0f);
     float specular = 0.0f;

     if (lambertian > 0.0f) {
        specular =
            blinnPhong(dir, surface, lightDir, 1.0f /*surface.shininess*/);
    }
     Vec3f lightColor = light.intensity;
     Vec3f diffuseComponent =
        diffuseColor * lambertian * lightColor * light.lightPower / distance;
     Vec3f specComponent =
        specColor * specular * lightColor * light.lightPower / distance;
     return diffuseComponent + specComponent;
}

/**
 * Calcualte diffuse and glossy using Phong
 */
Vec3f Tracer::diffuseAndGlossy(const Vec3f &dir, SurfaceData &data,
                               const int depth) {
    const Material *object = data.hit->getObject();
    Vec3f diffuse = data.hit->evalDiffuseColor(data.st);
    Vec3f final = Vec3f(0.0f, 0.0f, 0.0f);
    Vec3f shadowPointOrig = biasedOrigin(dir, data);
    for (int i = 0; i < g_scene.lightCnt; ++i) {
        Vec3f lightDir = g_scene.lights[i].position - data.hitPoint;
        float lightDistance = lightDir.magnitude();
        lightDir = lightDir.normalized();
        float lambert = fmaxf(0.f, lightDir.dot(data.n));

        Ray shadowRay(shadowPointOrig, lightDir, lightDistance);
        bool visible = !trace(shadowRay, SurfaceData());

        if (visible) {
            final += shade(dir, data, g_scene.lights[i], diffuse, diffuse,
                           lightDir, lightDistance);
        }





        
    //// apply gamma correction (assume ambientColor, diffuseColor and
        //// specColor have been linearized, i.e. have no gamma correction in
        //// them)
        // const float screenGamma = 2.2;
        // Vec3f colorGammaCorrected;
        // colorGammaCorrected[X] = powf(colorLinear[X], 1.0f / screenGamma);
        // colorGammaCorrected[Y] = powf(colorLinear[Y], 1.0f / screenGamma);
        // colorGammaCorrected[Z] = powf(colorLinear[Z], 1.0f / screenGamma);
        //// use the gamma corrected color in the fragment
        // return colorGammaCorrected;







        // float lambert = fmaxf(0.f, lightDir.dot(data.n));

        // const Vec3f intensity = g_scene.lights[i].intensity;
        // diffuse += intensity * lambert * visible;


    }

    return final;
}

Vec3f Tracer::biasedOrigin(const Vec3f &direction, const SurfaceData &data) {
    if (direction.dot(data.n) < 0.0f) {
        return data.hitPoint + data.n * g_options.shadowBias;
    } else {
        return data.hitPoint - data.n * g_options.shadowBias;
    }
}
Vec3f Tracer::invBiasedOrigin(const Vec3f &direction, const SurfaceData &data) {
    if (direction.dot(data.n) < 0.0f) {
        return data.hitPoint - data.n * g_options.shadowBias;
    } else {
        return data.hitPoint + data.n * g_options.shadowBias;
    }
}