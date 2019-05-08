#ifndef stuffr
#define stuffr

#include "MaterialType.cuh"
#include "tracer.cuh"

template <class Shape>
class DiffuseAndGlossy : public MaterialType<DiffuseAndGlossy<Shape>, Shape> {
    Vec3f eval(const Trace<Shape> *traceInst, const Vec3f &dir, uint32_t &index,
               Vec2f &uv, Vec2f &st, const Shape *hitObject,
               const Vec3f &hitPoint, const Vec3f &N, const int depth) {
        Vec3f hitColor = traceInst->options->backgroundColor;
        // [comment]
        // We use the Phong illumation model int the default case. The
        // phong model is composed of a diffuse and a specular
        // reflection component.
        // [/comment]
        Vec3f lightAmt = Vec<float, 3>(0.0f, 0.0f, 0.0f);
        Vec3f specularColor = Vec<float, 3>(0.f, 0.0f, 0.0f);
        Vec3f shadowPointOrig = (dir.dot(N) < 0)
                                    ? N * traceInst->options->bias
                                    : hitPoint - N * traceInst->options->bias;
        // [comment]
        // Loop over all lights in the scene and sum their contribution
        // up We also apply the lambert cosine law here though we
        // haven't explained yet what this means.
        // [/comment]
        for (uint32_t i = 0; i < traceInst->lightCnt; ++i) {
            Vec3f lightDir = traceInst->lights[i].position - hitPoint;
            // square of the distance between hitPoint and the light
            float lightDistance2 = lightDir.dot(lightDir);
            lightDir = lightDir.normalized();
            float LdotN = fmaxf(0.f, lightDir.dot(N));
            const Shape *shadowHitObject = nullptr;
            float tNearShadow = INF;
            // is the point in shadow, and is the nearest occluding
            // object closer to the object than the light itself?
            bool inShadow =
                traceInst->trace(shadowPointOrig, lightDir, tNearShadow, index,
                                 uv, &shadowHitObject) &&
                tNearShadow * tNearShadow < lightDistance2;
            lightAmt += traceInst->lights[i].intensity * LdotN * (1 - inShadow);
            Vec3f reflectionDirection = (-lightDir).reflect(N);
            float dotp = fmaxf(0.f, -reflectionDirection.dot(dir));
            specularColor += powf(dotp, hitObject->specularExponent) *
                             traceInst->lights[i].intensity;
        }
        hitColor = lightAmt * hitObject->evalDiffuseColor(st) * hitObject->Kd;
        hitColor += specularColor * hitObject->Ks;
        return hitColor;
    }
};

template <class Shape>
class ReflectionAndRefraction
    : public MaterialType<ReflectionAndRefraction<Shape>, Shape> {
    Vec3f eval(const Trace<Shape> *traceInst, const Vec3f &dir, uint32_t &index,
               Vec2f &uv, Vec2f &st, const Shape *hitObject,
               const Vec3f &hitPoint, const Vec3f &N, const int depth) {
        Vec3f hitColor = traceInst->options->backgroundColor;
        Vec3f reflectionDirection = N.reflect(dir).normalized();
        Vec3f refractionDirection =
            refract(dir, N, hitObject->ior).normalized();
        Vec3f reflectionRayOrig = (reflectionDirection.dot(N) < 0)
                                      ? hitPoint - N * traceInst->options->bias
                                      : hitPoint + N * traceInst->options->bias;
        Vec3f refractionRayOrig = (refractionDirection.dot(N) < 0)
                                      ? hitPoint - N * traceInst->options->bias
                                      : hitPoint + N * traceInst->options->bias;
        Vec3f reflectionColor = traceInst->castRay(
            reflectionRayOrig, reflectionDirection, depth + 1);
        Vec3f refractionColor = traceInst->castRay(
            refractionRayOrig, refractionDirection, depth + 1);
        float kr;
        fresnel(dir, N, hitObject->ior, kr);
        hitColor = reflectionColor * kr + refractionColor * (1 - kr);
        return hitColor;
    }
};

template <class Shape>
class Reflection : public MaterialType<Reflection<Shape>, Shape> {
    Vec3f eval(const Trace<Shape> *traceInst, const Vec3f &dir, uint32_t &index,
               Vec2f &uv, Vec2f &st, const Shape *hitObject,
               const Vec3f &hitPoint, const Vec3f &N, const int depth) {
        Vec3f hitColor = traceInst->options->backgroundColor;
        float kr = 0.5f;
        // fresnel(dir, N, hitObject->ior, kr);
        Vec3f reflectionDirection = dir.reflect(N);
        Vec3f reflectionRayOrig = (reflectionDirection.dot(N) < 0)
                                      ? hitPoint + N * traceInst->options->bias
                                      : hitPoint - N * traceInst->options->bias;
        hitColor =
            traceInst->castRay(reflectionRayOrig.normalized(),
                               reflectionDirection.normalized(), depth + 1) *
            kr;
        return hitColor;
    }
};

#endif  //
