#ifndef SHAPE_CUH
#define SHAPE_CUH

#include "Mesh.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"

struct Shape {
    enum ShapeKind { NOT_SET = -1, SPHERE, MESH };
    ShapeKind kind;
    union {
        Sphere sphere;
        Mesh mesh;
    };
    Material material;
    BoundingBox bb;

    __device__ Shape() { kind = NOT_SET; }

    __host__ Shape(const Shape &s) {
        kind = s.kind;
        if (kind == SPHERE) {
            sphere = s.sphere;
        } else if (kind == MESH) {
            mesh = s.mesh;
        }
        material = s.material;
        bb = s.bb;
    }

    __host__ Shape(Sphere sphere) {
        kind = SPHERE;
        this->sphere = sphere;
        Vec3f min = sphere.center - sphere.radius;
        Vec3f max = sphere.center + sphere.radius;
        this->bb = BoundingBox(min, max);
    }

    __host__ Shape(Mesh mesh, BoundingBox bb)
        : kind(MESH), mesh(mesh), bb(bb) {}

    __device__ Vec2f getStCoords(const Triangle &triangle,
                                 const Vec2f &uv) const {
        if (kind == Shape::SPHERE) {
            // Could use some standard sphere mapping instead...
            return Vec2f(1.0f, 1.0f);
        } else if (kind == Shape::MESH) {
            return mesh.getStCoords(triangle, uv);
        }
        // Famous words, should never happen :D
        return Vec2f(1.0f, 1.0f);
    }

    __device__ Vec3f getNormal(const Vec3f &P, const Triangle &triangle,
                               const Vec2f &uv, const Vec2f &st) const {
        if (kind == Shape::SPHERE) {
            return sphere.getNormal(P);
        } else if (kind == Shape::MESH) {
            return mesh.getNormal(triangle, uv, st);
        }
    }

    __device__ float getSpecularMask(const Vec2f &st) const {
        if (kind == Shape::SPHERE) {
            return material.Ks;
        } else if (kind == Shape::MESH) {
            return mesh.getSpecularMask(st);
        }
    }

    __device__ const Material *getObject() const { return &material; }

    __device__ Vec3f evalDiffuseColor(const Vec2f &st) const {
        if (kind == Shape::SPHERE) {
            return material.evalDiffuseColor(st);
        } else if (kind == Shape::MESH) {
            return mesh.evalDiffuseColor(st);
        }
    }

    __device__ bool intersect(Ray &ray, SurfaceData &data) const {
        if (kind == Shape::SPHERE) {
            return sphere.intersect(ray);
        } else if (kind == Shape::MESH) {
            return mesh.intersect(ray, data);
        }
    }
};

#endif  // !SHAPE_CUH
