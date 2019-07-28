#ifndef SHAPE_CUH
#define SHAPE_CUH

#include "Mesh.cuh"
#include "Sphere.cuh"

struct Shape {
    enum ShapeKind { NOT_SET = -1, SPHERE, MESH };
    ShapeKind kind;
    union {
        Sphere sphere;
        Mesh mesh;
    };
    Object material;

    __device__ Shape() { kind = NOT_SET; }

    __host__ Shape(const Shape &s) {
        kind = s.kind;
        if (kind == SPHERE) {
            sphere = s.sphere;
        } else if (kind == MESH) {
            mesh = s.mesh;
        }
        material = s.material;
    }

    __host__ Shape(Sphere sphere) {
        kind = SPHERE;
        this->sphere = sphere;
    }

    __host__ Shape(Mesh mesh) {
        kind = MESH;
        this->mesh = mesh;
    }

    __device__ void getSurfaceProperties(const Vec3f &P, const Vec3f &I,
                                         const uint32_t &index, const Vec2f &uv,
                                         Vec3f &N, Vec2f &st) const {
        if (kind == Shape::SPHERE) {
            sphere.getSurfaceProperties(P, I, index, uv, N, st);
        } else if (kind == Shape::MESH) {
            mesh.getSurfaceProperties(P, I, index, uv, N, st);
        }
    }

    __device__ const Object *getObject() const { return &material; }

    __device__ Vec3f evalDiffuseColor(const Vec2f &vec) const {
        if (kind == Shape::SPHERE) {
            return material.evalDiffuseColor(vec);
        } else if (kind == Shape::MESH) {
            return mesh.evalDiffuseColor(vec);
        }
    }

    __device__ bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear,
                              int &index, Vec2f &uv) const {
        if (kind == Shape::SPHERE) {
            return sphere.intersect(orig, dir, tnear, index, uv);
        } else if (kind == Shape::MESH) {
            return mesh.intersect(orig, dir, tnear, index, uv);
        }
    }
};

#endif  // !SHAPE_CUH
