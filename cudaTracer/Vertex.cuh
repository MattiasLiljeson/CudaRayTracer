#ifndef VERTEX_H
#define VERTEX_H

#include "Vec.cuh"

struct Vertex {
    enum { X, Y, Z };
    enum { U, V };
    Vec3f normal;
    Vec3f position;
    Vec2f texCoord;

    __host__ __device__ Vertex() {
        // Set default values which are easily identifiable
        position = Vec3f(1.23f,2.34f,3.45f);
        normal = Vec3f(4.56f,5.67f,6.78f);
        texCoord = Vec2f(7.89f,8.90f);
    }
    __host__ __device__ Vertex(float p_posX, float p_posY, float p_posZ,
                              float p_normX,
           float p_normY, float p_normZ, float p_texU, float p_texV) {
        position= Vec3f(p_posX,p_posY,p_posZ);
        normal = Vec3f(p_normX, p_normY, p_normZ);
        texCoord = Vec2f(p_texU, p_texV);
    }
    __host__ __device__ Vertex(Vec3f pos, Vec2f st) {
        position = pos;
        normal = Vec3f(0.0f, 1.0f, 0.0f);
        texCoord = st;
    }
    __host__ __device__ Vertex(Vec3f pos, Vec3f norm, Vec2f st) {
        position = pos;
        normal = norm;
        texCoord = st;
    }
};

#endif