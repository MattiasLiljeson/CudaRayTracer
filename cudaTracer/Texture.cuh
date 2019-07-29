#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <cuda_runtime.h>

struct Texture {
    int width;
    int height;
    unsigned char *data;

    __host__ __device__ Texture() {
        width = -1;
        height = -1;
        data = NULL;
    }

    __host__ __device__ Texture(unsigned width, unsigned height,
                                unsigned char *data) {
        this->data = data;
        this->width = width;
        this->height = height;
    }

    __device__ Vec3f sample(float u, float v) const {
        // nearest neighbor for now. TODO: support bilinear at least
        int x = (int)round(u * width) % width;
        int y = (int)round(v * height) % height;
        float r = data[y * width * 4 + x * 4] / 255.0f;
        float g = data[y * width * 4 + x * 4 + 1] / 255.0f;
        float b = data[y * width * 4 + x * 4 + 2] / 255.0f;
        return Vec3f(r, g, b);
        // return Vec3f(x/512.0f, , 1.0f);
    }
};

#endif  // !TEXTURE_CUH
