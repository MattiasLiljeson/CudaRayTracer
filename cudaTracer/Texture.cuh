#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <cuda_runtime.h>

struct Texture {
    int width;
    int height;
    unsigned char *data;
    enum AddressingMode { WRAP, CLAMP };
    AddressingMode mode;

    __host__ __device__ Texture() {
        width = -1;
        height = -1;
        data = NULL;
    }

    __host__ __device__ Texture(unsigned width, unsigned height,
                                unsigned char *data, AddressingMode mode)
        : data(data), width(width), height(height), mode(mode) {}

    __device__ Vec3f sample(const Vec2f &uv) const {
        return getBilinearFilteredPixelColor(uv[0], uv[1]);
    }

    __device__ Vec3f nearestNeightbour(float u, float v) const {
        int x = (int)round(u * width) % width;
        int y = (int)round(v * height) % height;
        return getCol(x, y);
    }

    __device__ Vec3f getBilinearFilteredPixelColor(float u, float v) const {
        u = u * width - 0.5;
        v = v * height - 0.5;
        int x = floor(u);
        int y = floor(v);
        double u_ratio = u - x;
        double v_ratio = v - y;
        double u_opposite = 1 - u_ratio;
        double v_opposite = 1 - v_ratio;
        Vec3f result =
            (getCol(x, y) * u_opposite + getCol(x + 1, y) * u_ratio) *
                v_opposite +
            (getCol(x, y + 1) * u_opposite + getCol(x + 1, y + 1) * u_ratio) *
                v_ratio;
        return result;
    }

    __device__ Vec3f getCol(int x, int y) const {
        // clamp-mode. Could be set to e.g. wrap or mirror
        switch (mode) {
            case CLAMP:
                x = x < 0 ? 0 : x;
                x = x > width - 1 ? width - 1 : x;
                y = y < 0 ? 0 : y;
                y = y > height - 1 ? height - 1 : y;
                break;
            case WRAP:
                x = x % width;
                y = y % height;
            default:
                break;
        }
        float r = data[y * width * 4 + x * 4] / 255.0f;
        float g = data[y * width * 4 + x * 4 + 1] / 255.0f;
        float b = data[y * width * 4 + x * 4 + 2] / 255.0f;
        return Vec3f(r, g, b);
    }
};

#endif  // !TEXTURE_CUH
