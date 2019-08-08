#ifndef VEC
#define VEC

#include <cuda_runtime.h>

namespace vectorAxes {
enum EUCLIDEAN { X, Y, Z, W };
enum UV { U, V };
enum ST { S, T };
}  // namespace vectorAxes

template <typename T, int Size>
class Vec {
   public:
    //private:
    T data[Size];

   public:
    enum AXIS { X, Y, Z, W };

    template <typename... Arg>
    __host__ __device__ constexpr Vec(const Arg&... args) : data{args...} {}

    __host__ __device__ constexpr Vec(const T d[Size]) {
        for (int i = 0; i < Size; ++i) {
            data[i] = d[i];
        }
    }

    __host__ __device__ constexpr Vec(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] = other.data[i];
        }
    }

    __host__ __device__ constexpr Vec() {}

    __host__ __device__ Vec operator*(const float& r) const {
        Vec v(const_cast<Vec const&>(*this));
        for (T& d : v.data) {
            d *= r;
        }
        return v;
    }

    __host__ __device__ Vec operator*(const Vec& other) const {
        Vec v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] *= other.data[i];
        }
        return v;
    }

    __host__ __device__ Vec operator/(const float& r) const {
        Vec v(const_cast<Vec const&>(*this));
        for (T& d : v.data) {
            d /= r;
        }
        return v;
    }

    __host__ __device__ Vec operator/(const Vec& other) const {
        Vec v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] /= other.data[i];
        }
        return v;
    }

    __host__ __device__ Vec operator-(const float& r) const {
        Vec v(const_cast<Vec const&>(*this));
        for (T& d : v.data) {
            d -= r;
        }
        return v;
    }

    __host__ __device__ Vec operator-(const Vec& other) const {
        Vec v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] -= other.data[i];
        }
        return v;
    }

    __host__ __device__ Vec operator+(const float& r) const {
        Vec v(const_cast<Vec const&>(*this));
        for (T& d : v.data) {
            d += r;
        }
        return v;
    }

    __host__ __device__ Vec operator+(const Vec& other) const {
        Vec v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] += other.data[i];
        }
        return v;
    }

    __host__ __device__ Vec operator-() const {
        Vec v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = -data[i];
        }
        return v;
    }

    __host__ __device__ Vec& operator+=(const float& r) {
        for (int i = 0; i < Size; ++i) {
            data[i] += r;
        }
        return *this;
    }

    __host__ __device__ Vec& operator+=(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    __host__ __device__ Vec& operator-=(const float& r) {
        for (int i = 0; i < Size; ++i) {
            data[i] -= r;
        }
        return *this;
    }

    __host__ __device__ Vec& operator-=(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    __host__ __device__ Vec& operator*=(const float& r) {
        for (int i = 0; i < Size; ++i) {
            data[i] *= r;
        }
        return *this;
    }

    __host__ __device__ Vec& operator*=(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    __host__ __device__ Vec& operator/=(const float& r) {
        for (int i = 0; i < Size; ++i) {
            data[i] /= r;
        }
        return *this;
    }

    __host__ __device__ Vec& operator/=(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    __host__ __device__ T& operator[](int idx) { return data[idx]; }

    __host__ __device__ const T& operator[](int idx) const { return data[idx]; }

    __host__ __device__ friend Vec operator+(const float& r, const Vec& other) {
        Vec v(other);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = other.data[i] + r;
        }
        return v;
    }
    __host__ __device__ friend Vec operator-(const float& r, const Vec& other) {
        Vec v(other);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = other.data[i] - r;
        }
        return v;
    }
    __host__ __device__ friend Vec operator*(const float& r, const Vec& other) {
        Vec v(other);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = other.data[i] * r;
        }
        return v;
    }
    __host__ __device__ friend Vec operator/(const float& r, const Vec& other) {
        Vec v(other);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = other.data[i] / r;
        }
        return v;
    }

    __host__ __device__ friend bool operator==(const Vec& lhs, const Vec& rhs) {
        for (int i = 0; i < Size; ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ friend bool operator!=(const Vec& lhs, const Vec& rhs) {
        return !(lhs == rhs);
    }

    __host__ __device__ float magnitude() const {
        float mag2 = 0.0f;
        for (int i = 0; i < Size; ++i) {
            mag2 += data[i] * data[i];
        }
        return sqrtf(mag2);
    }

    __host__ __device__ Vec normalized() const {
        float mag2 = 0.0f;
        for (int i = 0; i < Size; ++i) {
            mag2 += data[i] * data[i];
        }
        if (mag2 > 0) {
            float invMag = 1 / sqrtf(mag2);
            return Vec(*this) * invMag;
        }
        return Vec(*this);
    }

    __host__ __device__ T dot(const Vec& other) const {
        T sum = 0;
        for (int i = 0; i < Size; ++i) {
            sum += data[i] * other.data[i];
        }
        return sum;
    }

    __host__ __device__ Vec cross(const Vec& other) const {
        assert(Size == 3);  // only valid for 3d space
        Vec v;
        v[X] = data[Y] * other[Z] - data[Z] * other[Y];
        v[Y] = data[Z] * other[X] - data[X] * other[Z];
        v[Z] = data[X] * other[Y] - data[Y] * other[X];
        return v;
    }

    __host__ __device__ Vec reflect(const Vec& N) const {
        return *this - N * 2 * dot(N);
    }

    /*
    __host__ __device__ static Vec<float, 3> vec3f(float x, float y, float z) {
        return Vec<float, 3>(x, y, z);
    }

    __host__ __device__ static Vec<float, 3> vec3f(float x) {
        return Vec<float, 3>(x, x, x);
    }

    __host__ __device__ static Vec<float, 3> vec3f() { return vec3f(0.0f); }*/
};

typedef Vec<float, 3> Vec3f;
typedef Vec<float, 2> Vec2f;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 2> Vec2i;

//__host__ __device__ Vec<float, 3> vec3f(float x, float y, float z);
//
//__host__ __device__ Vec<float, 3> vec3f(float x);
//
//__host__ __device__ Vec<float, 3> vec3f();

#endif  // VEC
