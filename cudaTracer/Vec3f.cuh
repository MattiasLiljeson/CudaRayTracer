#ifndef VEC3F
#define VEC3F

#include <cuda_runtime.h>

template <typename T, int Size>
class Vec {
	public:
	enum AXIS { X, Y, Z, W };

    template <typename... Arg>
    __host__ __device__ constexpr Vec(const Arg &... args) : data{args...} {}

 	__device__ constexpr Vec(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] = other.data[i];
        }
	}

	__device__ Vec<T, Size> operator*(const float& r) const {
        Vec<T, Size> v(const_cast<Vec& const>(*this));
        for (T &d : v.data) {
            d *= r;
        }
        return v;
	}

	__device__ Vec operator*(const Vec<T, Size>& other) const {
        Vec<T, Size> v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] *= other.data[i];
        }
        return v;
	}

	__device__ Vec operator-(const Vec& other) const {
        Vec<T, Size> v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] -= other.data[i];
        }
        return v;
	}

	__device__ Vec operator+(const Vec& other) const {
        Vec<T, Size> v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] += other.data[i];
        }
        return v;
	}

	__device__ Vec operator-() const {
        Vec<T, Size> v(*this);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = -data[i];
        }
        return v;
	}

	__device__ Vec& operator+=(const Vec& other) {
        for (int i = 0; i < Size; ++i) {
            data[i] += other.data[i];
        }
        return *this;
	}
	__device__ friend Vec operator*(const float& r, const Vec& other){
        Vec<T, Size> v(other);
        for (int i = 0; i < Size; ++i) {
            v.data[i] = other.data[i] * r;
        }
        return v;
	}

	__device__ Vec normalized() const{
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

    __device__ float dotProduct(const Vec& other) const{
        float sum = 0.0f;
        for (int i = 0; i < Size; ++i) {
            sum += data[i] * other.data[i];
        }
        return sum;
    }

    __device__ Vec reflect(const Vec& N) const{
        return *this - N * 2 * dotProduct(N);
    }

	//private:
	T data[Size];

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

//__host__ __device__ Vec<float, 3> vec3f(float x, float y, float z);
//
//__host__ __device__ Vec<float, 3> vec3f(float x);
//
//__host__ __device__ Vec<float, 3> vec3f();

#endif  // !VEC3F
