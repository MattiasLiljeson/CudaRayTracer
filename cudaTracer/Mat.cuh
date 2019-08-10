#ifndef MAT_CUH
#define MAT_CUH

#include "Vec.cuh"
#include "globals.h"

template <typename T>
class Mat44 {
   public:
    T x[4][4];

    __device__ __host__ constexpr Mat44() {}

    __device__ __host__ constexpr Mat44(T r1[4], T r2[4], T r3[4], T r4[4]) {
        x[0][0] = r1[0];
        x[0][1] = r1[1];
        x[0][2] = r1[2];
        x[0][3] = r1[3];
        x[1][0] = r2[0];
        x[1][1] = r2[1];
        x[1][2] = r2[2];
        x[1][3] = r2[3];
        x[2][0] = r3[0];
        x[2][1] = r3[1];
        x[2][2] = r3[2];
        x[2][3] = r3[3];
        x[3][0] = r4[0];
        x[3][1] = r4[1];
        x[3][2] = r4[2];
        x[3][3] = r4[3];
    }

    __device__ __host__ constexpr Mat44(T a, T b, T c, T d, T e, T f, T g, T h,
                                        T i, T j, T k, T l, T m, T n, T o,
                                        T p) {
        x[0][0] = a;
        x[0][1] = b;
        x[0][2] = c;
        x[0][3] = d;
        x[1][0] = e;
        x[1][1] = f;
        x[1][2] = g;
        x[1][3] = h;
        x[2][0] = i;
        x[2][1] = j;
        x[2][2] = k;
        x[2][3] = l;
        x[3][0] = m;
        x[3][1] = n;
        x[3][2] = o;
        x[3][3] = p;
    }

    __device__ __host__ const T *operator[](int i) const { return x[i]; }
    __device__ __host__ T *operator[](int i) { return x[i]; }

    // Multiply the current matrix with another matrix (rhs)
    __device__ __host__ Mat44 operator*(const Mat44 &v) const {
        Mat44 tmp;
        multiply(*this, v, tmp);
        return tmp;
    }

    __device__ __host__ friend bool operator==(const Mat44 &lhs,
                                               const Mat44 &rhs) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (lhs[i][j] != rhs[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    __device__ __host__ friend bool operator!=(const Mat44 &lhs,
                                               const Mat44 &rhs) {
        return !(lhs == rhs);
    }

    __device__ __host__ static void multiply(const Mat44<T> &a, const Mat44 &b,
                                             Mat44 &c) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] +
                          a[i][2] * b[2][j] + a[i][3] * b[3][j];
            }
        }
    }

    // \brief return a transposed copy of the current matrix as a new matrix
    __device__ __host__ Mat44 transposed() const {
        Mat44 t;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                t[i][j] = x[j][i];
            }
        }
        return t;
    }

    // \brief transpose itself
    __device__ __host__ Mat44 &transpose() {
        Mat44 tmp(x[0][0], x[1][0], x[2][0], x[3][0], x[0][1], x[1][1], x[2][1],
                  x[3][1], x[0][2], x[1][2], x[2][2], x[3][2], x[0][3], x[1][3],
                  x[2][3], x[3][3]);
        *this = tmp;
        return *this;
    }

    //[comment]
    // This method needs to be used for point-matrix multiplication. Keep in
    // mind we don't make the distinction between points and vectors at least
    // from a programming point of view, as both (as well as normals) are
    // declared as Vec3. However, mathematically they need to be treated
    // differently. Points can be translated when translation for vectors is
    // meaningless. Furthermore, points are implicitly be considered as having
    // homogeneous coordinates. Thus the w coordinates needs to be computed and
    // to convert the coordinates from homogeneous back to Cartesian
    // coordinates, we need to divided x, y z by w.
    //
    // The coordinate w is more often than not equals to 1, but it can be
    // different than 1 especially when the matrix is projective matrix
    // (perspective projection matrix).
    //[/comment]
    template <typename S>
    __device__ __host__ Vec<S, 3> multPoint(const Vec<S, 3> &src) const {
        S a, b, c, w;
        a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0];
        b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1];
        c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2];
        w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3];
        return Vec<S, 3>(a / w, b / w, c / w);
    }

    //[comment]
    // This method needs to be used for vector-matrix multiplication. Look at
    // the differences with the previous method (to compute a point-matrix
    // multiplication). We don't use the coefficients in the matrix that account
    // for translation (x[3][0], x[3][1], x[3][2]) and we don't compute w.
    //[/comment]
    template <typename S>
    __device__ __host__ Vec<S, 3> multVec(const Vec<S, 3> &src) const {
        S a, b, c;
        a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0];
        b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1];
        c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2];
        return Vec<S, 3>(a, b, c);
    }

    __device__ __host__ static Vec3f transformNormal(const Mat44 mat,
                                                     const Vec3f &normal) {
        Mat44 inv = mat.inversed();
        Mat44 transInv = inv.transposed();
        return transInv.multVecMatrix(normal);
    }

    //[comment]
    // Compute the inverse of the matrix using the Gauss-Jordan (or reduced row)
    // elimination method. We didn't explain in the lesson on Geometry how the
    // inverse of matrix can be found. Don't worry at this point if you don't
    // understand how this works. But we will need to be able to compute the
    // inverse of matrices in the first lessons of the "Foundation of 3D
    // Rendering" section, which is why we've added this code. For now, you can
    // just use it and rely on it for doing what it's supposed to do. If you
    // want to learn how this works though, check the lesson on called Matrix
    // Inverse in the "Mathematics and Physics of Computer Graphics" section.
    //[/comment]
    __device__ __host__ Mat44 inversed() const {
        int i, j, k;
        Mat44 s = identity();
        Mat44 t(*this);

        // Forward elimination
        for (i = 0; i < 3; i++) {
            int pivot = i;

            T pivotsize = t[i][i];

            if (pivotsize < 0) pivotsize = -pivotsize;

            for (j = i + 1; j < 4; j++) {
                T tmp = t[j][i];

                if (tmp < 0) tmp = -tmp;

                if (tmp > pivotsize) {
                    pivot = j;
                    pivotsize = tmp;
                }
            }

            if (pivotsize == 0) {
                // Cannot invert singular matrix
                return identity();
            }

            if (pivot != i) {
                for (j = 0; j < 4; j++) {
                    T tmp;

                    tmp = t[i][j];
                    t[i][j] = t[pivot][j];
                    t[pivot][j] = tmp;

                    tmp = s[i][j];
                    s[i][j] = s[pivot][j];
                    s[pivot][j] = tmp;
                }
            }

            for (j = i + 1; j < 4; j++) {
                T f = t[j][i] / t[i][i];

                for (k = 0; k < 4; k++) {
                    t[j][k] -= f * t[i][k];
                    s[j][k] -= f * s[i][k];
                }
            }
        }

        // Backward substitution
        for (i = 3; i >= 0; --i) {
            T f;

            if ((f = t[i][i]) == 0) {
                // Cannot invert singular matrix
                return identity();
            }

            for (j = 0; j < 4; j++) {
                t[i][j] /= f;
                s[i][j] /= f;
            }

            for (j = 0; j < i; j++) {
                f = t[j][i];

                for (k = 0; k < 4; k++) {
                    t[j][k] -= f * t[i][k];
                    s[j][k] -= f * s[i][k];
                }
            }
        }

        return s;
    }

    // \brief set current matrix to its inverse
    __device__ __host__ const Mat44<T> &invert() {
        *this = inversed();
        return *this;
    }

    /*__host__ friend std::ostream &operator<<(std::ostream &s,
                                                        const Mat44 &m) {
        std::ios_base::fmtflags oldFlags = s.flags();
        int width = 12;  // total with of the displayed number
        s.precision(5);  // control the number of displayed decimals
        s.setf(std::ios_base::fixed);

        s << "(" << std::setw(width) << m[0][0] << " " << std::setw(width)
          << m[0][1] << " " << std::setw(width) << m[0][2] << " "
          << std::setw(width) << m[0][3] << "\n"
          <<

            " " << std::setw(width) << m[1][0] << " " << std::setw(width)
          << m[1][1] << " " << std::setw(width) << m[1][2] << " "
          << std::setw(width) << m[1][3] << "\n"
          <<

            " " << std::setw(width) << m[2][0] << " " << std::setw(width)
          << m[2][1] << " " << std::setw(width) << m[2][2] << " "
          << std::setw(width) << m[2][3] << "\n"
          <<

            " " << std::setw(width) << m[3][0] << " " << std::setw(width)
          << m[3][1] << " " << std::setw(width) << m[3][2] << " "
          << std::setw(width) << m[3][3] << ")\n";

        s.flags(oldFlags);
        return s;
    }*/

    __device__ __host__ static Mat44 identity() {
        Mat44 x;
        x[0][0] = 1;
        x[0][1] = 0;
        x[0][2] = 0;
        x[0][3] = 0;
        x[1][0] = 0;
        x[1][1] = 1;
        x[1][2] = 0;
        x[1][3] = 0;
        x[2][0] = 0;
        x[2][1] = 0;
        x[2][2] = 1;
        x[2][3] = 0;
        x[3][0] = 0;
        x[3][1] = 0;
        x[3][2] = 0;
        x[3][3] = 1;
        return x;
    }

    __device__ __host__ static Mat44 rotation(const Vec3f &axis,
                                              const float &angle) {
        Mat44 m;
        float sinA = sin(ToRadian(angle));
        float cosA = cos(ToRadian(angle));
        float ux = axis[Vec3f::X];
        float uy = axis[Vec3f::Y];
        float uz = axis[Vec3f::Z];

        m[0][0] = cosA + ux * ux * (1.0f - cosA);
        m[0][1] = ux * uy * (1 - cosA) - uz * sinA;
        m[0][2] = ux * uz * (1 - cosA) + uy * sinA;
        m[0][3] = 0;

        m[1][0] = uy * ux * (1 - cosA) + uz * sinA;
        m[1][1] = cosA + uy * uy * (1 - cosA);
        m[1][2] = uy * uz * (1 - cosA) - ux * sinA;
        m[1][3] = 0;

        m[2][0] = uz * ux * (1 - cosA) - uy * sinA;
        m[2][1] = uz * uy * (1 - cosA) + ux * sinA;
        m[2][2] = cosA + uz * uz * (1 - cosA);
        m[2][3] = 0;

        m[3][0] = 0;
        m[3][1] = 0;
        m[3][2] = 0;
        m[3][3] = 1;

        return m;
    }

    __device__ __host__ static Mat44 rotationX(const float &angle) {
        float sinA = sin(ToRadian(angle));
        float cosA = cos(ToRadian(angle));
        return Mat44(1.0f, 0.0f, 0.0f, 0.0f,   //
                     0.0f, cosA, -sinA, 0.0f,  //
                     0.0f, sinA, cosA, 0.0f,   //
                     0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ __host__ static Mat44 rotationY(const float &angle) {
        float sinA = sin(ToRadian(angle));
        float cosA = cos(ToRadian(angle));
        return Mat44(cosA, 0.0f, sinA, 0.0f,   //
                     0.0f, 1.0f, 0.0f, 0.0f,   //
                     -sinA, 0.0f, cosA, 0.0f,  //
                     0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ __host__ static Mat44 rotationZ(const float &angle) {
        float sinA = sin(ToRadian(angle));
        float cosA = cos(ToRadian(angle));
        return Mat44(cosA, -sinA, 0.0f, 0.0f,  //
                     sinA, cosA, 0.0f, 0.0f,   //
                     0.0f, 0.0f, 1.0f, 0.0f,   //
                     0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ __host__ static Mat44 translate(const float x, const float y,
                                               const float z) {
        return Mat44(1.0f, 0.0f, 0.0f, 0.0f,  //
                     0.0f, 1.0f, 0.0f, 0.0f,  //
                     0.0f, 0.0f, 1.0f, 0.0f,  //
                     x, y, z, 1.0f);
    }

    __device__ __host__ static Mat44 scale(const float x, const float y,
                                           const float z) {
        return Mat44(x, 0.0f, 0.0f, 0.0f,  //
                     0.0f, y, 0.0f, 0.0f,  //
                     0.0f, 0.0f, z, 0.0f,  //
                     0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ __host__ static Mat44 scale(const float scale) {
        return Mat44(1.0f, 0.0f, 0.0f, 0.0f,  //
                     0.0f, 1.0f, 0.0f, 0.0f,  //
                     0.0f, 0.0f, 1.0f, 0.0f,  //
                     0.0f, 0.0f, 0.0f, 1.0f / scale);
    }
};

typedef Mat44<float> Mat44f;

#endif