#pragma once
#include <cmath>
#include <cstdint>
#include <memory.h>
#include <type_traits>

#if defined(__x86_64) || defined(__amd64)
#include <smmintrin.h>
#endif


namespace mgm {

    //=========
    // VECTORS
    //=========


    template<size_t S, class T>
    class vec {
        template<class ... Ts>
        void init(size_t& i, const T x, const Ts ... xs) {
            data[i] = x;
            init(++i, xs...);
        }
        void init(size_t& i, const T x1, const T x2) {
            data[i] = x1;
            data[i + 1] = x2;
        }

        public:
        T data[S]{};

        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 1, int>::type = 0>
        T& x() { return data[0]; }
        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 2, int>::type = 0>
        T& y() { return data[1]; }
        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 3, int>::type = 0>
        T& z() { return data[2]; }
        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 4, int>::type = 0>
        T& w() { return data[3]; }

        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 1, int>::type = 0>
        const T& x() const { return data[0]; }
        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 2, int>::type = 0>
        const T& y() const { return data[1]; }
        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 3, int>::type = 0>
        const T& z() const { return data[2]; }
        template<size_t VectorSize = S, typename std::enable_if<VectorSize >= 4, int>::type = 0>
        const T& w() const { return data[3]; }

        vec(const vec<S, T>&) = default;
        vec(vec<S, T>&&) = default;
        vec& operator=(const vec<S, T>&) = default;
        vec& operator=(vec<S, T>&&) = default;

        template<size_t VectorSize =S, class ... Ts, typename std::enable_if<VectorSize >= 5, int>::type = 0>
        vec(const T x, const Ts ... xs) {
            static_assert(sizeof...(Ts) + 1 == S);
            size_t i = 0;
            init(i, x, xs...);
        }

        template<size_t VectorSize =S, typename std::enable_if<VectorSize == 2, int>::type = 0>
        vec(const T x, const T y) {
            this->x() = x;
            this->y() = y;
        }
        template<size_t VectorSize =S, typename std::enable_if<VectorSize == 3, int>::type = 0>
        vec(const T x, const T y, const T z) {
            this->x() = x;
            this->y() = y;
            this->z() = z;
        }
        template<size_t VectorSize =S, typename std::enable_if<VectorSize == 4, int>::type = 0>
        vec(const T x, const T y, const T z, const T w) {
            this->x() = x;
            this->y() = y;
            this->z() = z;
            this->w() = w;
        }

        vec(const T x = T()) {
            for (T& p : data)
                p = x;
        }

        explicit vec(const T* k) {
            memcpy(data, k, S * sizeof(T));
        }

        T& operator[](const size_t i) {
            if (i < S)
                return data[i];
            return data[S - 1];
        }

        const T& operator[](const size_t i) const {
            if (i < S)
                return data[i];
            return data[S - 1];
        }

        vec<S, T> operator+(const vec<S, T>& v) const {
            vec<S, T> res{};
            for (size_t i = 0; i < S; i++)
                res[i] = data[i] + v[i];
            return res;
        }
        vec<S, T> operator-(const vec<S, T>& v) const {
            vec<S, T> res{};
            for (size_t i = 0; i < S; i++)
                res[i] = data[i] - v[i];
            return res;
        }
        vec<S, T> operator*(const vec<S, T>& v) const {
            vec<S, T> res{};
            for (size_t i = 0; i < S; i++)
                res[i] = data[i] * v[i];
            return res;
        }
        vec<S, T> operator/(const vec<S, T>& v) const {
            vec<S, T> res{};
            for (size_t i = 0; i < S; i++)
                res[i] = data[i] / v[i];
            return res;
        }

        vec<S, T>& operator+=(const vec<S, T>& v) {
            for (size_t i = 0; i < S; i++)
                data[i] += v.data[i];
            return *this;
        }
        vec<S, T>& operator-=(const vec<S, T>& v) {
            for (size_t i = 0; i < S; i++)
                data[i] -= v.data[i];
            return *this;
        }
        vec<S, T>& operator*=(const vec<S, T>& v) {
            for (size_t i = 0; i < S; i++)
                data[i] *= v.data[i];
            return *this;
        }
        vec<S, T>& operator/=(const vec<S, T>& v) {
            for (size_t i = 0; i < S; i++)
                data[i] /= v.data[i];
            return *this;
        }

        T* begin() { return data; }
        T* end() { return data + S; }

        const T* begin() const { return data; }
        const T* end() const{ return data + S; }

        /**
         * @brief Calculate the dot product between this vector and another
         * 
         * @param v The second vector in the dot product operation
         */
        T dot(const vec<S, T>& v) const {
            T res = data[0] * v.data[0];
            for (size_t i = 1; i < S; i++)
                res += data[i] * v.data[i];
            return res;
        }

        /**
         * @brief Calculate the length of the vector
         */
        T length() const {
            return std::sqrt(this->dot(*this));
        }

        /**
         * @brief Calculate the distance between this vector and another
         * 
         * @param v The vector to calculate the distance to
         */
        T distance_to(const vec<S, T>& v) const {
            return (v - *this).length();
        }

        /**
         * @brief Return a normalized version of this vector
         * 
         * @return The normalized vector
         */
        vec<S, T> normalized() const {
            return *this / this->length();
        }

        /**
         * @brief Normalize this vector, then return a reference to it
         * 
         * @return A reference to this vector, after normalizing it
         */
        vec<S, T>& normalize() {
            return *this /= this->length();
        }

        /**
         * @brief Return the direction from this vector to another
         * 
         * @param v The vector to trace to
         */
        vec<S, T> direction_to(const vec<S, T>& v) const {
            return (v - *this).normalized();
        }

        private:
        static T max(const T& k1, const T& k2) {
            if (k1 > k2)
                return k1;
            return k2;
        }
        static T min(const T& k1, const T& k2) {
            if (k1 < k2)
                return k1;
            return k2;
        }

        public:
        /**
         * @brief Return the a vector with all values minimum between the two
         * 
         * @param v1 The first vector
         * @param v2 The second vector
         */
        static vec<S, T> max(const vec<S, T>& v1, const vec<S, T>& v2) {
            vec<S, T> res{};
            for (size_t i = 0; i < S; i++)
                res[i] = max(v1.data[i], v2.data[i]);
            return res;
        }

        /**
         * @brief Return the a vector with all values maximum between the two
         * 
         * @param v1 The first vector
         * @param v2 The second vector
         */
        static vec<S, T> min(const vec<S, T>& v1, const vec<S, T>& v2) {
            vec<S, T> res{};
            for (size_t i = 0; i < S; i++)
                res[i] = min(v1.data[i], v2.data[i]);
            return res;
        }

        /**
         * @brief Return a clamped version of this vector
         * 
         * @param low The lowest to clamp to
         * @param high The highest to clamp to
         * @return A vector with all values clamped between the two other vectors
         */
        vec<S, T> clamped(const vec<S, T>& low, const vec<S, T>& high) const {
            return max(min(*this, high), low);
        }

        /**
         * @brief Clamp this vector between two vectors, and return a reference to it
         * 
         * @param low The lowest to clamp to
         * @param high The highest to clamp to
         * @return A reference to this vector, after clamping
         */
        vec<S, T>& clamp(const vec<S, T>& low, const vec<S, T>& high) {
            *this = max(min(*this, high), low);
            return *this;
        }
    };

#if defined(__x86_64) || defined(__amd64)
    template<>
    inline vec<4, float> vec<4, float>::operator+(const vec<4, float> &v) const {
        const __m128 m = _mm_add_ps(*(__m128*)data, *(__m128*)v.data);
        return vec<4, float> { (float*)&m };
    }
    template<>
    inline vec<4, float> vec<4, float>::operator-(const vec<4, float> &v) const {
        const __m128 m = _mm_sub_ps(*(__m128*)data, *(__m128*)v.data);
        return vec<4, float> { (float*)&m };
    }
    template<>
    inline vec<4, float> vec<4, float>::operator*(const vec<4, float> &v) const {
        const __m128 m = _mm_mul_ps(*(__m128*)data, *(__m128*)v.data);
        return vec<4, float> { (float*)&m };
    }
    template<>
    inline vec<4, float> vec<4, float>::operator/(const vec<4, float> &v) const {
        const __m128 m = _mm_div_ps(*(__m128*)data, *(__m128*)v.data);
        return vec<4, float> { (float*)&m };
    }

    template<>
    inline vec<4, float>& vec<4, float>::operator+=(const vec<4UL, float> &v) {
        *(__m128*)data = _mm_add_ps(*(__m128*)data, *(__m128*)v.data);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator-=(const vec<4UL, float> &v) {
        *(__m128*)data = _mm_sub_ps(*(__m128*)data, *(__m128*)v.data);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator*=(const vec<4UL, float> &v) {
        *(__m128*)data = _mm_mul_ps(*(__m128*)data, *(__m128*)v.data);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator/=(const vec<4UL, float> &v) {
        *(__m128*)data = _mm_div_ps(*(__m128*)data, *(__m128*)v.data);
        return *this;
    }

    template<>
    inline vec<4, float> vec<4, float>::max(const vec<4UL, float> &v1, const vec<4UL, float> &v2) {
        const __m128& m = _mm_max_ps(*(__m128*)v1.data, *(__m128*)v2.data);
        return vec<4, float> { (float*)&m };
    }
    template<>
    inline vec<4, float> vec<4, float>::min(const vec<4UL, float> &v1, const vec<4UL, float> &v2) {
        const __m128& m = _mm_min_ps(*(__m128*)v1.data, *(__m128*)v2.data);
        return vec<4, float> { (float*)&m };
    }
#endif


    using vec2f = vec<2, float>;
    using vec3f = vec<3, float>;
    using vec4f = vec<4, float>;
    using vec2d = vec<2, double>;
    using vec3d = vec<3, double>;
    using vec4d = vec<4, double>;

    using vec2u8 = vec<2, uint8_t>;
    using vec3u8 = vec<3, uint8_t>;
    using vec4u8 = vec<4, uint8_t>;
    using vec2i8 = vec<2, int8_t>;
    using vec3i8 = vec<3, int8_t>;
    using vec4i8 = vec<4, int8_t>;

    using vec2u16 = vec<2, uint16_t>;
    using vec3u16 = vec<3, uint16_t>;
    using vec4u16 = vec<4, uint16_t>;
    using vec2i16 = vec<2, int16_t>;
    using vec3i16 = vec<3, int16_t>;
    using vec4i16 = vec<4, int16_t>;

    using vec2u32 = vec<2, uint32_t>;
    using vec3u32 = vec<3, uint32_t>;
    using vec4u32 = vec<4, uint32_t>;
    using vec2i32 = vec<2, int32_t>;
    using vec3i32 = vec<3, int32_t>;
    using vec4i32 = vec<4, int32_t>;

    using vec2u64 = vec<2, uint64_t>;
    using vec3u64 = vec<3, uint64_t>;
    using vec4u64 = vec<4, uint64_t>;
    using vec2i64 = vec<2, int64_t>;
    using vec3i64 = vec<3, int64_t>;
    using vec4i64 = vec<4, int64_t>;



    //==========
    // MATRICES
    //==========

    template<size_t l, size_t c, class T>
    class mat {
        template<class ... Ts>
        void init(size_t& i, const T x, const Ts ... xs) {
            ((T*)data)[i] = x;
            init(++i, xs...);
        }
        void init(size_t& i, const T x) {
            ((T*)data)[i] = x;
        }

        public:

        vec<c, T> data[l];

        mat(const mat<l, c, T>&) = default;
        mat(mat<l, c, T>&&) = default;
        mat& operator=(const mat<l, c, T>&) = default;
        mat& operator=(mat<l, c, T>&&) = default;

        template<class ... Ts>
        mat(const T x, const Ts ... xs) {
            size_t i = 0;
            init(i, x, xs...);
        }

        explicit mat(const T x = T()) {
            for (size_t i = 0; i < l && i < c; i++)
                data[i][i] = x;
        }

        explicit mat(const T* k) {
            memcpy(data, k, l * c * sizeof(T));
        }

        vec<c, T>& operator[](const size_t i) {
            if (i < l)
                return data[i];
            return data[l - 1];
        }
        const vec<c, T>& operator[](const size_t i) const {
            if (i < l)
                return data[i];
            return data[l - 1];
        }

        mat<l, c, T> operator+(const mat<l, c, T>& m) const {
            mat<l, c, T> res{};
            for (size_t i = 0; i < l; i++)
                res[i] = data[i] + m[i];
            return res;
        }
        mat<l, c, T> operator-(const mat<l, c, T>& m) const {
            mat<l, c, T> res{};
            for (size_t i = 0; i < l; i++)
                res[i] = data[i] - m[i];
            return res;
        }

        template<size_t l2, size_t c2, typename std::enable_if<c == l2, int>::type = 0>
        mat<l, c2, T> operator*(const mat<l2, c2, T>& m) const {
            mat<l, c2, T> res{};
            for (int i = 0; i < l; i++)
                for (int j = 0; j < c2; j++)
                    for (int k = 0; k < c; k++)
                        res[i][j] += data[i][k] * m[k][j];
            return res;
        }

        mat<l, c, T>& operator+=(const mat<l, c, T>& m) {
            for (size_t i = 0; i < l; i++)
                data[i] += m[i];
            return *this;
        }
        mat<l, c, T>& operator-=(const mat<l, c, T>& m) {
            for (size_t i = 0; i < l; i++)
                data[i] -= m[i];
            return *this;
        }

        vec<c, T>* begin() { return data; }
        vec<c, T>* end() { return data + l; }

        const vec<c, T>* begin() const { return data; }
        const vec<c, T>* end() const{ return data + l; }

        /**
         * @brief Return a transposed version of the matrix
         */
        mat<c, l, T> transposed() const {
            mat<c, l, T> res{};
            for (size_t i = 0; i < c; i++)
                for (size_t j = 0; j < l; j++)
                    res[i][j] = data[j][i];
            return res;
        }

        /**
         * @brief Remove the column(x) and line(y) from the matrix, and return the new matrix
         * 
         * @param pos 
         * @return mat<l - 1, c - 1, T> 
         */
        mat<l - 1, c - 1, T> submat(const vec2u64& pos) const {
            mat<l - 1, c - 1, T> res{};
            for (size_t i = 0; i < l - 1; i++) {
                for (size_t j = 0; j < c - 1; j++) {
                    res[i][j] = data[i + (i >= pos.y())][j + (j >= pos.x())];
                }
            }
            return res;
        }

        /**
         * @brief Calculate the determinant of the matrix
         */
        template<size_t Lines = l, size_t Columns = c, typename std::enable_if<Lines >= 3 && Columns == Lines, int>::type = 0>
        T det() const {
            T res{};
            bool ff = false;
            for (size_t i = 0; i < c; i++) {
                if (ff)
                    res -= data[i][0] * submat(vec2u64(i, 0)).det();
                else
                    res += data[i][0] * submat(vec2u64(i, 0)).det();
                ff = ! ff;
            }
            return res;
        }

        /**
         * @brief Calculate the determinant of the matrix
         */
        template<size_t Lines = l, size_t Columns = c, typename std::enable_if<Lines == 2 && Columns == 2, int>::type = 0>
        T det() const {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }

        /**
         * @brief Generate a 2D rotation matrix with angle and scale (scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 2 && Columns == 2
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_rotation2d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                cos, -sin,
                sin, cos
            };
        }

        /**
         * @brief Generate a 2D rotation matrix with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_rotation2d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                cos, -sin, T(),
                sin, cos, T(),
                T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix for the X axis with angle and scale (scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_x_rotation3d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                (T)1, T(), T(),
                T(), cos, -sin,
                T(), sin, cos
            };
        }

        /**
         * @brief Generate a 3D rotation matrix for the Y axis with angle and scale (scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_y_rotation3d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                cos, T(), sin,
                T(), (T)1, T(),
                -sin, T(), cos
            };
        }

        /**
         * @brief Generate a 3D rotation matrix for the Z axis with angle and scale (scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_z_rotation3d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                cos, -sin, T(),
                sin, cos, T(),
                T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix using a precalculated sin and cos for the X axis with angle and scale (scale is 1.0)
         * 
         * @param sin Sine of the angle
         * @param cos Cosine of the angle
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_x_rotation3d(T sin, T cos) {
            return mat<c, l, T>{
                (T)1, T(), T(),
                T(), cos, -sin,
                T(), sin, cos
            };
        }

        /**
         * @brief Generate a 3D rotation matrix using a precalculated sin and cos for the Y axis with angle and scale (scale is 1.0)
         * 
         * @param sin Sine of the angle
         * @param cos Cosine of the angle
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_y_rotation3d(T sin, T cos) {
            return mat<c, l, T>{
                cos, T(), sin,
                T(), (T)1, T(),
                -sin, T(), cos
            };
        }

        /**
         * @brief Generate a 3D rotation matrix using a precalculated sin and cos for the Z axis with angle and scale (scale is 1.0)
         * 
         * @param sin Sine of the angle
         * @param cos Cosine of the angle
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 3 && Columns == 3
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_z_rotation3d(T sin, T cos) {
            return mat<c, l, T>{
                cos, -sin, T(),
                sin, cos, T(),
                T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix for the X axis with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_x_rotation3d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                (T)1, T(), T(), T(),
                T(), cos, -sin, T(),
                T(), sin, cos, T(),
                T(), T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix for the Y axis with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_y_rotation3d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                cos, T(), sin, T(),
                T(), (T)1, T(), T(),
                -sin, T(), cos, T(),
                T(), T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix for the Z axis with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param angle The angle to use
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_z_rotation3d(T angle) {
            const T cos = std::cos(angle);
            const T sin = std::sin(angle);
            return mat<c, l, T>{
                cos, -sin, T(), T(),
                sin, cos, T(), T(),
                T(), T(), (T)1, T(),
                T(), T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix using a precalculated sin and cos for the X axis with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param sin Sine of the angle
         * @param cos Cosine of the angle
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_x_rotation3d(T sin, T cos) {
            return mat<c, l, T>{
                (T)1, T(), T(), T(),
                T(), cos, -sin, T(),
                T(), sin, cos, T(),
                T(), T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix using a precalculated sin and cos for the Y axis with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param sin Sine of the angle
         * @param cos Cosine of the angle
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_y_rotation3d(T sin, T cos) {
            return mat<c, l, T>{
                cos, T(), sin, T(),
                T(), (T)1, T(), T(),
                -sin, T(), cos, T(),
                T(), T(), T(), (T)1
            };
        }

        /**
         * @brief Generate a 3D rotation matrix using a precalculated sin and cos for the Z axis with angle, position, scale and skew (position and skew are 0.0, scale is 1.0)
         * 
         * @param sin Sine of the angle
         * @param cos Cosine of the angle
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_z_rotation3d(T sin, T cos) {
            return mat<c, l, T>{
                cos, -sin, T(), T(),
                sin, cos, T(), T(),
                T(), T(), (T)1, T(),
                T(), T(), T(), (T)1
            };
        }

        /**
         * @brief Rotate the matrix in 2D and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 2 && Columns == 2 || Lines == 3 && Columns == 3)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate2d(T angle) {
            *this = gen_rotation2d(angle) * (*this);
            return *this;
        }

        /**
         * @brief Return a rotated version of this matrix in 2D
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 2 && Columns == 2 || Lines == 3 && Columns == 3)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated2d(T angle) const {
            return gen_rotation2d(angle) * (*this);
        }

        /**
         * @brief Rotate the matrix in 3D in the order XYZ and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate3d_xyz(const vec<3, T>& axis) {
            *this = gen_x_rotation3d(axis.x())
                  * gen_y_rotation3d(axis.y())
                  * gen_z_rotation3d(axis.z()) * (*this);
            return *this;
        }

        /**
         * @brief Rotate the matrix in 3D in the order XZY and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate3d_xzy(const vec<3, T>& axis) {
            *this = gen_x_rotation3d(axis.x())
                  * gen_z_rotation3d(axis.z())
                  * gen_y_rotation3d(axis.y()) * (*this);
            return *this;
        }

        /**
         * @brief Rotate the matrix in 3D in the order YXZ and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate3d_yxz(const vec<3, T>& axis) {
            *this = gen_y_rotation3d(axis.y())
                  * gen_x_rotation3d(axis.x())
                  * gen_z_rotation3d(axis.z()) * (*this);
            return *this;
        }

        /**
         * @brief Rotate the matrix in 3D in the order YZX and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate3d_yzx(const vec<3, T>& axis) {
            *this = gen_y_rotation3d(axis.y())
                  * gen_z_rotation3d(axis.z())
                  * gen_x_rotation3d(axis.x()) * (*this);
            return *this;
        }

        /**
         * @brief Rotate the matrix in 3D in the order ZXY and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate3d_zxy(const vec<3, T>& axis) {
            *this = gen_z_rotation3d(axis.z())
                  * gen_x_rotation3d(axis.x())
                  * gen_y_rotation3d(axis.y()) * (*this);
            return *this;
        }

        /**
         * @brief Rotate the matrix in 3D in the order ZYX and return a reference to it after it has been rotated
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T>& rotate3d_zyx(const vec<3, T>& axis) {
            *this = gen_z_rotation3d(axis.z())
                  * gen_y_rotation3d(axis.y())
                  * gen_x_rotation3d(axis.x()) * (*this);
            return *this;
        }

        /**
         * @brief Return a rotated version of this matrix in 2D in the order XYZ
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated3d_xyz(const vec<3, T>& axis) const {
            return gen_x_rotation3d(axis.x())
                 * gen_y_rotation3d(axis.y())
                 * gen_z_rotation3d(axis.z()) * (*this);
        }

        /**
         * @brief Return a rotated version of this matrix in 2D in the order XZY
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated3d_xzy(const vec<3, T>& axis) const {
            return gen_x_rotation3d(axis.x())
                 * gen_z_rotation3d(axis.z())
                 * gen_y_rotation3d(axis.y()) * (*this);
        }

        /**
         * @brief Return a rotated version of this matrix in 2D in the order YXZ
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated3d_yxz(const vec<3, T>& axis) const {
            return gen_y_rotation3d(axis.y())
                 * gen_x_rotation3d(axis.x())
                 * gen_z_rotation3d(axis.z()) * (*this);
        }

        /**
         * @brief Return a rotated version of this matrix in 2D in the order YZX
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated3d_yzx(const vec<3, T>& axis) const {
            return gen_y_rotation3d(axis.y())
                 * gen_z_rotation3d(axis.z())
                 * gen_x_rotation3d(axis.x()) * (*this);
        }

        /**
         * @brief Return a rotated version of this matrix in 2D in the order ZXY
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated3d_zxy(const vec<3, T>& axis) const {
            return gen_z_rotation3d(axis.z())
                 * gen_x_rotation3d(axis.x())
                 * gen_y_rotation3d(axis.y()) * (*this);
        }

        /**
         * @brief Return a rotated version of this matrix in 2D in the order ZYX
         * 
         * @param angle (in radians) The angle to rotate by
         */
        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<(Lines == 3 && Columns == 3 || Lines == 4 && Columns == 4)
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        mat<l, c, T> rotated3d_zyx(const vec<3, T>& axis) const {
            return gen_z_rotation3d(axis.z())
                 * gen_y_rotation3d(axis.y())
                 * gen_x_rotation3d(axis.x()) * (*this);
        }
    };

    using mat2f = mat<2, 2, float>;
    using mat3f = mat<3, 3, float>;
    using mat4f = mat<4, 4, float>;
    using mat2d = mat<2, 2, double>;
    using mat3d = mat<3, 3, double>;
    using mat4d = mat<4, 4, double>;

    using mat2u8 = mat<2, 2, uint8_t>;
    using mat3u8 = mat<3, 3, uint8_t>;
    using mat4u8 = mat<4, 4, uint8_t>;
    using mat2i8 = mat<2, 2, int8_t>;
    using mat3i8 = mat<3, 3, int8_t>;
    using mat4i8 = mat<4, 4, int8_t>;

    using mat2u16 = mat<2, 2, uint16_t>;
    using mat3u16 = mat<3, 3, uint16_t>;
    using mat4u16 = mat<4, 4, uint16_t>;
    using mat2i16 = mat<2, 2, int16_t>;
    using mat3i16 = mat<3, 3, int16_t>;
    using mat4i16 = mat<4, 4, int16_t>;

    using mat2u32 = mat<2, 2, uint32_t>;
    using mat3u32 = mat<3, 3, uint32_t>;
    using mat4u32 = mat<4, 4, uint32_t>;
    using mat2i32 = mat<2, 2, int32_t>;
    using mat3i32 = mat<3, 3, int32_t>;
    using mat4i32 = mat<4, 4, int32_t>;

    using mat2u64 = mat<2, 2, uint64_t>;
    using mat3u64 = mat<3, 3, uint64_t>;
    using mat4u64 = mat<4, 4, uint64_t>;
    using mat2i64 = mat<2, 2, int64_t>;
    using mat3i64 = mat<3, 3, int64_t>;
    using mat4i64 = mat<4, 4, int64_t>;
}
