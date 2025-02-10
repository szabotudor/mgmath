#pragma once
#include <cmath>
#include <cstdint>
#include <memory.h>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <cassert>

#if defined(__x86_64) || defined(__amd64) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>
#include <smmintrin.h>
#endif

#define ASSURE_SIZE(SIZE) size_t VectorSize = S, typename std::enable_if<VectorSize >= SIZE, bool>::type = true
#define ASSURE_EXACT_SIZE(SIZE) size_t VectorSize = S, typename std::enable_if<VectorSize == SIZE, bool>::type = true


namespace mgm {

    //=========
    // VECTORS
    //=========


    template<size_t S, typename T>
    class vec_storage {
        public:

        T _data[S];

        template<ASSURE_SIZE(1)>
        T& _x() { return _data[0]; }
        template<ASSURE_SIZE(2)>
        T& _y() { return _data[1]; }
        template<ASSURE_SIZE(3)>
        T& _z() { return _data[2]; }
        template<ASSURE_SIZE(4)>
        T& _w() { return _data[3]; }

        template<ASSURE_SIZE(1)>
        const T& _x() const { return _data[0]; }
        template<ASSURE_SIZE(2)>
        const T& _y() const { return _data[1]; }
        template<ASSURE_SIZE(3)>
        const T& _z() const { return _data[2]; }
        template<ASSURE_SIZE(4)>
        const T& _w() const { return _data[3]; }

        T& operator[](const size_t i) {
            #if !defined(NDEBUG)
            if (i > S)
                throw std::runtime_error{"Index out of range"};
            #endif
            return _data[i];
        }
        const T& operator[](const size_t i) const {
            #if !defined(NDEBUG)
            if (i > S)
                throw std::runtime_error{"Index out of range"};
            #endif
            return _data[i];
        }

        vec_storage(const T& k = T{}) : _data(k) {}
        
        template<typename... Ts>
        vec_storage(Ts&&... args) : _data(std::forward<Ts>(args)...) {}

        vec_storage(const vec_storage&) = default;
        vec_storage(vec_storage&&) = default;
        vec_storage& operator=(const vec_storage&) = default;
        vec_storage& operator=(vec_storage&&) = default;

        T* data() { return _data; }
        const T* data() const { return _data; }
    };

    template<typename T>
    class vec_storage<4, T> {
        public:

        T x{}, y{}, z{}, w{};

        T& _x() { return x; }
        T& _y() { return y; }
        T& _z() { return z; }
        T& _w() { return w; }

        const T& _x() const { return x; }
        const T& _y() const { return y; }
        const T& _z() const { return z; }
        const T& _w() const { return w; }

        T& operator[](const size_t i) {
            switch (i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                case 3: return w;
                default:
                    #if !defined(NDEBUG)
                    throw std::runtime_error{"Index out of range"};
                    #else
                    return x;
                    #endif
            }
        }
        const T& operator[](const size_t i) const {
            switch (i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                case 3: return w;
                default:
                    #if !defined(NDEBUG)
                    throw std::runtime_error{"Index out of range"};
                    #else
                    return x;
                    #endif
            }
        }

        vec_storage(const T& k = T{}) : x(k), y(k), z(k), w(k) {}

        vec_storage(const T& x_v, const T& y_v, const T& z_v = 0, const T& w_v = 0) : x(x_v), y(y_v), z(z_v), w(w_v) {}

        vec_storage(const vec_storage&) = default;
        vec_storage(vec_storage&&) = default;
        vec_storage& operator=(const vec_storage&) = default;
        vec_storage& operator=(vec_storage&&) = default;

        T* data() { return (T*)this; }
        const T* data() const { return (const T*)this; }
    };
    template<typename T>
    class vec_storage<3, T> {
        public:

        T x{}, y{}, z{};

        T& _x() { return x; }
        T& _y() { return y; }
        T& _z() { return z; }

        const T& _x() const { return x; }
        const T& _y() const { return y; }
        const T& _z() const { return z; }

        T& operator[](const size_t i) {
            switch (i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                default:
                    #if !defined(NDEBUG)
                    throw std::runtime_error{"Index out of range"};
                    #else
                    return x;
                    #endif
            }
        }
        const T& operator[](const size_t i) const {
            switch (i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                default:
                    #if !defined(NDEBUG)
                    throw std::runtime_error{"Index out of range"};
                    #else
                    return x;
                    #endif
            }
        }

        vec_storage(const T& k = T{}) : x(k), y(k), z(k) {}

        vec_storage(const T& x_v, const T& y_v, const T& z_v = 0) : x(x_v), y(y_v), z(z_v) {}

        vec_storage(const vec_storage&) = default;
        vec_storage(vec_storage&&) = default;
        vec_storage& operator=(const vec_storage&) = default;
        vec_storage& operator=(vec_storage&&) = default;

        T* data() { return (T*)this; }
        const T* data() const { return (const T*)this; }
    };
    template<typename T>
    class vec_storage<2, T> {
        public:

        T x{}, y{};

        T& _x() { return x; }
        T& _y() { return y; }

        const T& _x() const { return x; }
        const T& _y() const { return y; }

        T& operator[](const size_t i) {
            switch (i) {
                case 0: return x;
                case 1: return y;
                default:
                    #if !defined(NDEBUG)
                    throw std::runtime_error{"Index out of range"};
                    #else
                    return x;
                    #endif
            }
        }
        const T& operator[](const size_t i) const {
            switch (i) {
                case 0: return x;
                case 1: return y;
                default:
                    #if !defined(NDEBUG)
                    throw std::runtime_error{"Index out of range"};
                    #else
                    return x;
                    #endif
            }
        }

        vec_storage(const T& k = T{}) : x(k), y(k) {}

        vec_storage(const T& x_v, const T& y_v) : x(x_v), y(y_v) {}

        vec_storage(const vec_storage&) = default;
        vec_storage(vec_storage&&) = default;
        vec_storage& operator=(const vec_storage&) = default;
        vec_storage& operator=(vec_storage&&) = default;

        T* data() { return (T*)this; }
        const T* data() const { return (const T*)this; }
    };

    template<size_t S, class T>
    class vec : public vec_storage<S, T> {
        public:

        template<ASSURE_SIZE(1)>
        T& _x() { return vec_storage<S, T>::_x(); }
        template<ASSURE_SIZE(2)>
        T& _y() { return vec_storage<S, T>::_y(); }
        template<ASSURE_SIZE(3)>
        T& _z() { return vec_storage<S, T>::_z(); }
        template<ASSURE_SIZE(4)>
        T& _w() { return vec_storage<S, T>::_w(); }

        template<ASSURE_SIZE(1)>
        const T& _x() const { return vec_storage<S, T>::_x(); }
        template<ASSURE_SIZE(2)>
        const T& _y() const { return vec_storage<S, T>::_y(); }
        template<ASSURE_SIZE(3)>
        const T& _z() const { return vec_storage<S, T>::_z(); }
        template<ASSURE_SIZE(4)>
        const T& _w() const { return vec_storage<S, T>::_w(); }

        T* data() { return vec_storage<S, T>::data(); }
        const T* data() const { return vec_storage<S, T>::data(); }


        template <typename...>
        struct TypeList {};

        template <typename U, typename... Ts>
        struct Append;

        template <typename U, typename... Ts>
        struct Append<U, TypeList<Ts...>> {
            using Type = TypeList<Ts..., U>;
        };

        template <size_t n, typename U = TypeList<>>
        struct IntListGenerator {
            using Type = typename IntListGenerator<n - 1, typename Append<T, U>::Type>::Type;
        };

        template <typename U>
        struct IntListGenerator<0, U> {
            using Type = U;
        };

        template <size_t n>
        using IntList = typename IntListGenerator<n>::Type;

        static inline void add_one(const T& a, const T& b, T& r, size_t& i) { r = a + b; ++i; }
        static inline void sub_one(const T& a, const T& b, T& r, size_t& i) { r = a - b; ++i; }
        static inline void mul_one(const T& a, const T& b, T& r, size_t& i) { r = a * b; ++i; }
        static inline void div_one(const T& a, const T& b, T& r, size_t& i) { r = a / b; ++i; }
        static inline void mod_one(const T& a, const T& b, T& r, size_t& i) { r = a % b; ++i; }
        static inline void eq_one(const T& a, const T& b, bool& r, size_t& i) { r = r && a == b; ++i; }


        template<typename... Ts>
        static inline void add(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (add_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }
        template<typename... Ts>
        static inline void sub(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (sub_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }
        template<typename... Ts>
        static inline void mul(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (mul_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }
        template<typename... Ts>
        static inline void div(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (div_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }
        template<typename... Ts>
        static inline void mod(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (mod_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }
        template<typename... Ts>
        static inline bool eq(const T* a, const T* b, TypeList<Ts...>) {
            size_t i = 0;
            bool result = true;
            (eq_one((const Ts&)a[i], (const Ts&)b[i], result, i), ...);
            return result;
        }

        static inline void real_dot(const T& a, const T& b, T& r, size_t& i) {
            r += a * b;
            ++i;
        }
        template<typename... Ts>
        static inline T real_dot(const T* a, const T* b, TypeList<Ts...>) {
            size_t i = 0;
            T sum = 0;
            ((real_dot((const Ts&)a[i], (const Ts&)b[i], (Ts&)sum, i)), ...);
            return sum;
        }

        static inline void max_one(const T& a, const T& b, T& r, size_t& i) { r = a > b ? a : b; ++i; }
        static inline void min_one(const T& a, const T& b, T& r, size_t& i) { r = a < b ? a : b; ++i; }

        template<typename... Ts>
        static inline void max(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (max_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }
        template<typename... Ts>
        static inline void min(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (min_one((const Ts&)a[i], (const Ts&)b[i], (Ts&)r[i], i), ...);
        }

        public:

#if defined(MGMATH_SWIZZLE)
        template<ASSURE_SIZE(2)> vec<2, T> xx() const { return vec<2, T>{_x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<2, T> xy() const { return vec<2, T>{_x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<2, T> xz() const { return vec<2, T>{_x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<2, T> xw() const { return vec<2, T>{_x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<2, T> yx() const { return vec<2, T>{_y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<2, T> yy() const { return vec<2, T>{_y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<2, T> yz() const { return vec<2, T>{_y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<2, T> yw() const { return vec<2, T>{_y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<2, T> zx() const { return vec<2, T>{_z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<2, T> zy() const { return vec<2, T>{_z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<2, T> zz() const { return vec<2, T>{_z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<2, T> zw() const { return vec<2, T>{_z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<2, T> wx() const { return vec<2, T>{_w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<2, T> wy() const { return vec<2, T>{_w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<2, T> wz() const { return vec<2, T>{_w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<2, T> ww() const { return vec<2, T>{_w(), _w()}; }

        template<ASSURE_SIZE(2)> vec<3, T> xxx() const { return vec<3, T>{_x(), _x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> xxy() const { return vec<3, T>{_x(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xxz() const { return vec<3, T>{_x(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xxw() const { return vec<3, T>{_x(), _x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<3, T> xyx() const { return vec<3, T>{_x(), _y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> xyy() const { return vec<3, T>{_x(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xyz() const { return vec<3, T>{_x(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xyw() const { return vec<3, T>{_x(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xzx() const { return vec<3, T>{_x(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xzy() const { return vec<3, T>{_x(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xzz() const { return vec<3, T>{_x(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xzw() const { return vec<3, T>{_x(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xwx() const { return vec<3, T>{_x(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xwy() const { return vec<3, T>{_x(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xwz() const { return vec<3, T>{_x(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xww() const { return vec<3, T>{_x(), _w(), _w()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yxx() const { return vec<3, T>{_y(), _x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yxy() const { return vec<3, T>{_y(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yxz() const { return vec<3, T>{_y(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yxw() const { return vec<3, T>{_y(), _x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yyx() const { return vec<3, T>{_y(), _y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yyy() const { return vec<3, T>{_y(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yyz() const { return vec<3, T>{_y(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yyw() const { return vec<3, T>{_y(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yzx() const { return vec<3, T>{_y(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yzy() const { return vec<3, T>{_y(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yzz() const { return vec<3, T>{_y(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yzw() const { return vec<3, T>{_y(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> ywx() const { return vec<3, T>{_y(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> ywy() const { return vec<3, T>{_y(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> ywz() const { return vec<3, T>{_y(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yww() const { return vec<3, T>{_y(), _w(), _w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zxx() const { return vec<3, T>{_z(), _x(), _x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zxy() const { return vec<3, T>{_z(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zxz() const { return vec<3, T>{_z(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zxw() const { return vec<3, T>{_z(), _x(), _w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zyx() const { return vec<3, T>{_z(), _y(), _x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zyy() const { return vec<3, T>{_z(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zyz() const { return vec<3, T>{_z(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zyw() const { return vec<3, T>{_z(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zzx() const { return vec<3, T>{_z(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zzy() const { return vec<3, T>{_z(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zzz() const { return vec<3, T>{_z(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zzw() const { return vec<3, T>{_z(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zwx() const { return vec<3, T>{_z(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zwy() const { return vec<3, T>{_z(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zwz() const { return vec<3, T>{_z(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zww() const { return vec<3, T>{_z(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxx() const { return vec<3, T>{_w(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxy() const { return vec<3, T>{_w(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxz() const { return vec<3, T>{_w(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxw() const { return vec<3, T>{_w(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyx() const { return vec<3, T>{_w(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyy() const { return vec<3, T>{_w(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyz() const { return vec<3, T>{_w(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyw() const { return vec<3, T>{_w(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzx() const { return vec<3, T>{_w(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzy() const { return vec<3, T>{_w(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzz() const { return vec<3, T>{_w(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzw() const { return vec<3, T>{_w(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wwx() const { return vec<3, T>{_w(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wwy() const { return vec<3, T>{_w(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wwz() const { return vec<3, T>{_w(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> www() const { return vec<3, T>{_w(), _w(), _w()}; }

        template<ASSURE_SIZE(2)> vec<4, T> xxxx() const { return vec<4, T>{_x(), _x(), _x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xxxy() const { return vec<4, T>{_x(), _x(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxxz() const { return vec<4, T>{_x(), _x(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxxw() const { return vec<4, T>{_x(), _x(), _x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xxyx() const { return vec<4, T>{_x(), _x(), _y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xxyy() const { return vec<4, T>{_x(), _x(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxyz() const { return vec<4, T>{_x(), _x(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxyw() const { return vec<4, T>{_x(), _x(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxzx() const { return vec<4, T>{_x(), _x(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxzy() const { return vec<4, T>{_x(), _x(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxzz() const { return vec<4, T>{_x(), _x(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxzw() const { return vec<4, T>{_x(), _x(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxwx() const { return vec<4, T>{_x(), _x(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxwy() const { return vec<4, T>{_x(), _x(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxwz() const { return vec<4, T>{_x(), _x(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxww() const { return vec<4, T>{_x(), _x(), _w(), _w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyxx() const { return vec<4, T>{_x(), _y(), _x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyxy() const { return vec<4, T>{_x(), _y(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyxz() const { return vec<4, T>{_x(), _y(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyxw() const { return vec<4, T>{_x(), _y(), _x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyyx() const { return vec<4, T>{_x(), _y(), _y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyyy() const { return vec<4, T>{_x(), _y(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyyz() const { return vec<4, T>{_x(), _y(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyyw() const { return vec<4, T>{_x(), _y(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyzx() const { return vec<4, T>{_x(), _y(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyzy() const { return vec<4, T>{_x(), _y(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyzz() const { return vec<4, T>{_x(), _y(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyzw() const { return vec<4, T>{_x(), _y(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xywx() const { return vec<4, T>{_x(), _y(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xywy() const { return vec<4, T>{_x(), _y(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xywz() const { return vec<4, T>{_x(), _y(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyww() const { return vec<4, T>{_x(), _y(), _w(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzxx() const { return vec<4, T>{_x(), _z(), _x(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzxy() const { return vec<4, T>{_x(), _z(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzxz() const { return vec<4, T>{_x(), _z(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzxw() const { return vec<4, T>{_x(), _z(), _x(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzyx() const { return vec<4, T>{_x(), _z(), _y(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzyy() const { return vec<4, T>{_x(), _z(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzyz() const { return vec<4, T>{_x(), _z(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzyw() const { return vec<4, T>{_x(), _z(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzzx() const { return vec<4, T>{_x(), _z(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzzy() const { return vec<4, T>{_x(), _z(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzzz() const { return vec<4, T>{_x(), _z(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzzw() const { return vec<4, T>{_x(), _z(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzwx() const { return vec<4, T>{_x(), _z(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzwy() const { return vec<4, T>{_x(), _z(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzwz() const { return vec<4, T>{_x(), _z(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzww() const { return vec<4, T>{_x(), _z(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxx() const { return vec<4, T>{_x(), _w(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxy() const { return vec<4, T>{_x(), _w(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxz() const { return vec<4, T>{_x(), _w(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxw() const { return vec<4, T>{_x(), _w(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyx() const { return vec<4, T>{_x(), _w(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyy() const { return vec<4, T>{_x(), _w(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyz() const { return vec<4, T>{_x(), _w(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyw() const { return vec<4, T>{_x(), _w(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzx() const { return vec<4, T>{_x(), _w(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzy() const { return vec<4, T>{_x(), _w(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzz() const { return vec<4, T>{_x(), _w(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzw() const { return vec<4, T>{_x(), _w(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwwx() const { return vec<4, T>{_x(), _w(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwwy() const { return vec<4, T>{_x(), _w(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwwz() const { return vec<4, T>{_x(), _w(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwww() const { return vec<4, T>{_x(), _w(), _w(), _w()}; }

        template<ASSURE_SIZE(2)> vec<4, T> yxxx() const { return vec<4, T>{_y(), _x(), _x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yxxy() const { return vec<4, T>{_y(), _x(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxxz() const { return vec<4, T>{_y(), _x(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxxw() const { return vec<4, T>{_y(), _x(), _x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yxyx() const { return vec<4, T>{_y(), _x(), _y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yxyy() const { return vec<4, T>{_y(), _x(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxyz() const { return vec<4, T>{_y(), _x(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxyw() const { return vec<4, T>{_y(), _x(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxzx() const { return vec<4, T>{_y(), _x(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxzy() const { return vec<4, T>{_y(), _x(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxzz() const { return vec<4, T>{_y(), _x(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxzw() const { return vec<4, T>{_y(), _x(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxwx() const { return vec<4, T>{_y(), _x(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxwy() const { return vec<4, T>{_y(), _x(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxwz() const { return vec<4, T>{_y(), _x(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxww() const { return vec<4, T>{_y(), _x(), _w(), _w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyxx() const { return vec<4, T>{_y(), _y(), _x(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyxy() const { return vec<4, T>{_y(), _y(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyxz() const { return vec<4, T>{_y(), _y(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyxw() const { return vec<4, T>{_y(), _y(), _x(), _w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyyx() const { return vec<4, T>{_y(), _y(), _y(), _x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyyy() const { return vec<4, T>{_y(), _y(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyyz() const { return vec<4, T>{_y(), _y(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyyw() const { return vec<4, T>{_y(), _y(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyzx() const { return vec<4, T>{_y(), _y(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyzy() const { return vec<4, T>{_y(), _y(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyzz() const { return vec<4, T>{_y(), _y(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyzw() const { return vec<4, T>{_y(), _y(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yywx() const { return vec<4, T>{_y(), _y(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yywy() const { return vec<4, T>{_y(), _y(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yywz() const { return vec<4, T>{_y(), _y(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyww() const { return vec<4, T>{_y(), _y(), _w(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzxx() const { return vec<4, T>{_y(), _z(), _x(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzxy() const { return vec<4, T>{_y(), _z(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzxz() const { return vec<4, T>{_y(), _z(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzxw() const { return vec<4, T>{_y(), _z(), _x(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzyx() const { return vec<4, T>{_y(), _z(), _y(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzyy() const { return vec<4, T>{_y(), _z(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzyz() const { return vec<4, T>{_y(), _z(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzyw() const { return vec<4, T>{_y(), _z(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzzx() const { return vec<4, T>{_y(), _z(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzzy() const { return vec<4, T>{_y(), _z(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzzz() const { return vec<4, T>{_y(), _z(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzzw() const { return vec<4, T>{_y(), _z(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzwx() const { return vec<4, T>{_y(), _z(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzwy() const { return vec<4, T>{_y(), _z(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzwz() const { return vec<4, T>{_y(), _z(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzww() const { return vec<4, T>{_y(), _z(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxx() const { return vec<4, T>{_y(), _w(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxy() const { return vec<4, T>{_y(), _w(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxz() const { return vec<4, T>{_y(), _w(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxw() const { return vec<4, T>{_y(), _w(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyx() const { return vec<4, T>{_y(), _w(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyy() const { return vec<4, T>{_y(), _w(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyz() const { return vec<4, T>{_y(), _w(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyw() const { return vec<4, T>{_y(), _w(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzx() const { return vec<4, T>{_y(), _w(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzy() const { return vec<4, T>{_y(), _w(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzz() const { return vec<4, T>{_y(), _w(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzw() const { return vec<4, T>{_y(), _w(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywwx() const { return vec<4, T>{_y(), _w(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywwy() const { return vec<4, T>{_y(), _w(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywwz() const { return vec<4, T>{_y(), _w(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywww() const { return vec<4, T>{_y(), _w(), _w(), _w()}; }

        template<ASSURE_SIZE(3)> vec<4, T> zxxx() const { return vec<4, T>{_z(), _x(), _x(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxxy() const { return vec<4, T>{_z(), _x(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxxz() const { return vec<4, T>{_z(), _x(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxxw() const { return vec<4, T>{_z(), _x(), _x(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxyx() const { return vec<4, T>{_z(), _x(), _y(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxyy() const { return vec<4, T>{_z(), _x(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxyz() const { return vec<4, T>{_z(), _x(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxyw() const { return vec<4, T>{_z(), _x(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxzx() const { return vec<4, T>{_z(), _x(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxzy() const { return vec<4, T>{_z(), _x(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxzz() const { return vec<4, T>{_z(), _x(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxzw() const { return vec<4, T>{_z(), _x(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxwx() const { return vec<4, T>{_z(), _x(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxwy() const { return vec<4, T>{_z(), _x(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxwz() const { return vec<4, T>{_z(), _x(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxww() const { return vec<4, T>{_z(), _x(), _w(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyxx() const { return vec<4, T>{_z(), _y(), _x(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyxy() const { return vec<4, T>{_z(), _y(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyxz() const { return vec<4, T>{_z(), _y(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyxw() const { return vec<4, T>{_z(), _y(), _x(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyyx() const { return vec<4, T>{_z(), _y(), _y(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyyy() const { return vec<4, T>{_z(), _y(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyyz() const { return vec<4, T>{_z(), _y(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyyw() const { return vec<4, T>{_z(), _y(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyzx() const { return vec<4, T>{_z(), _y(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyzy() const { return vec<4, T>{_z(), _y(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyzz() const { return vec<4, T>{_z(), _y(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyzw() const { return vec<4, T>{_z(), _y(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zywx() const { return vec<4, T>{_z(), _y(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zywy() const { return vec<4, T>{_z(), _y(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zywz() const { return vec<4, T>{_z(), _y(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyww() const { return vec<4, T>{_z(), _y(), _w(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzxx() const { return vec<4, T>{_z(), _z(), _x(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzxy() const { return vec<4, T>{_z(), _z(), _x(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzxz() const { return vec<4, T>{_z(), _z(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzxw() const { return vec<4, T>{_z(), _z(), _x(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzyx() const { return vec<4, T>{_z(), _z(), _y(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzyy() const { return vec<4, T>{_z(), _z(), _y(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzyz() const { return vec<4, T>{_z(), _z(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzyw() const { return vec<4, T>{_z(), _z(), _y(), _w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzzx() const { return vec<4, T>{_z(), _z(), _z(), _x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzzy() const { return vec<4, T>{_z(), _z(), _z(), _y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzzz() const { return vec<4, T>{_z(), _z(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzzw() const { return vec<4, T>{_z(), _z(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzwx() const { return vec<4, T>{_z(), _z(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzwy() const { return vec<4, T>{_z(), _z(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzwz() const { return vec<4, T>{_z(), _z(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzww() const { return vec<4, T>{_z(), _z(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxx() const { return vec<4, T>{_z(), _w(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxy() const { return vec<4, T>{_z(), _w(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxz() const { return vec<4, T>{_z(), _w(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxw() const { return vec<4, T>{_z(), _w(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyx() const { return vec<4, T>{_z(), _w(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyy() const { return vec<4, T>{_z(), _w(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyz() const { return vec<4, T>{_z(), _w(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyw() const { return vec<4, T>{_z(), _w(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzx() const { return vec<4, T>{_z(), _w(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzy() const { return vec<4, T>{_z(), _w(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzz() const { return vec<4, T>{_z(), _w(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzw() const { return vec<4, T>{_z(), _w(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwwx() const { return vec<4, T>{_z(), _w(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwwy() const { return vec<4, T>{_z(), _w(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwwz() const { return vec<4, T>{_z(), _w(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwww() const { return vec<4, T>{_z(), _w(), _w(), _w()}; }

        template<ASSURE_SIZE(4)> vec<4, T> wxxx() const { return vec<4, T>{_w(), _x(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxxy() const { return vec<4, T>{_w(), _x(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxxz() const { return vec<4, T>{_w(), _x(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxxw() const { return vec<4, T>{_w(), _x(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyx() const { return vec<4, T>{_w(), _x(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyy() const { return vec<4, T>{_w(), _x(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyz() const { return vec<4, T>{_w(), _x(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyw() const { return vec<4, T>{_w(), _x(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzx() const { return vec<4, T>{_w(), _x(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzy() const { return vec<4, T>{_w(), _x(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzz() const { return vec<4, T>{_w(), _x(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzw() const { return vec<4, T>{_w(), _x(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxwx() const { return vec<4, T>{_w(), _x(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxwy() const { return vec<4, T>{_w(), _x(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxwz() const { return vec<4, T>{_w(), _x(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxww() const { return vec<4, T>{_w(), _x(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxx() const { return vec<4, T>{_w(), _y(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxy() const { return vec<4, T>{_w(), _y(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxz() const { return vec<4, T>{_w(), _y(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxw() const { return vec<4, T>{_w(), _y(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyx() const { return vec<4, T>{_w(), _y(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyy() const { return vec<4, T>{_w(), _y(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyz() const { return vec<4, T>{_w(), _y(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyw() const { return vec<4, T>{_w(), _y(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzx() const { return vec<4, T>{_w(), _y(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzy() const { return vec<4, T>{_w(), _y(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzz() const { return vec<4, T>{_w(), _y(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzw() const { return vec<4, T>{_w(), _y(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wywx() const { return vec<4, T>{_w(), _y(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wywy() const { return vec<4, T>{_w(), _y(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wywz() const { return vec<4, T>{_w(), _y(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyww() const { return vec<4, T>{_w(), _y(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxx() const { return vec<4, T>{_w(), _z(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxy() const { return vec<4, T>{_w(), _z(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxz() const { return vec<4, T>{_w(), _z(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxw() const { return vec<4, T>{_w(), _z(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyx() const { return vec<4, T>{_w(), _z(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyy() const { return vec<4, T>{_w(), _z(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyz() const { return vec<4, T>{_w(), _z(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyw() const { return vec<4, T>{_w(), _z(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzx() const { return vec<4, T>{_w(), _z(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzy() const { return vec<4, T>{_w(), _z(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzz() const { return vec<4, T>{_w(), _z(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzw() const { return vec<4, T>{_w(), _z(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzwx() const { return vec<4, T>{_w(), _z(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzwy() const { return vec<4, T>{_w(), _z(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzwz() const { return vec<4, T>{_w(), _z(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzww() const { return vec<4, T>{_w(), _z(), _w(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxx() const { return vec<4, T>{_w(), _w(), _x(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxy() const { return vec<4, T>{_w(), _w(), _x(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxz() const { return vec<4, T>{_w(), _w(), _x(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxw() const { return vec<4, T>{_w(), _w(), _x(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyx() const { return vec<4, T>{_w(), _w(), _y(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyy() const { return vec<4, T>{_w(), _w(), _y(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyz() const { return vec<4, T>{_w(), _w(), _y(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyw() const { return vec<4, T>{_w(), _w(), _y(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzx() const { return vec<4, T>{_w(), _w(), _z(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzy() const { return vec<4, T>{_w(), _w(), _z(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzz() const { return vec<4, T>{_w(), _w(), _z(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzw() const { return vec<4, T>{_w(), _w(), _z(), _w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwwx() const { return vec<4, T>{_w(), _w(), _w(), _x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwwy() const { return vec<4, T>{_w(), _w(), _w(), _y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwwz() const { return vec<4, T>{_w(), _w(), _w(), _z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwww() const { return vec<4, T>{_w(), _w(), _w(), _w()}; }

        template<ASSURE_SIZE(3)> vec(const vec<2, T>& v, const T& z) : vec_storage<S, T>{v._x(), v._y(), z} {}
        template<ASSURE_SIZE(3)> vec(const T& x, const vec<2, T>& v) : vec_storage<S, T>{x, v._x(), v._y()} {}
        template<ASSURE_SIZE(4)> vec(const vec<2, T>& v1, const vec<2, T>& v2) : vec_storage<S, T>{v1._x(), v1._y(), v2._x(), v2._y()} {}
        template<ASSURE_SIZE(4)> vec(const vec<2, T>& v, const T& z, const T& w) : vec_storage<S, T>{v._x(), v._y(), z, w} {}
        template<ASSURE_SIZE(4)> vec(const T& x, const vec<2, T>& v, const T& w) : vec_storage<S, T>{x, v._x(), v._y(), w} {}
        template<ASSURE_SIZE(4)> vec(const T& x, const T& y, const vec<2, T>& v) : vec_storage<S, T>{x, y, v._x(), v._y()} {}
        template<ASSURE_SIZE(4)> vec(const vec<3, T>& v, const T& w) : vec_storage<S, T>{v._x(), v._y(), v._z(), w} {}
        template<ASSURE_SIZE(4)> vec(const T& x, const vec<3, T>& v) : vec_storage<S, T>{x, v._x(), v._y(), v._z()} {}
#endif

        vec(const vec<S, T>& v) : vec_storage<S, T>(static_cast<const vec_storage<S, T>&>(v)) {}
        vec(vec<S, T>&& v) : vec_storage<S, T>(static_cast<vec_storage<S, T>&&>(v)) {}
        vec& operator=(const vec<S, T>& v) {
            if (this != &v) {
                this->~vec();
                new (this) vec(v);
            }
            return *this;
        }
        vec& operator=(vec<S, T>&& v) {
            if (this != &v) {
                this->~vec();
                new (this) vec(v);
            }
            return *this;
        }

        template<class ... Ts, ASSURE_SIZE(5)>
        vec(const Ts ... xs) {
            static_assert(sizeof...(Ts) + 1 == S, "Incorrect number of arguments to vec constructor");
            size_t i = 0;
            ((data()[i++] = xs), ...);
        }

        template<ASSURE_EXACT_SIZE(2)>
        vec(const T x, const T y) {
            _x() = x;
            _y() = y;
        }
        template<ASSURE_EXACT_SIZE(3)>
        vec(const T x, const T y, const T z) {
            _x() = x;
            _y() = y;
            _z() = z;
        }
        template<ASSURE_EXACT_SIZE(4)>
        vec(const T x, const T y, const T z, const T w) {
            _x() = x;
            _y() = y;
            _z() = z;
            _w() = w;
        }

        vec(const T& k = T{}) : vec_storage<S, T>(k) {}

        explicit vec(const T* k) {
            memcpy(data(), k, S * sizeof(T));
        }

        T& operator[](const size_t i) { return vec_storage<S, T>::operator[](i); }
        const T& operator[](const size_t i) const { return vec_storage<S, T>::operator[](i); }

        vec<S, T> operator+(const vec<S, T>& v) const {
            vec<S, T> res;
            add(data(), v.data(), res.data(), IntList<S>{});
            return res;
        }
        vec<S, T> operator-(const vec<S, T>& v) const {
            vec<S, T> res{};
            sub(data(), v.data(), res.data(), IntList<S>{});
            return res;
        }
        vec<S, T> operator-() const {
            return vec<S, T>{} - *this;
        }
        vec<S, T> operator*(const vec<S, T>& v) const {
            vec<S, T> res{};
            mul(data(), v.data(), res.data(), IntList<S>{});
            return res;
        }
        vec<S, T> operator/(const vec<S, T>& v) const {
            vec<S, T> res{};
            div(data(), v.data(), res.data(), IntList<S>{});
            return res;
        }
        vec<S, T> operator%(const T& k) const {
            vec<S, T> res;
            mod(data(), k, res.data(), IntList<S>{});
            return res;
        }
        bool operator==(const vec<S, T>& v) const {
            return eq(data(), v.data(), IntList<S>{});
        }
        bool operator!=(const vec<S, T>& v) const {
            return !eq(data(), v.data(), IntList<S>{});
        }

        vec<S, T>& operator+=(const vec<S, T>& v) {
            add(data(), v.data(), data(), IntList<S>{});
            return *this;
        }
        vec<S, T>& operator-=(const vec<S, T>& v) {
            sub(data(), v.data(), data(), IntList<S>{});
            return *this;
        }
        vec<S, T>& operator*=(const vec<S, T>& v) {
            mul(data(), v.data(), data(), IntList<S>{});
            return *this;
        }
        vec<S, T>& operator/=(const vec<S, T>& v) {
            div(data(), v.data(), data(), IntList<S>{});
            return *this;
        }
        vec<S, T>& operator%=(const T& k) {
            mod(data(), k, data(), IntList<S>{});
            return *this;
        }

        friend vec<S, T> operator+(const T& l, const vec<S, T>& r) {
            return vec<S, T>{l} + r;
        }
        friend vec<S, T> operator-(const T& l, const vec<S, T>& r) {
            return vec<S, T>{l} - r;
        }
        friend vec<S, T> operator*(const T& l, const vec<S, T>& r) {
            return vec<S, T>{l} * r;
        }
        friend vec<S, T> operator/(const T& l, const vec<S, T>& r) {
            return vec<S, T>{l} / r;
        }

        T* begin() { return data(); }
        T* end() { return data() + S; }

        const T* begin() const { return data(); }
        const T* end() const{ return data() + S; }

        friend std::ostream& operator<<(std::ostream& os, const vec<S, T>& v) {
            os << "(";
            for (size_t i = 0; i < S; i++) {
                os << v._data[i];
                if (i != S - 1)
                    os << ", ";
            }
            os << ")";
            return os;
        }
        friend std::istream& operator>>(std::istream& is, vec<S, T>& v) {
            for (size_t i = 0; i < S; i++)
                is >> v._data[i];
            return is;
        }

        /**
         * @brief Calculate the dot product between this vector and another
         * 
         * @param v The second vector in the dot product operation
         */
        T dot(const vec<S, T>& v) const {
            return real_dot(data(), v.data(), IntList<S>{});
        }

        /**
         * @brief Calculate the length of the vector
         */
        T length() const {
            return std::sqrt(this->dot(*this));
        }

        /**
         * @brief Calculate the squared length of the vector (faster than the actual length, useful for fast comparisons)
         */
        T length_squared() const {
            return this->dot(*this);
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

        /**
         * @brief Return the a vector with all values minimum between the two
         * 
         * @param v1 The first vector
         * @param v2 The second vector
         */
        static vec<S, T> max(const vec<S, T>& v1, const vec<S, T>& v2) {
            vec<S, T> res;
            max(v1.data(), v2.data(), res.data(), IntList<S>{});
            return res;
        }

        /**
         * @brief Return the a vector with all values maximum between the two
         * 
         * @param v1 The first vector
         * @param v2 The second vector
         */
        static vec<S, T> min(const vec<S, T>& v1, const vec<S, T>& v2) {
            vec<S, T> res;
            min(v1.data(), v2.data(), res.data(), IntList<S>{});
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

        /**
         * @brief Perform a linear interpolation frmo this vector to another destination vector
         * 
         * @param destination The vector to interpolate towards
         * @param weight The amount to interpolate by
         * @return The result of the interpolation
         */
        vec<S, T> lerp(const vec<S, T>& destination, T weight) const {
            return *this + weight * (destination - *this);
        }
    };

#if (defined(__x86_64) || defined(__amd64) || defined(_M_X64) || defined(_M_AMD64)) && defined(MGMATH_SIMD)
    template<>
    inline vec<2, float>::vec(const float& k) {
        const __m128 a = _mm_set1_ps(k);
        _mm_storel_pi(reinterpret_cast<__m64*>(data()), a);
    }
    template<>
    inline vec<4, float>::vec(const float& k) {
        const __m128 a = _mm_set1_ps(k);
        _mm_store_ps(data(), a);
    }

    template<>
    inline vec<2, float> vec<2, float>::operator+(const vec<2, float>& v) const {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_add_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data()), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::operator-(const vec<2, float>& v) const {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_sub_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data()), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::operator*(const vec<2, float>& v) const {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_mul_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data()), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::operator/(const vec<2, float>& v) const {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_div_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data()), res);
        return r;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator+=(const vec<2, float>& v) {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_add_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data()), res);
        return *this;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator-=(const vec<2, float>& v) {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_sub_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data()), res);
        return *this;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator*=(const vec<2, float>& v) {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_mul_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data()), res);
        return *this;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator/=(const vec<2, float>& v) {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data()));
        const __m128 res = _mm_div_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data()), res);
        return *this;
    }

    template<>
    inline vec<2, float> vec<2, float>::max(const vec<2UL, float> &v1, const vec<2UL, float> &v2) {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v1.data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v2.data()));
        const __m128 res = _mm_max_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data()), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::min(const vec<2UL, float> &v1, const vec<2UL, float> &v2) {
        const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v1.data()));
        const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v2.data()));
        const __m128 res = _mm_min_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data()), res);
        return r;
    }


    template<>
    inline vec<3, float> vec<3, float>::operator+(const vec<3, float>& v) const {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_add_ps(a, b);
        vec<3, float> r;
        memcpy(r.data(), &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::operator-(const vec<3, float>& v) const {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_sub_ps(a, b);
        vec<3, float> r;
        memcpy(r.data(), &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::operator*(const vec<3, float>& v) const {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_mul_ps(a, b);
        vec<3, float> r;
        memcpy(r.data(), &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::operator/(const vec<3, float>& v) const {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_div_ps(a, b);
        vec<3, float> r;
        memcpy(r.data(), &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator+=(const vec<3, float>& v) {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_add_ps(a, b);
        memcpy(data(), &res, sizeof(float) * 3);
        return *this;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator-=(const vec<3, float>& v) {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_sub_ps(a, b);
        memcpy(data(), &res, sizeof(float) * 3);
        return *this;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator*=(const vec<3, float>& v) {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_mul_ps(a, b);
        memcpy(data(), &res, sizeof(float) * 3);
        return *this;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator/=(const vec<3, float>& v) {
        const __m128 a = _mm_set_ps(0, data()[2], data()[1], data()[0]);
        const __m128 b = _mm_set_ps(0, v.data()[2], v.data()[1], v.data()[0]);
        const __m128 res = _mm_div_ps(a, b);
        memcpy(data(), &res, sizeof(float) * 3);
        return *this;
    }

    template<>
    inline vec<3, float> vec<3, float>::max(const vec<3UL, float> &v1, const vec<3UL, float> &v2) {
        const __m128 a = _mm_set_ps(0, v1.data()[2], v1.data()[1], v1.data()[0]);
        const __m128 b = _mm_set_ps(0, v2.data()[2], v2.data()[1], v2.data()[0]);
        const __m128 res = _mm_max_ps(a, b);
        vec<3, float> r;
        memcpy(r.data(), &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::min(const vec<3UL, float> &v1, const vec<3UL, float> &v2) {
        const __m128 a = _mm_set_ps(0, v1.data()[2], v1.data()[1], v1.data()[0]);
        const __m128 b = _mm_set_ps(0, v2.data()[2], v2.data()[1], v2.data()[0]);
        const __m128 res = _mm_min_ps(a, b);
        vec<3, float> r;
        memcpy(r.data(), &res, sizeof(float) * 3);
        return r;
    }


    template<>
    inline vec<4, float> vec<4, float>::operator+(const vec<4, float>& v) const {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_add_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data(), res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::operator-(const vec<4, float>& v) const {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_sub_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data(), res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::operator*(const vec<4, float>& v) const {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_mul_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data(), res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::operator/(const vec<4, float>& v) const {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_div_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data(), res);
        return r;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator+=(const vec<4, float>& v) {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_add_ps(a, b);
        _mm_storeu_ps(data(), res);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator-=(const vec<4, float>& v) {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_sub_ps(a, b);
        _mm_storeu_ps(data(), res);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator*=(const vec<4, float>& v) {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_mul_ps(a, b);
        _mm_storeu_ps(data(), res);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator/=(const vec<4, float>& v) {
        const __m128 a = _mm_loadu_ps(data());
        const __m128 b = _mm_loadu_ps(v.data());
        const __m128 res = _mm_div_ps(a, b);
        _mm_storeu_ps(data(), res);
        return *this;
    }

    template<>
    inline vec<4, float> vec<4, float>::max(const vec<4UL, float> &v1, const vec<4UL, float> &v2) {
        const __m128 a = _mm_loadu_ps(v1.data());
        const __m128 b = _mm_loadu_ps(v2.data());
        const __m128 res = _mm_max_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data(), res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::min(const vec<4UL, float> &v1, const vec<4UL, float> &v2) {
        const __m128 a = _mm_loadu_ps(v1.data());
        const __m128 b = _mm_loadu_ps(v2.data());
        const __m128 res = _mm_min_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data(), res);
        return r;
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
            if constexpr (l == c)
                for (size_t i = 0; i < l; i++)
                    data[i][i] = x;
            else
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
            for (size_t i = 0; i < l; i++)
                for (size_t j = 0; j < c2; j++)
                    for (size_t k = 0; k < c; k++)
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
                    res[i][j] = data[i + (i >= pos.y)][j + (j >= pos.x)];
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

        template<size_t Lines = l, size_t Columns = c, class Type = T,
            typename std::enable_if<Lines == 4 && Columns == 4
            && (std::is_same<Type, float>::value || std::is_same<Type, double>::value), int>::type = 0>
        static mat<l, c, T> gen_perspective_projection(T fov, T aspect, T near, T far) {
            T tan_half_fov = std::tan(fov / T(2));
            return mat<4, 4, T>{
                T(1) / (aspect * tan_half_fov), T(0),                 T(0),                                T(0),
                T(0),                           T(1) / tan_half_fov,  T(0),                                T(0),
                T(0),                           T(0),                 -(far + near) / (far - near),        -(T(2) * far * near) / (far - near),
                T(0),                           T(0),                 T(-1), T(0)
            };
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



    //=============
    // QUATERNIONS
    //=============

    template<typename T>
    class quat : public vec<4, T> {
        public:

        using vec<4, T>::x;
        using vec<4, T>::y;
        using vec<4, T>::z;
        using vec<4, T>::w;

        using vec<4, T>::vec;

        quat() : vec<4, T>(T(0), T(0), T(0), T(1)) {}

        explicit quat(const vec<4, T>& v) : vec<4, T>(v) {}
        operator vec<4, T>() const { return this->xyzw(); }

        quat<T> operator*(const quat<T>& q) const {
            return {
                w * q.x + x * q.w + y * q.z - z * q.y,
                w * q.y + y * q.w + z * q.x - x * q.z,
                w * q.z + z * q.w + x * q.y - y * q.x,
                w * q.w - x * q.x - y * q.y - z * q.z
            };
        }
        quat<T>& operator*=(const quat<T>& q) {
            return *this = *this * q;
        }

        /**
         * @brief Calculate the quaternion's conjugate `q*`
         */
        quat<T> conjugate() const {
            return quat{-x, -y, -z, w};
        }

        /**
         * @brief Calculate the quaternion's magnitude (equivalent to length)
         */
        T norm() const {
            return vec<4, T>::length();
        }

        /**
         * @brief Calculate the quaternion's inverse `(q*) / (q.norm()^2)`
         */
        quat<T> inv() const {
            const auto len_sq = vec<4, T>::length_squared();
            if (len_sq == 0)
                throw std::runtime_error("Cannot invert zero quaternion");
            return static_cast<quat>(conjugate() / len_sq);
        }

        /**
         * @brief Rotate a vector using this quaternion
         * 
         * @param v The vector to rotate
         * @return The rotated version of the vector
         */
        vec<3, T> rotate(const vec<3, T>& v) const {
            const auto res = *this * quat<T>{v, 0.0f} * conjugate();
            return vec<3, T>{res.x, res.y, res.z};
        }

        /**
         * @brief Rotate a vector using this quaternion, making sure it is a normalized quaternion first, suitable for rotations
         * 
         * @param v The vector to rotate
         * @return The rotate version of the vector
         */
        vec<3, T> rotate_safe(const vec<3, T>& v) const {
            const auto res = *this * quat<T>{v, 0.0f} * inv();
            return vec<3, T>{res.x, res.y, res.z};
        }
        
        /**
         * @brief Generate a quaternion from an angle rotated around a given axis (normalized direction vector)
         * 
         * @param axis The axis to rotate around
         * @param angle The angle in radians to rotate by
         * @return The calculated quaternion
         */
        static inline quat<T> from_angle(const vec<3, T>& axis, T angle) {
            if (angle == 0)
                return quat<T>{0, 0, 0, 1};

            const auto ha = angle / 2;
            const auto s = std::sin(ha);
            const auto c = std::cos(ha);
            return quat<T>{axis * s, c};
        }

        /**
         * @brief Generate a quaternion from an angle rotated around a given axis, making sure the axis is a valid normalized vector, suitable for reprezenting axis
         * 
         * @param axis The axis to rotate around
         * @param angle The angle in radians to rotate by
         * @return The calculated quaternion
         */
        static inline quat<T> from_angle_safe(const vec<3, T>& axis, const T angle) {
            if (angle == T(0))
                return quat<T>{T(0), T(0), T(0), T(1)};

            if (angle > std::numbers::pi * T(2))
                angle = angle - std::numbers::pi * T(2) * std::floor(angle / (std::numbers::pi * T(2)));
            else if (angle < -std::numbers::pi * T(2))
                angle = angle + std::numbers::pi * T(2) * std::ceil(angle / (std::numbers::pi * T(2)));

            const auto ha = angle / T(2);
            const auto s = std::sin(ha);
            const auto c = std::cos(ha);
            return quat<T>{axis.normalized() * s, c};
        }

        /**
         * @brief Generate a rotation matrix from this quaternion, that will rotate a vector the same way this quaternion would
         */
        mat<4, 4, T> as_rotation_mat4() const {
            return mat<4, 4, T>{
                T(1) - T(2) * (y * y + z * z), T(2) * (x * y - z * w),     T(2) * (x * z + y * w),     T(0),
                T(2) * (x * y + z * w),     T(1) - T(2) * (x * x + z * z), T(2) * (y * z - x * w),     T(0),
                T(2) * (x * z - y * w),     T(2) * (y * z + x * w),     T(1) - T(2) * (x * x + y * y), T(0),
                T(0),                       T(0),                       T(0),                       T(1)
            };
        }
        /**
         * @brief Generate a rotation matrix from this quaternion, that will rotate a vector the same way this quaternion would
         */
        mat<3, 3, T> as_rotation_mat3() const {
            return mat<3, 3, T>{
                T(1) - T(2) * (y * y + z * z), T(2) * (x * y - z * w),     T(2) * (x * z + y * w),
                T(2) * (x * y + z * w),     T(1) - T(2) * (x * x + z * z), T(2) * (y * z - x * w),
                T(2) * (x * z - y * w),     T(2) * (y * z + x * w),     T(1) - T(2) * (x * x + y * y)
            };
        }

        /**
         * @brief Perform a spherical linear interpolation (slerp) from this quaternion to a destination quaternion
         * 
         * @param destination The destination to interpolate towards
         * @param weight The amount to interpolate by
         * @return The result of the interpolation
         */
        quat<T> slerp(quat<T> destination, T weight) const {
            auto d = this->dot(destination);

            if (d < T(0)) {
                d = -d;
                destination = -destination;
            }

            // Because of loss of precision
            static constexpr auto THRESHOLD = T(0.9995);
            if (d > THRESHOLD)
                return static_cast<quat<T>>(this->lerp(destination).normalized());

            const auto theta = std::acos(d) * weight;

            return *this * std::cos(theta) + (destination - *this * d).normalized() * std::sin(theta);
        }
    };

    using quatf = quat<float>;
    using quatd = quat<double>;
}
