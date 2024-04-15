#pragma once
#include <cmath>
#include <cstdint>
#include <memory.h>
#include <ostream>
#include <type_traits>
#include <cassert>

#if defined(__x86_64) || defined(__amd64)
#include <xmmintrin.h>
#include <smmintrin.h>
#endif

#define ASSURE_SIZE(SIZE) size_t VectorSize = S, typename std::enable_if<VectorSize >= SIZE, bool>::type = true
#define FORCE_SIZE(SIZE) size_t VectorSize = S, typename std::enable_if<VectorSize == SIZE, bool>::type = true


namespace mgm {

    //=========
    // VECTORS
    //=========


    template<size_t S, class T>
    class vec {
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

        template<typename... Ts>
        static inline void add(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = (const Ts&)a[i] + (const Ts&)b[i]), ...);
        }
        template<typename... Ts>
        static inline void sub(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = (const Ts&)a[i] - (const Ts&)b[i]), ...);
        }
        template<typename... Ts>
        static inline void mul(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = (const Ts&)a[i] * (const Ts&)b[i]), ...);
        }
        template<typename... Ts>
        static inline void div(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = (const Ts&)a[i] / (const Ts&)b[i]), ...);
        }
        template<typename... Ts>
        static inline void mod(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = (const Ts&)a[i] % (const Ts&)b[i]), ...);
        }
        template<typename... Ts>
        static inline bool eq(const T* a, const T* b, TypeList<Ts...>) {
            size_t i = 0;
            return (((Ts&)a[i] == (Ts&)b[i]) && ...);
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

        template<typename... Ts>
        static inline void max(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = ((const Ts&)a[i] > (const Ts&)b[i]) ? (const Ts&)a[i] : (const Ts&)b[i]), ...);
        }
        template<typename... Ts>
        static inline void min(const T* a, const T* b, T* r, TypeList<Ts...>) {
            size_t i = 0;
            (((Ts&)r[i++] = ((const Ts&)a[i] < (const Ts&)b[i]) ? (const Ts&)a[i] : (const Ts&)b[i]), ...);
        }

        public:
        T data[S];

        template<ASSURE_SIZE(1)>
        T& x() { return data[0]; }
        template<ASSURE_SIZE(2)>
        T& y() { return data[1]; }
        template<ASSURE_SIZE(3)>
        T& z() { return data[2]; }
        template<ASSURE_SIZE(4)>
        T& w() { return data[3]; }

        template<ASSURE_SIZE(1)>
        const T& x() const { return data[0]; }
        template<ASSURE_SIZE(2)>
        const T& y() const { return data[1]; }
        template<ASSURE_SIZE(3)>
        const T& z() const { return data[2]; }
        template<ASSURE_SIZE(4)>
        const T& w() const { return data[3]; }

#if defined(MGMATH_SWIZZLE)
        template<ASSURE_SIZE(2)> vec<2, T> xx() const { return vec<2, T>{x(), x()}; }
        template<ASSURE_SIZE(2)> vec<2, T> xy() const { return vec<2, T>{x(), y()}; }
        template<ASSURE_SIZE(3)> vec<2, T> xz() const { return vec<2, T>{x(), y()}; }
        template<ASSURE_SIZE(4)> vec<2, T> xw() const { return vec<2, T>{x(), w()}; }
        template<ASSURE_SIZE(2)> vec<2, T> yx() const { return vec<2, T>{y(), x()}; }
        template<ASSURE_SIZE(2)> vec<2, T> yy() const { return vec<2, T>{y(), y()}; }
        template<ASSURE_SIZE(3)> vec<2, T> yz() const { return vec<2, T>{y(), z()}; }
        template<ASSURE_SIZE(4)> vec<2, T> yw() const { return vec<2, T>{y(), w()}; }
        template<ASSURE_SIZE(3)> vec<2, T> zx() const { return vec<2, T>{z(), x()}; }
        template<ASSURE_SIZE(3)> vec<2, T> zy() const { return vec<2, T>{z(), y()}; }
        template<ASSURE_SIZE(3)> vec<2, T> zz() const { return vec<2, T>{z(), z()}; }
        template<ASSURE_SIZE(4)> vec<2, T> zw() const { return vec<2, T>{z(), w()}; }
        template<ASSURE_SIZE(4)> vec<2, T> wx() const { return vec<2, T>{w(), x()}; }
        template<ASSURE_SIZE(4)> vec<2, T> wy() const { return vec<2, T>{w(), y()}; }
        template<ASSURE_SIZE(4)> vec<2, T> wz() const { return vec<2, T>{w(), z()}; }
        template<ASSURE_SIZE(4)> vec<2, T> ww() const { return vec<2, T>{w(), w()}; }

        template<ASSURE_SIZE(2)> vec<3, T> xxx() const { return vec<3, T>{x(), x(), x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> xxy() const { return vec<3, T>{x(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xxz() const { return vec<3, T>{x(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xxw() const { return vec<3, T>{x(), x(), w()}; }
        template<ASSURE_SIZE(2)> vec<3, T> xyx() const { return vec<3, T>{x(), y(), x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> xyy() const { return vec<3, T>{x(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xyz() const { return vec<3, T>{x(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xyw() const { return vec<3, T>{x(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xzx() const { return vec<3, T>{x(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xzy() const { return vec<3, T>{x(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> xzz() const { return vec<3, T>{x(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xzw() const { return vec<3, T>{x(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xwx() const { return vec<3, T>{x(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xwy() const { return vec<3, T>{x(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xwz() const { return vec<3, T>{x(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> xww() const { return vec<3, T>{x(), w(), w()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yxx() const { return vec<3, T>{y(), x(), x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yxy() const { return vec<3, T>{y(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yxz() const { return vec<3, T>{y(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yxw() const { return vec<3, T>{y(), x(), w()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yyx() const { return vec<3, T>{y(), y(), x()}; }
        template<ASSURE_SIZE(2)> vec<3, T> yyy() const { return vec<3, T>{y(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yyz() const { return vec<3, T>{y(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yyw() const { return vec<3, T>{y(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yzx() const { return vec<3, T>{y(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yzy() const { return vec<3, T>{y(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> yzz() const { return vec<3, T>{y(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yzw() const { return vec<3, T>{y(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> ywx() const { return vec<3, T>{y(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> ywy() const { return vec<3, T>{y(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> ywz() const { return vec<3, T>{y(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> yww() const { return vec<3, T>{y(), w(), w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zxx() const { return vec<3, T>{z(), x(), x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zxy() const { return vec<3, T>{z(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zxz() const { return vec<3, T>{z(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zxw() const { return vec<3, T>{z(), x(), w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zyx() const { return vec<3, T>{z(), y(), x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zyy() const { return vec<3, T>{z(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zyz() const { return vec<3, T>{z(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zyw() const { return vec<3, T>{z(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zzx() const { return vec<3, T>{z(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zzy() const { return vec<3, T>{z(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<3, T> zzz() const { return vec<3, T>{z(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zzw() const { return vec<3, T>{z(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zwx() const { return vec<3, T>{z(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zwy() const { return vec<3, T>{z(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zwz() const { return vec<3, T>{z(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> zww() const { return vec<3, T>{z(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxx() const { return vec<3, T>{w(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxy() const { return vec<3, T>{w(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxz() const { return vec<3, T>{w(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wxw() const { return vec<3, T>{w(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyx() const { return vec<3, T>{w(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyy() const { return vec<3, T>{w(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyz() const { return vec<3, T>{w(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wyw() const { return vec<3, T>{w(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzx() const { return vec<3, T>{w(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzy() const { return vec<3, T>{w(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzz() const { return vec<3, T>{w(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wzw() const { return vec<3, T>{w(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wwx() const { return vec<3, T>{w(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wwy() const { return vec<3, T>{w(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<3, T> wwz() const { return vec<3, T>{w(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<3, T> www() const { return vec<3, T>{w(), w(), w()}; }

        template<ASSURE_SIZE(2)> vec<4, T> xxxx() const { return vec<4, T>{x(), x(), x(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xxxy() const { return vec<4, T>{x(), x(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxxz() const { return vec<4, T>{x(), x(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxxw() const { return vec<4, T>{x(), x(), x(), w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xxyx() const { return vec<4, T>{x(), x(), y(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xxyy() const { return vec<4, T>{x(), x(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxyz() const { return vec<4, T>{x(), x(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxyw() const { return vec<4, T>{x(), x(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxzx() const { return vec<4, T>{x(), x(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxzy() const { return vec<4, T>{x(), x(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xxzz() const { return vec<4, T>{x(), x(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxzw() const { return vec<4, T>{x(), x(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxwx() const { return vec<4, T>{x(), x(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxwy() const { return vec<4, T>{x(), x(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxwz() const { return vec<4, T>{x(), x(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xxww() const { return vec<4, T>{x(), x(), w(), w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyxx() const { return vec<4, T>{x(), y(), x(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyxy() const { return vec<4, T>{x(), y(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyxz() const { return vec<4, T>{x(), y(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyxw() const { return vec<4, T>{x(), y(), x(), w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyyx() const { return vec<4, T>{x(), y(), y(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> xyyy() const { return vec<4, T>{x(), y(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyyz() const { return vec<4, T>{x(), y(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyyw() const { return vec<4, T>{x(), y(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyzx() const { return vec<4, T>{x(), y(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyzy() const { return vec<4, T>{x(), y(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xyzz() const { return vec<4, T>{x(), y(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyzw() const { return vec<4, T>{x(), y(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xywx() const { return vec<4, T>{x(), y(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xywy() const { return vec<4, T>{x(), y(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xywz() const { return vec<4, T>{x(), y(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xyww() const { return vec<4, T>{x(), y(), w(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzxx() const { return vec<4, T>{x(), z(), x(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzxy() const { return vec<4, T>{x(), z(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzxz() const { return vec<4, T>{x(), z(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzxw() const { return vec<4, T>{x(), z(), x(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzyx() const { return vec<4, T>{x(), z(), y(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzyy() const { return vec<4, T>{x(), z(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzyz() const { return vec<4, T>{x(), z(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzyw() const { return vec<4, T>{x(), z(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzzx() const { return vec<4, T>{x(), z(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzzy() const { return vec<4, T>{x(), z(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> xzzz() const { return vec<4, T>{x(), z(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzzw() const { return vec<4, T>{x(), z(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzwx() const { return vec<4, T>{x(), z(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzwy() const { return vec<4, T>{x(), z(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzwz() const { return vec<4, T>{x(), z(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xzww() const { return vec<4, T>{x(), z(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxx() const { return vec<4, T>{x(), w(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxy() const { return vec<4, T>{x(), w(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxz() const { return vec<4, T>{x(), w(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwxw() const { return vec<4, T>{x(), w(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyx() const { return vec<4, T>{x(), w(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyy() const { return vec<4, T>{x(), w(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyz() const { return vec<4, T>{x(), w(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwyw() const { return vec<4, T>{x(), w(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzx() const { return vec<4, T>{x(), w(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzy() const { return vec<4, T>{x(), w(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzz() const { return vec<4, T>{x(), w(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwzw() const { return vec<4, T>{x(), w(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwwx() const { return vec<4, T>{x(), w(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwwy() const { return vec<4, T>{x(), w(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwwz() const { return vec<4, T>{x(), w(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> xwww() const { return vec<4, T>{x(), w(), w(), w()}; }

        template<ASSURE_SIZE(2)> vec<4, T> yxxx() const { return vec<4, T>{y(), x(), x(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yxxy() const { return vec<4, T>{y(), x(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxxz() const { return vec<4, T>{y(), x(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxxw() const { return vec<4, T>{y(), x(), x(), w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yxyx() const { return vec<4, T>{y(), x(), y(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yxyy() const { return vec<4, T>{y(), x(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxyz() const { return vec<4, T>{y(), x(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxyw() const { return vec<4, T>{y(), x(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxzx() const { return vec<4, T>{y(), x(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxzy() const { return vec<4, T>{y(), x(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yxzz() const { return vec<4, T>{y(), x(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxzw() const { return vec<4, T>{y(), x(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxwx() const { return vec<4, T>{y(), x(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxwy() const { return vec<4, T>{y(), x(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxwz() const { return vec<4, T>{y(), x(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yxww() const { return vec<4, T>{y(), x(), w(), w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyxx() const { return vec<4, T>{y(), y(), x(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyxy() const { return vec<4, T>{y(), y(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyxz() const { return vec<4, T>{y(), y(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyxw() const { return vec<4, T>{y(), y(), x(), w()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyyx() const { return vec<4, T>{y(), y(), y(), x()}; }
        template<ASSURE_SIZE(2)> vec<4, T> yyyy() const { return vec<4, T>{y(), y(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyyz() const { return vec<4, T>{y(), y(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyyw() const { return vec<4, T>{y(), y(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyzx() const { return vec<4, T>{y(), y(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyzy() const { return vec<4, T>{y(), y(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yyzz() const { return vec<4, T>{y(), y(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyzw() const { return vec<4, T>{y(), y(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yywx() const { return vec<4, T>{y(), y(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yywy() const { return vec<4, T>{y(), y(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yywz() const { return vec<4, T>{y(), y(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yyww() const { return vec<4, T>{y(), y(), w(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzxx() const { return vec<4, T>{y(), z(), x(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzxy() const { return vec<4, T>{y(), z(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzxz() const { return vec<4, T>{y(), z(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzxw() const { return vec<4, T>{y(), z(), x(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzyx() const { return vec<4, T>{y(), z(), y(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzyy() const { return vec<4, T>{y(), z(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzyz() const { return vec<4, T>{y(), z(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzyw() const { return vec<4, T>{y(), z(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzzx() const { return vec<4, T>{y(), z(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzzy() const { return vec<4, T>{y(), z(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> yzzz() const { return vec<4, T>{y(), z(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzzw() const { return vec<4, T>{y(), z(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzwx() const { return vec<4, T>{y(), z(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzwy() const { return vec<4, T>{y(), z(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzwz() const { return vec<4, T>{y(), z(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> yzww() const { return vec<4, T>{y(), z(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxx() const { return vec<4, T>{y(), w(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxy() const { return vec<4, T>{y(), w(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxz() const { return vec<4, T>{y(), w(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywxw() const { return vec<4, T>{y(), w(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyx() const { return vec<4, T>{y(), w(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyy() const { return vec<4, T>{y(), w(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyz() const { return vec<4, T>{y(), w(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywyw() const { return vec<4, T>{y(), w(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzx() const { return vec<4, T>{y(), w(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzy() const { return vec<4, T>{y(), w(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzz() const { return vec<4, T>{y(), w(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywzw() const { return vec<4, T>{y(), w(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywwx() const { return vec<4, T>{y(), w(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywwy() const { return vec<4, T>{y(), w(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywwz() const { return vec<4, T>{y(), w(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> ywww() const { return vec<4, T>{y(), w(), w(), w()}; }

        template<ASSURE_SIZE(3)> vec<4, T> zxxx() const { return vec<4, T>{z(), x(), x(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxxy() const { return vec<4, T>{z(), x(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxxz() const { return vec<4, T>{z(), x(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxxw() const { return vec<4, T>{z(), x(), x(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxyx() const { return vec<4, T>{z(), x(), y(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxyy() const { return vec<4, T>{z(), x(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxyz() const { return vec<4, T>{z(), x(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxyw() const { return vec<4, T>{z(), x(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxzx() const { return vec<4, T>{z(), x(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxzy() const { return vec<4, T>{z(), x(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zxzz() const { return vec<4, T>{z(), x(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxzw() const { return vec<4, T>{z(), x(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxwx() const { return vec<4, T>{z(), x(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxwy() const { return vec<4, T>{z(), x(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxwz() const { return vec<4, T>{z(), x(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zxww() const { return vec<4, T>{z(), x(), w(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyxx() const { return vec<4, T>{z(), y(), x(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyxy() const { return vec<4, T>{z(), y(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyxz() const { return vec<4, T>{z(), y(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyxw() const { return vec<4, T>{z(), y(), x(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyyx() const { return vec<4, T>{z(), y(), y(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyyy() const { return vec<4, T>{z(), y(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyyz() const { return vec<4, T>{z(), y(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyyw() const { return vec<4, T>{z(), y(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyzx() const { return vec<4, T>{z(), y(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyzy() const { return vec<4, T>{z(), y(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zyzz() const { return vec<4, T>{z(), y(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyzw() const { return vec<4, T>{z(), y(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zywx() const { return vec<4, T>{z(), y(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zywy() const { return vec<4, T>{z(), y(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zywz() const { return vec<4, T>{z(), y(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zyww() const { return vec<4, T>{z(), y(), w(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzxx() const { return vec<4, T>{z(), z(), x(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzxy() const { return vec<4, T>{z(), z(), x(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzxz() const { return vec<4, T>{z(), z(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzxw() const { return vec<4, T>{z(), z(), x(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzyx() const { return vec<4, T>{z(), z(), y(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzyy() const { return vec<4, T>{z(), z(), y(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzyz() const { return vec<4, T>{z(), z(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzyw() const { return vec<4, T>{z(), z(), y(), w()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzzx() const { return vec<4, T>{z(), z(), z(), x()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzzy() const { return vec<4, T>{z(), z(), z(), y()}; }
        template<ASSURE_SIZE(3)> vec<4, T> zzzz() const { return vec<4, T>{z(), z(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzzw() const { return vec<4, T>{z(), z(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzwx() const { return vec<4, T>{z(), z(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzwy() const { return vec<4, T>{z(), z(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzwz() const { return vec<4, T>{z(), z(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zzww() const { return vec<4, T>{z(), z(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxx() const { return vec<4, T>{z(), w(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxy() const { return vec<4, T>{z(), w(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxz() const { return vec<4, T>{z(), w(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwxw() const { return vec<4, T>{z(), w(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyx() const { return vec<4, T>{z(), w(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyy() const { return vec<4, T>{z(), w(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyz() const { return vec<4, T>{z(), w(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwyw() const { return vec<4, T>{z(), w(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzx() const { return vec<4, T>{z(), w(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzy() const { return vec<4, T>{z(), w(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzz() const { return vec<4, T>{z(), w(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwzw() const { return vec<4, T>{z(), w(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwwx() const { return vec<4, T>{z(), w(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwwy() const { return vec<4, T>{z(), w(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwwz() const { return vec<4, T>{z(), w(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> zwww() const { return vec<4, T>{z(), w(), w(), w()}; }

        template<ASSURE_SIZE(4)> vec<4, T> wxxx() const { return vec<4, T>{w(), x(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxxy() const { return vec<4, T>{w(), x(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxxz() const { return vec<4, T>{w(), x(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxxw() const { return vec<4, T>{w(), x(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyx() const { return vec<4, T>{w(), x(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyy() const { return vec<4, T>{w(), x(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyz() const { return vec<4, T>{w(), x(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxyw() const { return vec<4, T>{w(), x(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzx() const { return vec<4, T>{w(), x(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzy() const { return vec<4, T>{w(), x(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzz() const { return vec<4, T>{w(), x(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxzw() const { return vec<4, T>{w(), x(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxwx() const { return vec<4, T>{w(), x(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxwy() const { return vec<4, T>{w(), x(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxwz() const { return vec<4, T>{w(), x(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wxww() const { return vec<4, T>{w(), x(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxx() const { return vec<4, T>{w(), y(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxy() const { return vec<4, T>{w(), y(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxz() const { return vec<4, T>{w(), y(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyxw() const { return vec<4, T>{w(), y(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyx() const { return vec<4, T>{w(), y(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyy() const { return vec<4, T>{w(), y(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyz() const { return vec<4, T>{w(), y(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyyw() const { return vec<4, T>{w(), y(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzx() const { return vec<4, T>{w(), y(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzy() const { return vec<4, T>{w(), y(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzz() const { return vec<4, T>{w(), y(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyzw() const { return vec<4, T>{w(), y(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wywx() const { return vec<4, T>{w(), y(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wywy() const { return vec<4, T>{w(), y(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wywz() const { return vec<4, T>{w(), y(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wyww() const { return vec<4, T>{w(), y(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxx() const { return vec<4, T>{w(), z(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxy() const { return vec<4, T>{w(), z(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxz() const { return vec<4, T>{w(), z(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzxw() const { return vec<4, T>{w(), z(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyx() const { return vec<4, T>{w(), z(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyy() const { return vec<4, T>{w(), z(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyz() const { return vec<4, T>{w(), z(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzyw() const { return vec<4, T>{w(), z(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzx() const { return vec<4, T>{w(), z(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzy() const { return vec<4, T>{w(), z(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzz() const { return vec<4, T>{w(), z(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzzw() const { return vec<4, T>{w(), z(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzwx() const { return vec<4, T>{w(), z(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzwy() const { return vec<4, T>{w(), z(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzwz() const { return vec<4, T>{w(), z(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wzww() const { return vec<4, T>{w(), z(), w(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxx() const { return vec<4, T>{w(), w(), x(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxy() const { return vec<4, T>{w(), w(), x(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxz() const { return vec<4, T>{w(), w(), x(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwxw() const { return vec<4, T>{w(), w(), x(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyx() const { return vec<4, T>{w(), w(), y(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyy() const { return vec<4, T>{w(), w(), y(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyz() const { return vec<4, T>{w(), w(), y(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwyw() const { return vec<4, T>{w(), w(), y(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzx() const { return vec<4, T>{w(), w(), z(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzy() const { return vec<4, T>{w(), w(), z(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzz() const { return vec<4, T>{w(), w(), z(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwzw() const { return vec<4, T>{w(), w(), z(), w()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwwx() const { return vec<4, T>{w(), w(), w(), x()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwwy() const { return vec<4, T>{w(), w(), w(), y()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwwz() const { return vec<4, T>{w(), w(), w(), z()}; }
        template<ASSURE_SIZE(4)> vec<4, T> wwww() const { return vec<4, T>{w(), w(), w(), w()}; }

        template<ASSURE_SIZE(3)> vec(const vec<2, T>& v, const T& z) : data{{v.x(), v.y(), z}} {}
        template<ASSURE_SIZE(3)> vec(const T& x, const vec<2, T>& v) : data{{x, v.x(), v.y()}} {}
        template<ASSURE_SIZE(4)> vec(const vec<2, T>& v1, const vec<2, T>& v2) : data{{v1.x(), v1.y(), v2.x(), v2.y()}} {}
        template<ASSURE_SIZE(4)> vec(const vec<2, T>& v, const T& z, const T& w) : data{{v.x(), v.y(), z, w}} {}
        template<ASSURE_SIZE(4)> vec(const T& x, const vec<2, T>& v, const T& w) : data{{x, v.x(), v.y(), w}} {}
        template<ASSURE_SIZE(4)> vec(const T& x, const T& y, const vec<2, T>& v) : data{{x, y, v.x(), v.y()}} {}
        template<ASSURE_SIZE(4)> vec(const vec<3, T>& v, const T& w) : data{{v.x(), v.y(), v.z(), w}} {}
        template<ASSURE_SIZE(4)> vec(const T& x, const vec<3, T>& v) : data{{x, v.x(), v.y(), v.z()}} {}
#endif

        vec(const vec<S, T>& v) { memcpy(data, v.data, S * sizeof(T)); }
        vec(vec<S, T>&& v) { memcpy(data, v.data, S * sizeof(T)); }
        vec& operator=(const vec<S, T>& v) {
            if (this != &v)
                memcpy(data, v.data, S * sizeof(T));
            return *this;
        }
        vec& operator=(vec<S, T>&& v) {
            if (this != &v)
                memcpy(data, v.data, S * sizeof(T));
            return *this;
        }

        template<class ... Ts, ASSURE_SIZE(5)>
        vec(const Ts ... xs) {
            static_assert(sizeof...(Ts) + 1 == S, "Incorrect number of arguments to vec constructor");
            size_t i = 0;
            ((data[i++] = xs), ...);
        }

        template<FORCE_SIZE(2)>
        vec(const T x, const T y) {
            this->x() = x;
            this->y() = y;
        }
        template<FORCE_SIZE(3)>
        vec(const T x, const T y, const T z) {
            this->x() = x;
            this->y() = y;
            this->z() = z;
        }
        template<FORCE_SIZE(4)>
        vec(const T x, const T y, const T z, const T w) {
            this->x() = x;
            this->y() = y;
            this->z() = z;
            this->w() = w;
        }

        vec(const T& k = T{}) {
            for (T& p : data)
                p = k;
        }

        explicit vec(const T* k) {
            memcpy(data, k, S * sizeof(T));
        }

        T& operator[](const size_t i) {
#if !defined(NDEBUG)
            if (i < S)
#endif
                return data[i];
#if !defined(NDEBUG)
            assert(false && "Index out of bounds");
            return data[0];
#endif
        }

        const T& operator[](const size_t i) const {
#if !defined(NDEBUG)
            if (i < S)
#endif
                return data[i];
#if !defined(NDEBUG)
            assert(false && "Index out of bounds");
            return data[0];
#endif
        }

        vec<S, T> operator+(const vec<S, T>& v) const {
            vec<S, T> res;
            add(data, v.data, res.data, IntList<S>{});
            return res;
        }
        vec<S, T> operator-(const vec<S, T>& v) const {
            vec<S, T> res{};
            sub(data, v.data, res.data, IntList<S>{});
            return res;
        }
        vec<S, T> operator*(const vec<S, T>& v) const {
            vec<S, T> res{};
            mul(data, v.data, res.data, IntList<S>{});
            return res;
        }
        vec<S, T> operator/(const vec<S, T>& v) const {
            vec<S, T> res{};
            div(data, v.data, res.data, IntList<S>{});
            return res;
        }
        vec<S, T> operator%(const T& k) const {
            vec<S, T> res;
            mod(data, k, res.data, IntList<S>{});
            return res;
        }
        bool operator==(const vec<S, T>& v) const {
            return eq(data, v.data, IntList<S>{});
        }
        bool operator!=(const vec<S, T>& v) const {
            return !eq(data, v.data, IntList<S>{});
        }

        vec<S, T>& operator+=(const vec<S, T>& v) {
            add(data, v.data, data, IntList<S>{});
            return *this;
        }
        vec<S, T>& operator-=(const vec<S, T>& v) {
            sub(data, v.data, data, IntList<S>{});
            return *this;
        }
        vec<S, T>& operator*=(const vec<S, T>& v) {
            mul(data, v.data, data, IntList<S>{});
            return *this;
        }
        vec<S, T>& operator/=(const vec<S, T>& v) {
            div(data, v.data, data, IntList<S>{});
            return *this;
        }
        vec<S, T>& operator%=(const T& k) {
            mod(data, k, data, IntList<S>{});
            return *this;
        }

        T* begin() { return data; }
        T* end() { return data + S; }

        const T* begin() const { return data; }
        const T* end() const{ return data + S; }

        friend std::ostream& operator<<(std::ostream& os, const vec<S, T>& v) {
            os << "(";
            for (size_t i = 0; i < S; i++) {
                os << v.data[i];
                if (i != S - 1)
                    os << ", ";
            }
            os << ")";
            return os;
        }
        friend std::istream& operator>>(std::istream& is, vec<S, T>& v) {
            for (size_t i = 0; i < S; i++)
                is >> v.data[i];
            return is;
        }

        /**
         * @brief Calculate the dot product between this vector and another
         * 
         * @param v The second vector in the dot product operation
         */
        T dot(const vec<S, T>& v) const {
            return real_dot(data, v.data, IntList<S>{});
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

        /**
         * @brief Return the a vector with all values minimum between the two
         * 
         * @param v1 The first vector
         * @param v2 The second vector
         */
        static vec<S, T> max(const vec<S, T>& v1, const vec<S, T>& v2) {
            vec<S, T> res;
            max(v1.data, v2.data, res.data, IntList<S>{});
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
            min(v1.data, v2.data, res.data, IntList<S>{});
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

#if (defined(__x86_64) || defined(__amd64)) && defined(MGMATH_SIMD)
    template<>
    inline vec<2, float>::vec(const float& k) {
        __m128 a = _mm_set1_ps(k);
        _mm_storel_pi(reinterpret_cast<__m64*>(data), a);
    }
    template<>
    inline vec<4, float>::vec(const float& k) {
        __m128 a = _mm_set1_ps(k);
        _mm_store_ps(data, a);
    }

    template<>
    inline vec<2, float> vec<2, float>::operator+(const vec<2, float>& v) const {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_add_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::operator-(const vec<2, float>& v) const {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_sub_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::operator*(const vec<2, float>& v) const {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_mul_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::operator/(const vec<2, float>& v) const {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_div_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data), res);
        return r;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator+=(const vec<2, float>& v) {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_add_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data), res);
        return *this;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator-=(const vec<2, float>& v) {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_sub_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data), res);
        return *this;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator*=(const vec<2, float>& v) {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_mul_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data), res);
        return *this;
    }
    template<>
    inline vec<2, float>& vec<2, float>::operator/=(const vec<2, float>& v) {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v.data));
        __m128 res = _mm_div_ps(a, b);
        _mm_storel_pi(reinterpret_cast<__m64*>(data), res);
        return *this;
    }

    template<>
    inline vec<2, float> vec<2, float>::max(const vec<2UL, float> &v1, const vec<2UL, float> &v2) {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v1.data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v2.data));
        __m128 res = _mm_max_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data), res);
        return r;
    }
    template<>
    inline vec<2, float> vec<2, float>::min(const vec<2UL, float> &v1, const vec<2UL, float> &v2) {
        __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v1.data));
        __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(v2.data));
        __m128 res = _mm_min_ps(a, b);
        vec<2, float> r;
        _mm_storel_pi(reinterpret_cast<__m64*>(r.data), res);
        return r;
    }


    template<>
    inline vec<3, float> vec<3, float>::operator+(const vec<3, float>& v) const {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_add_ps(a, b);
        vec<3, float> r;
        memcpy(r.data, &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::operator-(const vec<3, float>& v) const {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_sub_ps(a, b);
        vec<3, float> r;
        memcpy(r.data, &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::operator*(const vec<3, float>& v) const {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_mul_ps(a, b);
        vec<3, float> r;
        memcpy(r.data, &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::operator/(const vec<3, float>& v) const {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_div_ps(a, b);
        vec<3, float> r;
        memcpy(r.data, &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator+=(const vec<3, float>& v) {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_add_ps(a, b);
        memcpy(data, &res, sizeof(float) * 3);
        return *this;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator-=(const vec<3, float>& v) {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_sub_ps(a, b);
        memcpy(data, &res, sizeof(float) * 3);
        return *this;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator*=(const vec<3, float>& v) {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_mul_ps(a, b);
        memcpy(data, &res, sizeof(float) * 3);
        return *this;
    }
    template<>
    inline vec<3, float>& vec<3, float>::operator/=(const vec<3, float>& v) {
        __m128 a = _mm_set_ps(0, data[2], data[1], data[0]);
        __m128 b = _mm_set_ps(0, v.data[2], v.data[1], v.data[0]);
        __m128 res = _mm_div_ps(a, b);
        memcpy(data, &res, sizeof(float) * 3);
        return *this;
    }

    template<>
    inline vec<3, float> vec<3, float>::max(const vec<3UL, float> &v1, const vec<3UL, float> &v2) {
        __m128 a = _mm_set_ps(0, v1.data[2], v1.data[1], v1.data[0]);
        __m128 b = _mm_set_ps(0, v2.data[2], v2.data[1], v2.data[0]);
        __m128 res = _mm_max_ps(a, b);
        vec<3, float> r;
        memcpy(r.data, &res, sizeof(float) * 3);
        return r;
    }
    template<>
    inline vec<3, float> vec<3, float>::min(const vec<3UL, float> &v1, const vec<3UL, float> &v2) {
        __m128 a = _mm_set_ps(0, v1.data[2], v1.data[1], v1.data[0]);
        __m128 b = _mm_set_ps(0, v2.data[2], v2.data[1], v2.data[0]);
        __m128 res = _mm_min_ps(a, b);
        vec<3, float> r;
        memcpy(r.data, &res, sizeof(float) * 3);
        return r;
    }


    template<>
    inline vec<4, float> vec<4, float>::operator+(const vec<4, float>& v) const {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_add_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data, res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::operator-(const vec<4, float>& v) const {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_sub_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data, res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::operator*(const vec<4, float>& v) const {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_mul_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data, res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::operator/(const vec<4, float>& v) const {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_div_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data, res);
        return r;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator+=(const vec<4, float>& v) {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_add_ps(a, b);
        _mm_storeu_ps(data, res);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator-=(const vec<4, float>& v) {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_sub_ps(a, b);
        _mm_storeu_ps(data, res);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator*=(const vec<4, float>& v) {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_mul_ps(a, b);
        _mm_storeu_ps(data, res);
        return *this;
    }
    template<>
    inline vec<4, float>& vec<4, float>::operator/=(const vec<4, float>& v) {
        __m128 a = _mm_loadu_ps(data);
        __m128 b = _mm_loadu_ps(v.data);
        __m128 res = _mm_div_ps(a, b);
        _mm_storeu_ps(data, res);
        return *this;
    }

    template<>
    inline vec<4, float> vec<4, float>::max(const vec<4UL, float> &v1, const vec<4UL, float> &v2) {
        __m128 a = _mm_loadu_ps(v1.data);
        __m128 b = _mm_loadu_ps(v2.data);
        __m128 res = _mm_max_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data, res);
        return r;
    }
    template<>
    inline vec<4, float> vec<4, float>::min(const vec<4UL, float> &v1, const vec<4UL, float> &v2) {
        __m128 a = _mm_loadu_ps(v1.data);
        __m128 b = _mm_loadu_ps(v2.data);
        __m128 res = _mm_min_ps(a, b);
        vec<4, float> r;
        _mm_storeu_ps(r.data, res);
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
