#pragma once
#include <cstdint>
#include <cmath>

#ifndef ACCURACY
#define ACCURACY 1
#endif // !ACCURACY



namespace mgm {
	//==============
	// VECTOR CLASS
	//==============
	
	template<class T> class Vec2Base;
	template<class T> class Vec3Base;
	template<class T> class Vec4Base;

	template<uint8_t S, class T>
	class vec {
		void init(uint8_t& i, T x) {
			data[i] = x;
		}

		template<class ... Ts>
		void init(uint8_t& i, T x, Ts ... xs) {
			data[i] = x;
			init(++i, xs...);
		}

		public:
		T data[S];

		T& operator[](uint8_t i) {
			if (i < S)
				return data[i];
			else
				return data[S - 1];
		}

		template<class ... Ts>
		vec(T x, Ts ... xs) {
			static_assert(sizeof...(Ts) / sizeof(T) <= S, "To many arguments for vector");
			data[0] = x;
			uint8_t i = 1;
			init(i, xs...);
		}

		template<uint8_t Sw, class Tw>
		vec(vec<Sw, Tw> v) {
			static_assert(Sw <= S, "Vector to big");
			for (uint8_t i = 0; i < Sw; i++)
				data[i] = v.data[i];
			for (uint8_t i = Sw; i < S; i++)
				data[i] = (T)0;
		}

		vec(Vec2Base<T> v) {
			static_assert(S >= 2, "Vector not big enough to copy from vec2");
			data[0] = v.x;
			data[1] = v.y;
		}

		vec(Vec3Base<T> v) {
			static_assert(S >= 3, "Vector not big enough to copy from vec3");
			data[0] = v.x;
			data[1] = v.y;
			data[2] = v.z;
		}

		vec(Vec4Base<T> v) {
			static_assert(S >= 4, "Vector not big enough to copy from vec4");
			data[0] = v.x;
			data[1] = v.y;
			data[2] = v.z;
			data[3] = v.w;
		}

		vec(T k = (T)0) {
			for (uint8_t i = 0; i < S; i++)
				data[i] = k;
		}

		vec(const vec<S, T>& v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] = v.data[i];
		}
		vec(vec<S, T>& v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] = v.data[i];
		}
		vec operator=(const vec<S, T>& v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] = v.data[i];
			return *this;
		}
		vec operator=(vec<S, T>& v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] = v.data[i];
			return *this;
		}

		vec<S, T> operator+(vec<S, T> v) {
			vec<S, T> res{};
			for (uint8_t i = 0; i < S; i++)
				res.data[i] = data[i] + v.data[i];
			return res;
		}
		vec<S, T> operator-(vec<S, T> v) {
			vec<S, T> res{};
			for (uint8_t i = 0; i < S; i++)
				res.data[i] = data[i] - v.data[i];
			return res;
		}
		vec<S, T> operator*(vec<S, T> v) {
			vec<S, T> res{};
			for (uint8_t i = 0; i < S; i++)
				res.data[i] = data[i] * v.data[i];
			return res;
		}
		vec<S, T> operator/(vec<S, T> v) {
			vec<S, T> res{};
			for (uint8_t i = 0; i < S; i++)
				res.data[i] = data[i] / v.data[i];
			return res;
		}

		vec<S, T> operator+=(vec<S, T> v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] += v.data[i];
			return *this;
		}
		vec<S, T> operator-=(vec<S, T> v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] -= v.data[i];
			return *this;
		}
		vec<S, T> operator*=(vec<S, T> v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] *= v.data[i];
			return *this;
		}
		vec<S, T> operator/=(vec<S, T> v) {
			for (uint8_t i = 0; i < S; i++)
				data[i] /= v.data[i];
			return *this;
		}

		inline T dot(vec<S, T> v) {
			T res = data[0] * v.data[0];
			for (uint8_t i = 1; i < S; i++)
				res += data[i] * v.data[i];
			return res;
		}
		inline T length() { return sqrt(this->dot(*this)); }
		inline T distanceTo(vec<S, T> v) { return (v - (*this)).length(); };
		inline void normalize() { *this = (*this) / length(); }
		inline vec<S, T> normalized() { return (*this) / length(); }
		inline T directionTo(vec<S, T> v) { return (v - (*this)).normalized(); }
		inline vec<S, T> clamped(vec<S, T> v1, vec<S, T> v2) {
			vec<S, T> res{*this};
			for (uint8_t i = 0; i < S; i++) {
				if (res.data[i] < v1.data[i])
					res.data[i] = v1.data;
				else if (res.data[i] > v2.data[i])
					res.data[i] = v2.data[i];
			}
			return res;
		}
		inline vec<S, T> clamp(vec<S, T> v1, vec<S, T> v2) { *this = this->clamped(v1, v2); }
	};

	template<class T>
	class Vec2Base {
		public:
		T x, y;

		template<class Tw>
		Vec2Base(Vec2Base<Tw> v): x{(T)v.x}, y{(T)v.y} {}

		Vec2Base(T k = (T)0): x{k}, y{k} {}
		Vec2Base(T vx, T vy): x{vx}, y{vy} {}
		Vec2Base(const Vec2Base& v): x{v.x}, y{v.y} {}
		Vec2Base(vec<2, T> v): x{v[0]}, y{v[1]} {}
		Vec2Base<T>& operator=(const Vec2Base<T>& v) { x = v.x; y = v.y; return *this; }

		Vec2Base operator+(Vec2Base<T> v) { return Vec2Base(x + v.x, y + v.y); }
		Vec2Base operator-(Vec2Base<T> v) { return Vec2Base(x - v.x, y - v.y); }
		Vec2Base operator*(Vec2Base<T> v) { return Vec2Base(x * v.x, y * v.y); }
		Vec2Base operator/(Vec2Base<T> v) { return Vec2Base(x / v.x, y / v.y); }
		Vec2Base operator+=(Vec2Base<T> v) { x += v.x; y += v.y; return *this; }
		Vec2Base operator-=(Vec2Base<T> v) { x -= v.x; y -= v.y; return *this; }
		Vec2Base operator*=(Vec2Base<T> v) { x *= v.x; y *= v.y; return *this; }
		Vec2Base operator/=(Vec2Base<T> v) { x /= v.x; y /= v.y; return *this; }

		inline T dot(Vec2Base<T> v) { return x * v.x + y * v.y; }
		inline T length() { return sqrt(this->dot(*this)); }
		inline T distanceTo(Vec2Base<T> v) { return (v - (*this)).length(); };
		inline void normalize() { *this = (*this) / length(); }
		inline Vec2Base<T> normalized() { return (*this) / length(); }
		inline T directionTo(Vec2Base<T> v) { return (v - (*this)).normalized(); }
		inline Vec2Base<T> clamped(Vec2Base<T> v1, Vec2Base<T> v2) {
			Vec2Base<T> res{ *this };

			if (res.x < v1.x)
				res.x = v1.x;
			else if (res.x > v2.x)
				res.x = v2.x;

			if (res.y < v1.y)
				res.y = v1.y;
			else if (res.y > v2.y)
				res.y = v2.y;

			return res;
		}
		inline void clamp(Vec2Base<T> v1, Vec2Base<T> v2) { *this = this->clamped(v1, v2); }
	};

	template<class T>
	class Vec3Base {
		public:
		T x, y, z;

		template<class Tw>
		Vec3Base(Vec3Base<Tw> v): x{(T)v.x}, y{(T)v.y}, z{(T)v.z} {}

		Vec3Base(T k = (T)0): x{k}, y{k}, z{k} {}
		Vec3Base(T vx, T vy, T vz = (T)0): x{vx}, y{vy}, z{vz} {}
		Vec3Base(Vec2Base<T> v, T vz = (T)0): x{v.x}, y{v.y}, z{vz} {}
		Vec3Base(T vx, Vec2Base<T> v): x{vx}, y{v.x}, z{v.y} {}
		Vec3Base(const Vec3Base& v): x{v.x}, y{v.y}, z{v.z} {}
		Vec3Base(vec<3, T> v): x{v[0]}, y{v[1]}, z{v[2]} {}
		Vec3Base<T>& operator=(const Vec3Base<T>& v) { x = v.x; y = v.y; z = v.z; return *this; }

		Vec3Base operator+(Vec3Base<T> v) { return Vec3Base(x + v.x, y + v.y, z + v.z); }
		Vec3Base operator-(Vec3Base<T> v) { return Vec3Base(x - v.x, y - v.y, z - v.z); }
		Vec3Base operator*(Vec3Base<T> v) { return Vec3Base(x * v.x, y * v.y, z * v.z); }
		Vec3Base operator/(Vec3Base<T> v) { return Vec3Base(x / v.x, y / v.y, z / v.z); }
		Vec3Base operator+=(Vec3Base<T> v) { *this = (*this) + v; return *this; }
		Vec3Base operator-=(Vec3Base<T> v) { *this = (*this) - v; return *this; }
		Vec3Base operator*=(Vec3Base<T> v) { *this = (*this) * v; return *this; }
		Vec3Base operator/=(Vec3Base<T> v) { *this = (*this) / v; return *this; }

		inline T dot(Vec3Base<T> v) { return x * v.x + y * v.y + z * v.z; }
		inline T length() { return sqrt(this->dot(*this)); }
		inline T distanceTo(Vec3Base<T> v) { return (v - (*this)).length(); }
		inline void normalize() { *this = (*this) / length(); }
		inline Vec3Base<T> normalized() { return (*this) / length(); }
		inline T directionTo(Vec3Base<T> v) { return (v - (*this)).normalized(); }
		inline Vec3Base<T> clamped(Vec3Base<T> v1, Vec3Base<T> v2) {
			Vec3Base<T> res{ *this };

			if (res.x < v1.x)
				res.x = v1.x;
			else if (res.x > v2.x)
				res.x = v2.x;

			if (res.y < v1.y)
				res.y = v1.y;
			else if (res.y > v2.y)
				res.y = v2.y;

			if (res.z < v1.z)
				res.z = v1.z;
			else if (res.z > v2.z)
				res.z = v2.z;

			return res;
		}
		inline void clamp(Vec3Base<T> v1, Vec3Base<T> v2) { *this = this->clamped(v1, v2); }
	};

	template<class T>
	class Vec4Base {
		public:
		T x, y, z, w;

		template<class Tw>
		Vec4Base(Vec4Base<Tw> v): x{(T)v.x}, y{(T)v.y}, z{(T)v.z}, w{(T)v.w} {}

		Vec4Base(T k = (T)0): x{k}, y{k}, z{k}, w{k} {}
		Vec4Base(T vx, T vy, T vz = (T)0, T vw = (T)0): x{vx}, y{vy}, z{vz}, w{vw} {}
		Vec4Base(Vec2Base<T> v, T vz = (T)0, T vw = (T)0): x{v.x}, y{v.y}, z{vz}, w{vw} {}
		Vec4Base(T vx, Vec2Base<T> v, T vw = (T)0): x{vx}, y{v.x}, z{v.y}, w{vw} {}
		Vec4Base(T vx, T vy, Vec2Base<T> v): x{vx}, y{vy}, z{v.x}, w{v.y} {}
		Vec4Base(Vec2Base<T> v1, Vec2Base<T> v2): x{v1.x}, y{v1.y}, z{v2.x}, w{v2.y} {}
		Vec4Base(Vec3Base<T> v, T vw = (T)0): x{v.x}, y{v.y}, z{v.z}, w{vw} {}
		Vec4Base(T vx, Vec3Base<T> v): x{vx}, y{v.x}, z{v.y}, w{v.z} {}
		Vec4Base(const Vec4Base& v): x{v.x}, y{v.y}, z{v.z}, w{v.w} {}
		Vec4Base(vec<4, T> v): x{v[0]}, y{v[1]}, z{v[2]}, w{v[3]} {}
		Vec4Base<T>& operator=(const Vec4Base<T>& v) { x = v.x; y = v.y; z = v.z; w = v.w; return *this; }

		Vec4Base operator+(Vec4Base<T> v) { return Vec4Base(x + v.x, y + v.y, z + v.z, w + v.w); }
		Vec4Base operator-(Vec4Base<T> v) { return Vec4Base(x - v.x, y - v.y, z - v.z, w - v.w); }
		Vec4Base operator*(Vec4Base<T> v) { return Vec4Base(x * v.x, y * v.y, z * v.z, w * v.w); }
		Vec4Base operator/(Vec4Base<T> v) { return Vec4Base(x / v.x, y / v.y, z / v.z, w / v.w); }
		Vec4Base operator+=(Vec4Base<T> v) { *this = (*this) + v; return *this; }
		Vec4Base operator-=(Vec4Base<T> v) { *this = (*this) - v; return *this; }
		Vec4Base operator*=(Vec4Base<T> v) { *this = (*this) * v; return *this; }
		Vec4Base operator/=(Vec4Base<T> v) { *this = (*this) / v; return *this; }

		inline T dot(Vec4Base<T> v) { return x * v.x + y * v.y + z * v.z + w * v.w; }
		inline T length() { return sqrt(this->dot(*this)); }
		inline T distanceTo(Vec4Base<T> v) { return (v - (*this)).length(); }
		inline void normalize() { *this = (*this) / length(); }
		inline Vec4Base<T> normalized() { return (*this) / length(); }
		inline T directionTo(Vec4Base<T> v) { return (v - (*this)).normalized(); }
		inline Vec4Base<T> clamped(Vec4Base<T> v1, Vec4Base<T> v2) {
			Vec4Base<T> res{*this};

			if (res.x < v1.x)
				res.x = v1.x;
			else if (res.x > v2.x)
				res.x = v2.x;

			if (res.y < v1.y)
				res.y = v1.y;
			else if (res.y > v2.y)
				res.y = v2.y;

			if (res.z < v1.z)
				res.z = v1.z;
			else if (res.z > v2.z)
				res.z = v2.z;

			if (res.w < v1.w)
				res.w = v1.w;
			else if (res.w > v2.w)
				res.w = v2.w;

			return res;
		}
		inline void clamp(Vec4Base<T> v1, Vec4Base<T> v2) { *this = this->clamped(v1, v2); }
	};

	using vec2 = Vec2Base<float>;
	using vec3 = Vec3Base<float>;
	using vec4 = Vec4Base<float>;
	using dvec2 = Vec2Base<double>;
	using dvec3 = Vec3Base<double>;
	using dvec4 = Vec4Base<double>;

	using i8vec2 = Vec2Base<int8_t>;
	using i8vec3 = Vec3Base<int8_t>;
	using i8vec4 = Vec4Base<int8_t>;
	using ui8vec2 = Vec2Base<uint8_t>;
	using ui8vec3 = Vec3Base<uint8_t>;
	using ui8vec4 = Vec4Base<uint8_t>;

	using i16vec2 = Vec2Base<int16_t>;
	using i16vec3 = Vec3Base<int16_t>;
	using i16vec4 = Vec4Base<int16_t>;
	using ui16vec2 = Vec2Base<uint16_t>;
	using ui16vec3 = Vec3Base<uint16_t>;
	using ui16vec4 = Vec4Base<uint16_t>;

	using i32vec2 = Vec2Base<int32_t>;
	using i32vec3 = Vec3Base<int32_t>;
	using i32vec4 = Vec4Base<int32_t>;
	using ui32vec2 = Vec2Base<uint32_t>;
	using ui32vec3 = Vec3Base<uint32_t>;
	using ui32vec4 = Vec4Base<uint32_t>;

	using i64vec2 = Vec2Base<int64_t>;
	using i64vec3 = Vec3Base<int64_t>;
	using i64vec4 = Vec4Base<int64_t>;
	using ui64vec2 = Vec2Base<uint64_t>;
	using ui64vec3 = Vec3Base<uint64_t>;
	using ui64vec4 = Vec4Base<uint64_t>;

	using ivec2 = i32vec2;
	using ivec3 = i32vec3;
	using ivec4 = i32vec4;
	using uivec2 = ui32vec2;
	using uivec3 = ui32vec3;
	using uivec4 = ui32vec4;



	//==============
	// MATRIX CLASS
	//==============

	template<uint8_t i, uint8_t j, class T = float>
	class mat {
		inline void init(uint8_t& n, T x) {
			((T*)(data))[n] = x;
		}

		template<class ... Ts>
		inline void init(uint8_t& n, T x, Ts ... xs) {
			((T*)(data))[n] = x;
			init(++n, xs...);
		}
		
		public:
		vec<j, T> data[i];

		mat(const mat<i, j, T>& m);
		mat(mat<i, j, T>& m);
		void operator=(const mat<i, j, T>& m);
		vec<j, T>& operator[](uint8_t x);

		mat(T w = (T)0);
		mat(T* data);

		template<class T_w>
		mat(mat<i, j, T_w>& m) {
			for (uint8_t x = 0; x < i; x++)
				((T*)(data))[x] = ((T*)(m.data))[x];
		}

		template<class ... Ts>
		mat(T x, Ts ... xs) {
			static_assert(sizeof...(Ts) + 1 == i * j, "Number of elements in matrix initializer is not eual num matrix size");
			uint8_t n = 0;
			init(n, x, xs...);
		}

		inline mat<i, j, T> operator+(mat<i, j, T> m);
		inline mat<i, j, T> operator-(mat<i, j, T> m);
		inline mat<i, i, T> operator*(mat<j, i, T> m);
		inline vec<j, T> operator*(vec<j, T> v);

		inline mat<i, j, T> operator+(T w);
		inline mat<i, j, T> operator-(T w);
		inline mat<i, j, T> operator*(T w);
		inline mat<i, j, T> operator/(T w);

		inline void operator+=(mat<i, j, T> m);
		inline void operator-=(mat<i, j, T> m);

		inline void operator+=(T w);
		inline void operator-=(T w);
		inline void operator*=(T w);
		inline void operator/=(T w);

		/**
		 * @brief Fill the matrix with the given value
		 * 
		 * @param w The value to fill the matrix with
		 */
		void fill(T w);
	};


	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T>::mat(const mat<i, j, T>& m) {
		for (uint8_t x = 0; x < i; x++)
			data[x] = m.data[x];
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T>::mat(mat<i, j, T>& m) {
		for (uint8_t x = 0; x < i; x++)
			data[x] = m.data[x];
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator=(const mat<i, j, T>& m) {
		for (uint8_t x = 0; x < i; x++)
			data[x] = m.data[x];
	}

	template<uint8_t i, uint8_t j, class T>
	vec<j, T>& mat<i, j, T>::operator[](uint8_t x) {
		if (x < j)
			return data[x];
		else
			return data[j - 1];
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T>::mat(T w) {
		for (uint8_t x = 0; x < i; x++) {
			data[x] = vec<j, T>((T)0);
			data[x][x] = w;
		}
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T>::mat(T* data) {
		memcpy(mat::data, data, i * j * sizeof(T));
	}


	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T> mat<i, j, T>::operator+(mat<i, j, T> m) {
		mat<i, j, T> res{*this};
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(res.data))[x] += ((T*)(m.data))[x];
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T> mat<i, j, T>::operator-(mat<i, j, T> m) {
		mat<i, j, T> res{*this};
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(res.data))[x] -= ((T*)(m.data))[x];
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, i, T> mat<i, j, T>::operator*(mat<j, i, T> m) {
		mat<i, i, T> res{(T)0};
		for (uint8_t x = 0; x < i; x++)
			for (uint8_t y = 0; y < i; y++)
				for (uint8_t n = 0; n < j; n++)
					res[x][y] += data[x][n] * m.data[n][y];
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	vec<j, T> mat<i, j, T>::operator*(vec<j, T> v) {
		vec<j, T> res{};
		for (uint8_t x = 0; x < i; x++)
			for (uint8_t y = 0; y < j; y++)
				res[x] += data[x][y] * v[y];
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T> mat<i, j, T>::operator+(T w) {
		mat<i, j, T> res{*this};
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(this->data))[x] += w;
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T> mat<i, j, T>::operator-(T w) {
		mat<i, j, T> res{*this};
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(this->data))[x] -= w;
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T> mat<i, j, T>::operator*(T w) {
		mat<i, j, T> res{*this};
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(this->data))[x] *= w;
		return res;
	}

	template<uint8_t i, uint8_t j, class T>
	mat<i, j, T> mat<i, j, T>::operator/(T w) {
		mat<i, j, T> res{*this};
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(this->data))[x] /= w;
		return res;
	}


	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator+=(mat<i, j, T> m) {
		*this = *this + m;
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator-=(mat<i, j, T> m) {
		*this = *this - m;
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator+=(T w) {
		*this = *this + w;
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator-=(T w) {
		*this = *this - w;
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator*=(T w) {
		*this = *this * w;
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::operator/=(T w) {
		*this = *this / w;
	}

	template<uint8_t i, uint8_t j, class T>
	void mat<i, j, T>::fill(T w) {
		for (uint8_t x = 0; x < i * j; x++)
			((T*)(this->data))[x] = w;
	}


	using mat2 = mat<2, 2, float>;
	using mat3 = mat<3, 3, float>;
	using mat4 = mat<4, 4, float>;

	using dmat2 = mat<2, 2, double>;
	using dmat3 = mat<3, 3, double>;
	using dmat4 = mat<4, 4, double>;

	using i8mat2 = mat<2, 2, int8_t>;
	using i8mat3 = mat<3, 3, int8_t>;
	using i8mat4 = mat<4, 4, int8_t>;
	using ui8mat2 = mat<2, 2, uint8_t>;
	using ui8mat3 = mat<3, 3, uint8_t>;
	using ui8mat4 = mat<4, 4, uint8_t>;

	using i16mat2 = mat<2, 2, int16_t>;
	using i16mat3 = mat<3, 3, int16_t>;
	using i16mat4 = mat<4, 4, int16_t>;
	using ui16mat2 = mat<2, 2, uint16_t>;
	using ui16mat3 = mat<3, 3, uint16_t>;
	using ui16mat4 = mat<4, 4, uint16_t>;

	using i32mat2 = mat<2, 2, int32_t>;
	using i32mat3 = mat<3, 3, int32_t>;
	using i32mat4 = mat<4, 4, int32_t>;
	using ui32mat2 = mat<2, 2, uint32_t>;
	using ui32mat3 = mat<3, 3, uint32_t>;
	using ui32mat4 = mat<4, 4, uint32_t>;

	using i64mat2 = mat<2, 2, int64_t>;
	using i64mat3 = mat<3, 3, int64_t>;
	using i64mat4 = mat<4, 4, int64_t>;
	using ui64mat2 = mat<2, 2, uint64_t>;
	using ui64mat3 = mat<3, 3, uint64_t>;
	using ui64mat4 = mat<4, 4, uint64_t>;

	using imat2 = i32mat2;
	using imat3 = i32mat3;
	using imat4 = i32mat4;
	using uimat2 = ui32mat2;
	using uimat3 = ui32mat3;
	using uimat4 = ui32mat4;



	//=================
	// TRANSFORM CLASS
	//=================

	template<class T = float>
	class transform {
		public:
		enum class Order {
			XYZ,
			XZY,
			YXZ,
			YZX,
			ZXY,
			ZYX
		};

		mat<4, 4, float> matrix{ (T)1 };

		transform() {}

		transform<T> operator*(transform<T> t);
		transform<T>& operator*=(transform<T> t);

		transform(const transform& t);
		transform& operator=(const transform& t);

		void translate(vec<3, T> v);
		transform<T> translated(vec<3, T> v);
		void scale(vec<3, T> v);
		transform<T> scaled(vec<3, T> v);
		void rotate(vec<3, T> v, Order axisOrder = Order::XYZ);
		transform<T> rotated(vec<3, T> v, Order axisOrder = Order::XYZ);
		void rotateLook(vec<3, T> v, Order axisOrder = Order::XYZ);
		transform<T> rotatedLook(vec<3, T> v, Order axisOrder = Order::XYZ);

		vec<3, T> getTranslation() { return vec<3, T>(matrix[0][3], matrix[1][3], matrix[2][3]); }

		T* data();


		static inline mat<4, 4, T> genRotationMatrix_X(T rotation);
		static inline mat<4, 4, T> genRotationMatrix_Y(T rotation);
		static inline mat<4, 4, T> genRotationMatrix_Z(T rotation);
		static inline mat<4, 4, T> genRotationMatrix(vec<3, T> rotation, Order axisOrder = Order::XYZ);
		static transform<T> genProjection(T fov, T width, T height, T far_view = (T)100.0, T n = (T)0.1) {
			transform<T> res{};
			#if ACCURACY <= 1
			float tfp2 = tanf(fov / 2.0f);
			res.matrix = (mat<4, 4, T>)mat<4, 4, float>(
				1.0f / ((width / height) * tfp2), 0.0f,        0.0f,                           0.0f,
				0.0f,                             1.0f / tfp2, 0.0f,                           0.0f,
				0.0f,                             0.0f,        -((far_view + n) / (far_view - n)), -(2.0f * far_view * n / (far_view - n)),
				0.0f,                             0.0f,        -1.0f,                          0.0f
				);
			#else
			double tfp2 = tan(fov / 2.0);
			res.matrix = (mat<4, 4, T>)mat<4, 4, double>(
				1.0 / ((width / height) * tfp2), 0.0, 0.0, 0.0,
				0.0,                           1.0 / tfp2, 0.0,                            0.0,
				0.0,                           0.0,        -((far + near) / (far - near)), -(2.0 * far * near / (far - near)),
				0.0,                           0.0,        -1.0f,                          0.0
				);
			#endif
			return res;
		}
	};

	template<class T>
	transform<T> transform<T>::operator*(transform<T> t) {
		transform<T> res{ *this };
		res.matrix = res.matrix * t.matrix;
		return res;
	}

	template<class T>
	transform<T>& transform<T>::operator*=(transform<T> t) {
		*this = (*this) * (*this);
		return *this;
	}

	template<class T>
	transform<T>::transform(const transform& t) {
		matrix = t.matrix;
	}

	template<class T>
	transform<T>& transform<T>::operator=(const transform& t) {
		matrix = t.matrix;
		return *this;
	}

	template<class T>
	void transform<T>::translate(vec<3, T> v) {
		matrix[0][3] += v[0];
		matrix[1][3] += v[1];
		matrix[2][3] += v[2];
	}

	template<class T>
	transform<T> transform<T>::translated(vec<3, T> v) {
		transform res{ *this };
		res.matrix[0][3] += v[0];
		res.matrix[1][3] += v[1];
		res.matrix[2][3] += v[2];
		return res;
	}

	template<class T>
	void transform<T>::scale(vec<3, T> v) {
		matrix[0][0] *= v[0];
		matrix[1][1] *= v[1];
		matrix[2][2] *= v[2];
	}

	template<class T>
	transform<T> transform<T>::scaled(vec<3, T> v) {
		transform res{ *this };
		res.matrix[0][0] *= v[0];
		res.matrix[1][1] *= v[1];
		res.matrix[2][2] *= v[2];
		return res;
	}

	template<class T>
	void transform<T>::rotate(vec<3, T> v, Order axisOrder) {
		matrix = matrix * genRotationMatrix(v, axisOrder);
	}

	template<class T>
	transform<T> transform<T>::rotated(vec<3, T> v, Order axisOrder) {
		transform res{*this};
		res.matrix = matrix * genRotationMatrix(v, axisOrder);
		return res;
	}

	template<class T>
	void transform<T>::rotateLook(vec<3, T> v, Order axisOrder) {
		matrix = genRotationMatrix(v, axisOrder) * matrix;
	}

	template<class T>
	transform<T> transform<T>::rotatedLook(vec<3, T> v, Order axisOrder) {
		transform res{ *this };
		res.matrix = genRotationMatrix(v, axisOrder) * matrix;
		return res;
	}

	template<class T>
	T* transform<T>::data() {
		return (T*)&matrix;
	}

	template<class T>
	mat<4, 4, T> transform<T>::genRotationMatrix_X(T rotation) {
		#if ACCURACY <= 1
		float s = sinf((float)rotation);
		float c = cosf((float)rotation);
		return (mat<4, 4, T>)mat<4, 4, float>(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, c,    s,    0.0f,
			0.0f, -s,   c,    0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
			);
		#else
		double s = sin((double)rotation);
		double c = cos((double)rotation);
		return (mat<4, 4, T>)mat<4, 4, double>(
			1.0, 0.0, 0.0, 0.0,
			0.0, c,   s,   0.0,
			0.0, -s,  c,   0.0,
			0.0, 0.0, 0.0, 1.0
			);
		#endif
	}

	template<class T>
	mat<4, 4, T> transform<T>::genRotationMatrix_Y(T rotation) {
		#if ACCURACY <= 1
		float s = sinf((float)rotation);
		float c = cosf((float)rotation);
		return (mat<4, 4, T>)mat<4, 4, float>(
			c,    0.0f, -s,   0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			s,    0.0f, c,    0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
			);
		#else
		double s = sin((double)rotation);
		double c = cos((double)rotation);
		return (mat<4, 4, T>)mat<4, 4, double>(
			c,   0.0, -s,  0.0,
			0.0, 1.0, 0.0, 0.0,
			s,   0.0, c,   0.0,
			0.0, 0.0, 0.0, 1.0
			);
		#endif
	}

	template<class T>
	mat<4, 4, T> transform<T>::genRotationMatrix_Z(T rotation) {
		#if ACCURACY <= 1
		float s = sinf((float)rotation);
		float c = cosf((float)rotation);
		return (mat<4, 4, T>)mat<4, 4, float>(
			c,    -s,   0.0f, 0.0f,
			s,    c,    0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
			);
		#else
		double s = sin((double)rotation);
		double c = cos((double)rotation);
		return (mat<4, 4, T>)mat<4, 4, double>(
			c,   -s,  0.0, 0.0,
			s,   c,   0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0
			);
		#endif
	}

	template<class T>
	mat<4, 4, T> transform<T>::genRotationMatrix(vec<3, T> rotation, Order axisOrder) {
		switch (axisOrder) {
			case Order::XYZ:
				return genRotationMatrix_X(rotation[0])
					* genRotationMatrix_Y(rotation[1])
					* genRotationMatrix_Z(rotation[2]);
				break;
			case Order::XZY:
				return genRotationMatrix_X(rotation[0])
					* genRotationMatrix_Z(rotation[2])
					* genRotationMatrix_Y(rotation[1]);
				break;
			case Order::YXZ:
				return genRotationMatrix_Y(rotation[1])
					* genRotationMatrix_X(rotation[0])
					* genRotationMatrix_Z(rotation[2]);
				break;
			case Order::YZX:
				return genRotationMatrix_Y(rotation[1])
					* genRotationMatrix_Z(rotation[2])
					* genRotationMatrix_X(rotation[0]);
				break;
			case Order::ZXY:
				return genRotationMatrix_Z(rotation[2])
					* genRotationMatrix_X(rotation[0])
					* genRotationMatrix_Y(rotation[1]);
				break;
			case Order::ZYX:
				return genRotationMatrix_Z(rotation[2])
					* genRotationMatrix_Y(rotation[1])
					* genRotationMatrix_X(rotation[0]);
				break;
			default:
				return mat<4, 4, T>();
				break;
		}
	}

	using transform32 = transform<float>;
	using transform64 = transform<double>;

	template<class T>
	struct TransformParent {
		transform<T> trans{};
		TransformParent<T>* parent = nullptr;

		transform<T> getTransform() {
			if (parent == nullptr)
				return trans;
			else
				return parent->getTransform() * trans;
		}
	};

	using TransformParent32 = TransformParent<float>;
	using TransformParent64 = TransformParent<double>;
}
