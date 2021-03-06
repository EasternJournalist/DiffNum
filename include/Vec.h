#pragma once


#include <initializer_list>
#include <ostream>

#include <Common.h>
#include <myarray.h>
#include <DiffMath.h>

namespace Common
{
	
	template<typename T, ptrdiff_t N>
	struct vec : public array<T, N>
	{
		using array_type = array<T, N>;

		__HOST_DEVICE__ vec() : array<T, N>() { }

		__HOST_DEVICE__ vec(const T* _data) {
			if (_data == nullptr) {
				this->_Elems[0] = NaN<T>;
				return;
			}
			for (ptrdiff_t i = 0; i < N; i++)
				(*this)[i] = _data[i];
		}

		__HOST_DEVICE__ vec(const vec<T, N>& _Right) : array_type(_Right) {}

		__HOST_DEVICE__ vec(const array_type& _Right) : array_type(_Right) {}

		__HOST_DEVICE__ vec(const std::initializer_list<T> _List) : array_type(_List) {}

		template<class... _Args>
		__HOST_DEVICE__ vec(const _Args... args) {
			static_assert(sizeof...(_Args) == N, "Number of parameters must be the same with the length of vec.");
			FArgtoArray((T*)this, args...);
		}

		__HOST_DEVICE__ vec(const T _Val) { fill(_Val); }

		template<ptrdiff_t N1, ptrdiff_t N2>
		__HOST_DEVICE__  vec(const vec<T, N1>& v1, const vec<T, N2>& v2) {
			static_assert(N1 + N2 == N, "Cannot combine two vecs");
			for (ptrdiff_t i = 0; i < N1; i++)
				(*this)[i] = v1[i];
			for (ptrdiff_t i = N1; i < N; i++)
				(*this)[i] = v2[i - N1];
		}

		template<ptrdiff_t N1>
		__HOST_DEVICE__ vec(const vec<T, N1>& v1, const T v2) {
			static_assert(N1 + 1 == N, "Cannot combine two vec");
			for (ptrdiff_t i = 0; i < N1; i++)
				(*this)[i] = v1[i];
			(*this)[N1] = v2;
		}

		template<ptrdiff_t M>
		__HOST_DEVICE__ const vec<T, N>& operator=(const vec<T, M>& v) {
			static const ptrdiff_t K = M < N ? M : N;
			for (ptrdiff_t i = 0; i < K; i++)
				(*this)[i] = v[i];
			return *this;
		}
		template<ptrdiff_t M>
		__HOST_DEVICE__ const vec<T, N>& operator=(const array<T, M>& v) {
			static const ptrdiff_t K = M < N ? M : N;
			for (ptrdiff_t i = 0; i < K; i++)
				(*this)[i] = v[i];
			return *this;
		}

		__HOST_DEVICE__ vec<T, N> operator+(const vec<T, N>& v2) const {
			vec<T, N>  v3;
			for (ptrdiff_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] + v2[i];
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator-(const vec<T, N>& v2) const {
			vec<T, N>  v3;
			for (ptrdiff_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] - v2[i];
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator-() const {
			vec<T, N>  v3;
			for (ptrdiff_t i = 0; i < N; i++)
				v3[i] = -this->_Elems[i];
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator*(const T v2) const {
			vec<T, N> v3;
			for (ptrdiff_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] * v2;
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator/(const T v2) const {
			vec<T, N> v3;
			for (ptrdiff_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] / v2;
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator*(const vec<T, N>& v2) const {
			vec<T, N> v3;
			for (ptrdiff_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] * v2[i];
			return v3;
		}
		__HOST_DEVICE__ const vec<T, N>& operator -= (const vec<T, N>& v2) {
			for (ptrdiff_t i = 0; i < N; i++)
				this->_Elems[i] -= v2[i];
			return *this;
		}
		__HOST_DEVICE__ const vec<T, N>& operator += (const vec<T, N>& v2) {
			for (ptrdiff_t i = 0; i < N; i++)
				this->_Elems[i] += v2[i];
			return *this;
		}
		__HOST_DEVICE__ bool operator==(const vec<T, N>& v2) {
			for (ptrdiff_t i = 0; i < N; i++) {
				if (this->_Elems[i] != v2[i])
					return false;
			}
			return true;
		}
		__HOST_DEVICE__ void fill(const T v) {
			for (T* p = (T*)this, *pend = (T*)this + N; p != pend; p++)
				*p = v;
		}

		__HOST_DEVICE__ T norm2() const {
			T ret = T(0);
			for (ptrdiff_t i = 0; i < N; i++)
				ret += this->_Elems[i] * this->_Elems[i];
			return ret;
		}

		__HOST_DEVICE__ T norm() const {
			return Math<T>::Sqrt(norm2());
		}

		__HOST_DEVICE__ vec<T, N> normalize() const {
			T nrm = norm();
			vec<T, N> ret;
			for (ptrdiff_t i = 0; i < N; i++)
				ret[i] = this->_Elems[i] / nrm;
			return ret;
		}

		__HOST_DEVICE__ static const vec<T, 3> cross(const vec<T, 3>& v1, const vec<T, 3>& v2) {
			return { v1[1] * v2[2] - v1[2] * v2[1],
				v1[2] * v2[0] - v1[0] * v2[2],
				v1[0] * v2[1] - v1[1] * v2[0]
			};
		}

		__HOST_DEVICE__ static const T dot(const vec<T, N>& v1, const vec<T, N>& v2) {
			T ret(0);
			for (ptrdiff_t i = 0; i < N; i++)
				ret += v1[i] * v2[i];
			return ret;
		}

		__HOST_DEVICE__ ~vec() { }

		__HOST_DEVICE__ bool Valid() const {
			return Math<T>::IsNaN(this->_Elems[0]);
		}
	};

	template<typename T, ptrdiff_t N>
	std::ostream& operator << (std::ostream& ostrm, const vec<T, N>& v) {
		if (!v.Valid()) {
			ostrm << "Invalid";
		}
		else {
			for (ptrdiff_t i = 0; i < N; i++)
				ostrm << v[i] << ", ";
		}
		return ostrm;
	}
}