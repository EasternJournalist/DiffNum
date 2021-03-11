#pragma once


#include <initializer_list>
#include <ostream>

#include <Common.h>
#include <myarray.h>
#include <DiffMath.h>

namespace Common
{
	
	template<typename T, size_t N>
	struct vec : public array<T, N>
	{
		__HOST_DEVICE__ vec() : array<T, N>() { }
		__HOST_DEVICE__ vec(const T* _data) {
			if (_data == nullptr) {
				this->_Elems[0] = NaN<T>;
				return;
			}
			for (size_t i = 0; i < N; i++)
				(*this)[i] = _data[i];
		}

		__HOST_DEVICE__ vec(const vec<T, N>& v) {
			for (size_t i = 0; i < N; i++)
				(*this)[i] = v[i];
		}

		__HOST_ONLY__ vec(const std::initializer_list<T> elem_list) {
			assert(elem_list.size() <= N && "Initializer list is too long");
			size_t i = 0;
			for (auto it : elem_list)
				this->_Elems[i++] = it;
		}

		__HOST_DEVICE__ vec(const T v) {
			for (T* p = this->_Elems, *pend = this->_Elems + N; p != pend; p++)
				*p = v;
		}

		template<size_t N1, size_t N2>
		__HOST_DEVICE__  vec(const vec<T, N1>& v1, const vec<T, N2>& v2) {
			static_assert(N1 + N2 == N, "Cannot combine two vecs");
			for (size_t i = 0; i < N1; i++)
				(*this)[i] = v1[i];
			for (size_t i = N1; i < N; i++)
				(*this)[i] = v2[i - N1];
		}

		template<size_t N1>
		__HOST_DEVICE__ vec(const vec<T, N1>& v1, const T v2) {
			static_assert(N1 + 1 == N, "Cannot combine two vec");
			for (size_t i = 0; i < N1; i++)
				(*this)[i] = v1[i];
			(*this)[N1] = v2;
		}

		template<size_t M>
		__HOST_DEVICE__ const vec<T, N>& operator=(const vec<T, M>& v) {
			for (size_t i = 0; i < M; i++)
				(*this)[i] = v[i];
			return *this;
		}
		template<size_t M>
		__HOST_DEVICE__ const vec<T, N>& operator=(const array<T, M>& v) {
			for (size_t i = 0; i < M; i++)
				(*this)[i] = v[i];
			return *this;
		}

		__HOST_DEVICE__ vec<T, N> operator+(const vec<T, N>& v2) const {
			vec<T, N>  v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] + v2[i];
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator-(const vec<T, N>& v2) const {
			vec<T, N>  v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] - v2[i];
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator-() const {
			vec<T, N>  v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = -this->_Elems[i];
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator*(const T v2) const {
			vec<T, N> v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] * v2;
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator/(const T v2) const {
			vec<T, N> v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] / v2;
			return v3;
		}
		__HOST_DEVICE__ vec<T, N> operator*(const vec<T, N>& v2) const {
			vec<T, N> v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] * v2[i];
			return v3;
		}
		__HOST_DEVICE__ const vec<T, N>& operator -= (const vec<T, N>& v2) {
			for (size_t i = 0; i < N; i++)
				this->_Elems[i] -= v2[i];
			return *this;
		}
		__HOST_DEVICE__ const vec<T, N>& operator += (const vec<T, N>& v2) {
			for (size_t i = 0; i < N; i++)
				this->_Elems[i] += v2[i];
			return *this;
		}
		__HOST_DEVICE__ const bool operator==(const vec<T, N>& v2) {
			for (size_t i = 0; i < N; i++) {
				if (this->_Elems[i] != v2[i])
					return false;
			}
			return true;
		}
		__HOST_DEVICE__ void fill(const T v) {
			for (T* p = this->_Elems, *pend = this->_Elems + N; p != pend; p++)
				*p = v;
		}

		__HOST_DEVICE__ T norm2() const {
			T ret = static_cast<T>(0);
			for (size_t i = 0; i < N; i++)
				ret += this->_Elems[i] * this->_Elems[i];
			return ret;
		}

		__HOST_DEVICE__ T norm() const {
			return Math<T>::Sqrt(norm2());
		}

		__HOST_DEVICE__ vec<T, N> normalize() const {
			T nrm = norm();
			vec<T, N> ret;
			for (size_t i = 0; i < N; i++)
				ret[i] = this->_Elems[i] / nrm;
			return ret;
		}

		__HOST_DEVICE__ static const vec<T, 3> cross(vec<T, 3> v1, vec<T, 3>v2) {
			return { v1[1] * v2[2] - v1[2] * v2[1],
				v1[2] * v2[0] - v1[0] * v2[2],
				v1[0] * v2[1] - v1[1] * v2[0]
			};
		}

		__HOST_DEVICE__ static const T dot(const vec<T, N>& v1, const vec<T, N>& v2) {
			T ret(0.);
			for (size_t i = 0; i < N; i++)
				ret += v1[i] * v2[i];
			return ret;
		}

		__HOST_DEVICE__ ~vec() { }

		__HOST_DEVICE__ bool Valid() const {
			return Math<T>::IsNaN(this->_Elems[0]);
		}
	};

	template<typename T, size_t N>
	std::ostream& operator << (std::ostream& ostrm, const vec<T, N>& v) {
		if (!v.Valid()) {
			ostrm << "Invalid";
		}
		else {
			for (size_t i = 0; i < N; i++)
				ostrm << v[i] << ", ";
		}
		return ostrm;
	}
}