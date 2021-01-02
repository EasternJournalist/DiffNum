#pragma once

#include <array>
#include <initializer_list>
#include <ostream>

#include <DiffBasic.h>
#include <DiffMath.h>

namespace DiffNum
{
	template<typename T, size_t N>
	struct Vec : public std::array<T, N>
	{
		Vec() : std::array<T, N>() { }
		Vec(const T* _data) {
			if (_data == nullptr) {
				this->_Elems[0] = NaN<T>;
				return;
			}
			for (size_t i = 0; i < N; i++)
				(*this)[i] = _data[i];
		}

		Vec(const Vec<T, N>& v) {
			for (size_t i = 0; i < N; i++)
				(*this)[i] = v[i];
		}

		Vec(const std::initializer_list<T> elem_list) {
			assert(elem_list.size() <= N && "Initializer list is too long");
			size_t i = 0;
			for (auto it : elem_list)
				this->_Elems[i++] = it;
		}

		Vec(const T v) {
			for (T* p = this->_Elems, *pend = this->_Elems + N; p != pend; p++)
				*p = v;
		}

		template<size_t N1, size_t N2>
		Vec(const Vec<T, N1>& v1, const Vec<T, N2>& v2) {
			static_assert(N1 + N2 == N, "Cannot combine two Vecs");
			for (size_t i = 0; i < N1; i++)
				(*this)[i] = v1[i];
			for (size_t i = N1; i < N; i++)
				(*this)[i] = v2[i - N1];
		}

		template<size_t N1>
		Vec(const Vec<T, N1>& v1, const T v2) {
			static_assert(N1 + 1 == N, "Cannot combine two Vec");
			for (size_t i = 0; i < N1; i++)
				(*this)[i] = v1[i];
			(*this)[N1] = v2;
		}

		template<size_t M>
		const Vec<T, N>& operator=(const Vec<T, M>& v) {
			for (size_t i = 0; i < M; i++)
				(*this)[i] = v[i];
			return *this;
		}
		template<size_t M>
		const Vec<T, N>& operator=(const std::array<T, M>& v) {
			for (size_t i = 0; i < M; i++)
				(*this)[i] = v[i];
			return *this;
		}

		Vec<T, N> operator+(const Vec<T, N>& v2) const {
			Vec<T, N>  v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] + v2[i];
			return v3;
		}
		Vec<T, N> operator-(const Vec<T, N>& v2) const {
			Vec<T, N>  v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] - v2[i];
			return v3;
		}
		Vec<T, N> operator-() const {
			Vec<T, N>  v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = -this->_Elems[i];
			return v3;
		}
		Vec<T, N> operator*(const T v2) const {
			Vec<T, N> v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] * v2;
			return v3;
		}
		Vec<T, N> operator/(const T v2) const {
			Vec<T, N> v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] / v2;
			return v3;
		}
		Vec<T, N> operator*(const Vec<T, N>& v2) const {
			Vec<T, N> v3;
			for (size_t i = 0; i < N; i++)
				v3[i] = this->_Elems[i] * v2[i];
			return v3;
		}
		const Vec<T, N>& operator -= (const Vec<T, N>& v2) {
			for (size_t i = 0; i < N; i++)
				this->_Elems[i] -= v2[i];
			return *this;
		}
		const Vec<T, N>& operator += (const Vec<T, N>& v2) {
			for (size_t i = 0; i < N; i++)
				this->_Elems[i] += v2[i];
			return *this;
		}
		const bool operator==(const Vec<T, N>& v2) {
			for (size_t i = 0; i < N; i++) {
				if (this->_Elems[i] != v2[i])
					return false;
			}
			return true;
		}
		void fill(const T v) {
			for (T* p = this->_Elems, *pend = this->_Elems + N; p != pend; p++)
				*p = v;
		}

		T norm2() const {
			T ret = static_cast<T>(0);
			for (size_t i = 0; i < N; i++)
				ret += this->_Elems[i] * this->_Elems[i];
			return ret;
		}
		T norm() const {
			return Sqrt(norm2());
		}
		Vec<T, N> normalize() const {
			T nrm = norm();
			Vec<T, N> ret;
			for (size_t i = 0; i < N; i++)
				ret[i] = this->_Elems[i] / nrm;
			return ret;
		}

		static const Vec<T, 3> cross(Vec<T, 3> v1, Vec<T, 3>v2) {
			return { v1[1] * v2[2] - v1[2] * v2[1],
				v1[2] * v2[0] - v1[0] * v2[2],
				v1[0] * v2[1] - v1[1] * v2[0]
			};
		}

		static const T dot(const Vec<T, N>& v1, const Vec<T, N>& v2) {
			T ret(0.);
			for (size_t i = 0; i < N; i++)
				ret += v1[i] * v2[i];
			return ret;
		}

		template<typename T2>
		Vec<T2, N> cast_to() const {
			Vec<T2, N> ret;
			for (size_t i = 0; i < N; i++)
				ret[i] = static_cast<T2>(this->_Elems[i]);
			return ret;
		}

		~Vec() { }

		bool constexpr Valid() const {
			return !Decl::IsNaN<T>(this->_Elems[0]);
		}
	};

	template<typename T, size_t N>
	std::ostream& operator << (std::ostream& ostrm, const Vec<T, N>& v) {
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