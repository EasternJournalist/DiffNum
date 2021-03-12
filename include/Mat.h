#pragma once

#include <Common.h>
#include <Vec.h>


namespace Common {
	// 小型稠密矩阵的模板
	template<typename T, ptrdiff_t NRow, ptrdiff_t NCol>
	struct mat : public array<array<T, NCol>, NRow>{
		using array_type = array<array<T, NCol>, NRow>;
		static const ptrdiff_t _Size = NRow * NCol;

		__HOST_DEVICE__ mat() : array_type() {}

		__HOST_DEVICE__ mat(std::initializer_list<T> init_list) {
			ptrdiff_t i = 0;
			for (auto it : init_list)
				((T*)this)[i++] = it;
		}

		__HOST_DEVICE__ mat(const mat<T, NRow, NCol>& _Right) :array_type(_Right) { }

		__HOST_DEVICE__ mat(const T val) { fill(val);}

		__HOST_DEVICE__ ~mat() { }

		__HOST_DEVICE__ const mat<T, NRow, NCol>& operator =(const mat<T, NRow, NCol>& m) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = ((T*)&m)[i];
			return (*this);
		}
		__HOST_DEVICE__ const mat<T, NRow, NCol>& operator =(const array<vec<T, NCol>, NRow>& m) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = ((T*)&m)[i];
			return (*this);
		}

		__HOST_DEVICE__ void fill(const T _Val) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = _Val;
		}

		__HOST_DEVICE__  vec<T, NRow> dot(const vec<T, NCol>& v) const {
			vec<T, NRow> ret;
			for (ptrdiff_t i = 0; i < NRow; i++) {
				ret[i] = T(0);
				for (ptrdiff_t j = 0; j < NCol; j++)
					ret[i] += (*this)[i][j] * v[j];
			}
			return ret;
		}

		__HOST_DEVICE__ mat<T, NRow, NCol> operator*(const T t) const {
			mat<T, NRow, NCol> ret;
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)(&ret))[i] = ((T*)this)[i] * t;
			return ret;
		}

		__HOST_DEVICE__ mat<T, NRow, NCol> operator+(const mat<T, NRow, NCol>& m2) const {
			mat<T, NRow, NCol> ret;
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)(&ret))[i] = ((T*)this)[i] + ((T*)&m2)[i];
			return ret;
		}

		__HOST_DEVICE__ mat<T, NRow, NCol> operator-(const mat<T, NRow, NCol>& m2) const {
			mat<T, NRow, NCol> ret;
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)(&ret))[i] = ((T*)this)[i] - ((T*)&m2)[i];
			return ret;
		}

		template<ptrdiff_t L>
		__HOST_DEVICE__ mat<T, NRow, L> operator*(const mat<T, NCol, L>& m2) const {
			mat<T, NRow, L> ret;
			ret.fill(T(0));
			for (ptrdiff_t i = 0; i < NRow; i++)
				for (ptrdiff_t j = 0; j < L; j++)
					for (ptrdiff_t k = 0; k < NCol; k++)
						ret[i][j] += (*this)[i][k] * m2[k][j];
			return ret;
		}

		__HOST_DEVICE__ mat<T, NCol, NRow> transpose() const {
			mat<T, NCol, NRow> ret;
			for (ptrdiff_t i = 0; i < NRow; i++)
				for (ptrdiff_t j = 0; j < NCol; j++)
					ret[j][i] = (*this)[i][j];
			return ret;
		}

		__HOST_DEVICE__ mat<T, NCol, NRow> inverse() const {
			static_assert(NRow == NCol, "Inverse of non-square matrix is not supported");
			mat<T, NCol, NRow> ret;
			mat<T, NCol, NRow> ma(*this);
			ptrdiff_t firstrownot0, i, j;

			for (i = 0; i < NCol; i++) {
				for (firstrownot0 = i; T(0) == ma[firstrownot0][i] && firstrownot0 < NCol; firstrownot0++);
				assert(firstrownot0 >= NCol && "Singular matrix.");
				if (firstrownot0 != i) {
					ma.SR(i, firstrownot0);
					ret.SR(i, firstrownot0);
				}
				for (j = i + 1; j < NCol; j++) {
					if (ma(j, i) != 0) {
						ma.TR(j, i, -ma[j][i] / ma[i][i]);
						ret.TR(j, i, -ma[j][i] / ma[i][i]);
					}
				}
			}
			for (i = NCol; i >= 0; i--) {
			 	ret.DR(i, T(1) / ma(i, i));
				for (j = i - 1; j >= 1; j--) {
					if (ma[j][i] != 0)
						ret.TR(j, i, -ma[j][i]);
				}
			}
			return ret;
		}

		__HOST_DEVICE__ void SR(ptrdiff_t i, ptrdiff_t j) {
			for (ptrdiff_t k = 0; k < NCol; k++) {
				T _Tmp = (*this)[i][k];
				(*this)[i][k] = (*this)[j][k];
				(*this)[j][k] = _Tmp;
			}
		}

		__HOST_DEVICE__ void TR(ptrdiff_t i, ptrdiff_t j, const T val) {
			for (ptrdiff_t k = 0; k < NCol; k++) {
				(*this)[i][k] += val * (*this)[j][k];
			}
		}

		__HOST_DEVICE__ void DR(ptrdiff_t i, const T val) {
			for (ptrdiff_t k = 0; k < NCol; k++) {
				(*this)[i][k] *= val;
			}
		}

		__HOST_DEVICE__ T dot_s(const vec<T, NRow>& v1, const vec<T, NCol>& v2) const {
			T ret(0);
			for (ptrdiff_t i = 0; i < NRow; i++)
				for (ptrdiff_t j = 0; j < NCol; j++)
					ret += v1[i] * (*this)[i][j] * v2[j];
			return ret;
		}

		__HOST_DEVICE__ T dot_ss(const vec<T, NRow>& v) const {
			T ret(0);
			for (ptrdiff_t i = 0; i < NRow; i++)
				for (ptrdiff_t j = 0; j < NRow; j++)
					ret += v[i] * v[j] * (*this)[i][j];
			return ret;
		}

		__HOST_DEVICE__ mat<T, NRow, NCol> Identity() {
			static_assert(NRow == NCol, "Identity matrix should be square matrix.");
			mat<T, NRow, NCol> m;
			m.fill(T(0));
			for (ptrdiff_t i = 0; i < NRow; i++)
				m[i][i] = T(1);
			return m;
		}

		__HOST_DEVICE__ static mat<T, NRow, NRow> Diag(const vec<T, NRow>& diag) {
			mat<T, NRow, NRow> ret(0.f);
			for (ptrdiff_t i = 0; i < NRow; i++)
				ret[i][i] = diag[i];
			return ret;
		}

		__HOST_DEVICE__ T det(void) const {
			static_assert(NRow == NCol, "Identity matrix should be square matrix.");
			T ans(1);
			mat<T, NCol, NRow> ma(*this);
			for (ptrdiff_t i = 0; i < NCol; i++) {
				ptrdiff_t firstrownot0;
				for (firstrownot0 = i;  T(0) == ma[firstrownot0][i] && firstrownot0 < NRow; firstrownot0++);
				if (firstrownot0 == NRow) firstrownot0 = i;
				ans *= ma[firstrownot0][i];
				if (firstrownot0 != i) {
					for (ptrdiff_t k = i + 1; k < NCol; k++) {
						T _Tmp = ma[i][k];
						ma[i][k] = ma[firstrownot0][k];
						ma[firstrownot0][k] = _Tmp;
					}
				}
				for (ptrdiff_t j = i + 1; j < NRow; j++) {
					if (ma[j][i] != T(0)) {
						for (ptrdiff_t k = i + 1; k < NCol; k++)
							ma[j][k] += (-ma[j][i] / ma[i][i]) * ma[i][k];
					}
				}
			}
		}

		__HOST_DEVICE__ constexpr ptrdiff_t size() const {
			return _Size;
		}

		__HOST_DEVICE__ T* data() const {
			return (T*)this;
		}

	};


	template<typename T>
	struct mat<T, 3, 3> : public array<array<T, 3>, 3>{
		using array_type = array<array<T, 3>, 3>;
		static const ptrdiff_t _Size = 9;

		__HOST_DEVICE__ mat() : array_type() {}

		__HOST_DEVICE__ mat(std::initializer_list<T> init_list) {
			ptrdiff_t i = 0;
			for (auto it : init_list)
				((T*)this)[i++] = it;
		}

		__HOST_DEVICE__ mat(const mat<T, 3, 3>& _Right) :array_type(_Right) { }

		__HOST_DEVICE__ mat(const T val) { fill(val); }

		__HOST_DEVICE__ ~mat() { }

		__HOST_DEVICE__ const mat<T, 3, 3>& operator =(const mat<T, 3, 3>& m) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = ((T*)&m)[i];
			return (*this);
		}
		__HOST_DEVICE__ const mat<T, 3, 3>& operator =(const array<vec<T, 3>, 3>& m) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = ((T*)&m)[i];
			return (*this);
		}

		__HOST_DEVICE__ void fill(const T _Val) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = _Val;
		}

		__HOST_DEVICE__  vec<T, 3> dot(const vec<T, 3>& v) const {
			vec<T, 3> ret;
			for (ptrdiff_t i = 0; i < 3; i++) {
				ret[i] = T(0);
				for (ptrdiff_t j = 0; j < 3; j++)
					ret[i] += (*this)[i][j] * v[j];
			}
			return ret;
		}

		__HOST_DEVICE__ mat<T, 3, 3> operator*(const T t) const {
			mat<T, 3, 3> ret;
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)(&ret))[i] = ((T*)this)[i] * t;
			return ret;
		}

		__HOST_DEVICE__ mat<T, 3, 3> operator+(const mat<T, 3, 3>& m2) const {
			mat<T, 3, 3> ret;
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)(&ret))[i] = ((T*)this)[i] + ((T*)&m2)[i];
			return ret;
		}

		__HOST_DEVICE__ mat<T, 3, 3> operator-(const mat<T, 3, 3>& m2) const {
			mat<T, 3, 3> ret;
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)(&ret))[i] = ((T*)this)[i] - ((T*)&m2)[i];
			return ret;
		}

		template<ptrdiff_t L>
		__HOST_DEVICE__ mat<T, 3, L> operator*(const mat<T, 3, L>& m2) const {
			mat<T, 3, L> ret;
			ret.fill(T(0));
			for (ptrdiff_t i = 0; i < 3; i++)
				for (ptrdiff_t j = 0; j < L; j++)
					for (ptrdiff_t k = 0; k < 3; k++)
						ret[i][j] += (*this)[i][k] * m2[k][j];
			return ret;
		}

		__HOST_DEVICE__ mat<T, 3, 3> transpose() const {
			mat<T, 3, 3> ret;
			for (ptrdiff_t i = 0; i < 3; i++)
				for (ptrdiff_t j = 0; j < 3; j++)
					ret[j][i] = (*this)[i][j];
			return ret;
		}

		__HOST_DEVICE__ mat<T, 3, 3> inverse() const {
			mat<T, 3, 3> ret;
			for (ptrdiff_t i = 0; i < 3; i++)
				for (ptrdiff_t j = 0; j < 3; j++)
					ret[j][i] = ((*this)[(i + 1) % 3][(j + 1) % 3] * (*this)[(i + 2) % 3][(j + 2) % 3] - (*this)[(i + 1) % 3][(j + 2) % 3] * (*this)[(i + 2) % 3][(j + 1) % 3]);
			T _Det = 0.f;
			for (ptrdiff_t i = 0; i < 3; i++) {
				_Det += (*this)[0][i] * ret[i][0];
			}
			for (ptrdiff_t i = 0; i < 3; i++)
				for (ptrdiff_t j = 0; j < 3; j++)
					ret[i][j] /= _Det;
			return ret;
		}

		__HOST_DEVICE__ void SR(ptrdiff_t i, ptrdiff_t j) {
			for (ptrdiff_t k = 0; k < 3; k++) {
				T _Tmp = (*this)[i][k];
				(*this)[i][k] = (*this)[j][k];
				(*this)[j][k] = _Tmp;
			}
		}

		__HOST_DEVICE__ void TR(ptrdiff_t i, ptrdiff_t j, const T val) {
			for (ptrdiff_t k = 0; k < 3; k++) {
				(*this)[i][k] += val * (*this)[j][k];
			}
		}

		__HOST_DEVICE__ void DR(ptrdiff_t i, const T val) {
			for (ptrdiff_t k = 0; k < 3; k++) {
				(*this)[i][k] *= val;
			}
		}

		__HOST_DEVICE__ T dot_s(const vec<T, 3>& v1, const vec<T, 3>& v2) const {
			T ret(0);
			for (ptrdiff_t i = 0; i < 3; i++)
				for (ptrdiff_t j = 0; j < 3; j++)
					ret += v1[i] * (*this)[i][j] * v2[j];
			return ret;
		}

		__HOST_DEVICE__ T dot_ss(const vec<T, 3>& v) const {
			T ret(0);
			for (ptrdiff_t i = 0; i < 3; i++)
				for (ptrdiff_t j = 0; j < 3; j++)
					ret += v[i] * v[j] * (*this)[i][j];
			return ret;
		}

		__HOST_DEVICE__ mat<T, 3, 3> Identity() {
			mat<T, 3, 3> m;
			m.fill(T(0));
			for (ptrdiff_t i = 0; i < 3; i++)
				m[i][i] = T(1);
			return m;
		}

		__HOST_DEVICE__ static mat<T, 3, 3> Diag(const vec<T, 3>& diag) {
			mat<T, 3, 3> ret(0.f);
			for (ptrdiff_t i = 0; i < 3; i++)
				ret[i][i] = diag[i];
			return ret;
		}

		__HOST_DEVICE__ T det(void) const {
			return (*this)[0][0] * ((*this)[1][1] * (*this)[2][2] - (*this)[1][2] * (*this)[2][1]) - (*this)[0][1] * ((*this)[1][0] * (*this)[2][2] - (*this)[1][2] * (*this)[2][0]) + (*this)[0][2] * ((*this)[1][0] * (*this)[2][1] - (*this)[1][1] * (*this)[2][0]);
		}

		__HOST_DEVICE__ constexpr ptrdiff_t size() const {
			return _Size;
		}

		__HOST_DEVICE__ T* data() const {
			return (T*)this;
		}
	};


	template<typename T, ptrdiff_t NRow, ptrdiff_t NCol>
	std::ostream& operator << (std::ostream& ostrm, const mat<T, NRow, NCol>& m) {
		for (ptrdiff_t i = 0; i < NRow; i++) {
			for (ptrdiff_t j = 0; j < NCol; j++)
				ostrm << m[i][j] << ", ";

			ostrm << std::endl;
		}
		return ostrm;
	}

	typedef mat<float, 3, 3> matf3;
	typedef mat<float, 4, 4> matf4;
	typedef mat<float, 4, 3> matf43;
}