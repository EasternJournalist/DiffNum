#pragma once

#include <cuda_runtime.h>  //Í·ÎÄ¼þ
#include <ostream>
#include <assert.h>

namespace DiffNum {

	template<class T, size_t N>
	struct array_cuda {

		T _Elems[N];


		__host__ __device__ array_cuda() {}


		__host__ __device__ array_cuda(const array_cuda<T, N>& _Right) {
			for (size_t i = 0; i < N; i++)
				_Elems[i] = _Right._Elems[i];
		}


		__host__ __device__ const T& operator[](const size_t _Pos) const {
			return _Elems[_Pos];
		}


		__host__ __device__ T& operator[](const size_t _Pos) {
			return _Elems[_Pos];
		}


		__host__ __device__ const array_cuda<T, N>& operator=(const array_cuda<T, N>& _Right) {
			for (size_t i = 0; i < N; i++)
				_Elems[i] = _Right._Elems[i];
			return *this;
		}
	};

	/// <summary>
	/// Differentiable varible (fixed variable to be studied, and faster). The numerical value and the derivatives will be automatically evaluated simultaneously.
	/// The gradients on target variable must be specified before any computation. You may use DiffArrayVar_cuda.SetVar or 
	/// class DiffManager to help initalize and deal with the variables that you study. 
	/// </summary>
	/// <typeparam name="n_type">The type of numerical value. eg. float, double</typeparam>
	/// <typeparam name="size"> The size of gradient vector </typeparam>
	template <typename n_type, size_t size>
	struct DiffArrayVar_cuda {

		n_type value;
		array_cuda<n_type, size> gradient;


		__host__ __device__ DiffArrayVar_cuda() {}


		__host__ __device__ DiffArrayVar_cuda(const n_type value) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}


		__host__ __device__ DiffArrayVar_cuda(n_type value, const array_cuda<n_type, size>& gradient) : value(value), gradient(gradient) {}


		__host__ __device__ DiffArrayVar_cuda(const n_type value, const size_t as_var_idx) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator=(const DiffArrayVar_cuda<n_type, size>& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator=(const n_type _Right) {
			value = _Right;
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}

		__host__ __device__ void SetVar(const size_t as_var_idx) {
			assert(as_var_idx < size);
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator+(const DiffArrayVar_cuda<n_type, size>& _Right) const {

			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = value + _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] + _Right.gradient[i];

			return ret;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator-(const DiffArrayVar_cuda<n_type, size>& _Right) const {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = value - _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] - _Right.gradient[i];

			return ret;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator-() const {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = -value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = -gradient[i];

			return ret;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator*(const DiffArrayVar_cuda<n_type, size>& _Right) const {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = value * _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];

			return ret;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator/(const DiffArrayVar_cuda<n_type, size>& _Right) const {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = value / _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);

			return ret;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator+(const n_type _Right) const {
			DiffArrayVar_cuda<n_type, size> ret(value + _Right, gradient);
			return ret;
		}


		__host__ __device__ static friend inline DiffArrayVar_cuda<n_type, size> operator+(const n_type _Left, const DiffArrayVar_cuda<n_type, size>& _Right) {
			return _Right + _Left;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator-(const n_type _Right) const {
			DiffArrayVar_cuda<n_type, size> ret(value - _Right, gradient);
			return ret;
		}


		__host__ __device__ static friend inline DiffArrayVar_cuda<n_type, size> operator-(const n_type _Left, const DiffArrayVar_cuda<n_type, size>& _Right) {
			DiffArrayVar_cuda<n_type, size> ret(_Left - _Right.value, _Right.size);

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator*(const n_type _Right) const {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = value * _Right;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] * _Right;
			return ret;
		}


		__host__ __device__ static friend inline DiffArrayVar_cuda<n_type, size> operator*(const n_type _Left, const DiffArrayVar_cuda<n_type, size>& _Right) {
			return _Right * _Left;
		}


		__host__ __device__ DiffArrayVar_cuda<n_type, size> operator/(const n_type _Right) const {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = value / _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}


		__host__ __device__ static friend inline DiffArrayVar_cuda<n_type, size> operator/(const n_type _Left, const DiffArrayVar_cuda<n_type, size>& _Right) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = _Left / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator+=(const DiffArrayVar_cuda<n_type, size>& _Right) {
			value += _Right.value;
			for (size_t i = 0; i < size; i++) 
				gradient[i] += _Right.gradient[i];
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator-=(const DiffArrayVar_cuda<n_type, size>& _Right) {
			value -= _Right.value;
			for (size_t i = 0; i < size; i++) 
				gradient[i] -= _Right.gradient[i];
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator*=(const DiffArrayVar_cuda<n_type, size>& _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			value *= _Right.value;
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator/=(const DiffArrayVar_cuda<n_type, size>& _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			value /= _Right.value;
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator*=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] *= _Right;
			value *= _Right;
			return *this;
		}


		__host__ __device__ const DiffArrayVar_cuda<n_type, size>& operator/=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] /= _Right;
			value /= _Right;
			return *this;
		}

	};

	template<typename n_type, size_t size>
	std::ostream& operator << (std::ostream& ostrm, const DiffArrayVar_cuda<n_type, size>& v) {
		ostrm << v.value << "(";
		for (size_t i = 0; i < size - 1; i++)
			ostrm << v.gradient[i] << ", ";
		ostrm << v.gradient[size - 1] << ")";
		return ostrm;
	}
}