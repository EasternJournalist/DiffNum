#pragma once

#include <cuda_runtime.h>  //Í·ÎÄ¼þ
#include <ostream>
#include <sstream>
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
	/// The gradients on target variable must be specified before any computation. You may use DiffVar_cuda.setVar or 
	/// class DiffManager to help initalize and deal with the variables that you study. 
	/// </summary>
	/// <typeparam name="n_type">The type of numerical value. eg. float, double</typeparam>
	/// <typeparam name="size"> The size of gradient vector </typeparam>
	template <typename d_type, size_t size>
	struct DiffVar_cuda {
		static_assert(size > 0, "The size of DiffVar_cuda gradients must be a positive integer.");

		using s_type = DiffVar_cuda<d_type, size>;
		using n_type = d_type;

		d_type value;
		array_cuda<d_type, size> gradient;

		__host__ __device__ n_type getValue() const {
			return value;
		}

		__host__ __device__ const d_type& operator[](size_t var_idx) const {
			return gradient[var_idx];
		}

		__host__ __device__ DiffVar_cuda() {}


		__host__ __device__ DiffVar_cuda(const n_type value) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}


		__host__ __device__ DiffVar_cuda(n_type value, const array_cuda<n_type, size>& gradient) : value(value), gradient(gradient) {}


		__host__ __device__ DiffVar_cuda(const n_type value, const size_t as_var_idx) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		__host__ __device__ const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}


		__host__ __device__ const s_type& operator=(const n_type _Right) {
			value = _Right;
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}

		__host__ __device__ void setVar(const size_t as_var_idx) {
			assert(as_var_idx < size);
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		__host__ __device__ s_type operator+(const s_type& _Right) const {

			s_type ret;
			ret.value = value + _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] + _Right.gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator-(const s_type& _Right) const {
			s_type ret;
			ret.value = value - _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] - _Right.gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator-() const {
			s_type ret;
			ret.value = -value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = -gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator*(const s_type& _Right) const {
			s_type ret;
			ret.value = value * _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator/(const s_type& _Right) const {
			s_type ret;
			ret.value = value / _Right.value;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);

			return ret;
		}


		__host__ __device__ s_type operator+(const n_type _Right) const {
			s_type ret(value + _Right, gradient);
			return ret;
		}


		__host__ __device__ static friend inline s_type operator+(const n_type _Left, const s_type& _Right) {
			return _Right + _Left;
		}


		__host__ __device__ s_type operator-(const n_type _Right) const {
			s_type ret(value - _Right, gradient);
			return ret;
		}


		__host__ __device__ static friend inline s_type operator-(const n_type _Left, const s_type& _Right) {
			s_type ret(_Left - _Right.value, _Right.size);

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}


		__host__ __device__ s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;

			for (size_t i = 0; i < size; i++) 
				ret.gradient[i] = gradient[i] * _Right;
			return ret;
		}


		__host__ __device__ static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}


		__host__ __device__ s_type operator/(const n_type _Right) const {
			s_type ret;
			ret.value = value / _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}


		__host__ __device__ static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}


		__host__ __device__ const s_type& operator+=(const s_type& _Right) {
			value += _Right.value;
			for (size_t i = 0; i < size; i++) 
				gradient[i] += _Right.gradient[i];
			return *this;
		}


		__host__ __device__ const s_type& operator-=(const s_type& _Right) {
			value -= _Right.value;
			for (size_t i = 0; i < size; i++) 
				gradient[i] -= _Right.gradient[i];
			return *this;
		}


		__host__ __device__ const s_type& operator*=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			value *= _Right.value;
			return *this;
		}


		__host__ __device__ const s_type& operator/=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			value /= _Right.value;
			return *this;
		}


		__host__ __device__ const s_type& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}


		__host__ __device__ const s_type& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}


		__host__ __device__ const s_type& operator*=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] *= _Right;
			value *= _Right;
			return *this;
		}


		__host__ __device__ const s_type& operator/=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) 
				gradient[i] /= _Right;
			value /= _Right;
			return *this;
		}

		__host__ const std::string toString() const {
			std::stringstream ss;
			ss << value << "(";
			for (size_t i = 0; i < size - 1; i++)
				ss << gradient[i] << ", ";
			ss << gradient.back() << ")";
			return ss.str();
		}


		__host__ const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			for (size_t i = 0; i < size - 1; i++)
				ss << gradient[i] << ", ";
			ss << gradient.back() << ")";
			return ss.str();
		}

	};

	template <typename r_type, size_t size, size_t r_size>
	struct DiffVar_cuda<DiffVar_cuda<r_type, r_size>, size> {
		static_assert(size == r_size, "The sizes of gradients must be the same when using DiffVar recursively.");

		using n_type = typename DiffVar_cuda<r_type, size>::n_type;
		using d_type = DiffVar_cuda<r_type, size>;
		using s_type = DiffVar_cuda<d_type, size>;

		d_type value;
		array_cuda<d_type, size> gradient;

		__host__ __device__ n_type getValue() const {
			return value.getValue();
		}

		__host__ __device__ const d_type& operator[](size_t var_idx) const {
			return gradient[var_idx];
		}

		__host__ __device__ DiffVar_cuda() {}


		__host__ __device__ DiffVar_cuda(const n_type value) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}


		__host__ __device__ DiffVar_cuda(n_type value, const array_cuda<d_type, size>& gradient) : value(value), gradient(gradient) {}


		__host__ __device__ DiffVar_cuda(const n_type value, const size_t as_var_idx) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		__host__ __device__ const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}


		__host__ __device__ const s_type& operator=(const n_type _Right) {
			value = _Right;
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}

		__host__ __device__ void setVar(const size_t as_var_idx) {
			assert(as_var_idx < size);
			value.setVar(as_var_idx);
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		__host__ __device__ s_type operator+(const s_type& _Right) const {

			s_type ret;
			ret.value = value + _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = gradient[i] + _Right.gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator-(const s_type& _Right) const {
			s_type ret;
			ret.value = value - _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = gradient[i] - _Right.gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator-() const {
			s_type ret;
			ret.value = -value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator*(const s_type& _Right) const {
			s_type ret;
			ret.value = value * _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];

			return ret;
		}


		__host__ __device__ s_type operator/(const s_type& _Right) const {
			s_type ret;
			ret.value = value / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);

			return ret;
		}


		__host__ __device__ s_type operator+(const n_type _Right) const {
			s_type ret(value + _Right, gradient);
			return ret;
		}


		__host__ __device__ static friend inline s_type operator+(const n_type _Left, const s_type& _Right) {
			return _Right + _Left;
		}


		__host__ __device__ s_type operator-(const n_type _Right) const {
			s_type ret(value - _Right, gradient);
			return ret;
		}


		__host__ __device__ static friend inline s_type operator-(const n_type _Left, const s_type& _Right) {
			s_type ret(_Left - _Right.value, _Right.size);

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}


		__host__ __device__ s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = gradient[i] * _Right;
			return ret;
		}


		__host__ __device__ static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}


		__host__ __device__ s_type operator/(const n_type _Right) const {
			s_type ret;
			ret.value = value / _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}


		__host__ __device__ static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}


		__host__ __device__ const s_type& operator+=(const s_type& _Right) {
			value += _Right.value;
			for (size_t i = 0; i < size; i++)
				gradient[i] += _Right.gradient[i];
			return *this;
		}


		__host__ __device__ const s_type& operator-=(const s_type& _Right) {
			value -= _Right.value;
			for (size_t i = 0; i < size; i++)
				gradient[i] -= _Right.gradient[i];
			return *this;
		}


		__host__ __device__ const s_type& operator*=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++)
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			value *= _Right.value;
			return *this;
		}


		__host__ __device__ const s_type& operator/=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++)
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			value /= _Right.value;
			return *this;
		}


		__host__ __device__ const s_type& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}


		__host__ __device__ const s_type& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}


		__host__ __device__ const s_type& operator*=(const n_type _Right) {
			for (size_t i = 0; i < size; i++)
				gradient[i] *= _Right;
			value *= _Right;
			return *this;
		}


		__host__ __device__ const s_type& operator/=(const n_type _Right) {
			for (size_t i = 0; i < size; i++)
				gradient[i] /= _Right;
			value /= _Right;
			return *this;
		}

		__host__ const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			for (size_t i = 0; i < size - 1; i++)
				ss << gradient[i].toString_grad() << ", ";
			ss << gradient.back().toString_grad() << ")";
			return ss.str();
		}


		__host__ const std::string toString() const {
			std::stringstream ss;
			ss << value.toString();
			ss << toString_grad();
			return ss.str();
		}

	};


	template<typename n_type, size_t size>
	std::ostream& operator << (std::ostream& ostrm, const DiffVar_cuda<n_type, size>& v) {
		ostrm << v.value << "(";
		for (size_t i = 0; i < size - 1; i++)
			ostrm << v.gradient[i] << ", ";
		ostrm << v.gradient[size - 1] << ")";
		return ostrm;
	}
}