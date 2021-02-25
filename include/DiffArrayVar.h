#pragma once

#include <array>
#include <assert.h>
#include <ostream>

namespace DiffNum {
	/// <summary>
	/// Differentiable varible (fixed variable to be studied, and faster). The numerical value and the derivatives will be automatically evaluated simultaneously.
	/// The gradients on target variable must be specified before any computation. You may use DiffArrayVar.SetVar or 
	/// class DiffManager to help initalize and deal with the variables that you study. 
	/// </summary>
	/// <typeparam name="n_type">The type of numerical value. eg. float, double</typeparam>
	/// <typeparam name="size"> The size of gradient vector </typeparam>
	template <typename n_type, size_t size>
	struct DiffArrayVar {
		
		DiffArrayVar() {}
		DiffArrayVar(const n_type value) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}
		DiffArrayVar(n_type value, const std::array<n_type, size>& gradient) : value(value), gradient(gradient) {}
		DiffArrayVar(n_type value, size_t as_var_idx) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		const DiffArrayVar<n_type, size>& operator=(const DiffArrayVar<n_type, size>& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator=(const n_type _Right) {
			value = _Right;
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}

		void SetVar(const size_t num_var, const size_t as_var_idx) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		DiffArrayVar<n_type, size> operator+(const DiffArrayVar<n_type, size>& _Right) const {

			DiffArrayVar<n_type, size> ret;
			ret.value = value + _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}


		DiffArrayVar<n_type, size> operator-(const DiffArrayVar<n_type, size>& _Right) const {
			DiffArrayVar<n_type, size> ret;
			ret.value = value - _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}


		DiffArrayVar<n_type, size> operator-() const {
			DiffArrayVar<n_type, size> ret;
			ret.value = -value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}


		DiffArrayVar<n_type, size> operator*(const DiffArrayVar<n_type, size>& _Right) const {
			DiffArrayVar<n_type, size> ret;
			ret.value = value * _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}


		DiffArrayVar<n_type, size> operator/(const DiffArrayVar<n_type, size>& _Right) const {
			DiffArrayVar<n_type, size> ret;
			ret.value = value / _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			return ret;
		}


		DiffArrayVar<n_type, size> operator+(const n_type _Right) const {
			DiffArrayVar<n_type, size> ret(value + _Right, gradient);
			return ret;
		}


		static friend inline DiffArrayVar<n_type, size> operator+(const n_type _Left, const DiffArrayVar<n_type, size>& _Right) {
			return _Right + _Left;
		}


		DiffArrayVar<n_type, size> operator-(const n_type _Right) const {
			DiffArrayVar<n_type, size> ret(value - _Right, gradient);
			return ret;
		}


		static friend inline DiffArrayVar<n_type, size> operator-(const n_type _Left, const DiffArrayVar<n_type, size>& _Right) {
			DiffArrayVar<n_type, size> ret(_Left - _Right.value, _Right.size);

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}


		DiffArrayVar<n_type, size> operator*(const n_type _Right) const {
			DiffArrayVar<n_type, size> ret;
			ret.value = value * _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}


		static friend inline DiffArrayVar<n_type, size> operator*(const n_type _Left, const DiffArrayVar<n_type, size>& _Right) {
			return _Right * _Left;
		}


		DiffArrayVar<n_type, size> operator/(const n_type _Right) const {
			DiffArrayVar<n_type, size> ret;
			ret.value = value / _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}


		static friend inline DiffArrayVar<n_type, size> operator/(const n_type _Left, const DiffArrayVar<n_type, size>& _Right) {
			DiffArrayVar<n_type, size> ret;
			ret.value = _Left / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}


		const DiffArrayVar<n_type, size>& operator+=(const DiffArrayVar<n_type, size>& _Right) {
			value += _Right.value;
			for (size_t i = 0; i < size; i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator-=(const DiffArrayVar<n_type, size>& _Right) {
			value -= _Right.value;
			for (size_t i = 0; i < size; i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator*=(const DiffArrayVar<n_type, size>& _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator/=(const DiffArrayVar<n_type, size>& _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			value /= _Right.value;
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator*=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}


		const DiffArrayVar<n_type, size>& operator/=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}


		n_type value;
		std::array<n_type, size> gradient;
	};

	template<typename n_type, size_t size>
	std::ostream& operator << (std::ostream& ostrm, const DiffArrayVar<n_type, size>& v) {
		ostrm << v.value << "(";
		for (size_t i = 0; i < size - 1; i++)
			ostrm << v.gradient[i] << ", ";
		ostrm << v.gradient.back() << ")";
		return ostrm;
	}
}