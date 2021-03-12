#pragma once

#include <assert.h>
#include <ostream>
#include <sstream>
#include <Common.h>
#include <myarray.h>

namespace DiffNum {


	/// <summary>
	/// Differentiable varible (fixed variable to be studied, and faster). The numerical value and the derivatives will be automatically evaluated simultaneously.
	/// The gradients on target variable must be specified before any computation. You may use DiffVar.setVar or 
	/// class DiffManager to help initalize and deal with the variables that you study. 
	/// </summary>
	/// <typeparam name="d_type">The type of numerical value. eg. float, double</typeparam>
	/// <typeparam name="size"> The size of gradient vector </typeparam>
	template <class d_type, ptrdiff_t size>
	struct DiffVar {
		static_assert(size > 0, "The size of array-like DiffVar gradients must be a positive integer.\
			Do you refer to vector-like DiffVar? You may forget to include \"DiffVar_vec.h\".");
		
		using n_type = d_type;
		using s_type = DiffVar<d_type, size>;

		__HOST_DEVICE__ DiffVar() {}
		__HOST_DEVICE__ DiffVar(const n_type value) : value(value) {
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}
		__HOST_DEVICE__ DiffVar(n_type value, const Common::array<n_type, size>& gradient) : value(value), gradient(gradient) {}
		__HOST_DEVICE__ DiffVar(n_type value, ptrdiff_t as_var_idx) : value(value) {
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}

		__HOST_DEVICE__ n_type getValue() const {
			return value;
		}

		__HOST_DEVICE__ bool operator <(const s_type& _Right) const {
			return value < _Right.value;
		}

		__HOST_DEVICE__ bool operator <=(const s_type& _Right) const {
			return value <= _Right.value;
		}

		__HOST_DEVICE__ bool operator >(const s_type& _Right) const {
			return value > _Right.value;
		}

		__HOST_DEVICE__ bool operator >=(const s_type& _Right) const {
			return value >= _Right.value;
		}

		__HOST_DEVICE__ bool operator ==(const s_type& _Right) const {
			return value == _Right.value;
		}

		__HOST_DEVICE__ bool operator !=(const s_type& _Right) const {
			return value != _Right.value;
		}

		__HOST_DEVICE__ bool operator <(const n_type _Right) const {
			return value < _Right;
		}

		__HOST_DEVICE__ bool operator <=(const n_type _Right) const {
			return value <= _Right;
		}

		__HOST_DEVICE__ bool operator >(const n_type _Right) const {
			return value > _Right;
		}

		__HOST_DEVICE__ bool operator >=(const n_type _Right) const {
			return value >= _Right;
		}

		__HOST_DEVICE__ bool operator ==(const n_type _Right) const {
			return value == _Right;
		}

		__HOST_DEVICE__ bool operator !=(const n_type _Right) const {
			return value != _Right;
		}

		__HOST_DEVICE__ const d_type& operator[](ptrdiff_t var_idx) const {
			return gradient[var_idx];
		}

		__HOST_DEVICE__ const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator=(const n_type _Right) {
			value = _Right;
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}

		__HOST_DEVICE__ void setVar(const ptrdiff_t as_var_idx) {
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}

		__HOST_DEVICE__ s_type operator+(const s_type& _Right) const {

			s_type ret;
			ret.value = value + _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator-(const s_type& _Right) const {
			s_type ret;
			ret.value = value - _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator-() const {
			s_type ret;
			ret.value = -value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator*(const s_type& _Right) const {
			s_type ret;
			ret.value = value * _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator/(const s_type& _Right) const {
			s_type ret;
			ret.value = value / _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator+(const n_type _Right) const {
			s_type ret(value + _Right, gradient);
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator+(const n_type _Left, const s_type& _Right) {
			return _Right + _Left;
		}

		__HOST_DEVICE__ s_type operator-(const n_type _Right) const {
			s_type ret(value - _Right, gradient);
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator-(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left - _Right.value;

			for (ptrdiff_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}

		__HOST_DEVICE__ s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}

		__HOST_DEVICE__ s_type operator/(const n_type _Right) const {
			s_type ret;
			ret.value = value / _Right;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;

			for (ptrdiff_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}

		__HOST_DEVICE__ const s_type& operator+=(const s_type& _Right) {
			value += _Right.value;
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator-=(const s_type& _Right) {
			value -= _Right.value;
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator*=(const s_type& _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator/=(const s_type& _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			value /= _Right.value;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator*=(const n_type _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator/=(const n_type _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}

		__HOST_ONLY__ const std::string toString() const {
			std::stringstream ss;
			ss << value << "(";
			for (ptrdiff_t i = 0; i < size - 1; i++)
				ss << gradient[i] << ", ";
			ss << gradient.back() << ")";
			return ss.str();
		}

		__HOST_ONLY__ const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			for (ptrdiff_t i = 0; i < size - 1; i++)
				ss << gradient[i] << ", ";
			ss << gradient.back() << ")";
			return ss.str();
		}

		d_type value;
		Common::array<d_type, size> gradient;
	};


	template <class r_type, ptrdiff_t size, ptrdiff_t r_size>
	struct DiffVar<DiffVar<r_type, r_size>, size> {
		static_assert(size == r_size, "The sizes of gradients must be the same when using DiffVar recursively.");

		using n_type = typename DiffVar<r_type, size>::n_type;
		using d_type = DiffVar<r_type, size>;
		using s_type = DiffVar<DiffVar<r_type, size>, size>;
		

		__HOST_DEVICE__ DiffVar() {}
		__HOST_DEVICE__ DiffVar(const n_type value) : value(value) {
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}
		__HOST_DEVICE__ DiffVar(const d_type& value, const Common::array<d_type, size>& gradient) : value(value), gradient(gradient) {}
		__HOST_DEVICE__ DiffVar(n_type value, ptrdiff_t as_var_idx) : value(value) {
			this->value.setVar(as_var_idx);
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}

		__HOST_DEVICE__ n_type getValue()const {
			return value.getValue();
		}

		__HOST_DEVICE__ bool operator <(const s_type& _Right) const {
			return getValue() < _Right.getValue();
		}

		__HOST_DEVICE__ bool operator <=(const s_type& _Right) const {
			return getValue() <= _Right.getValue();
		}

		__HOST_DEVICE__ bool operator >(const s_type& _Right) const {
			return getValue() > _Right.getValue();
		}

		__HOST_DEVICE__ bool operator >=(const s_type& _Right) const {
			return getValue() >= _Right.getValue();
		}

		__HOST_DEVICE__ bool operator ==(const s_type& _Right) const {
			return getValue() == _Right.getValue();
		}

		__HOST_DEVICE__ bool operator <(const n_type _Right) const {
			return getValue() < _Right;
		}

		__HOST_DEVICE__ bool operator <=(const n_type _Right) const {
			return getValue() <= _Right;
		}

		__HOST_DEVICE__ bool operator >(const n_type _Right) const {
			return getValue() > _Right;
		}

		__HOST_DEVICE__ bool operator >=(const n_type _Right) const {
			return getValue() >= _Right;
		}

		__HOST_DEVICE__ bool operator ==(const n_type _Right) const {
			return getValue() == _Right;
		}

		__HOST_DEVICE__ const d_type& operator[](ptrdiff_t var_idx) const {
			return gradient[var_idx];
		}

		__HOST_DEVICE__ const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator=(const n_type _Right) {
			value = _Right;
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}

		__HOST_DEVICE__ void setVar(const ptrdiff_t as_var_idx) {
			value.setVar(as_var_idx);
			for (ptrdiff_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}

		__HOST_DEVICE__ s_type operator+(const s_type& _Right) const {

			s_type ret;
			ret.value = value + _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator-(const s_type& _Right) const {
			s_type ret;
			ret.value = value - _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator-() const {
			s_type ret;
			ret.value = -value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator*(const s_type& _Right) const {
			s_type ret;
			ret.value = value * _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator/(const s_type& _Right) const {
			s_type ret;
			ret.value = value / _Right.value;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			return ret;
		}

		__HOST_DEVICE__ s_type operator+(const n_type _Right) const {
			s_type ret(value + _Right, gradient);
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator+(const n_type _Left, const s_type& _Right) {
			return _Right + _Left;
		}

		__HOST_DEVICE__ s_type operator-(const n_type _Right) const {
			s_type ret(value - _Right, gradient);
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator-(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left - _Right.value;

			for (ptrdiff_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}

		__HOST_DEVICE__ s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}

		__HOST_DEVICE__ s_type operator/(const n_type _Right) const {
			s_type ret;
			ret.value = value / _Right;

			for (ptrdiff_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}

		__HOST_DEVICE__ static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;

			for (ptrdiff_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}

		__HOST_DEVICE__ const s_type& operator+=(const s_type& _Right) {
			value += _Right.value;
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator-=(const s_type& _Right) {
			value -= _Right.value;
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator*=(const s_type& _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator/=(const s_type& _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			value /= _Right.value;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator*=(const n_type _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}

		__HOST_DEVICE__ const s_type& operator/=(const n_type _Right) {
			for (ptrdiff_t i = 0; i < size; i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}

		__HOST_ONLY__ const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			for (ptrdiff_t i = 0; i < size - 1; i++)
				ss << gradient[i].toString_grad() << ", ";
			ss << gradient.back().toString_grad() << ")";
			return ss.str();
		}

		__HOST_ONLY__ const std::string toString() const {
			std::stringstream ss;
			ss << value.toString();
			ss << toString_grad();
			return ss.str();
		}

		d_type value;
		Common::array<d_type, size> gradient;
	};

	template<typename d_type, ptrdiff_t size>
	std::ostream& operator << (std::ostream& ostrm, const DiffVar<d_type, size>& v) {
		ostrm << v.toString();
		return ostrm;
	}
}