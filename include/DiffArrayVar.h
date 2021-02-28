#pragma once

#include <array>
#include <assert.h>
#include <ostream>
#include <sstream>

namespace DiffNum {


	/// <summary>
	/// Differentiable varible (fixed variable to be studied, and faster). The numerical value and the derivatives will be automatically evaluated simultaneously.
	/// The gradients on target variable must be specified before any computation. You may use DiffArrayVar.SetVar or 
	/// class DiffManager to help initalize and deal with the variables that you study. 
	/// </summary>
	/// <typeparam name="d_type">The type of numerical value. eg. float, double</typeparam>
	/// <typeparam name="size"> The size of gradient vector </typeparam>
	template <class d_type, size_t size>
	struct DiffArrayVar {
		using n_type = d_type;
		using s_type = DiffArrayVar<d_type, size>;

		DiffArrayVar() {}
		DiffArrayVar(const n_type value) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}
		DiffArrayVar(n_type value, const std::array<n_type, size>& gradient) : value(value), gradient(gradient) {}
		DiffArrayVar(n_type value, size_t as_var_idx) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}

		n_type getValue() const {
			return value;
		}

		const d_type& operator[](size_t var_idx) const {
			return gradient[var_idx];
		}

		const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}


		const s_type& operator=(const n_type _Right) {
			value = _Right;
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}


		void SetVar(const size_t as_var_idx) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}
		

		s_type operator+(const s_type& _Right) const {

			s_type ret;
			ret.value = value + _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}


		s_type operator-(const s_type& _Right) const {
			s_type ret;
			ret.value = value - _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}


		s_type operator-() const {
			s_type ret;
			ret.value = -value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}


		s_type operator*(const s_type& _Right) const {
			s_type ret;
			ret.value = value * _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}


		s_type operator/(const s_type& _Right) const {
			s_type ret;
			ret.value = value / _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			return ret;
		}


		s_type operator+(const n_type _Right) const {
			s_type ret(value + _Right, gradient);
			return ret;
		}


		static friend inline s_type operator+(const n_type _Left, const s_type& _Right) {
			return _Right + _Left;
		}


		s_type operator-(const n_type _Right) const {
			s_type ret(value - _Right, gradient);
			return ret;
		}


		static friend inline s_type operator-(const n_type _Left, const s_type& _Right) {
			s_type ret(_Left - _Right.value, _Right.size);

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}


		s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}


		static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}


		s_type operator/(const n_type _Right) const {
			s_type ret;
			ret.value = value / _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}


		static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}


		const s_type& operator+=(const s_type& _Right) {
			value += _Right.value;
			for (size_t i = 0; i < size; i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}


		const s_type& operator-=(const s_type& _Right) {
			value -= _Right.value;
			for (size_t i = 0; i < size; i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}


		const s_type& operator*=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}


		const s_type& operator/=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			value /= _Right.value;
			return *this;
		}


		const s_type& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}


		const s_type& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}


		const s_type& operator*=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}


		const s_type& operator/=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}


		const std::string toString() const {
			std::stringstream ss;
			ss << value << "(";
			for (size_t i = 0; i < size - 1; i++)
				ss << gradient[i] << ", ";
			ss << gradient.back() << ")";
			return ss.str();
		}


		const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			for (size_t i = 0; i < size - 1; i++)
				ss << gradient[i] << ", ";
			ss << gradient.back() << ")";
			return ss.str();
		}

		d_type value;
		std::array<d_type, size> gradient;
	};


	template <class r_type, size_t size>
	struct DiffArrayVar<DiffArrayVar<r_type, size>, size> {
		using n_type = typename DiffArrayVar<r_type, size>::n_type;
		using d_type = DiffArrayVar<r_type, size>;
		using s_type = DiffArrayVar<DiffArrayVar<r_type, size>, size>;
		

		DiffArrayVar() {}
		DiffArrayVar(const n_type value) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
		}
		DiffArrayVar(n_type value, const std::array<n_type, size>& gradient) : value(value), gradient(gradient) {}
		DiffArrayVar(n_type value, size_t as_var_idx) : value(value) {
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		n_type getValue()const {
			return value.getValue();
		}


		const d_type& operator[](size_t var_idx) const {
			return gradient[var_idx];
		}


		const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}


		const s_type& operator=(const n_type _Right) {
			value = _Right;
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			return *this;
		}


		void SetVar(const size_t as_var_idx) {
			value.SetVar(as_var_idx);
			for (size_t i = 0; i < size; i++) gradient[i] = n_type(0);
			gradient[as_var_idx] = n_type(1);
		}


		s_type operator+(const s_type& _Right) const {

			s_type ret;
			ret.value = value + _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}


		s_type operator-(const s_type& _Right) const {
			s_type ret;
			ret.value = value - _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}


		s_type operator-() const {
			s_type ret;
			ret.value = -value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}


		s_type operator*(const s_type& _Right) const {
			s_type ret;
			ret.value = value * _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}


		s_type operator/(const s_type& _Right) const {
			s_type ret;
			ret.value = value / _Right.value;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			return ret;
		}


		s_type operator+(const n_type _Right) const {
			s_type ret(value + _Right, gradient);
			return ret;
		}


		static friend inline s_type operator+(const n_type _Left, const s_type& _Right) {
			return _Right + _Left;
		}


		s_type operator-(const n_type _Right) const {
			s_type ret(value - _Right, gradient);
			return ret;
		}


		static friend inline s_type operator-(const n_type _Left, const s_type& _Right) {
			s_type ret(_Left - _Right.value, _Right.size);

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}


		s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}


		static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}


		s_type operator/(const n_type _Right) const {
			s_type ret;
			ret.value = value / _Right;

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}


		static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;

			for (size_t i = 0; i < size; i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}


		const s_type& operator+=(const s_type& _Right) {
			value += _Right.value;
			for (size_t i = 0; i < size; i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}


		const s_type& operator-=(const s_type& _Right) {
			value -= _Right.value;
			for (size_t i = 0; i < size; i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}


		const s_type& operator*=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}


		const s_type& operator/=(const s_type& _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			value /= _Right.value;
			return *this;
		}


		const s_type& operator+=(const n_type _Right) {
			value += _Right;
			return *this;
		}


		const s_type& operator-=(const n_type _Right) {
			value -= _Right;
			return *this;
		}


		const s_type& operator*=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}


		const s_type& operator/=(const n_type _Right) {
			for (size_t i = 0; i < size; i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}


		const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			for (size_t i = 0; i < size - 1; i++)
				ss << gradient[i].toString_grad() << ", ";
			ss << gradient.back().toString_grad() << ")";
			return ss.str();
		}


		const std::string toString() const {
			std::stringstream ss;
			ss << value.toString();
			ss << toString_grad();
			return ss.str();
		}

		d_type value;
		std::array<d_type, size> gradient;
	};

	template<typename d_type, size_t size>
	std::ostream& operator << (std::ostream& ostrm, const DiffArrayVar<d_type, size>& v) {
		ostrm << v.toString();
		return ostrm;
	}
}