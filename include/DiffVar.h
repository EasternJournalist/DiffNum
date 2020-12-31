#pragma once

#include <vector>
#include <assert.h>
#include <ostream>

namespace DiffNum {

	template <typename n_type>
	struct DiffVar {
		using num_type = n_type;
		DiffVar() {}
		DiffVar(const num_type value) : value(value) {}
		explicit DiffVar(const size_t num_var) : gradient(num_var) {}
		DiffVar(num_type value, size_t num_var) :value(value), gradient(num_var) {}
		DiffVar(num_type value, std::vector<num_type> gradient) :value(value), gradient(gradient) {}

		DiffVar(num_type value, size_t num_var, size_t as_var_idx) : value(value), gradient(num_var, static_cast<num_type>(0)) {
			gradient[as_var_idx] = static_cast<num_type>(1);
		}

		const DiffVar<n_type>& operator=(const DiffVar<n_type>& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}
		const DiffVar<n_type>& operator=(const num_type _Right) {
			value = _Right;
			gradient.clear();
			return *this;
		}

		void SetVar(const size_t num_var) {
			gradient.clear();
			gradient.resize(num_var, static_cast<num_type>(0));
		}
		void SetVar(const size_t num_var, const size_t as_var_idx) {
			gradient.clear();
			gradient.resize(num_var, static_cast<num_type>(0));
			gradient[as_var_idx] = static_cast<num_type>(1);
		}

		DiffVar<n_type> operator+(const DiffVar<n_type>& _Right) const {
			if (gradient.empty())
				return _Right + value;
			if (_Right.gradient.empty())
				return operator+(_Right.value);
			assert(gradient.size() == _Right.gradient.size());
			DiffVar<n_type> ret(gradient.size());
			ret.value = value + _Right.value;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}
		DiffVar<n_type> operator-(const DiffVar<n_type>& _Right) const {
			if (gradient.empty())
				return value - _Right;
			if (_Right.gradient.empty())
				return operator-(_Right.value);
			assert(gradient.size() == _Right.gradient.size());
			DiffVar<n_type> ret(gradient.size());
			ret.value = value - _Right.value;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}
		DiffVar<n_type> operator-() const {
			DiffVar<n_type> ret(gradient.size());
			ret.value = -value;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}
		DiffVar<n_type> operator*(const DiffVar<n_type>& _Right) const {
			if (gradient.empty())
				return _Right * value;
			if (_Right.gradient.empty())
				return operator*(_Right.value);
			assert(gradient.size() == _Right.gradient.size());
			DiffVar<n_type> ret(gradient.size());
			ret.value = value * _Right.value;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}
		DiffVar<n_type> operator/(const DiffVar<n_type>& _Right) const {
			if (gradient.empty())
				return value /  _Right;
			if (_Right.gradient.empty())
				return operator/(_Right.value);
			assert(gradient.size() == _Right.gradient.size());
			DiffVar<n_type> ret(gradient.size());
			ret.value = value / _Right.value;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			return ret;
		}
		

		DiffVar<n_type> operator+(num_type _Right) const {
			DiffVar<n_type> ret(value + _Right, gradient);
			return ret;
		}
		static friend inline DiffVar<n_type> operator+(num_type _Left, const DiffVar<n_type>& _Right) {
			return _Right + _Left;
		}
		DiffVar<n_type> operator-(num_type _Right) const {
			DiffVar<n_type> ret(value - _Right, gradient);
			return ret;
		}
		static friend inline DiffVar<n_type> operator-(num_type _Left, const DiffVar<n_type>& _Right) {
			DiffVar<n_type> ret(_Right.gradient.size());
			ret.value = _Left - _Right.value;
			for (size_t i = 0; i < _Right.gradient.size(); i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}
		DiffVar<n_type> operator*(num_type _Right) const {
			DiffVar<n_type> ret(gradient.size());
			ret.value = value * _Right;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}
		static friend inline DiffVar<n_type> operator*(num_type _Left, const DiffVar<n_type>& _Right) {
			return _Right * _Left;
		}
		DiffVar<n_type> operator/(num_type _Right) const {
			DiffVar<n_type> ret(gradient.size());
			ret.value = value / _Right;
			for (size_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}
		static friend inline DiffVar<n_type> operator/(num_type _Left, const DiffVar<n_type>& _Right) {
			DiffVar<n_type> ret(_Right.gradient.size());
			ret.value = _Left / _Right.value;
			for (size_t i = 0; i < _Right.gradient.size(); i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}
		const DiffVar<n_type>& operator+=(const DiffVar<n_type>& _Right) {
			if (_Right.gradient.empty()) 
				return operator+=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), num_type(0));
			assert(gradient.size() == _Right.gradient.size());
			value += _Right.value;
			for (size_t i = 0; i < gradient.size(); i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}
		const DiffVar<n_type>& operator-=(const DiffVar<n_type>& _Right) {
			if (_Right.gradient.empty())
				return operator-=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), num_type(0));
			assert(gradient.size() == _Right.gradient.size());
			value -= _Right.value;
			for (size_t i = 0; i < gradient.size(); i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}
		const DiffVar<n_type>& operator*=(const DiffVar<n_type>& _Right) {
			if (_Right.gradient.empty())
				return operator*=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), num_type(0));
			assert(gradient.size() == _Right.gradient.size());	
			for (size_t i = 0; i < gradient.size(); i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}
		const DiffVar<n_type>& operator/=(const DiffVar<n_type>& _Right) {
			if (_Right.gradient.empty())
				return operator/=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), num_type(0));
			assert(gradient.size() == _Right.gradient.size());
			for (size_t i = 0; i < gradient.size(); i++) {
				gradient[i] = (gradient[i] * _Right.value - value * _Right.gradient[i]) / (_Right.value * _Right.value);
			}
			value /= _Right.value;
			return *this;
		}

		const DiffVar<n_type>& operator+=(num_type _Right) {
			value += _Right;
			return *this;
		}
		const DiffVar<n_type>& operator-=(num_type _Right) {
			value -= _Right;
			return *this;
		}
		const DiffVar<n_type>& operator*=(num_type _Right) {
			for (size_t i = 0; i < gradient.size(); i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}
		const DiffVar<n_type>& operator/=(num_type _Right) {
			for (size_t i = 0; i < gradient.size(); i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}
		num_type value;
		std::vector<num_type> gradient;
	};

	template<typename n_type>
	std::ostream& operator << (std::ostream& ostrm, const DiffVar<n_type>& v) {
		ostrm << v.value << "(";
		for (size_t i = 0; i < v.gradient.size()-1; i++)
			ostrm << v.gradient[i] << ", ";
		ostrm << v.gradient.back() << ")";
		return ostrm;
	}
}