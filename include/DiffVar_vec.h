#pragma once

#include <vector>
#include <assert.h>
#include <ostream>

namespace DiffNum {

	/// <summary>
	/// Differentiable varible. The numerical value and the derivatives will be automatically evaluated simultaneously.
	/// The gradients on target variable must be specified before any computation. You may use DiffVar.setVar or 
	/// class DiffManager to help initalize and deal with the variables that you study. 
	/// </summary>
	/// <typeparam name="n_type">The type of numerical value. eg. float, double</typeparam>
	template <class d_type>
	struct DiffVar<d_type, 0> {
		using n_type = d_type;
		using s_type = DiffVar<d_type, 0>;


		DiffVar() {}
		DiffVar(const n_type value) : value(value) {}
		DiffVar(n_type value, ptrdiff_t num_var) :value(value), gradient(num_var) {}
		DiffVar(n_type value, const std::vector<n_type>& gradient) :value(value), gradient(gradient) {}
		DiffVar(n_type value, ptrdiff_t num_var, ptrdiff_t as_var_idx) : value(value), gradient(num_var, n_type(0)) {
			gradient[as_var_idx] = n_type(1);
		}

		n_type getValue() const {
			return value;
		}

		bool operator <(const s_type& _Right) const {
			return value < _Right.value;
		}

		bool operator <=(const s_type& _Right) const {
			return value <= _Right.value;
		}

		bool operator >(const s_type& _Right) const {
			return value > _Right.value;
		}

		bool operator >=(const s_type& _Right) const {
			return value >= _Right.value;
		}

		bool operator ==(const s_type& _Right) const {
			return value == _Right.value;
		}

		bool operator !=(const s_type& _Right) const {
			return value != _Right.value;
		}

		bool operator <(const n_type _Right) const {
			return value < _Right;
		}

		bool operator <=(const n_type _Right) const {
			return value <= _Right;
		}

		bool operator >(const n_type _Right) const {
			return value > _Right;
		}

		bool operator >=(const n_type _Right) const {
			return value >= _Right;
		}

		bool operator ==(const n_type _Right) const {
			return value == _Right;
		}

		bool operator !=(const n_type _Right) const {
			return value != _Right;
		}

		const d_type& operator[](ptrdiff_t var_idx) const {
			return gradient[var_idx];
		}

		const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}

		const s_type& operator=(const n_type _Right) {
			value = _Right;
			gradient.clear();
			return *this;
		}

		void setVar(const ptrdiff_t num_var) {
			gradient.clear();
			gradient.resize(num_var, n_type(0));
		}

		void setVar(const ptrdiff_t num_var, const ptrdiff_t as_var_idx) {
			gradient.clear();
			gradient.resize(num_var, n_type(0));
			gradient[as_var_idx] = n_type(1);
		}

		s_type operator+(const s_type& _Right) const {
			if (gradient.empty())
				return _Right + value;
			if (_Right.gradient.empty())
				return operator+(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret(value + _Right.value, gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}

		s_type operator-(const s_type& _Right) const {
			if (gradient.empty())
				return value - _Right;
			if (_Right.gradient.empty())
				return operator-(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret(value - _Right.value, gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}

		s_type operator-() const {
			s_type ret(-value, gradient.size());
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}

		s_type operator*(const s_type& _Right) const {
			if (gradient.empty())
				return _Right * value;
			if (_Right.gradient.empty())
				return operator*(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret(value * _Right.value, gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}

		s_type operator/(const s_type& _Right) const {
			if (gradient.empty())
				return value /  _Right;
			if (_Right.gradient.empty())
				return operator/(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret(value / _Right.value, gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
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
			s_type ret(_Left - _Right.value, _Right.gradient.size());

			for (ptrdiff_t i = 0; i < _Right.gradient.size(); i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}

		s_type operator*(const n_type _Right) const {
			s_type ret(value * _Right, gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] * _Right;
			}
			return ret;
		}

		static friend inline s_type operator*(const n_type _Left, const s_type& _Right) {
			return _Right * _Left;
		}

		s_type operator/(const n_type _Right) const {
			s_type ret(value / _Right, gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}

		static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret(_Left / _Right.value, _Right.gradient.size());

			for (ptrdiff_t i = 0; i < _Right.gradient.size(); i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}

		const s_type& operator+=(const s_type& _Right) {
			if (_Right.gradient.empty()) 
				return operator+=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			value += _Right.value;
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}

		const s_type& operator-=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator-=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			value -= _Right.value;
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}

		const s_type& operator*=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator*=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());	

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}

		const s_type& operator/=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator/=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
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
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}

		const s_type& operator/=(const n_type _Right) {
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}

		const std::string toString() const {
			std::stringstream ss;
			ss << value << "(";
			if (gradient.size() > 0) {
				for (ptrdiff_t i = 0; i < gradient.size() - 1; i++)
					ss << gradient[i] << ", ";
				ss << gradient.back();
			}
			ss << ")";
			return ss.str();
		}

		const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			if (gradient.size() > 0) {
				for (ptrdiff_t i = 0; i < gradient.size() - 1; i++)
					ss << gradient[i] << ", ";
				ss << gradient.back();
			}
			ss << gradient.back() << ")";
			return ss.str();
		}

		d_type value;
		std::vector<d_type> gradient;
	};

	template <typename r_type>
	struct DiffVar<DiffVar<r_type, 0>, 0> {
		using n_type = typename DiffVar<r_type, 0>::n_type;
		using d_type = DiffVar<r_type, 0>;
		using s_type = DiffVar<d_type, 0>;

		DiffVar() {}
		DiffVar(const n_type value) : value(value) {}
		DiffVar(n_type value, ptrdiff_t num_var) :value(value), gradient(num_var) {}
		DiffVar(const d_type& value, const std::vector<d_type>& gradient) :value(value), gradient(gradient) {}
		DiffVar(n_type value, ptrdiff_t num_var, ptrdiff_t as_var_idx) : value(d_type(value)), gradient(num_var, d_type(0)) {
			this->value.setVar(num_var, as_var_idx);
			gradient[as_var_idx] = n_type(1);
		}

		n_type getValue() const {
			return value.getValue();
		}

		bool operator <(const s_type& _Right) const {
			return getValue() < _Right.getValue();
		}

		bool operator <=(const s_type& _Right) const {
			return getValue() <= _Right.getValue();
		}

		bool operator >(const s_type& _Right) const {
			return getValue() > _Right.getValue();
		}

		bool operator >=(const s_type& _Right) const {
			return getValue() >= _Right.getValue();
		}

		bool operator ==(const s_type& _Right) const {
			return getValue() == _Right.getValue();
		}

		bool operator <(const n_type _Right) const {
			return getValue() < _Right;
		}

		bool operator <=(const n_type _Right) const {
			return getValue() <= _Right;
		}

		bool operator >(const n_type _Right) const {
			return getValue() > _Right;
		}

		bool operator >=(const n_type _Right) const {
			return getValue() >= _Right;
		}

		bool operator ==(const n_type _Right) const {
			return getValue() == _Right;
		}

		const d_type& operator[](ptrdiff_t var_idx) const {
			return gradient[var_idx];
		}

		const s_type& operator=(const s_type& _Right) {
			value = _Right.value;
			gradient = _Right.gradient;
			return *this;
		}

		const s_type& operator=(const n_type _Right) {
			value = _Right;
			gradient.clear();
			return *this;
		}

		void setVar(const ptrdiff_t num_var) {
			gradient.clear();
			gradient.resize(num_var, d_type(0));
		}

		void setVar(const ptrdiff_t num_var, const ptrdiff_t as_var_idx) {
			gradient.clear();
			value.setVar(num_var, as_var_idx);
			gradient.resize(num_var, d_type(0));
			gradient[as_var_idx] = n_type(1);
		}

		s_type operator+(const s_type& _Right) const {
			if (gradient.empty())
				return _Right + value;
			if (_Right.gradient.empty())
				return operator+(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret;
			ret.value = value + _Right.value;
			ret.gradient.resize(gradient.size());
			
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] + _Right.gradient[i];
			}
			return ret;
		}

		s_type operator-(const s_type& _Right) const {
			if (gradient.empty())
				return value - _Right;
			if (_Right.gradient.empty())
				return operator-(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret; 
			ret.value = value - _Right.value;
			ret.gradient.resize(gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] - _Right.gradient[i];
			}
			return ret;
		}

		s_type operator-() const {
			s_type ret;
			ret.value = -value;
			ret.gradient.resize(gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = -gradient[i];
			}
			return ret;
		}

		s_type operator*(const s_type& _Right) const {
			if (gradient.empty())
				return _Right * value;
			if (_Right.gradient.empty())
				return operator*(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret;
			ret.value = value * _Right.value;
			ret.gradient.resize(gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			return ret;
		}

		s_type operator/(const s_type& _Right) const {
			if (gradient.empty())
				return value / _Right;
			if (_Right.gradient.empty())
				return operator/(_Right.value);
			assert(gradient.size() == _Right.gradient.size());

			s_type ret;
			ret.value = value / _Right.value;
			ret.gradient.resize(gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
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
			s_type ret;
			ret.value = _Left - _Right.value;
			ret.gradient.resize(_Right.gradient.size());

			for (ptrdiff_t i = 0; i < _Right.gradient.size(); i++)
				ret.gradient[i] = -_Right.gradient[i];
			return ret;
		}

		s_type operator*(const n_type _Right) const {
			s_type ret;
			ret.value = value * _Right;
			ret.gradient.resize(gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
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
			ret.gradient.resize(gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				ret.gradient[i] = gradient[i] / _Right;
			}
			return ret;
		}

		static friend inline s_type operator/(const n_type _Left, const s_type& _Right) {
			s_type ret;
			ret.value = _Left / _Right.value;
			ret.gradient.resize(_Right.gradient.size());

			for (ptrdiff_t i = 0; i < _Right.gradient.size(); i++)
				ret.gradient[i] = -(_Left / (_Right.value * _Right.value)) * _Right.gradient[i];
			return ret;
		}

		const s_type& operator+=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator+=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			value += _Right.value;
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] += _Right.gradient[i];
			}
			return *this;
		}

		const s_type& operator-=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator-=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			value -= _Right.value;
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] -= _Right.gradient[i];
			}
			return *this;
		}

		const s_type& operator*=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator*=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] = gradient[i] * _Right.value + value * _Right.gradient[i];
			}
			value *= _Right.value;
			return *this;
		}

		const s_type& operator/=(const s_type& _Right) {
			if (_Right.gradient.empty())
				return operator/=(_Right.value);
			if (gradient.empty())
				gradient.resize(_Right.gradient.size(), n_type(0));
			assert(gradient.size() == _Right.gradient.size());

			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
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
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] *= _Right;
			}
			value *= _Right;
			return *this;
		}

		const s_type& operator/=(const n_type _Right) {
			for (ptrdiff_t i = 0; i < gradient.size(); i++) {
				gradient[i] /= _Right;
			}
			value /= _Right;
			return *this;
		}

		const std::string toString_grad() const {
			std::stringstream ss;
			ss << "(";
			if (gradient.size() > 0) {
				for (ptrdiff_t i = 0; i < gradient.size() - 1; i++)
					ss << gradient[i].toString_grad() << ", ";
				ss << gradient.back().toString_grad();
			}
			ss << ")";
			return ss.str();
		}

		const std::string toString() const {
			std::stringstream ss;
			ss << value.toString();
			ss << toString_grad();
			return ss.str();
		}

		d_type value;
		std::vector<d_type> gradient;
	};

	template<class d_type>
	std::ostream& operator << (std::ostream& ostrm, const DiffVar<d_type, 0>& v) {
		ostrm << v.toString();
		return ostrm;
	}
}