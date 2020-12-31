#pragma once


#include <DiffVar.h>
#include <math.h>
#include <decl.h>

namespace DiffNum {
	static float Sqrt(const float& _X) { return sqrt(_X); }

	template<typename n_type>
	DiffVar<n_type> Sqrt(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = sqrt(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = _X.gradient[i] / (static_cast<n_type>(2) * ret.value);
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Sin(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = sin(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = cos(_X.value) * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Cos(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = cos(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = -sin(_X.value) * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Tan(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = tan(_X.value);
		n_type cosX = cos(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = _X.gradient[i] / (cosX * cosX);;
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Asin(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = asin(_X.value);
		n_type dasinX = static_cast<n_type>(1) / sqrt(static_cast<n_type>(1) - _X.value * _X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = dasinX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Acos(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = acos(_X.value);
		n_type dacosX = static_cast<n_type>(-1) / sqrt(static_cast<n_type>(1) - _X.value * _X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = dacosX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Atan(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = atan(_X.value);
		n_type datanX = static_cast<n_type>(1) / (static_cast<n_type>(1) + _X.value * _X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = datanX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Sinh(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = sinh(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = cosh(_X.value) * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Cosh(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = cosh(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = sinh(_X.value) * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Tanh(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = tanh(_X.value);
		n_type dtanhX = static_cast<n_type>(2) / (static_cast<n_type>(1) + cosh(static_cast<n_type>(2) * _X.value));
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = dtanhX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Asinh(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = asinh(_X.value);
		n_type dasinhX = static_cast<n_type>(1) / sqrt(static_cast<n_type>(1) + _X.value * _X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = dasinhX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Acosh(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = acosh(_X.value);
		n_type dacoshX = static_cast<n_type>(1) / sqrt(static_cast<n_type>(-1) + _X.value * _X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = dacoshX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Atanh(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = atanh(_X.value);
		n_type datanhX = static_cast<n_type>(1) / (static_cast<n_type>(1) - _X.value * _X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = datanhX * _X.gradient[i];
		}
		return ret;
	}

	template<typename n_type>
	DiffVar<n_type> Exp(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = exp(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = ret.value * _X.gradient[i];
		}
		return ret;
	}
	template<typename n_type>
	DiffVar<n_type> Log(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = log(_X.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = -_X.gradient[i] / _X.value;
		}
		return ret;
	}
	
	template<typename n_type>
	DiffVar<n_type> Pow(const DiffVar<n_type>& _X, const DiffVar<n_type>& _Y) {
		assert(_X.gradient.size() == _Y.gradient.size());
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = pow(_X.value, _Y.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + log(_X.value) * ret.value * _Y.gradient[i];
		}
		return ret;
	}
	template<typename n_type>
	DiffVar<n_type> Pow(const n_type& _X, const DiffVar<n_type>& _Y) {
		DiffVar<n_type> ret(_Y.gradient.size());
		ret.value = pow(_X, _Y.value);
		for (size_t i = 0; i < _Y.gradient.size(); i++) {
			ret.gradient[i] = log(_X) * ret.value * _Y.gradient[i];
		}
		return ret;
	}
	template<typename n_type>
	DiffVar<n_type> Pow(const DiffVar<n_type>& _X, const n_type& _Y) {
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = pow(_X.value, _Y);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = _Y * (ret.value / _X.value) * _X.gradient[i];
		}
		return ret;
	}
	template<typename n_type>
	DiffVar<n_type> Pow(const DiffVar<n_type>& _X, unsigned int _Y) {
		DiffVar<n_type> ret(_Y & 1 ? _X.value : static_cast<n_type>(1), _X.gradient.size());
		n_type p = _X.value * _X.value, _Yn = n_type(_Y);
		_Y >>= 1;
		while (_Y) {
			if (_Y & 1)
				ret.value *= p;
			p *= p;
			_Y >> 1;
		}
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = _Yn * (ret.value / _X.value) * _X.gradient[i];
		}
		return ret;
	}
}