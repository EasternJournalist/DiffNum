#pragma once


#include <DiffVar.h>
#include <math.h>
#include <DiffBasic.h>

namespace DiffNum {
	static float Sqrt(const float& _X) { return sqrt(_X); }
	static float Sin(const float& _X) { return sin(_X); }
	static float Cos(const float& _X) { return cos(_X); }
	static float Tan(const float& _X) { return tan(_X); }
	static float Asin(const float& _X) { return asin(_X); }
	static float Acos(const float& _X) { return acos(_X); }
	static float Atan(const float& _X) { return atan(_X); }
	static float Sinh(const float& _X) { return sinh(_X); }
	static float Cosh(const float& _X) { return cosh(_X); }
	static float Tanh(const float& _X) { return tanh(_X); }
	static float Asinh(const float& _X) { return asinh(_X); }
	static float Acosh(const float& _X) { return acosh(_X); }
	static float Atanh(const float& _X) { return atanh(_X); }
	static float Exp(const float& _X) { return exp(_X); }
	static float Log(const float& _X) { return log(_X); }
	static float Pow(const float& _X, const float& _Y) { return pow(_X, _Y); }

	static double Sqrt(const double& _X) { return sqrt(_X); }
	static double Sin(const double& _X) { return sin(_X); }
	static double Cos(const double& _X) { return cos(_X); }
	static double Tan(const double& _X) { return tan(_X); }
	static double Asin(const double& _X) { return asin(_X); }
	static double Acos(const double& _X) { return acos(_X); }
	static double Atan(const double& _X) { return atan(_X); }
	static double Sinh(const double& _X) { return sinh(_X); }
	static double Cosh(const double& _X) { return cosh(_X); }
	static double Tanh(const double& _X) { return tanh(_X); }
	static double Asinh(const double& _X) { return asinh(_X); }
	static double Acosh(const double& _X) { return acosh(_X); }
	static double Atanh(const double& _X) { return atanh(_X); }
	static double Exp(const double& _X) { return exp(_X); }
	static double Log(const double& _X) { return log(_X); }
	static double Pow(const double& _X, const double& _Y) { return pow(_X, _Y); }

	template<typename n_type>
	DiffVar<n_type> Abs(const DiffVar<n_type>& _X) {
		DiffVar<n_type> ret(_X.gradient.size());
		if (_X.value > 0) {
			ret.value = _X.value;
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i];
			}
		}
		else {
			ret.value = -_X.value;
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = -_X.gradient[i];
			}
		}
		return ret;
	}

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
			ret.gradient[i] = _X.gradient[i] / _X.value;
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
	DiffVar<n_type> Pow(const DiffVar<n_type>& _X, const DiffVar<n_type>& _Y) {
		if (_X.gradient.empty())
			return Pow(_X.value, _Y);
		if (_Y.gradient.empty())
			return Pow(_X, _Y.value);
		assert(_X.gradient.size() == _Y.gradient.size());
		DiffVar<n_type> ret(_X.gradient.size());
		ret.value = pow(_X.value, _Y.value);
		for (size_t i = 0; i < _X.gradient.size(); i++) {
			ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + log(_X.value) * ret.value * _Y.gradient[i];
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

	template<typename n_type>
	DiffVar<n_type> Max(const DiffVar<n_type>& _X, const DiffVar<n_type>& _Y) {
		if (_X.value > _Y.value) {
			return _X;
		}
		else {
			return _Y;
		}
	}
}