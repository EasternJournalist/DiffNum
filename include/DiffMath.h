#pragma once


#include <DiffVar.h>
#include <DiffArrayVar.h>
#include <math.h>
#include <DiffBasic.h>

namespace DiffNum {
	
	/// <summary>
	/// Math functions for numerical type.
	/// </summary>
	/// <typeparam name="n_type"></typeparam>
	template<class n_type>
	class Math {
		
	};

	template<>
	class Math<float> {
	public:
		static bool IsNaN(const float& _X) { return isnan(_X); }

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
	};

	template<>
	class Math<double> {
	public:
		static bool IsNaN(const double& _X) { return isnan(_X); }

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
	};


	template<class n_type>
	class Math<DiffVar<n_type>> {
		using MathofNum = Math<n_type>;
	public:

		static bool IsNaN(const DiffVar<n_type>& _X) {
			return MathofNum::IsNan(_X.value); 
		}


		static DiffVar<n_type> Abs(const DiffVar<n_type>& _X) {
			if (_X.value > n_type(0)) {
				return _X;
			}
			else {
				DiffVar<n_type> ret(-_X.value, _X.gradient.size());
				for (size_t i = 0; i < _X.gradient.size(); i++) {
					ret.gradient[i] = -_X.gradient[i];
				}
				return ret;
			}
		}


		static DiffVar<n_type> Sqrt(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Sqrt(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i] / (n_type(2) * ret.value);
			}
			return ret;
		}


		static DiffVar<n_type> Sin(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Sin(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Cos(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Cos(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Cos(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = -MathofNum::Sin(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Tan(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Tan(_X.value), _X.gradient.size());

			n_type cosX = MathofNum::Cos(_X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i] / (cosX * cosX);;
			}
			return ret;
		}


		static DiffVar<n_type> Asin(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Asin(_X.value), _X.gradient.size());

			n_type dasinX = n_type(1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dasinX * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Acos(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Acos(_X.value), _X.gradient.size());

			n_type dacosX = n_type(-1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dacosX * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Atan(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Atan(_X.value), _X.gradient.size());

			n_type datanX = n_type(1) / (n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = datanX * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Sinh(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Sinh(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Cosh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Cosh(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Cosh(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Sinh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Tanh(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Tanh(_X.value), _X.gradient.size());

			n_type dtanhX = n_type(2) / (n_type(1) + MathofNum::Cosh(n_type(2) * _X.value));
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dtanhX * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Asinh(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Asinh(_X.value), _X.gradient.size());

			n_type dasinhX = n_type(1) / MathofNum::Sqrt(n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dasinhX * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Acosh(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Acosh(_X.value), _X.gradient.size());

			n_type dacoshX = n_type(1) / MathofNum::Sqrt(n_type(-1) + _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dacoshX * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Atanh(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Atanh(_X.value), _X.gradient.size());

			n_type datanhX = n_type(1) / (n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = datanhX * _X.gradient[i];
			}
			return ret;
		}
		

		static DiffVar<n_type> Exp(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Exp(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = ret.value * _X.gradient[i];
			}
			return ret;
		}
		

		static DiffVar<n_type> Log(const DiffVar<n_type>& _X) {
			DiffVar<n_type> ret(MathofNum::Log(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i] / _X.value;
			}
			return ret;
		}


		static DiffVar<n_type> Pow(const n_type& _X, const DiffVar<n_type>& _Y) {
			DiffVar<n_type> ret(MathofNum::Pow(_X, _Y.value), _Y.gradient.size());

			for (size_t i = 0; i < _Y.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Log(_X) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Pow(const DiffVar<n_type>& _X, const n_type& _Y) {
			DiffVar<n_type> ret(MathofNum::Pow(_X.value, _Y), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _Y * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Pow(const DiffVar<n_type>& _X, const DiffVar<n_type>& _Y) {
			if (_X.gradient.empty())
				return Pow(_X.value, _Y);
			if (_Y.gradient.empty())
				return Pow(_X, _Y.value);
			assert(_X.gradient.size() == _Y.gradient.size());
			DiffVar<n_type> ret(MathofNum::Pow(_X.value, _Y.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + MathofNum::Log(_X.value) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Pow(const DiffVar<n_type>& _X, unsigned int _Y) {
			DiffVar<n_type> ret(_Y & 1 ? _X.value : n_type(1), _X.gradient.size());
			n_type p = _X.value * _X.value, _Yn = n_type(_Y);
			_Y >>= 1;
			while (_Y) {
				if (_Y & 1)
					ret.value *= p;
				p *= p;
				_Y >>= 1;
			}
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _Yn * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffVar<n_type> Max(const DiffVar<n_type>& _X, const DiffVar<n_type>& _Y) {
			if (_X.value > _Y.value) {
				return _X;
			}
			else {
				return _Y;
			}
		}
	};
	

	template<class n_type, size_t size>
	class Math<DiffArrayVar<n_type, size>> {
		using MathofNum = Math<n_type>;
	public:

		static bool IsNaN(const DiffArrayVar<n_type, size>& _X) {
			return MathofNum::IsNan(_X.value);
		}


		static DiffArrayVar<n_type, size> Abs(const DiffArrayVar<n_type, size>& _X) {
			if (_X.value > n_type(0)) {
				return _X;
			}
			else {
				DiffArrayVar<n_type, size> ret;
				ret.value = -_X.value;
				for (size_t i = 0; i < size; i++) {
					ret.gradient[i] = -_X.gradient[i];
				}
				return ret;
			}
		}


		static DiffArrayVar<n_type, size> Sqrt(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Sqrt(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / (n_type(2) * ret.value);
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Sin(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Sin(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Cos(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Cos(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Cos(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = -MathofNum::Sin(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Tan(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Tan(_X.value);

			n_type cosX = MathofNum::Cos(_X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / (cosX * cosX);;
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Asin(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Asin(_X.value);

			n_type dasinX = n_type(1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dasinX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Acos(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Acos(_X.value);

			n_type dacosX = n_type(-1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dacosX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Atan(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Atan(_X.value);

			n_type datanX = n_type(1) / (n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = datanX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Sinh(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Sinh(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Cosh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Cosh(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Cosh(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Sinh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Tanh(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Tanh(_X.value);

			n_type dtanhX = n_type(2) / (n_type(1) + MathofNum::Cosh(n_type(2) * _X.value));
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dtanhX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Asinh(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Asinh(_X.value);

			n_type dasinhX = n_type(1) / MathofNum::Sqrt(n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dasinhX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Acosh(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Acosh(_X.value);

			n_type dacoshX = n_type(1) / MathofNum::Sqrt(n_type(-1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dacoshX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Atanh(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Atanh(_X.value);

			n_type datanhX = n_type(1) / (n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = datanhX * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Exp(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Exp(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = ret.value * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Log(const DiffArrayVar<n_type, size>& _X) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Log(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / _X.value;
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Pow(const n_type& _X, const DiffArrayVar<n_type, size>& _Y) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Pow(_X, _Y.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Log(_X) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Pow(const DiffArrayVar<n_type, size>& _X, const n_type& _Y) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Pow(_X.value, _Y);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Y * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Pow(const DiffArrayVar<n_type, size>& _X, const DiffArrayVar<n_type, size>& _Y) {
			DiffArrayVar<n_type, size> ret;
			ret.value = MathofNum::Pow(_X.value, _Y.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + MathofNum::Log(_X.value) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Pow(const DiffArrayVar<n_type, size>& _X, unsigned int _Y) {
			DiffArrayVar<n_type, size> ret;
			ret.value = _Y & 1 ? _X.value : n_type(1);

			n_type p = _X.value * _X.value, _Yn = n_type(_Y);
			_Y >>= 1;
			while (_Y) {
				if (_Y & 1)
					ret.value *= p;
				p *= p;
				_Y >>= 1;
			}
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Yn * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		static DiffArrayVar<n_type, size> Max(const DiffArrayVar<n_type, size>& _X, const DiffArrayVar<n_type, size>& _Y) {
			if (_X.value > _Y.value) {
				return _X;
			}
			else {
				return _Y;
			}
		}
	};
}