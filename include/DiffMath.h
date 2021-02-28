#pragma once

#include <DiffBasic.h>
#include <DiffVar.h>
#include <DiffArrayVar.h>
#include <math.h>


namespace DiffNum {

	template <class d_type> const DiffVar<d_type> NaN<DiffVar<d_type>> = DiffVar<d_type>(NaN<typename DiffVar<d_type>::n_type>);
	template <class d_type, size_t size> const DiffArrayVar<d_type, size> NaN<DiffArrayVar<d_type, size>> = DiffArrayVar<d_type, size>(NaN<typename DiffArrayVar<d_type, size>::n_type>);

	template <class d_type> const DiffVar<d_type> Inf<DiffVar<d_type>> = DiffVar<d_type>(Inf<typename DiffVar<d_type>::n_type>);
	template <class d_type, size_t size> const DiffArrayVar<d_type, size> Inf<DiffArrayVar<d_type, size>> = DiffArrayVar<d_type, size>(Inf<typename DiffArrayVar<d_type, size>::n_type>);

	template <class d_type> const DiffVar<d_type> NegInf<DiffVar<d_type>> = DiffVar<d_type>(NegInf<typename DiffVar<d_type>::n_type>);
	template <class d_type, size_t size> const DiffArrayVar<d_type, size> NegInf<DiffArrayVar<d_type, size>> = DiffArrayVar<d_type, size>(NegInf<typename DiffArrayVar<d_type, size>::n_type>);

	template <class d_type> const DiffVar<d_type> Pi<DiffVar<d_type>> = DiffVar<d_type>(Pi<typename DiffVar<d_type>::n_type>);
	template <class d_type, size_t size> const DiffArrayVar<d_type, size> Pi<DiffArrayVar<d_type, size>> = DiffArrayVar<d_type, size>(Pi<typename DiffArrayVar<d_type, size>::n_type>);

	/// <summary>
	/// Static mathematical functions for DiffNum.
	/// </summary>
	/// <typeparam name="n_type"></typeparam>
	template<class n_type>
	class Math {};

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


	template<class d_type>
	class Math<DiffVar<d_type>> {
		using MathofNum = Math<d_type>;
		using n_type = typename DiffVar<d_type>::n_type;
		using s_type = DiffVar<d_type>;

	public:

		static bool IsNaN(const s_type& _X) {
			return MathofNum::IsNan(_X.value); 
		}


		static s_type Abs(const s_type& _X) {
			if (_X.value > n_type(0)) {
				return _X;
			}
			else {
				s_type ret(-_X.value, _X.gradient.size());
				for (size_t i = 0; i < _X.gradient.size(); i++) {
					ret.gradient[i] = -_X.gradient[i];
				}
				return ret;
			}
		}


		static s_type Sqrt(const s_type& _X) {
			s_type ret(MathofNum::Sqrt(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i] / (n_type(2) * ret.value);
			}
			return ret;
		}


		static s_type Sin(const s_type& _X) {
			s_type ret(MathofNum::Sin(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Cos(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Cos(const s_type& _X) {
			s_type ret(MathofNum::Cos(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = -MathofNum::Sin(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Tan(const s_type& _X) {
			s_type ret(MathofNum::Tan(_X.value), _X.gradient.size());

			d_type cosX = MathofNum::Cos(_X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i] / (cosX * cosX);;
			}
			return ret;
		}


		static s_type Asin(const s_type& _X) {
			s_type ret(MathofNum::Asin(_X.value), _X.gradient.size());

			d_type dasinX = n_type(1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dasinX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Acos(const s_type& _X) {
			s_type ret(MathofNum::Acos(_X.value), _X.gradient.size());

			d_type dacosX = n_type(-1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dacosX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Atan(const s_type& _X) {
			s_type ret(MathofNum::Atan(_X.value), _X.gradient.size());

			d_type datanX = n_type(1) / (n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = datanX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Sinh(const s_type& _X) {
			s_type ret(MathofNum::Sinh(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Cosh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Cosh(const s_type& _X) {
			s_type ret(MathofNum::Cosh(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Sinh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Tanh(const s_type& _X) {
			s_type ret(MathofNum::Tanh(_X.value), _X.gradient.size());

			d_type dtanhX = n_type(2) / (n_type(1) + MathofNum::Cosh(n_type(2) * _X.value));
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dtanhX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Asinh(const s_type& _X) {
			s_type ret(MathofNum::Asinh(_X.value), _X.gradient.size());

			d_type dasinhX = n_type(1) / MathofNum::Sqrt(n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dasinhX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Acosh(const s_type& _X) {
			s_type ret(MathofNum::Acosh(_X.value), _X.gradient.size());

			d_type dacoshX = n_type(1) / MathofNum::Sqrt(n_type(-1) + _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = dacoshX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Atanh(const s_type& _X) {
			s_type ret(MathofNum::Atanh(_X.value), _X.gradient.size());

			d_type datanhX = n_type(1) / (n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = datanhX * _X.gradient[i];
			}
			return ret;
		}
		

		static s_type Exp(const s_type& _X) {
			s_type ret(MathofNum::Exp(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = ret.value * _X.gradient[i];
			}
			return ret;
		}
		

		static s_type Log(const s_type& _X) {
			s_type ret(MathofNum::Log(_X.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _X.gradient[i] / _X.value;
			}
			return ret;
		}


		static s_type Pow(const n_type& _X, const s_type& _Y) {
			s_type ret(MathofNum::Pow(_X, _Y.value), _Y.gradient.size());

			for (size_t i = 0; i < _Y.gradient.size(); i++) {
				ret.gradient[i] = MathofNum::Log(_X) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static s_type Pow(const s_type& _X, const n_type& _Y) {
			s_type ret(MathofNum::Pow(_X.value, _Y), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _Y * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Pow(const s_type& _X, const s_type& _Y) {
			if (_X.gradient.empty())
				return Pow(_X.value, _Y);
			if (_Y.gradient.empty())
				return Pow(_X, _Y.value);
			assert(_X.gradient.size() == _Y.gradient.size());
			s_type ret(MathofNum::Pow(_X.value, _Y.value), _X.gradient.size());

			for (size_t i = 0; i < _X.gradient.size(); i++) {
				ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + MathofNum::Log(_X.value) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static s_type Pow(const s_type& _X, unsigned int _Y) {
			s_type ret(_Y & 1 ? _X.value : n_type(1), _X.gradient.size());
			d_type p = _X.value * _X.value, _Yn = n_type(_Y);
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


		static s_type Max(const s_type& _X, const s_type& _Y) {
			if (_X.value > _Y.value) {
				return _X;
			}
			else {
				return _Y;
			}
		}
	};
	

	template<class d_type, size_t size>
	class Math<DiffArrayVar<d_type, size>> {
		using MathofNum = Math<d_type>;
		using n_type = typename DiffArrayVar<d_type, size>::n_type;
		using s_type = DiffArrayVar<d_type, size>;


	public:

		static bool IsNaN(const s_type& _X) {
			return MathofNum::IsNan(_X.value);
		}


		static s_type Abs(const s_type& _X) {
			if (_X.value > n_type(0)) {
				return _X;
			}
			else {
				s_type ret;
				ret.value = -_X.value;
				for (size_t i = 0; i < size; i++) {
					ret.gradient[i] = -_X.gradient[i];
				}
				return ret;
			}
		}


		static s_type Sqrt(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Sqrt(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / (n_type(2) * ret.value);
			}
			return ret;
		}


		static s_type Sin(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Sin(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Cos(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Cos(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Cos(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = -MathofNum::Sin(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Tan(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Tan(_X.value);

			d_type cosX = MathofNum::Cos(_X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / (cosX * cosX);;
			}
			return ret;
		}


		static s_type Asin(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Asin(_X.value);

			d_type dasinX = n_type(1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dasinX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Acos(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Acos(_X.value);

			d_type dacosX = n_type(-1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dacosX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Atan(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Atan(_X.value);

			d_type datanX = n_type(1) / (n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = datanX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Sinh(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Sinh(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Cosh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Cosh(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Cosh(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Sinh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Tanh(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Tanh(_X.value);

			d_type dtanhX = n_type(2) / (n_type(1) + MathofNum::Cosh(n_type(2) * _X.value));
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dtanhX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Asinh(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Asinh(_X.value);

			d_type dasinhX = n_type(1) / MathofNum::Sqrt(n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dasinhX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Acosh(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Acosh(_X.value);

			d_type dacoshX = n_type(1) / MathofNum::Sqrt(n_type(-1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dacoshX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Atanh(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Atanh(_X.value);

			d_type datanhX = n_type(1) / (n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = datanhX * _X.gradient[i];
			}
			return ret;
		}


		static s_type Exp(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Exp(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = ret.value * _X.gradient[i];
			}
			return ret;
		}


		static s_type Log(const s_type& _X) {
			s_type ret;
			ret.value = MathofNum::Log(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / _X.value;
			}
			return ret;
		}


		static s_type Pow(const n_type& _X, const s_type& _Y) {
			s_type ret;
			ret.value = MathofNum::Pow(_X, _Y.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Log(_X) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static s_type Pow(const s_type& _X, const n_type& _Y) {
			s_type ret;
			ret.value = MathofNum::Pow(_X.value, _Y);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Y * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		static s_type Pow(const s_type& _X, const s_type& _Y) {
			s_type ret;
			ret.value = MathofNum::Pow(_X.value, _Y.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + MathofNum::Log(_X.value) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		static s_type Pow(const s_type& _X, unsigned int _Y) {
			s_type ret;
			ret.value = _Y & 1 ? _X.value : n_type(1);

			d_type p = _X.value * _X.value, _Yn = n_type(_Y);
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


		static s_type Max(const s_type& _X, const s_type& _Y) {
			if (_X.value > _Y.value) {
				return _X;
			}
			else {
				return _Y;
			}
		}
	};
}