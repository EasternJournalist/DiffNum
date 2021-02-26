#pragma once

#include <DiffBasic.h>
#include <DiffVar_cuda.cuh>
#include <cuda_runtime.h>

#include <math.h>

namespace DiffNum {

	template <class n_type, size_t size> const DiffArrayVar_cuda<n_type, size> NaN<DiffArrayVar_cuda<n_type, size>> = DiffArrayVar_cuda<n_type, size>(NaN<n_type>);
	template <class n_type, size_t size> const DiffArrayVar_cuda<n_type, size> Inf<DiffArrayVar_cuda<n_type, size>> = DiffArrayVar_cuda<n_type, size>(Inf<n_type>);
	template <class n_type, size_t size> const DiffArrayVar_cuda<n_type, size> NegInf<DiffArrayVar_cuda<n_type, size>> = DiffArrayVar_cuda<n_type, size>(NegInf<n_type>);
	template <class n_type, size_t size> const DiffArrayVar_cuda<n_type, size> Pi<DiffArrayVar_cuda<n_type, size>> = DiffArrayVar_cuda<n_type, size>(Pi<n_type>);

	/// <summary>
	/// Static mathematical functions for DiffVar in CUDA.
	/// </summary>
	/// <typeparam name="n_type"></typeparam>
	template<class n_type>
	class Math_cuda {};

	template<>
	class Math_cuda<float> {
	public:
		__host__ __device__ __forceinline__ static bool IsNaN(const float& _X) { return isnan(_X); }

		__host__ __device__ __forceinline__ static float Sqrt(const float& _X) { return sqrt(_X); }
		__host__ __device__ __forceinline__ static float Sin(const float& _X) { return sin(_X); }
		__host__ __device__ __forceinline__ static float Cos(const float& _X) { return cos(_X); }
		__host__ __device__ __forceinline__ static float Tan(const float& _X) { return tan(_X); }
		__host__ __device__ __forceinline__ static float Asin(const float& _X) { return asin(_X); }
		__host__ __device__ __forceinline__ static float Acos(const float& _X) { return acos(_X); }
		__host__ __device__ __forceinline__ static float Atan(const float& _X) { return atan(_X); }
		__host__ __device__ __forceinline__ static float Sinh(const float& _X) { return sinh(_X); }
		__host__ __device__ __forceinline__ static float Cosh(const float& _X) { return cosh(_X); }
		__host__ __device__ __forceinline__ static float Tanh(const float& _X) { return tanh(_X); }
		__host__ __device__ __forceinline__ static float Asinh(const float& _X) { return asinh(_X); }
		__host__ __device__ __forceinline__ static float Acosh(const float& _X) { return acosh(_X); }
		__host__ __device__ __forceinline__ static float Atanh(const float& _X) { return atanh(_X); }
		__host__ __device__ __forceinline__ static float Exp(const float& _X) { return exp(_X); }
		__host__ __device__ __forceinline__ static float Log(const float& _X) { return log(_X); }
		__host__ __device__ __forceinline__ static float Pow(const float& _X, const float& _Y) { return pow(_X, _Y); }
	};

	template<>
	class Math_cuda<double> {
	public:
		__host__ __device__ __forceinline__ static bool IsNaN(const double& _X) { return isnan(_X); }

		__host__ __device__ __forceinline__ static double Sqrt(const double& _X) { return sqrt(_X); }
		__host__ __device__ __forceinline__ static double Sin(const double& _X) { return sin(_X); }
		__host__ __device__ __forceinline__ static double Cos(const double& _X) { return cos(_X); }
		__host__ __device__ __forceinline__ static double Tan(const double& _X) { return tan(_X); }
		__host__ __device__ __forceinline__ static double Asin(const double& _X) { return asin(_X); }
		__host__ __device__ __forceinline__ static double Acos(const double& _X) { return acos(_X); }
		__host__ __device__ __forceinline__ static double Atan(const double& _X) { return atan(_X); }
		__host__ __device__ __forceinline__ static double Sinh(const double& _X) { return sinh(_X); }
		__host__ __device__ __forceinline__ static double Cosh(const double& _X) { return cosh(_X); }
		__host__ __device__ __forceinline__ static double Tanh(const double& _X) { return tanh(_X); }
		__host__ __device__ __forceinline__ static double Asinh(const double& _X) { return asinh(_X); }
		__host__ __device__ __forceinline__ static double Acosh(const double& _X) { return acosh(_X); }
		__host__ __device__ __forceinline__ static double Atanh(const double& _X) { return atanh(_X); }
		__host__ __device__ __forceinline__ static double Exp(const double& _X) { return exp(_X); }
		__host__ __device__ __forceinline__ static double Log(const double& _X) { return log(_X); }
		__host__ __device__ __forceinline__ static double Pow(const double& _X, const double& _Y) { return pow(_X, _Y); }
	};




	template<class n_type, size_t size>
	class Math_cuda<DiffArrayVar_cuda<n_type, size>> {
		using MathofNum = Math_cuda<n_type>;
	public:

		__host__ __device__ static bool IsNaN(const DiffArrayVar_cuda<n_type, size>& _X) {
			return MathofNum::IsNan(_X.value);
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Abs(const DiffArrayVar_cuda<n_type, size>& _X) {
			if (_X.value > n_type(0)) {
				return _X;
			}
			else {
				DiffArrayVar_cuda<n_type, size> ret;
				ret.value = -_X.value;
				for (size_t i = 0; i < size; i++) {
					ret.gradient[i] = -_X.gradient[i];
				}
				return ret;
			}
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Sqrt(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Sqrt(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / (n_type(2) * ret.value);
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Sin(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Sin(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Cos(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Cos(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Cos(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = -MathofNum::Sin(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Tan(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Tan(_X.value);

			n_type cosX = MathofNum::Cos(_X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / (cosX * cosX);;
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Asin(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Asin(_X.value);

			n_type dasinX = n_type(1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dasinX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Acos(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Acos(_X.value);

			n_type dacosX = n_type(-1) / MathofNum::Sqrt(n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dacosX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Atan(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Atan(_X.value);

			n_type datanX = n_type(1) / (n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = datanX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Sinh(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Sinh(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Cosh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Cosh(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Cosh(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Sinh(_X.value) * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Tanh(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Tanh(_X.value);

			n_type dtanhX = n_type(2) / (n_type(1) + MathofNum::Cosh(n_type(2) * _X.value));
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dtanhX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Asinh(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Asinh(_X.value);

			n_type dasinhX = n_type(1) / MathofNum::Sqrt(n_type(1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dasinhX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Acosh(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Acosh(_X.value);

			n_type dacoshX = n_type(1) / MathofNum::Sqrt(n_type(-1) + _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = dacoshX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Atanh(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Atanh(_X.value);

			n_type datanhX = n_type(1) / (n_type(1) - _X.value * _X.value);
			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = datanhX * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Exp(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Exp(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = ret.value * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Log(const DiffArrayVar_cuda<n_type, size>& _X) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Log(_X.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _X.gradient[i] / _X.value;
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Pow(const n_type& _X, const DiffArrayVar_cuda<n_type, size>& _Y) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Pow(_X, _Y.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = MathofNum::Log(_X) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Pow(const DiffArrayVar_cuda<n_type, size>& _X, const n_type& _Y) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Pow(_X.value, _Y);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Y * (ret.value / _X.value) * _X.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Pow(const DiffArrayVar_cuda<n_type, size>& _X, const DiffArrayVar_cuda<n_type, size>& _Y) {
			DiffArrayVar_cuda<n_type, size> ret;
			ret.value = MathofNum::Pow(_X.value, _Y.value);

			for (size_t i = 0; i < size; i++) {
				ret.gradient[i] = _Y.value * (ret.value / _X.value) * _X.gradient[i] + MathofNum::Log(_X.value) * ret.value * _Y.gradient[i];
			}
			return ret;
		}


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Pow(const DiffArrayVar_cuda<n_type, size>& _X, unsigned int _Y) {
			DiffArrayVar_cuda<n_type, size> ret;
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


		__host__ __device__ static DiffArrayVar_cuda<n_type, size> Max(const DiffArrayVar_cuda<n_type, size>& _X, const DiffArrayVar_cuda<n_type, size>& _Y) {
			if (_X.value > _Y.value) {
				return _X;
			}
			else {
				return _Y;
			}
		}
	};
}