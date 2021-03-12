#pragma once

#ifndef DIFFNUM_NO_CUDA
#define DIFFNUM_WITH_CUDA
#endif
#ifdef DIFFNUM_WITH_CUDA
#include <cuda_runtime.h>
#define __HOST_DEVICE__ __host__ __device__
#define __HOST_ONLY__ __host__
#else
#define __HOST_DEVICE__
#define __HOST_ONLY__
#endif


namespace Common {
	const int __NaNd = 0xFFC00000, __Infinityd = 0x7F800000, __Neg_Infinityd = 0xFF800000;
	const __int64 __NaNf = 0xFFF8000000000000, __Infinityf = 0x7FF0000000000000, __Neg_Infinityf = 0xFFF0000000000000;

	template <class value_type>
	const value_type NaN;

	template <class value_type>
	const value_type Inf;

	template <class value_type>
	const value_type NegInf;

	template <class value_type>
	const value_type Pi;

	template <> const float NaN<float> = *((float*)&__NaNf);
	template <> const double NaN<double> = *((double*)&__NaNd);

	template <> const float Inf<float> = *((float*)&__Infinityf);
	template <> const double Inf<double> = *((double*)&__Infinityd);

	template <> const float NegInf<float> = *((float*)&__Neg_Infinityf);
	template <> const double NegInf<double> = *((double*)&__Neg_Infinityd);

	template <> const float Pi<float> = static_cast<float>(3.1415926535897932385);
	template <> const double Pi<double> = static_cast<double>(3.1415926535897932385);

	template<class T, ptrdiff_t N> struct array;
	template<class T, ptrdiff_t N> struct vec;
	template<class T, ptrdiff_t NRow, ptrdiff_t NCol> struct mat;
}


namespace DiffNum {
	template <class d_type, ptrdiff_t size> struct DiffVar;
	template<class n_type> class Math;
}