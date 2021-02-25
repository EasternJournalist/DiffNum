#pragma once

#include <DiffVar.h>
#include <DiffArrayVar.h>

namespace DiffNum {
	const int __NaNd = 0xFFC00000, __Infinityd = 0x7F800000, __Neg_Infinityd = 0xFF800000;
	const __int64 __NaNf = 0xFFF8000000000000, __Infinityf = 0x7FF0000000000000, __Neg_Infinityf = 0xFFF0000000000000;

	typedef DiffVar<double> ddouble;
	typedef DiffVar<float> dfloat;

	
	template <class value_type>
	const value_type NaN;
	template <> const float NaN<float> = *((float*)&__NaNf);
	template <> const double NaN<double> = *((double*)&__NaNd);
	template <class n_type> const DiffVar<n_type> NaN<DiffVar<n_type>> = DiffVar<n_type>(NaN<n_type>);
	template <class n_type, size_t size> const DiffArrayVar<n_type, size> NaN<DiffArrayVar<n_type, size>> = DiffArrayVar<n_type, size>(NaN<n_type>);

	template <class value_type>
	const value_type Inf;
	template <> const float Inf<float> = *((float*)&__Infinityf);
	template <> const double Inf<double> = *((double*)&__Infinityd);
	template <class n_type> const DiffVar<n_type> Inf<DiffVar<n_type>> = DiffVar<n_type>(Inf<n_type>);
	template <class n_type, size_t size> const DiffArrayVar<n_type, size> Inf<DiffArrayVar<n_type, size>> = DiffArrayVar<n_type, size>(Inf<n_type>);

	template <class value_type>
	const value_type NegInf;
	template <> const float NegInf<float> = *((float*)&__Neg_Infinityf);
	template <> const double NegInf<double> = *((double*)&__Neg_Infinityd);
	template <class n_type> const DiffVar<n_type> NegInf<DiffVar<n_type>> = DiffVar<n_type>(NegInf<n_type>);
	template <class n_type, size_t size> const DiffArrayVar<n_type, size> NegInf<DiffArrayVar<n_type, size>> = DiffArrayVar<n_type, size>(NegInf<n_type>);

	template <class value_type>
	const value_type Pi;
	template <> const float Pi<float> = static_cast<float>(3.1415926535897932385);
	template <> const double Pi<double> = static_cast<double>(3.1415926535897932385);
	template <class n_type> const DiffVar<n_type> Pi<DiffVar<n_type>> = DiffVar<n_type>(Pi<n_type>);
	template <class n_type, size_t size> const DiffArrayVar<n_type, size> Pi<DiffArrayVar<n_type, size>> = DiffArrayVar<n_type, size>(Pi<n_type>);

}