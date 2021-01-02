#pragma once

#include <DiffVar.h>

namespace DiffNum {
	const int __NaNd = 0xFFC00000, __Infinityd = 0x7F800000, __Neg_Infinityd = 0xFF800000;
	const __int64 __NaNf = 0xFFF8000000000000, __Infinityf = 0x7FF0000000000000, __Neg_Infinityf = 0xFFF0000000000000;

	typedef DiffVar<double> DiffDouble;
	typedef DiffVar<float> DiffFloat;


	template <class value_type>
	const value_type NaN;
	template <> const float NaN<float> = *((float*)&__NaNf);
	template <> const double NaN<double> = *((double*)&__NaNd);
	template <> const DiffFloat NaN<DiffFloat> = DiffFloat(NaN<float>);
	template <> const DiffDouble NaN<DiffDouble> = DiffDouble(NaN<double>);

	template <class value_type>
	const value_type Inf;
	template <> const float Inf<float> = *((float*)&__Infinityf);
	template <> const double Inf<double> = *((double*)&__Infinityd);
	template <> const DiffFloat Inf<DiffFloat> = DiffFloat(Inf<float>);
	template <> const DiffDouble Inf<DiffDouble> = DiffDouble(Inf<double>);

	template <class value_type>
	const value_type NegInf;
	template <> const float NegInf<float> = *((float*)&__Neg_Infinityf);
	template <> const double NegInf<double> = *((double*)&__Neg_Infinityd);
	template <> const DiffFloat NegInf<DiffFloat> = DiffFloat(NegInf<float>);
	template <> const DiffDouble NegInf<DiffDouble> = DiffDouble(NegInf<double>);

	template <class value_type>
	const value_type Pi;
	template <> const float Pi<float> = static_cast<float>(3.1415926535897932385);
	template <> const double Pi<double> = static_cast<double>(3.1415926535897932385);
	template <> const DiffFloat Pi<DiffFloat> = DiffFloat(Pi<float>);
	template <> const DiffDouble Pi<DiffDouble> = DiffDouble(Pi<double>);

	class Decl {
	public:
		template <class value_type>
		static bool IsNaN(const value_type& _X) { static_assert(false, "This type is not supported."); }
		template <> static bool IsNaN<float>(const float& _X) { return isnan(_X); }
		template <> static bool IsNaN<double>(const double& _X) { return isnan(_X); }
		template <> static bool IsNaN<DiffFloat>(const DiffFloat& _X) { return isnan(_X.value); }
		template <> static bool IsNaN<DiffDouble>(const DiffDouble& _X) { return isnan(_X.value); }
	};
}