#pragma once

#include <Common.h>

namespace Common {

	template<ptrdiff_t... _Args>
	struct MaxArg;

	template<ptrdiff_t _First, ptrdiff_t... _Rest>
	struct MaxArg<_First, _Rest...> {
		using MaxArg_Rest = MaxArg<_Rest...>;
		static const ptrdiff_t value = _First > MaxArg_Rest::value ? _First : MaxArg_Rest::value;
	};

	template<ptrdiff_t _Last>
	struct MaxArg<_Last> {
		static const ptrdiff_t value = _Last;
	};

	template<ptrdiff_t _Pos, ptrdiff_t... _Args>
	struct ArgAt;

	template<ptrdiff_t _Pos, ptrdiff_t _First, ptrdiff_t... _Rest>
	struct ArgAt<_Pos, _First, _Rest...> {
		static_assert(_Pos >= 0, "");
		static const ptrdiff_t value = ArgAt<_Pos - 1, _Rest...>::value;
	};
	
	template<ptrdiff_t _First, ptrdiff_t... _Rest>
	struct ArgAt<0, _First, _Rest...> {
		static const ptrdiff_t value =  _First;
	};

	template<ptrdiff_t _Pos, ptrdiff_t _Last>
	struct ArgAt<_Pos, _Last> {
		static_assert(_Pos == 0, "");
		static const ptrdiff_t value = _Last;
	};


	template<ptrdiff_t _Last>
	__forceinline void TArgtoArray(ptrdiff_t* _Arr) {
		*_Arr = _Last;
	};

	template<ptrdiff_t _First, ptrdiff_t... _Rest>
	__forceinline void TArgtoArray(ptrdiff_t* _Arr) {
		*_Arr = _First;
		TArgtoArray<_Rest...>(_Arr + 1);
	};
	
	template<class _Ty>
	__HOST_DEVICE__ __forceinline void FArgtoArray(_Ty* _Arr, const _Ty _Last) {
		*_Arr = _Last;
	};

	template<class _Ty, class... _Args>
	__HOST_DEVICE__ __forceinline void FArgtoArray(_Ty* _Arr, const _Ty _First, const _Args... _Rest) {
		*_Arr = _First;
		FArgtoArray(_Arr + 1, _Rest...);
	};
	
}