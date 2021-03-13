#pragma once

#include <Common.h>

namespace Common {
	template<ptrdiff_t... _Args>
	struct LenArg;

	template<ptrdiff_t _First, ptrdiff_t... _Rest>
	struct LenArg<_First, _Rest...> {
		static const ptrdiff_t value = LenArg<_Rest...>::value + 1;
	};

	template<ptrdiff_t _Last>
	struct LenArg<_Last> {
		static const ptrdiff_t value = 1;
	};

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
	void ArgtoArray(ptrdiff_t* _Arr) {
		*_Arr = _Last;
	};

	template<ptrdiff_t _First, ptrdiff_t... _Rest>
	void ArgtoArray(ptrdiff_t* _Arr) {
		*_Arr = _First;
		ArgtoArray<_Rest...>(_Arr + 1);
	};

}