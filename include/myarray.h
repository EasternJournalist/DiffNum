#pragma once
#include <Common.h>
#include <Args.h>
#include <initializer_list>

namespace Common {


	template<class T, ptrdiff_t N>
	struct array<T, N> {
		static_assert(N > 0, "Shapes of array should be greater than 0.");
		using array_type = array<T, N>;

		static const ptrdiff_t _Dims = 1;
		template<ptrdiff_t _Dim> static const ptrdiff_t shape = ArgAt<_Dim, N>::value;
		static const ptrdiff_t _Size = N;

		T _Elems[N];

		__HOST_DEVICE__ array() {}

		__HOST_DEVICE__ array(const array_type& _Right) {
			for (ptrdiff_t i = 0; i < N; i++)
				_Elems[i] = _Right._Elems[i];
		}

		__HOST_DEVICE__ array(const std::initializer_list<T> _List) {
			assert(_List.size() <= _Size);
			T* p = (T*)this;
			for (const T& it : _List)
				*(p++) = it;
		}

		__HOST_DEVICE__ const T& operator[](const ptrdiff_t _Pos) const {
			assert(0 <= _Pos && _Pos < N);
			return _Elems[_Pos];
		}

		__HOST_DEVICE__ T& operator[](const ptrdiff_t _Pos) {
			assert(0 <= _Pos && _Pos < N);
			return _Elems[_Pos];
		}

		__HOST_DEVICE__ const array<T, N>& operator=(const array_type& _Right) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				_Elems[i] = _Right._Elems[i];
			return *this;
		}

		__HOST_DEVICE__ const T& front() const noexcept {
			return _Elems[0];
		}

		__HOST_DEVICE__ T& front() noexcept {
			return _Elems[0];
		}

		__HOST_DEVICE__ const T& back() const noexcept {
			return _Elems[N - 1];
		}

		__HOST_DEVICE__ T& back() noexcept {
			return _Elems[N - 1];
		}

		__HOST_DEVICE__ static constexpr ptrdiff_t dims() {
			return _Dims;
		}

		__HOST_DEVICE__ static constexpr ptrdiff_t size() {
			return _Size;
		}
	};

	template<class T, ptrdiff_t _FirstN, ptrdiff_t... _RestNs>
	struct array<T, _FirstN, _RestNs...> {
		static_assert(_FirstN > 0, "Shapes of array should be greater than 0.");

		using array_type = array<T, _FirstN, _RestNs...>;
		using d_array_type = array<T, _RestNs...>;

		static const ptrdiff_t _Dims = d_array_type::_Dims + 1;

		template<ptrdiff_t _Dim> static const ptrdiff_t shape = ArgAt<_Dim, _FirstN, _RestNs...>::value;
		
		static const ptrdiff_t _Size = d_array_type::_Size * _FirstN;


		d_array_type _Elems[_FirstN];

		__HOST_DEVICE__ array() {}

		__HOST_DEVICE__ array(const array_type& _Right) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = ((T*)&_Right)[i];
		}
		
		__HOST_DEVICE__ array(const std::initializer_list<T> _List) {
			assert(_List.size() <= _Size);
			T* p = (T*)this;
			for (const T& it : _List)
				*(p++) = it;
		}

		__HOST_DEVICE__ const d_array_type& operator[](const ptrdiff_t _Pos) const {
			assert(0 <= _Pos && _Pos < _FirstN);
			return _Elems[_Pos];
		}

		__HOST_DEVICE__ d_array_type& operator[](const ptrdiff_t _Pos) {
			assert(0 <= _Pos && _Pos < _FirstN);
			return _Elems[_Pos];
		}

		__HOST_DEVICE__ const array_type& operator=(const array_type& _Right) {
			for (ptrdiff_t i = 0; i < _Size; i++)
				((T*)this)[i] = ((T*)&_Right)[i];
			return *this;
		}

		__HOST_DEVICE__ const d_array_type& front() const noexcept {
			return _Elems[0];
		}

		__HOST_DEVICE__ d_array_type& front() noexcept {
			return _Elems[0];
		}

		__HOST_DEVICE__ const d_array_type& back() const noexcept {
			return _Elems[_FirstN - 1];
		}

		__HOST_DEVICE__ d_array_type& back() noexcept {
			return _Elems[_FirstN - 1];
		}

		__HOST_DEVICE__ static constexpr ptrdiff_t dims() {
			return _Dims;
		}

		__HOST_DEVICE__ static constexpr ptrdiff_t size() {
			return _Size;
		}
	};
}