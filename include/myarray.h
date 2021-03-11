#pragma once
#include <DiffBasic.h>

namespace Common {
	template<class T, size_t N>
	struct array {

		T _Elems[N];

		__HOST_DEVICE__ array() {}

		__HOST_DEVICE__ array(const array<T, N>& _Right) {
			for (size_t i = 0; i < N; i++)
				_Elems[i] = _Right._Elems[i];
		}

		__HOST_DEVICE__ const T& operator[](const size_t _Pos) const {
			assert(_Pos < N);
			return _Elems[_Pos];
		}

		__HOST_DEVICE__ T& operator[](const size_t _Pos) {
			assert(_Pos < N);
			return _Elems[_Pos];
		}

		__HOST_DEVICE__ const array<T, N>& operator=(const array<T, N>& _Right) {
			for (size_t i = 0; i < N; i++)
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

		__HOST_DEVICE__ constexpr size_t size() {
			return N - 1;
		}
	};
}