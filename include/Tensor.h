#pragma once
#include <Common.h>
#include <myarray.h>

namespace Common {

	template<class T, ptrdiff_t... _Ns>
	struct tensor: public array<T, _Ns...> {
		using tensor_type = tensor<T, _Ns...>;
		using array_type = array<T, _Ns...>;

		__HOST_DEVICE__ tensor() : array_type() {}

		__HOST_DEVICE__ tensor(const tensor_type& _Right) : array_type(_Right) {}

		__HOST_DEVICE__ tensor(const array_type& _Right) : array_type(_Right) {}

		__HOST_DEVICE__ tensor(const std::initializer_list<T> _List) : array_type(_List) {}

		__HOST_DEVICE__ void fill(const T _Val) {
			for (ptrdiff_t i = 0; i < array_type::_Size; i++)
				((T*)this)[i] = _Val;
		}
		
		__HOST_DEVICE__ tensor_type operator+(const tensor_type& _Right) const {
			tensor_type ret;
			for (ptrdiff_t i = 0; i < array_type::_Size; i++)
				((T*)&ret)[i] = ((T*)this)[i] + ((T*)&_Right)[i];
			return ret;
		}

		__HOST_DEVICE__ tensor_type operator-(const tensor_type& _Right) const {
			tensor_type ret;
			for (ptrdiff_t i = 0; i < array_type::_Size; i++)
				((T*)&ret)[i] = ((T*)this)[i] + ((T*)&_Right)[i];
			return ret;
		}

		__HOST_DEVICE__ tensor_type operator*(const T _Right) const {
			tensor_type ret;
			for (ptrdiff_t i = 0; i < array_type::_Size; i++)
				((T*)&ret)[i] = ((T*)this)[i] * _Right;
			return ret;
		}

		__HOST_DEVICE__ tensor_type operator/(const T _Right) const {
			tensor_type ret;
			for (ptrdiff_t i = 0; i < array_type::_Size; i++)
				((T*)&ret)[i] = ((T*)this)[i] / _Right;
			return ret;
		}

		/*template<ptrdiff_t... _Rule, ptrdiff_t... _Ns_Right>
		void einsum(const tensor<T, _Ns_Right...>& _Right) {
			array<ptrdiff_t, MaxArg<_Rule...>> i;
			array<ptrdiff_t, MaxArg<_Rule...>> range;

		}*/
	};

}