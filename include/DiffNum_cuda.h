#pragma once

#include <cuda_runtime.h>
#include <DiffBasic.h>
#include <DiffVar_cuda.h>
#include <DiffMath_cuda.h>


namespace DiffNum {

	template<size_t size>
	using dfloat_arr_cuda = DiffArrayVar_cuda<float, size>;

	template<size_t size>
	using ddouble_arr_cuda = DiffArrayVar_cuda<double, size>;

}