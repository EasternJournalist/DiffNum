#pragma once

#include <cuda_runtime.h>
#include <DiffBasic.h>
#include <DiffVar_cuda.cuh>
#include <DiffMath_cuda.cuh>


namespace DiffNum {

	template<size_t size>
	using dfloat_arr_cuda = DiffArrayVar_cuda<float, size>;

	template<size_t size>
	using ddouble_arr_cuda = DiffArrayVar_cuda<double, size>;

}