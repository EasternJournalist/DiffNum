#pragma once

#include <cuda_runtime.h>
#include <DiffBasic.h>
#include <DiffVar_cuda.h>
#include <DiffMath_cuda.h>


namespace DiffNum {

	template<size_t size>
	using dfloat_cuda = DiffVar_cuda<float, size>;

	template<size_t size>
	using ddouble_cuda = DiffVar_cuda<double, size>;

}