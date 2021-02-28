#pragma once
/*
 * By including this file, all headers needed for differentiable variable are included.
 * (Vec, Mat, and Tensor are not contained. )
 */
#include <DiffBasic.h>
#include <DiffVar_arr.h>
#include <DiffVar_vec.h>
#include <DiffMath.h>
#include <DiffManager.h>

namespace DiffNum {

	template <size_t size>
	using dfloat = DiffVar<float, size>;
	template <size_t size>
	using ddouble = DiffVar<double, size>;

	template <size_t size>
	using ddfloat = DiffVar<DiffVar<float, size>, size>;
	template <size_t size>
	using dddouble = DiffVar<DiffVar<double, size>, size>;

	template <size_t size>
	using dmathf = Math<DiffVar<float, size>>;
	template <size_t size>
	using dmathd = Math<DiffVar<double, size>>;
	
}