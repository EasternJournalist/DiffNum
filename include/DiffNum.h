#pragma once
/*
 * By including this file, all headers needed for differentiable variable are included.
 * (Vec, Mat, and Tensor are not contained. )
 */
#include <DiffBasic.h>
#include <DiffVar.h>
#include <DiffArrayVar.h>
#include <DiffMath.h>
#include <DiffManager.h>

namespace DiffNum {

	using dfloat = DiffVar<float>;
	using ddouble = DiffVar<double>;
	using ddfloat = DiffVar<DiffVar<float>>;
	using dddouble = DiffVar<DiffVar<double>>;

	template <size_t size>
	using dfloat_arr = DiffArrayVar<float, size>;
	template <size_t size>
	using ddouble_arr = DiffArrayVar<double, size>;

	template <size_t size>
	using ddfloat_arr = DiffArrayVar<DiffArrayVar<float, size>, size>;
	template <size_t size>
	using dddouble_arr = DiffArrayVar<DiffArrayVar<double, size>, size>;

	using dmathf = Math<DiffVar<float>>;
	using dmathd = Math<DiffVar<double>>;

}