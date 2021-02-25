#pragma once
/*
 * By including this file, all headers needed for differentiable variable are included.
 * (Vec, Mat, and Tensor are not contained. )
 */
#include <DiffVar.h>
#include <DiffArrayVar.h>
#include <DiffBasic.h>
#include <DiffMath.h>
#include <DiffManager.h>

namespace DiffNum {
	typedef DiffVar<double> ddouble;
	typedef DiffVar<float> dfloat;

	typedef Math<DiffVar<float>> dmathf;
	typedef Math<DiffVar<double>> dmathd;

}