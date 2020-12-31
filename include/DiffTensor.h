#pragma once

#include <Tensor.h>

namespace DiffNum {



	class DiffTensor {

	public:
		

	private:
		Tensor value;
		std::vector<Tensor> gradient;
	};
}