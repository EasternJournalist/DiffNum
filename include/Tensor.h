#pragma once
#include <vector>
#include <assert.h>


namespace DiffNum {

	class Tensor {
		using value_type = double;
	public:

	public:
		// Create a new tensor with shape
		Tensor(const std::vector<ptrdiff_t>& shape);
		// Shallow copy
		Tensor(const Tensor& _Right);
		explicit Tensor(value_type value);

		~Tensor();

		Tensor operator[](const std::vector<ptrdiff_t>& indices) const;
		
		const Tensor& operator=(const Tensor& _Right);
		
		const Tensor& operator=(value_type _Right_value);

		// Return a deep copy
		Tensor Copy() const;
		
		Tensor Reshape(const std::vector<ptrdiff_t>& shape_) const;

		const std::vector<ptrdiff_t> Shape() const;
		size_t Size() const;
		size_t Dim() const;

		operator value_type&() const;

	private:
		bool dense_storage;

		ptrdiff_t* ref_count;
		value_type* value_data;

		std::vector<ptrdiff_t> stride;
		std::vector<ptrdiff_t> shape;
		ptrdiff_t size;
	};

	inline Tensor::Tensor(const std::vector<ptrdiff_t>& shape) : shape(shape), dense_storage(true) {
		assert(shape.size() > 0);
		stride[shape.size() - 1] = 1;
		for (ptrdiff_t i = shape.size() - 2; i >= 0; i--) {
			assert(shape[i + 1] > 0);
			stride[i] = stride[i + 1] * shape[i + 1];
		}
		assert(shape[0] > 0);
		size = stride[0] * shape[0];
		ref_count = new ptrdiff_t(1);
		value_data = new value_type[size];
	}

	
	inline Tensor::Tensor(const Tensor& _Right) : 
		ref_count(_Right.ref_count), value_data(_Right.value_data), stride(_Right.stride), shape(_Right.shape), size(_Right.size), dense_storage(_Right.dense_storage) {
		(*ref_count)++;
	}

	inline Tensor::Tensor(value_type value) {
		size = 1;
		ref_count = new ptrdiff_t(1);
		value_data = new value_type[size];
	}

	inline Tensor::~Tensor() {
		(*ref_count)--;
		if (*ref_count == 0) {
			delete ref_count;
			delete[] value_data;
		}
	}
	inline const Tensor& Tensor::operator=(value_type _Right_value){
		if (shape.size() == 0)
			(*value_data) = _Right_value;
		
		return *this;
	}
	inline Tensor Tensor::Copy() const {
		Tensor copied(shape);
		memcpy(copied.value_data, value_data, sizeof(value_type) * size);
		return copied;
	}
	inline const std::vector<ptrdiff_t> Tensor::Shape() const{
		return shape;
	}
	inline size_t Tensor::Size() const {
		return size;
	}
	inline size_t Tensor::Dim() const {
		return shape.size();
	}
	inline Tensor::operator value_type&() const {
		assert(shape.size() == 0);
		return *value_data;
	}
}