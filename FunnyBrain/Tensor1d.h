#pragma once
#include "FunnyBrainInternal.h"

class Tensor1d {
public:
	float* tensor = nullptr; // the device pointer of the tensor
	int numFloats; // the total number of floats in the tensor
	size_t sizeInBytes;

	Tensor1d(const int numFloats) {
		this->numFloats = numFloats;
		this->sizeInBytes = numFloats * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
	}

	Tensor1d(const Tensor1d& tensor1d) {
		this->numFloats = tensor1d.numFloats;
		this->sizeInBytes = numFloats * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToHost(this->tensor, tensor1d.tensor, this->sizeInBytes);
	}

	Tensor1d(float* floatArray, const int numFloats) {
		this->numFloats = numFloats;
		this->sizeInBytes = numFloats * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToDevice(this->tensor, floatArray, this->sizeInBytes);
	}

	~Tensor1d() {
		Free(this->tensor);
	}

	/*
	Adds 1d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Add(const Tensor1d& a, const Tensor1d& b, Tensor1d& c) {
		if (a.numFloats != b.numFloats) {
			return 1;
		}
		AddArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
		return 0;
	}

	/*
	Subtracts 1d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Subtract(const Tensor1d& a, const Tensor1d& b, Tensor1d& c) {
		if (a.numFloats != b.numFloats) {
			return 1;
		}
		SubtractArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
		return 0;
	}

	/*
	Adds a constant value b to all the floats of the 1d tensor a
	*/
	void AddConstantVal(float b) {
		AddConstant(this->tensor, b, this->numFloats);
	}

	/*
	Multiplies a constant value b to all the floats of the 1d tensor a
	*/
	void MultiplyConstantVal(float b) {
		MultiplyConstant(this->tensor, b, this->numFloats);
	}

	/*
	Subtracts a constant value b from all the floats of the 1d tensor a
	*/
	void SubtractConstantVal(float b) {
		SubtractConstant(this->tensor, b, this->numFloats);
	}

	/*
	Sets the value of the 1d tensor of b equal to a
	b.tensor = a is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void SetValue(const float* a) {
		CopyHostToDevice(this->tensor, a, this->sizeInBytes);
	}

	/*
	Sets the value of a equal to the 1d tensor b
	a = b.tensor is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void GetValue(float* a) {
		CopyHostToDevice(this->tensor, a, this->sizeInBytes);
	}
};