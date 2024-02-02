#pragma once
#include "FunnyBrainInternal.h"

class Tensor1d {
public:
	float* tensor = nullptr; // the device pointer of the tensor
	int length;
	size_t sizeInBytes;

	Tensor1d(const int length) {
		this->length = length;
		this->sizeInBytes = length * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
	}

	Tensor1d(const Tensor1d& tensor1d) {
		this->length = tensor1d.length;
		this->sizeInBytes = length * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToHost(this->tensor, tensor1d.tensor, this->sizeInBytes);
	}

	Tensor1d(float* floatArray, const int length) {
		this->length = length;
		this->sizeInBytes = length * sizeof(float);
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
		if (a.length != b.length) {
			return 1;
		}
		AddArrays(a.tensor, b.tensor, c.tensor, a.length);
		return 0;
	}

	/*
	Subtracts 1d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Subtract(const Tensor1d& a, const Tensor1d& b, Tensor1d& c) {
		if (a.length != b.length) {
			return 1;
		}
		SubtractArrays(a.tensor, b.tensor, c.tensor, a.length);
		return 0;
	}

	/*
	Sets the value of the 1d tensor of b equal to a
	b.tensor = a is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)

	Returns 0- if the operation was performed successfully
	1- if the dimentions of a and b mismatch
	*/
	static int SetValue(const float* a, const int a_length, Tensor1d& b) {
		if (a_length != b.length) {
			return 1;
		}

		CopyHostToDevice(b.tensor, a, b.sizeInBytes);
		return 0;
	}

	/*
	Sets the value of a equal to the 1d tensor b
	a = b.tensor is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)

	Returns 0- if the operation was performed successfully
	1- if the dimentions of a and b mismatch
	*/
	static int GetValue(float* a, const int a_length, const Tensor1d& b) {
		if (a_length != b.length) {
			return 1;
		}

		CopyHostToDevice(b.tensor, a, b.sizeInBytes);
		return 0;
	}
};