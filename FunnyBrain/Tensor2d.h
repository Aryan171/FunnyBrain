#pragma once
#include "FunnyBrainInternal.h"

class Tensor2d {
public:
	float* tensor = nullptr; // the device pointer of the tensor
	int rows, columns;
	int numFloats;
	size_t sizeInBytes;

	Tensor2d(const int rows, const int columns) {
		this->rows = rows;
		this->columns = columns;
		this->sizeInBytes = rows * columns * sizeof(float);
		this->numFloats = rows * columns;
		this->tensor = (float*)Create(this->sizeInBytes);
	}

	Tensor2d(const Tensor2d& tensor2d) {
		this->rows = tensor2d.rows;
		this->columns = tensor2d.columns;
		this->sizeInBytes = this->rows * this->columns * sizeof(float);
		this->numFloats = rows * columns;
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToHost(this->tensor, tensor2d.tensor, this->sizeInBytes);
	}

	Tensor2d(float* floatArray, const int rows, const int columns) {
		this->rows = rows;
		this->columns = columns;
		this->sizeInBytes = this->rows * this->columns * sizeof(float);
		this->numFloats = rows * columns;
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToDevice(this->tensor, floatArray, this->sizeInBytes);
	}

	~Tensor2d() {
		Free(this->tensor);
	}

	/*
	Adds 2d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Add(const Tensor2d& a, const Tensor2d& b, Tensor2d& c) {
		if (a.rows != b.rows || a.rows != c.rows || a.columns != b.columns || a.columns != c.columns) {
			return 1;
		}
		AddArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
		return 0;
	}

	/*
	Subtracts 2d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Subtract(const Tensor2d& a, const Tensor2d& b, Tensor2d& c) {
		if (a.rows != b.rows || a.rows != c.rows || a.columns != b.columns || a.columns != c.columns) {
			return 1;
		}
		SubtractArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
	}

	/*
	Multiplies 2d tensors a and b and stores the result in c

	Returns 0- if the operation was performed successfully
	1- if the dimentions of a, b or c were incorrect
	NOTE- For matrix multiplication the number of columns in a must be equal to number of
	rows in b, number of rows in a must be equal to number of rows in c and the number of
	columns in b must be equal to the number of columns in c
	*/
	static int Multiply(const Tensor2d& a, const Tensor2d& b, Tensor2d& c) {
		if (a.rows != c.rows || a.columns != b.rows || b.columns != c.columns) {
			return 1;
		}
		Multiply2d(a.tensor, b.tensor, c.tensor, a.rows, a.columns, b.columns);
		return 0;
	}

	/*
	Adds a constant value b to all the floats of the 2d tensor a
	*/
	void AddConstantVal(float b) {
		AddConstant(this->tensor, b, this->numFloats);
	}

	/*
	Multiplies a constant value b to all the floats of the 2d tensor a
	*/
	void MultiplyConstantVal(float b) {
		MultiplyConstant(this->tensor, b, this->numFloats);
	}

	/*
	Subtracts a constant value b from all the floats of the 2d tensor a
	*/
	void SubtractConstantVal(float b) {
		SubtractConstant(this->tensor, b, this->numFloats);
	}

	/*
	Sets the value of the 2d tensor of b equal to a
	b.tensor = a is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void SetValue(const float* a) {
		CopyHostToDevice(this->tensor, a, this->sizeInBytes);
	}

	/*
	Sets the value of a equal to the 2d tensor b
	a = b.tensor is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void GetValue(float* a) {
		CopyDeviceToHost(a, this->tensor, this->sizeInBytes);
	}
};