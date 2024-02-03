#pragma once

class Tensor1d {
public:
	float* tensor = nullptr; // the device pointer of the tensor
	int numFloats; // the total number of floats in the tensor
	size_t sizeInBytes;

	Tensor1d(const int numFloats);

	Tensor1d(const Tensor1d& tensor1d);

	Tensor1d(float* floatArray, const int numFloats);

	~Tensor1d();

	/*
	Adds 1d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Add(const Tensor1d& a, const Tensor1d& b, Tensor1d& c);

	/*
	Subtracts 1d tensors a and b and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Subtract(const Tensor1d& a, const Tensor1d& b, Tensor1d& c);

	/*
	Adds a constant value b to all the floats of the 1d tensor a
	*/
	void AddConstantVal(float b);

	/*
	Multiplies a constant value b to all the floats of the 1d tensor a
	*/
	void MultiplyConstantVal(float b);

	/*
	Subtracts a constant value b from all the floats of the 1d tensor a
	*/
	void SubtractConstantVal(float b);

	/*
	Sets the value of the 1d tensor of b equal to a
	b.tensor = a is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void SetValue(const float* a);

	/*
	Sets the value of a equal to the 1d tensor b
	a = b.tensor is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void GetValue(float* a);

	/*
	Sets the values of all the floats of the 1d tensor to a random value in between minVal and maxVal
	*/
	void RandomizeValues(float minVal, float maxVal);
};