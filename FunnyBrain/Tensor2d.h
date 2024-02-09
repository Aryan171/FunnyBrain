#pragma once

/*
this is a 2 dimentional tensor
*/
class Tensor2d {
public:
	float* tensor = nullptr; // the device pointer of the tensor
	int rows, columns;
	int numFloats;
	size_t sizeInBytes;

	Tensor2d();

	~Tensor2d();

	/*
	performs a deep copy of the right hand side Tensor2d object into the left hand side Tensor2d object
	*/
	void operator=(const Tensor2d& t2d);

	/*
	Adds 2d tensors a and b asynchronously and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Add(const Tensor2d& a, const Tensor2d& b, Tensor2d& c);

	/*
	Subtracts 2d tensors a and b asynchronously and stores the result in c

	Returs 0- if the operation was performed successfully
	1- if the dimentions of a, b and c are not equal
	*/
	static int Subtract(const Tensor2d& a, const Tensor2d& b, Tensor2d& c);

	/*
	Multiplies 2d tensors a and b asynchronously and stores the result in c

	Returns 0- if the operation was performed successfully
	1- if the dimentions of a, b or c were incorrect
	NOTE- For matrix multiplication the number of columns in a must be equal to number of
	rows in b, number of rows in a must be equal to number of rows in c and the number of
	columns in b must be equal to the number of columns in c
	*/
	static int Multiply(const Tensor2d& a, const Tensor2d& b, Tensor2d& c);

	void Initialize(const int rows, const int columns);

	void Initialize(const Tensor2d& tensor2d);

	void Initialize(float* floatArray, const int rows, const int columns);

	/*
	Adds a constant value b to all the floats of the 2d tensor a asynchronously 
	*/
	void AddConstantVal(float b);

	/*
	Multiplies a constant value b to all the floats of the 2d tensor a asynchronously 
	*/
	void MultiplyConstantVal(float b);

	/*
	Subtracts a constant value b from all the floats of the 2d tensor a asynchronously 
	*/
	void SubtractConstantVal(float b);

	/*
	Sets the value of the 2d tensor of b equal to a synchronously
	b.tensor = a is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void SetValue(const float* a);

	/*
	Sets the value of a equal to the 2d tensor b synchronously
	a = b.tensor is wrong because b.tensor is a device pointer (a pointer pointing to an array
	in the device(gpu) memory)
	*/
	void GetValue(float* a);

	/*
	Sets the values of all the floats of the 2d tensor to a random value in between minVal and maxVal synchronously
	*/
	void RandomizeValues(float minVal, float maxVal);

	/*
	For every element in the 1d tensor adds a random value in between minVal and maxVal to the element synchronously
	*/
	void Mutate(float minVal, float maxVal);
};