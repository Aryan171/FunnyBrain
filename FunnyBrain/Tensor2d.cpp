#include "Tensor2d.h"
#include "FunnyBrainInternal.h"

Tensor2d::Tensor2d() {
	this->rows = 0;
	this->columns = 0;
	this->numFloats = 0;
	this->sizeInBytes = 0;
	this->tensor = nullptr;
}

Tensor2d::~Tensor2d() {
	Free(this->tensor);
}

void Tensor2d::operator=(const Tensor2d& t2d) {
	if(this != &t2d) {
		this->rows = t2d.rows;
		this->columns = t2d.columns;
		this->sizeInBytes = t2d.sizeInBytes;
		this->numFloats = t2d.numFloats;
		Free(this->tensor);
		this->tensor = (float*)Create(t2d.sizeInBytes);
		CopyDeviceToDevice(this->tensor, t2d.tensor, t2d.sizeInBytes);
	}
}

int Tensor2d::Add(const Tensor2d& a, const Tensor2d& b, Tensor2d& c) {
	if (a.rows != b.rows || a.rows != c.rows || a.columns != b.columns || a.columns != c.columns) {
		return 1;
	}
	AddArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
	return 0;
}

int Tensor2d::Subtract(const Tensor2d& a, const Tensor2d& b, Tensor2d& c) {
	if (a.rows != b.rows || a.rows != c.rows || a.columns != b.columns || a.columns != c.columns) {
		return 1;
	}
	SubtractArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
	return 0;
}

int Tensor2d::Multiply(const Tensor2d& a, const Tensor2d& b, Tensor2d& c) {
	if (a.rows != c.rows || a.columns != b.rows || b.columns != c.columns) {
		return 1;
	}
	Multiply2d(a.tensor, b.tensor, c.tensor, a.rows, a.columns, b.columns);
	return 0;
}

void Tensor2d::Initialize(const int rows, const int columns) {
	this->rows = rows;
	this->columns = columns;
	this->sizeInBytes = rows * columns * sizeof(float);
	this->numFloats = rows * columns;
	Free(this->tensor);
	this->tensor = (float*)Create(this->sizeInBytes);
}

void Tensor2d::Initialize(const Tensor2d& tensor2d) {
	if (this != &tensor2d) {
		this->rows = tensor2d.rows;
		this->columns = tensor2d.columns;
		this->sizeInBytes = this->rows * this->columns * sizeof(float);
		this->numFloats = rows * columns;
		this->tensor = (float*)Create(this->sizeInBytes);
		Free(this->tensor);
		CopyHostToHost(this->tensor, tensor2d.tensor, this->sizeInBytes);
	}
}

void Tensor2d::Initialize(float* floatArray, const int rows, const int columns) {
	this->rows = rows;
	this->columns = columns;
	this->sizeInBytes = this->rows * this->columns * sizeof(float);
	this->numFloats = rows * columns;
	this->tensor = (float*)Create(this->sizeInBytes);
	Free(this->tensor);
	CopyHostToDevice(this->tensor, floatArray, this->sizeInBytes);
}

void Tensor2d::AddConstantVal(float b) {
	AddConstant(this->tensor, b, this->numFloats);
}

void Tensor2d::MultiplyConstantVal(float b) {
	MultiplyConstant(this->tensor, b, this->numFloats);
}

void Tensor2d::SubtractConstantVal(float b) {
	SubtractConstant(this->tensor, b, this->numFloats);
}

void Tensor2d::SetValue(const float* a) {
	CopyHostToDevice(this->tensor, a, this->sizeInBytes);
}

void Tensor2d::GetValue(float* a) {
	CopyDeviceToHost(a, this->tensor, this->sizeInBytes);
}

void Tensor2d::RandomizeValues(float minVal, float maxVal) {
	float* dev_a = (float*)Create(this->sizeInBytes);
	GenerateRandom(dev_a, minVal, maxVal, this->numFloats);
	CopyDeviceToDevice(this->tensor, dev_a, this->sizeInBytes);
	Free(dev_a);
}

void Tensor2d::Mutate(float minVal, float maxVal) {
	float* dev_a = (float*)Create(this->sizeInBytes);
	GenerateRandom(dev_a, minVal, maxVal, this->numFloats);
	AddArrays(dev_a, this->tensor, this->tensor, this->numFloats);
	Wait();
	Free(dev_a);
}