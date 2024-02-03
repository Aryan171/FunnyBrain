#include "Tensor2d.h"
#include "FunnyBrainInternal.h"
#include "curand.h"
#include "chrono"

Tensor2d::Tensor2d(const int rows, const int columns) {
	this->rows = rows;
	this->columns = columns;
	this->sizeInBytes = rows * columns * sizeof(float);
	this->numFloats = rows * columns;
	this->tensor = (float*)Create(this->sizeInBytes);
}

Tensor2d::Tensor2d(const Tensor2d& tensor2d) {
	this->rows = tensor2d.rows;
	this->columns = tensor2d.columns;
	this->sizeInBytes = this->rows * this->columns * sizeof(float);
	this->numFloats = rows * columns;
	this->tensor = (float*)Create(this->sizeInBytes);
	CopyHostToHost(this->tensor, tensor2d.tensor, this->sizeInBytes);
}

Tensor2d::Tensor2d(float* floatArray, const int rows, const int columns) {
	this->rows = rows;
	this->columns = columns;
	this->sizeInBytes = this->rows * this->columns * sizeof(float);
	this->numFloats = rows * columns;
	this->tensor = (float*)Create(this->sizeInBytes);
	CopyHostToDevice(this->tensor, floatArray, this->sizeInBytes);
}

Tensor2d::~Tensor2d() {
	Free(this->tensor);
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
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, std::chrono::system_clock::now().time_since_epoch().count());
	curandGenerateUniform(generator, this->tensor, this->numFloats);
	this->MultiplyConstantVal(maxVal - minVal);
	this->AddConstantVal(minVal);
}