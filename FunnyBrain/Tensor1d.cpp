#include "FunnyBrainInternal.h"
#include "Tensor1d.h"
#include "curand.h"
#include "chrono"

Tensor1d::Tensor1d(const int numFloats) {
	this->numFloats = numFloats;
	this->sizeInBytes = numFloats * sizeof(float);
	this->tensor = (float*)Create(this->sizeInBytes);
}

Tensor1d::Tensor1d(const Tensor1d& tensor1d) {
	this->numFloats = tensor1d.numFloats;
	this->sizeInBytes = numFloats * sizeof(float);
	this->tensor = (float*)Create(this->sizeInBytes);
	CopyHostToHost(this->tensor, tensor1d.tensor, this->sizeInBytes);
}

Tensor1d::Tensor1d(float* floatArray, const int numFloats) {
	this->numFloats = numFloats;
	this->sizeInBytes = numFloats * sizeof(float);
	this->tensor = (float*)Create(this->sizeInBytes);
	CopyHostToDevice(this->tensor, floatArray, this->sizeInBytes);
}

Tensor1d::~Tensor1d() {
	Free(this->tensor);
}

int Tensor1d::Add(const Tensor1d& a, const Tensor1d& b, Tensor1d& c) {
	if (a.numFloats != b.numFloats) {
		return 1;
	}
	AddArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
	return 0;
}

int Tensor1d::Subtract(const Tensor1d& a, const Tensor1d& b, Tensor1d& c) {
	if (a.numFloats != b.numFloats) {
		return 1;
	}
	SubtractArrays(a.tensor, b.tensor, c.tensor, a.numFloats);
	return 0;
}

void Tensor1d::AddConstantVal(float b) {
	AddConstant(this->tensor, b, this->numFloats);
}

void Tensor1d::MultiplyConstantVal(float b) {
	MultiplyConstant(this->tensor, b, this->numFloats);
}

void Tensor1d::SubtractConstantVal(float b) {
	SubtractConstant(this->tensor, b, this->numFloats);
}

void Tensor1d::SetValue(const float* a) {
	CopyHostToDevice(this->tensor, a, this->sizeInBytes);
}

void Tensor1d::GetValue(float* a) {
	CopyDeviceToHost(a, this->tensor, this->sizeInBytes);
}
void Tensor1d::RandomizeValues(float minVal, float maxVal) {
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, std::chrono::system_clock::now().time_since_epoch().count());
	curandGenerateUniform(generator, this->tensor, this->numFloats);
	this->MultiplyConstantVal(maxVal - minVal);
	this->AddConstantVal(minVal);
}