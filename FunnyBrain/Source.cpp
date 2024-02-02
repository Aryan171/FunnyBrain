#include "curand.h"
#include "Tensor1d.h"
#include "Tensor2d.h"
#include "chrono"

curandGenerator_t generator;

void RandomizeValues(Tensor1d& tensor1d, float minVal, float maxVal) {
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, std::chrono::system_clock::now().time_since_epoch().count());
	curandGenerateUniform(generator, tensor1d.tensor, tensor1d.numFloats);
	tensor1d.MultiplyConstantVal(maxVal - minVal);
	tensor1d.AddConstantVal(minVal);
}

void RandomizeValues(Tensor2d& tensor2d, float minVal, float maxVal) {
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, std::chrono::system_clock::now().time_since_epoch().count());
	curandGenerateUniform(generator, tensor2d.tensor, tensor2d.numFloats);
	tensor2d.MultiplyConstantVal(maxVal - minVal);
	tensor2d.AddConstantVal(minVal);
}