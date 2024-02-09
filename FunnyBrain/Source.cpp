#include <iostream>
#include "NeuralNet.h"
#include "FunnyBrainInternal.h"

void print(float* a, int length);

int main() {
	NeuralNet nn;
	int size[] = { 1, 1 };
	nn.Initialize(size, 2);
	float inputs[] = { 1000 };
	Tensor1d inp;
	inp.Initialize(inputs, 1);
	nn.Randomize(-2, 2);
	nn.FeedForward(inp, RELU, 0);
	nn.GetLastLayer(inputs);
	std::cout << "output = " << std::endl;
	print(inputs, 1);
}

void print(float* a, int length) {
	for (int i = 0; i < length; ++i) {
		std::cout << a[i] << std::endl;
	}
}