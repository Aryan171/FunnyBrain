#include "NeuralNet.h"
#include "FunnyBrainInternal.h"

NeuralNet::NeuralNet() {
	this->biases = nullptr;
	this->weights = nullptr;
	this->shape = nullptr;
	this->outputs = nullptr;
	this->numLayers = 0;
}

NeuralNet::NeuralNet(const Tensor1d* biases, const Tensor2d* weights, const int* shape, int numLayers) {
	this->biases = new Tensor1d[numLayers];
	this->weights = new Tensor2d[numLayers - 1];
	this->shape = new int[numLayers];
	this->outputs = new Tensor1d[numLayers];
	this->numLayers = numLayers;

	for (int i = 0; i < numLayers - 1; ++i) {
		this->biases[i] = Tensor1d(biases[i]);
		this->weights[i] = Tensor2d(weights[i]);
		this->shape[i] = shape[i];
		this->outputs[i] = Tensor1d(shape[i]);
	}

	numLayers--;

	this->biases[numLayers] = Tensor1d(biases[numLayers]);
	this->shape[numLayers] = shape[numLayers];
	this->outputs[numLayers] = Tensor1d(shape[numLayers]);
}

NeuralNet::NeuralNet(const NeuralNet& nn) {
	this->biases = new Tensor1d[nn.numLayers];
	this->weights = new Tensor2d[nn.numLayers - 1];
	this->shape = new int[nn.numLayers];
	this->outputs = new Tensor1d[nn.numLayers];
	this->numLayers = nn.numLayers;

	for (int i = 0; i < nn.numLayers - 1; ++i) {
		this->biases[i] = Tensor1d(nn.biases[i]);
		this->weights[i] = Tensor2d(nn.weights[i]);
		this->shape[i] = shape[i];
		this->outputs[i] = Tensor1d(shape[i]);
	}

	this->biases[nn.numLayers - 1] = Tensor1d(nn.biases[nn.numLayers - 1]);
	this->shape[nn.numLayers - 1] = nn.shape[nn.numLayers - 1];
	this->outputs[nn.numLayers - 1] = Tensor1d(shape[nn.numLayers - 1]);
}

NeuralNet::NeuralNet(const int* shape, int numLayers) {
	this->biases = new Tensor1d[numLayers];
	this->weights = new Tensor2d[numLayers - 1];
	this->shape = new int[numLayers];
	this->outputs = new Tensor1d[numLayers];
	this->numLayers = numLayers;

	for (int i = 0; i < numLayers - 1; ++i) {
		this->biases[i] = Tensor1d(shape[i]);
		this->weights[i] = Tensor2d(shape[i + 1], shape[i]);
		this->shape[i] = shape[i];
		this->outputs[i] = Tensor1d(shape[i]);
	}

	numLayers--;

	this->biases[numLayers] = Tensor1d(shape[numLayers]);
	this->shape[numLayers] = shape[numLayers];
	this->outputs[numLayers] = Tensor1d(shape[numLayers]);
}

void NeuralNet::FeedForward(const Tensor1d& input, const int activationFunction, const float parameter) {
	CopyDeviceToDevice(outputs[0].tensor, input.tensor, input.sizeInBytes);
	Tensor1d::Add(this->outputs[0], this->biases[0], this->outputs[0]);
	ActivateLayer(this->outputs[0].tensor, activationFunction, parameter, this->outputs[0].numFloats);
	for (int i = 1; i < this->numLayers; ++i) {
		this->CalculateNextLayer(i, activationFunction, parameter);
		Wait();
	}
}

void NeuralNet::CalculateNextLayer(const int n, const int activationFunction, const float parameter) {
	CalculateLayer(this->outputs[n - 1].tensor, this->biases[n].tensor, this->weights[n - 1].tensor, 
		this->outputs[n].tensor, this->weights[n - 1].rows, this->weights[n - 1].columns, activationFunction, parameter);
}

void NeuralNet::GetLastLayer(float* a) {
	this->outputs[this->shape[this->numLayers - 1]].GetValue(a);
}

void NeuralNet::Copy(const NeuralNet& nn) {
	this->biases = new Tensor1d[nn.numLayers];
	this->weights = new Tensor2d[nn.numLayers - 1];
	this->shape = new int[nn.numLayers];
	this->outputs = new Tensor1d[nn.numLayers];
	this->numLayers = nn.numLayers;

	for (int i = 0; i < nn.numLayers - 1; ++i) {
		this->biases[i] = Tensor1d(nn.biases[i]);
		this->weights[i] = Tensor2d(nn.weights[i]);
		this->shape[i] = shape[i];
		this->outputs[i] = Tensor1d(shape[i]);
	}

	this->biases[nn.numLayers - 1] = Tensor1d(nn.biases[nn.numLayers - 1]);
	this->shape[nn.numLayers - 1] = nn.shape[nn.numLayers - 1];
	this->outputs[nn.numLayers - 1] = Tensor1d(shape[nn.numLayers - 1]);
}

void NeuralNet::Mutate(const float minVal, const float maxVal) {
	for (int i = 0; i < numLayers - 1; ++i) {
		this->biases[i].Mutate(minVal, maxVal);
		this->weights[i].Mutate(minVal, maxVal);
	}
	this->biases[numLayers - 1].Mutate(minVal, maxVal);
}

inline void NeuralNet::CopyAndMutate(const NeuralNet& nn, const float minVal, const float maxVal) {
	this->Copy(nn);
	this->Mutate(minVal, maxVal);
}