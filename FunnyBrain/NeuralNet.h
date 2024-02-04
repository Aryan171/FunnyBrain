#pragma once
#include "Tensor1d.h"
#include "Tensor2d.h"

class NeuralNet {
public:
	Tensor1d* outputs; // stores the values of every neuron
	Tensor1d* biases;
	Tensor2d* weights;

	int* shape; // stores the shape of the neural network
	int numLayers;

	NeuralNet();
	NeuralNet(const Tensor1d* biases, const Tensor2d* weights, const int* shape, const int numLayers);
	NeuralNet(const NeuralNet& nn);
	NeuralNet(const int* shape, const int numLayers);

	/*
	Runs the entire neural network once on the given input and fills the entire output array
	*/
	void FeedForward(const Tensor1d& input);

	/*
	Calculates the values of one layer asynchronously
	*/
	void CalculateNextLayer(const int n);

	/*
	Returns the output value of the last layer i.e. returns the output of the neural network
	Before calling this function make shure to call the FeedForward function and fill the array 
	*/
	void GetLastLayer(float* a);

	/*
	Copies a neural network
	*/
	void Copy(const NeuralNet& nn);
		
	/*
	Mutates all the weights and biases
	*/
	void Mutate(const float minVal, const float maxVal);

	/*
	Copies a neural network and then mutates itself 
	*/
	inline void CopyAndMutate(const NeuralNet& nn, const float minVal, const float maxVal);
};