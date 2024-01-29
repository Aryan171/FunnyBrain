#include "FunnyBrainInternal.h"

class Tensor2d {
public:
	float* tensor = nullptr;
	int rows, columns;
	size_t sizeInBytes;

	Tensor2d(const int rows, const int columns) {
		this->rows = rows;
		this->columns = columns;
		this->sizeInBytes = rows * columns * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
	}

	Tensor2d(const Tensor2d& tensor2d) {
		this->rows = tensor2d.rows;
		this->columns = tensor2d.columns;
		this->sizeInBytes = this->rows * this->columns * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToHost(this->tensor, tensor2d.tensor, this->sizeInBytes);
	}

	Tensor2d(float* floatArray, const int rows, const int columns) {
		this->rows = rows;
		this->columns = columns;
		this->sizeInBytes = this->rows * this->columns * sizeof(float);
		this->tensor = (float*)Create(this->sizeInBytes);
		CopyHostToDevice(this->tensor, floatArray, this->sizeInBytes);
	}

	void Add()
};