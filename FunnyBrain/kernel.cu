#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include "curand.h"
#include "chrono"

#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
const int SQRT_MAX_THREADS = static_cast<int>(sqrt(MAX_THREADS));
const dim3 THREADS_PER_BLOCK(SQRT_MAX_THREADS, SQRT_MAX_THREADS);

__global__ void CUDAAddArrays(const float* dev_a, const float* dev_b, float* dev_c, const int arrayLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < arrayLength) {
        dev_c[i] = dev_a[i] + dev_b[i];
    }
}

__global__ void CUDASubtractArrays(const float* dev_a, const float* dev_b, float* dev_c, const int arrayLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < arrayLength) {
        dev_c[i] = dev_a[i] - dev_b[i];
    }
}

__global__ void CUDAMultiplyArrays(const float* dev_a, const float* dev_b, float* dev_c, 
    const int a_rows, const int a_columns, const int b_columns) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < a_rows && column < b_columns) {
        // temporary variable used to compute the result
        float h = 0;
        for (int i = 0; i < a_columns; ++i) {
            h += dev_a[row * a_columns + i] * dev_b[i * b_columns + column];
        }

        dev_c[row * b_columns + column] = h;
    }   
}

__global__ void CUDACalculateLayer(const float* dev_pLayer, const float* dev_biases, 
    const float* dev_weights, float* dev_outputLayer, const int dev_weights_rows, const int dev_weights_columns) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < dev_weights_rows) {
        // temporary variable used to compute the result
        float h = 0;
        for (int i = 0; i < dev_weights_columns; ++i) {
            h += dev_weights[row * dev_weights_columns + i] * dev_pLayer[i];
        }

        dev_outputLayer[row] = h + dev_biases[row];
    }
}

__global__ void CUDAAddConstant(float* dev_a, float b, int arrayLength) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < arrayLength) {
        dev_a[index] += b;
    }
}

__global__ void CUDASubtractConstant(float* dev_a, float b, int arrayLength) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < arrayLength) {
        dev_a[index] -= b;
    }
}

__global__ void CUDAMultiplyConstant(float* dev_a, float b, int arrayLength) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < arrayLength) {
        dev_a[index] *= b;
    }
}

__host__ void  AddConstant(float* dev_a, float b, int arrayLength) {
    dim3 numBlocks((arrayLength + MAX_THREADS - 1) / MAX_THREADS);
    CUDAAddConstant << < numBlocks, MAX_THREADS >> > (dev_a, b, arrayLength);
}

__host__ void  SubtractConstant(float* dev_a, float b, int arrayLength) {
    dim3 numBlocks((arrayLength + MAX_THREADS - 1) / MAX_THREADS);
    CUDASubtractConstant << < numBlocks, MAX_THREADS >> > (dev_a, b, arrayLength);
}

__host__ void  MultiplyConstant(float* dev_a, float b, int arrayLength) {
    dim3 numBlocks((arrayLength + MAX_THREADS - 1) / MAX_THREADS);
    CUDAMultiplyConstant << < numBlocks, MAX_THREADS >> > (dev_a, b, arrayLength);
}

__host__ void AddArrays(const float* dev_a, const float* dev_b, float* dev_c, const int arrayLength) {
    dim3 numBlocks((arrayLength + MAX_THREADS - 1) / MAX_THREADS);
    CUDAAddArrays <<< numBlocks, MAX_THREADS >>> (dev_a, dev_b, dev_c, arrayLength);
}

__host__ void SubtractArrays(const float* dev_a, const float* dev_b, float* dev_c, const int arrayLength) {
    dim3 numBlocks((arrayLength + MAX_THREADS - 1) / MAX_THREADS);
    CUDASubtractArrays <<< numBlocks, MAX_THREADS >>> (dev_a, dev_b, dev_c, arrayLength);
}

__host__ void Multiply2d(const float* dev_a, const float* dev_b, float* dev_c, 
    const int a_rows, const int a_columns, const int b_columns) {
    dim3 numBlocks((b_columns + MAX_THREADS - 1) / MAX_THREADS,
        (a_rows + MAX_THREADS - 1) / MAX_THREADS);
    dim3 threadsPerBlock(b_columns < MAX_THREADS ? b_columns : MAX_THREADS,
        a_rows < MAX_THREADS ? a_rows : MAX_THREADS);
    CUDAMultiplyArrays <<<  numBlocks, threadsPerBlock >>> (dev_a, dev_b, dev_c, a_rows, a_columns, b_columns);
}

__host__ void CalculateLayer(const float* dev_pLayer, const float* dev_biases,
    const float* dev_weights, float* dev_outputLayer, const int dev_weights_rows, const int dev_weights_columns) {
    dim3 numBlocks((dev_weights_rows + MAX_THREADS - 1) / MAX_THREADS);
    dim3 threadsPerBlock(dev_weights_rows < MAX_THREADS ? dev_weights_rows : MAX_THREADS);
    CUDACalculateLayer << < numBlocks, threadsPerBlock >> > (dev_pLayer, dev_biases,
        dev_weights, dev_outputLayer, dev_weights_rows, dev_weights_columns);
}

__host__ void* Create(size_t sizeInBytes) {
    void* devPtr;
    cudaMalloc(&devPtr, sizeInBytes);
    return devPtr;
}

__host__ void GenerateRandom(float* dev_a, float minVal, float maxVal, int arrayLength) {
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, std::chrono::system_clock::now().time_since_epoch().count());
    curandGenerateUniform(generator, dev_a, arrayLength);
    MultiplyConstant(dev_a, maxVal - minVal, arrayLength);
    cudaDeviceSynchronize();
    AddConstant(dev_a, minVal, arrayLength);
    cudaDeviceSynchronize();
}

__host__ void CopyDeviceToHost(void* dst, const void* src, size_t sizeInBytes) {
    cudaMemcpy(dst, src, sizeInBytes, cudaMemcpyDeviceToHost);
}

__host__ void CopyHostToDevice(void* dst, const void* src, size_t sizeInBytes) {
    cudaMemcpy(dst, src, sizeInBytes, cudaMemcpyHostToDevice);
}

__host__ void CopyDeviceToDevice(void* dst, const void* src, size_t sizeInBytes) {
    cudaMemcpy(dst, src, sizeInBytes, cudaMemcpyDeviceToDevice);
}

__host__ void CopyHostToHost(void* dst, const void* src, size_t sizeInBytes) {
    cudaMemcpy(dst, src, sizeInBytes, cudaMemcpyHostToHost);
}

__host__ void Wait() {
    cudaDeviceSynchronize();
}

__host__ void Free(void* devPtr) {
    cudaFree(devPtr);
}