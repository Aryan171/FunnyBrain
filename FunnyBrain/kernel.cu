#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define MAX_THREADS 1024
const int SQRT_MAX_THREADS = static_cast<int>(sqrt(MAX_THREADS));

__global__ void CUDAAdd1d(const int* dev_a, const int* dev_b, int* dev_c, const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dev_c[i] = dev_a[i] + dev_b[i];
    }
}

__global__ void CUDASubtract1d(const int* dev_a, const int* dev_b, int* dev_c, const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dev_c[i] = dev_a[i] - dev_b[i];
    }
}

__global__ void CUDAAdd2d(const int** dev_a, const int** dev_b, int** dev_c, const int rows, const int columns) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < rows && column < columns) {
        dev_c[row][column] = dev_a[row][column] + dev_b[row][column];
    }
}

__global__ void CUDASubtract2d(const int** dev_a, const int** dev_b, int** dev_c, const int rows, const int columns) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < rows && column < columns) {
        dev_c[row][column] = dev_a[row][column] - dev_b[row][column];
    }
}

__host__ void Add1d(const int* dev_a, const int* dev_b, int* dev_c, const int size) {
    CUDAAdd1d << < size / MAX_THREADS, size < MAX_THREADS ? size, MAX_THREADS >> > (dev_a, dev_b, dev_c, size);
}

__host__ void Subtract1d(const int* dev_a, const int* dev_b, int* dev_c, const int size) {
    CUDASubtract1d << < size / MAX_THREADS, size < MAX_THREADS ? size, MAX_THREADS >> > (dev_a, dev_b, dev_c, size);
}

// rows is the number of arrays in the 2d array and columns is the number of elements in each row of an array
__host__ void Add2d(const int** dev_a, const int** dev_b, int** dev_c, const int rows, const int columns) {
    dim3 numBlocks((rows / SQRT_MAX_THREADS) + (rows % SQRT_MAX_THREADS == 0 ? 0 : 1), 
        (columns / SQRT_MAX_THREADS) + (columns % SQRT_MAX_THREADS == 0 ? 0 : 1));
    dim3 threadsPerBlock(32, 32);
    CUDAAdd2d << < numBlocks, threadsPerBlock >> > (dev_a, dev_b, dev_c, rows, columns);
}

// rows is the number of arrays in the 2d array and columns is the number of elements in each row of an array
__host__ void Subtract2d(const int** dev_a, const int** dev_b, int** dev_c, const int rows, const int columns) {
    dim3 numBlocks((rows / SQRT_MAX_THREADS) + (rows % SQRT_MAX_THREADS == 0 ? 0 : 1),
        (columns / SQRT_MAX_THREADS) + (columns % SQRT_MAX_THREADS == 0 ? 0 : 1));
    dim3 threadsPerBlock(32, 32);
    CUDAAdd2d << < numBlocks, threadsPerBlock >> > (dev_a, dev_b, dev_c, rows, columns);
}

__host__ void Create(const void* devPtr, const int size) {
    cudaMalloc(&devPtr, size);
}

__host__ void CopyDeviceToHost(void* dst, const void* src, const int size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

__host__ void CopyHostToDevice(void* dst, const void* src, const int size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

__host__ void CopyDeviceToDevice(void* dst, const void* src, const int size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

__host__ void CopyHostToHost(void* dst, const void* src, const int size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
}

__host__ void Free(void* devPtr) {
    cudaFree(devPtr);
}