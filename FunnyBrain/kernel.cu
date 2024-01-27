#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
const int SQRT_MAX_THREADS = static_cast<int>(sqrt(MAX_THREADS));

__global__ void CUDAAddArrays(const int* dev_a, const int* dev_b, int* dev_c, const int arrayLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < arrayLength) {
        dev_c[i] = dev_a[i] + dev_b[i];
    }
}

__global__ void CUDASubtractArrays(const int* dev_a, const int* dev_b, int* dev_c, const int arrayLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < arrayLength) {
        dev_c[i] = dev_a[i] - dev_b[i];
    }
}

__host__ void AddArrays(const int* dev_a, const int* dev_b, int* dev_c, const int arrayLength) {
    dim3 numBlocks((arrayLength / MAX_THREADS) + (arrayLength % MAX_THREADS == 0 ? 0 : 1));
    CUDAAddArrays << < numBlocks, MAX_THREADS >> > (dev_a, dev_b, dev_c, arrayLength);
}

__host__ void SubtractArrays(const int* dev_a, const int* dev_b, int* dev_c, const int arrayLength) {
    dim3 numBlocks((arrayLength / MAX_THREADS) + (arrayLength % MAX_THREADS == 0 ? 0 : 1));
    CUDASubtract1d << < numBlocks, MAX_THREADS >> > (dev_a, dev_b, dev_c, arrayLength);
}

__host__ void* Create(size_t sizeInBytes) {
    void* devPtr;
    cudaMalloc(&devPtr, sizeInBytes);
    return devPtr;
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

// Waits till all the threads in the gpu finish doing their work
__host__ void Wait() {
    cudaDeviceSynchronize();
}

__host__ void Free(void* devPtr) {
    cudaError a = cudaFree(devPtr);
}