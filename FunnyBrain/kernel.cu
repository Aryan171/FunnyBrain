#define MAX_THREADS 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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

__host__ void Add1d(const int* dev_a, const int* dev_b, int* dev_c, const int size) {
    CUDAAdd1d << < size / MAX_THREADS, size < MAX_THREADS ? size, MAX_THREADS >> > (dev_a, dev_b, dev_c, size);
}

__host__ void Subtract1d(const int* dev_a, const int* dev_b, int* dev_c, const int size) {
    CUDASubtract1d << < size / MAX_THREADS, size < MAX_THREADS ? size, MAX_THREADS >> > (dev_a, dev_b, dev_c, size);
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