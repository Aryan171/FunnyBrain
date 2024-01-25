#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void AddArrays(const int* a, const int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__host__ void add(const int* a, const int* b, int* c, const int size) {
    int* dev_a, 
        *dev_b,
        *dev_c;

    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_b, size * sizeof(int));
    cudaMalloc(&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    
    AddArrays <<<1, size >>> (dev_a, dev_b, dev_c);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}