#pragma once

/*
Adds a constant value to all the floats of the array dev_a
*/
void  AddConstant(float* dev_a, float b, int arrayLength);

/*
Subtracts a constant value from all the floats of the array dev_a
*/
void  SubtractConstant(float* dev_a, float b, int arrayLength);

/*
Multiplies a constant value to all the floats of the array dev_a
*/
void  MultiplyConstant(float* dev_a, float b, int arrayLength);

/*
Adds the arrays dev_a and dev_b and stores the result in dev_c
dev_a, dev_b and dev_c are device pointers
*/
void AddArrays(const float* dev_a, const float* dev_b, float* dev_c, const int arrayLength);

/*
Subtracts the arrays dev_a and dev_b and stores the result in dev_c
dev_a, dev_b and dev_c are device pointers
*/
void SubtractArrays(const float* dev_a, const float* dev_b, float* dev_c, const int arrayLength);

/*
Multiplies two matrices dev_a and dev_b and stores the result in matrix dev_c

dev_a, dev_b and dev_c are one dimentional arrays, with the rows written one
after other 

a_rows- number of columns in dev_a
a_columns- number of columns in dev_a
b_columns- number of columns in dev_b

when we multiply two matrices A(Rows = m, Columns = n) and B(Rows = p, Columns = q)
the resultant matrix has dimentions (Rows = m, Columns = q), and the matrix multiplication
is only possible if n = p
*/
void Multiply2d(const float* dev_a, const float* dev_b, float* dev_c,
    const int a_rows, const int a_columns, const int b_columns);
/*
Allocates sizeInBytes number of bytes of memory in device(gpu) and returns the device pointer
*/
void* Create(size_t sizeInBytes);

/*
Copies memory from src(device pointer) to dst(host pointer) (copies memory from gpu to cpu)
*/
void CopyDeviceToHost(void* dst, const void* src, size_t sizeInBytes);

/*
Copies memory from src(host pointer) to dst(source pointer) (copies memory from cpu to gpu)
*/
void CopyHostToDevice(void* dst, const void* src, size_t sizeInBytes);

/*
Copies memory from src(device pointer) to dst(device pointer) (copies memory from gpu to gpu)
*/
void CopyDeviceToDevice(void* dst, const void* src, size_t sizeInBytes);

/*
Copies memory from src(host pointer) to dst(host pointer) (copies memory from cpu to cpu)
*/
void CopyHostToHost(void* dst, const void* src, size_t sizeInBytes);

/*
Waits till all the threads in the gpu finish doing their work
*/
void Wait();

/*
Frees the memory spaces pointed to at by the device pointer devPtr
*/
void Free(void* devPtr);