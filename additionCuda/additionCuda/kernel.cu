#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <Windows.h>
#include <cuda.h>

using namespace std;

unsigned const int m = 4;
unsigned const int n = 16;
unsigned const int sizen = 16*16;

 float mat1[sizen];
 float mat2[sizen];
 float mat3[sizen];
 float mat4[sizen];

__global__ void matrixMultiplicationKernel(float* M1, float* M2, float* M3) {

	int r = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.x + blockDim.x * blockIdx.x;

	float Sum = 0;

	
	for (int i = 0; i < n; i++) {
		Sum += M1[r * n + i] * M2[i * n + c];
	}
	M3[r * n + c] = Sum;
}

void matrixMultiplication(float *A, float *B, float *C) {
	dim3 threadsPerBlock(n,n);
	dim3 blocksPerGrid(1, 1);
	if (n*n > 512) {
		threadsPerBlock.x = 16;
		threadsPerBlock.y = 16;
		blocksPerGrid.x = ceil(double(n) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(n) / double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> >(A, B, C);
}

void cpu_mult() {
	float sum;
	for (int row = 0; row<n; row++) {
		for (int col = 0; col<n; col++) {
			sum = 0.0;
			for (int a = 0; a<n; a++) {
				sum += mat1[row*n + a] * mat2[a*n + col];
			}
			mat4[row*n + col] = sum;	
		}		
	}
}



inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
	}
	return result;
}

void gpu_mult() {
	unsigned int bytes = sizen * sizeof(float);
	float *d_M1, *d_M2, *d_M3;

	checkCuda(cudaMalloc((float**)&d_M1, bytes));
	checkCuda(cudaMalloc((float**)&d_M2, bytes));
	checkCuda(cudaMalloc((float**)&d_M3, bytes));
	checkCuda(cudaMemcpy((void*)(d_M1), (void*)(mat1), bytes, cudaMemcpyHostToDevice)); 
	checkCuda(cudaMemcpy((void*)(d_M2), (void*)(mat1), bytes, cudaMemcpyHostToDevice));
	matrixMultiplication (d_M1, d_M2, d_M3);
	checkCuda(cudaMemcpy((void*)(mat3), (void*)(d_M3), bytes, cudaMemcpyDeviceToHost));
	
	cudaFree(d_M1);
	cudaFree(d_M2);
	cudaFree(d_M3);

}


void main() {
	
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			mat1[i*n + j] = sin(i+j);
			mat2[i*n + j] = cos(j+i);
		}
			
	}

	float *d_M1, *d_M2, *d_M3;
	
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			cout << mat1[i*n + j] << "  ";
		}
		cout << "\n";
	}
	cout << "\n";
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			cout << mat2[i*n + j] << "  ";
		}
		cout << "\n";
	}
	cout << "\n";
	gpu_mult();
	cpu_mult();

	for (size_t i = 0 ; i < n; i++)
	{
		for (size_t j = 0 ; j < n; j++)
		{
			cout << mat3[i*n + j] << "  " ;
		}
		cout << "\n";
	}
	cout << "\n";
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			cout << mat4[i*n + j] << "  ";
		}
		cout << "\n";
	}
	cout << "\n";

}




