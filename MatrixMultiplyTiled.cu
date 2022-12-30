#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

using std::vector;

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

class Matrix
{
public:
	const int M;
	const int N;
	vector<vector<float>> values;

	Matrix(vector<vector<float>> v, int m, int n) : values(v), M(m), N(n) {}

	void Print()
	{
		for (const vector<float>& vec : values)
		{
			for (const float& x : vec)
			{
				std::cout << x << " ";
			}
			std::cout << "\n";
		}
	}
};
#define tileSize 2

__global__ void multiply_gpu(float* out, float* a, float* b, const int M, const int N, const int K)
{
	__shared__ float a_tile[tileSize][tileSize];
	__shared__ float b_tile[tileSize][tileSize];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int out_row = by * tileSize + ty;
	int out_col = bx * tileSize + tx;

	float sum = 0.0;
	for (int m = 0; m < (N + tileSize - 1) / tileSize; m++)
	{
		int a_row = out_row;
		int a_col = m * tileSize + tx;
		int b_row = m * tileSize + ty;
		int b_col = out_col;

		a_tile[ty][tx] = (a_row < M && a_col < N) ? a[a_row * N + a_col] : 0;
		b_tile[ty][tx] = (b_row < N && b_col < K) ? b[b_row * K + b_col] : 0;
			
		__syncthreads();

		for (int i = 0; i < tileSize; i++)
		{
			sum += (b_tile[i][tx] * a_tile[ty][i]);		
		}
		__syncthreads();
	}
	if (out_row < M && out_col < K) {
		out[out_row * K + out_col] = sum;
	}
	

	
}
//start with just squares
Matrix multiply(Matrix& A, Matrix& B)
{
	cudaEvent_t start, stop;

	check(cudaEventCreate(&start));
	check(cudaEventCreate(&stop));
	check(cudaEventRecord(start, 0));

	int M = A.M;
	int N = A.N;
	int K = B.N;
	float* A_ptr = (float*)malloc(M * N * sizeof(float));
	float* B_ptr = (float*)malloc(N * K * sizeof(float));

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A_ptr[i * N + j] = A.values[i][j];
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			B_ptr[i * K + j] = B.values[i][j];
		}
	}

	float* a;
	float* b;
	float* out;
	check(cudaMalloc((void**)&a, M * N * sizeof(float)));
	check(cudaMalloc((void**)&b, N * K * sizeof(float)));
	check(cudaMalloc((void**)&out, M * K * sizeof(float)));

	check(cudaMemcpy(a, A_ptr, M * N * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(b, B_ptr, N * K * sizeof(float), cudaMemcpyHostToDevice));

	free(A_ptr);
	free(B_ptr);


	dim3 block((K + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);
	dim3 thread_dim(tileSize, tileSize);

	multiply_gpu << <block, thread_dim>> > (out, a, b, M, N, K);

	float* outCpu = (float*)malloc(M * K * sizeof(float));
	check(cudaMemcpy(outCpu, out, M * K * sizeof(float), cudaMemcpyDeviceToHost));

	Matrix C(vector<vector<float>>(M, vector<float>(K, 0)), M, K);

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < K; j++)
		{
			C.values[i][j] = outCpu[i * K + j];
		}
	}
	check(cudaEventRecord(stop, 0));
	check(cudaEventSynchronize(stop));
	float elapsedTime;
	check(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time to generate: %f ms\n", elapsedTime);
	check(cudaEventDestroy(start));
	check(cudaEventDestroy(stop));

	free(outCpu);
	check(cudaFree(out));
	check(cudaFree(a));
	check(cudaFree(b));

	return C;
}

int main(void)
{
	Matrix A({ {1,2,3},{4,5,6}, {7,8,9} }, 3, 3);
	Matrix B({ {1,0,0},{0,1, 1},{0,0, 1}}, 3,3);

	Matrix C = multiply(A, B);
	C.Print();


	return 0;
}