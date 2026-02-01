#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <cuda_runtime.h>

__global__ void matrixMultiplyGPU(float *d_A, float *d_B, float *d_C, int N) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
 
    if (row < N && col < N) { 
        float sum = 0.0f; 
        for (int k = 0; k < N; k++) { 
            sum += d_A[row * N + k] * d_B[k * N + col]; 
        } 
        d_C[row * N + col] = sum; 
    } 
} 

int main(int argc, char **argv) { 
    int N = (argc > 1) ? atoi(argv[1]) : 1024; // allow matrix size as input 
    size_t size = N * N * sizeof(float); 
    
    float *A = (float *)malloc(size); 
    float *B = (float *)malloc(size); 
    float *C = (float *)malloc(size); 
    
		float *d_A, *d_B, *d_C;
		cudaMalloc(&d_A, size);
		cudaMalloc(&d_B, size);
		cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; i++) { 
        A[i] = rand() % 100 / 100.0f; 
        B[i] = rand() % 100 / 100.0f; 
    } 
 
 		cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
		
		dim3 block(16, 16);
		dim3 grid((N + 15) / 16, (N + 15) / 16);
 
    clock_t start = clock(); 
		matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);    
		cudaDeviceSynchronize();
		clock_t end = clock(); 
		
		cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC; 
    fprintf(stderr, "GPU execution time (N=%d): %f seconds\n", N, elapsed);
    
    free(A); 
    free(B); 
    free(C); 
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("%f\n", elapsed);
    return 0; 
}
