#include <stdio.h>       
#include <stdlib.h>      
#include <time.h>        
#include <cuda_runtime.h> 

// #define performs a preprocessor text substitution, replacing a name with a literal value or code before compilation, without creating variables or types.
#define TILE_WIDTH 16

__global__ void matrixMultiplyGPU(float *d_A, float *d_B, float *d_C, int N) {
		
		/*
			allocates fast memory that is shared by all threads within the same block, 
			enabling them to cooperate and reuse data efficiently.
		*/
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH]; 
    
   
    int bx = blockIdx.x; int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y; 
 
    int Row = by * TILE_WIDTH + ty; 
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0; 
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) { 
        if (Row < N && (m*TILE_WIDTH+tx) < N) 
            ds_A[ty][tx] = d_A[Row * N + m * TILE_WIDTH + tx]; 
        else 
            ds_A[ty][tx] = 0.0f; 
 
        if (Col < N && (m*TILE_WIDTH+ty) < N) 
            ds_B[ty][tx] = d_B[(m*TILE_WIDTH + ty) * N + Col]; 
        else 
            ds_B[ty][tx] = 0.0f; 
 
			 /*
				 __syncthreads() is a block-level barrier that pauses each thread until 				
 all threads in the block reach the same point, 
				 ensuring shared-memory operations are safely completed before continuing.
			 */
        __syncthreads(); 
 
        for (int k = 0; k < TILE_WIDTH; ++k) 
            Pvalue += ds_A[ty][k] * ds_B[k][tx]; 
        __syncthreads(); 
    } 
 
    if (Row < N && Col < N) 
        d_C[Row * N + Col] = Pvalue; 
}


// Exposed C function for Python 
extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int 
N) { 
    size_t size = N * N * sizeof(float); 
    float *d_A, *d_B, *d_C; 
 
    cudaMalloc((void**)&d_A, size); 
    cudaMalloc((void**)&d_B, size); 
    cudaMalloc((void**)&d_C, size); 
 
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 
 
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); 
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / 
TILE_WIDTH); 
 
    matrixMultiplyGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N); 
    cudaDeviceSynchronize(); 
 
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); 
 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);  
    float *B = (float *)malloc(size);  
    float *C = (float *)malloc(size);  // Output matrix C
    
    for (int i = 0; i < N * N; i++) {

        A[i] = rand() % 100 / 100.0f;  
        B[i] = rand() % 100 / 100.0f; 
    }
    
    clock_t start = clock();
    
		gpu_matrix_multiply(A, B, C, N);
   
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    fprintf(stderr, "GPU execution time via exposed C API (N=%d): %f seconds\n", N, elapsed);
    
    free(A);
    free(B);
    free(C);
    
    printf("%f\n", elapsed);
    return 0;
 }
