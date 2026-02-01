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

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
		/*
			Memory whose size is determined at compile time 
			(stack, static arrays, CUDA shared memory) 
			must have fixed dimensions known before the program runs, 
			while memory allocated from the heap (via malloc or cudaMalloc) 
			can be sized dynamically at runtime.
		*/
    size_t size = N * N * sizeof(float);
    
    float *A = (float *)malloc(size);  
    float *B = (float *)malloc(size);  
    float *C = (float *)malloc(size);  // Output matrix C
    
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
    
    // The block dimensions must match the tile width
    dim3 block(TILE_WIDTH, TILE_WIDTH );
    
    // Configure grid dimensions: number of blocks needed to cover N×N matrix
    // (N + 15) / 16 rounds up to ensure all elements are covered
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH );
    
    clock_t start = clock();
    
    // Launch kernel on GPU with specified grid and block dimensions
    // <<<grid, block>>> is CUDA syntax for kernel launch configuration
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    
    // Wait for all GPU threads to complete
    cudaDeviceSynchronize();
    
    clock_t end = clock();
    
    // Copy result matrix from device (GPU) back to host (CPU)
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    fprintf(stderr, "GPU execution time with tiling (N=%d): %f seconds\n", N, elapsed);
    
    free(A);
    free(B);
    free(C);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("%f\n", elapsed);
    return 0;
 }
