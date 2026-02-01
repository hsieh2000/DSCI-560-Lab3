#include <stdio.h>
#include <stdlib.h>      
#include <time.h>     
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
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
	C[i] = 0.0f;
    }
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta  = 0.0f;    

    clock_t start = clock();
    
    cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  
            N, N, N,  
            &alpha,  
            d_B, N,  
            d_A, N,  
            &beta,  
            d_C, N); 
   
    // Wait for all GPU threads to complete
    cudaDeviceSynchronize();
    
    clock_t end = clock();
    
    // Copy result matrix from device (GPU) back to host (CPU)
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    fprintf(stderr, "cuBLAS execution time (N=%d): %f seconds\n", N, elapsed);
    
    free(A);
    free(B);
    free(C);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    printf("%f\n", elapsed);
    return 0;
 }
