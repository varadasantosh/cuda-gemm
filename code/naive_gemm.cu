#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void generateRandomMatrix(float* matrix, int rows, int cols, float min, float max) {
    for (int i = 0; i < rows * cols; i++) {
        float range = max - min;
        matrix[i] = ((float)rand() / RAND_MAX) * range + min;
    }
}

int main() {
    int M = 256, N = 256, K = 256;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    
    // Host matrices
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Generate random matrices
    printf("Generating random matrices...\n");
    srand(time(NULL));
    generateRandomMatrix(h_A, M, K, 1.0f, 10.0f);
    generateRandomMatrix(h_B, K, N, 1.0f, 10.0f);
    
    // Copy to device
    printf("Copying matrices to GPU...\n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Setup kernel launch parameters
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Calculate Time Taken
    printf("Running GEMM kernel...\n");
    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance
    long long ops = 2LL * M * N * K;
    double gflops = (ops / (milliseconds / 1000.0f)) / 1e9;
    
    printf("\n=== GEMM Performance Results ===\n");
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    printf("Operations: %lld\n", ops);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
