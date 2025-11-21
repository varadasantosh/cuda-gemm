#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>  

#define TILE_SIZE 32

// Naive GEMM kernel
__global__ void naive_gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
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

// Tiled GEMM kernel
__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int A_col = tile * TILE_SIZE + tx;
        A_tile[ty][tx] = (row < M && A_col < K) ? A[row * K + A_col] : 0.0f;
        
        int B_row = tile * TILE_SIZE + ty;
        B_tile[ty][tx] = (B_row < K && col < N) ? B[B_row * N + col] : 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void generateRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 9.0f + 1.0f;
    }
}

bool verifyResults(float* C1, float* C2, int size, float tolerance = 9e-2f) {
    int errors = 0;
    for (int i = 0; i < size && errors < 10; i++) {
        if (fabs(C1[i] - C2[i]) > tolerance) {
            printf("  Mismatch at %d: %.6f vs %.6f\n", i, C1[i], C2[i]);
            errors++;
        }
    }
    return errors == 0;
}

int main() {
    // Matrix dimensions
    int M = 1024, N = 1024, K = 1024;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║          GEMM Performance Comparison                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    printf("Matrix: A(%dx%d) × B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tiled = (float*)malloc(size_C);
    float *h_C_cublas = (float*)malloc(size_C);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Generate random matrices
    srand(42);
    generateRandomMatrix(h_A, M, K);
    generateRandomMatrix(h_B, K, N);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ═══════════════════════════════════════
    // 1. NAIVE GEMM
    // ═══════════════════════════════════════
    printf("┌─────────────────────────────────────────┐\n");
    printf("│ 1. NAIVE GEMM (No Optimization)         │\n");
    printf("└─────────────────────────────────────────┘\n");
    
    dim3 block_naive(32, 32);
    dim3 grid_naive((N + 31) / 32, (M + 31) / 32);
    
    // Warm up
    naive_gemm_kernel<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Time
    cudaEventRecord(start);
    naive_gemm_kernel<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("  Time:        %7.2f ms\n", naive_time);
    printf("  Speedup:     %7.2fx (baseline)\n\n", 1.0);
    
    // ═══════════════════════════════════════
    // 2. TILED GEMM
    // ═══════════════════════════════════════
    printf("┌─────────────────────────────────────────┐\n");
    printf("│ 2. TILED GEMM (Shared Memory)           │\n");
    printf("└─────────────────────────────────────────┘\n");
    
    dim3 block_tiled(TILE_SIZE, TILE_SIZE);
    dim3 grid_tiled((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warm up
    tiled_gemm_kernel<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Time
    cudaEventRecord(start);
    tiled_gemm_kernel<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time;
    cudaEventElapsedTime(&tiled_time, start, stop);
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("  Time:        %7.2f ms\n", tiled_time);
    printf("  Speedup:     %7.2fx vs Naive\n\n", naive_time / tiled_time);
    
    // ═══════════════════════════════════════
    // 3. cuBLAS (Production Library)
    // ═══════════════════════════════════════
    printf("┌─────────────────────────────────────────┐\n");
    printf("│ 3. cuBLAS (Production Optimized)        │\n");
    printf("└─────────────────────────────────────────┘\n");
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // cuBLAS uses column-major, so we compute: C = B^T * A^T = (A * B)^T
    // Then interpret result as row-major C
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Warm up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();
    
    // Time
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cublas_time;
    cudaEventElapsedTime(&cublas_time, start, stop);
    cudaMemcpy(h_C_cublas, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("  Time:        %7.2f ms\n", cublas_time);
    printf("  Speedup:     %7.2fx vs Naive\n", naive_time / cublas_time);
    printf("               %7.2fx vs Tiled\n\n", tiled_time / cublas_time);
    
    // ═══════════════════════════════════════
    // VERIFICATION
    // ═══════════════════════════════════════
    printf("┌─────────────────────────────────────────┐\n");
    printf("│ Result Verification                     │\n");
    printf("└─────────────────────────────────────────┘\n");
    
    bool tiled_correct = verifyResults(h_C_naive, h_C_tiled, M * N);
    printf("  Tiled vs Naive:  %s\n", tiled_correct ? "✓ PASS" : "✗ FAIL");
    
    bool cublas_correct = verifyResults(h_C_naive, h_C_cublas, M * N);
    printf("  cuBLAS vs Naive: %s\n\n", cublas_correct ? "✓ PASS" : "✗ FAIL");
    
    // ═══════════════════════════════════════
    // SUMMARY TABLE
    // ═══════════════════════════════════════
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║                    Performance Summary            ║\n");
    printf("╠════════════════╦══════════╦═══════════╦═══════════╣\n");
    printf("║ Implementation ║   Time   ║  Speedup  ║ Status    ║\n");
    printf("╠════════════════╬══════════╬═══════════╬═══════════╣\n");
    printf("║ Naive GEMM     ║ %6.2f ms║   1.00x   ║   ✓       ║\n",
           naive_time);
    printf("║ Tiled GEMM     ║ %6.2f ms║  %5.2fx   ║   %s       ║\n", 
           tiled_time, naive_time/tiled_time, 
           tiled_correct ? "✓" : "✗");
    printf("║ cuBLAS         ║ %6.2f ms║  %5.2fx   ║   %s       ║\n", 
           cublas_time, naive_time/cublas_time,
           cublas_correct ? "✓" : "✗");
    printf("╚════════════════╩══════════╩═══════════╩═══════════╩\n");
    
    // Cleanup
    free(h_A); free(h_B); 
    free(h_C_naive); free(h_C_tiled); free(h_C_cublas);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    return 0;
}
