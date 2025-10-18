#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Tiled GEMM kernel using 32x32 shared memory tiles
__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global indices for the output matrix C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from matrix A into shared memory
        int A_row = row;
        int A_col = tile * TILE_SIZE + tx;
        if (A_row < M && A_col < K) {
            A_tile[ty][tx] = A[A_row * K + A_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        
        // Load tile from matrix B into shared memory
        int B_row = tile * TILE_SIZE + ty;
        int B_col = col;
        if (B_row < K && B_col < N) {
            B_tile[ty][tx] = B[B_row * N + B_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Simple GEMM kernel for comparison
__global__ void simple_gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
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

void printMatrix(float* matrix, int rows, int cols, const char* name) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 8; i++) {
        for (int j = 0; j < cols && j < 8; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("...\n");
}

bool verifyResults(float* C1, float* C2, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C1[i] - C2[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions - should be multiples of TILE_SIZE for best performance
    int M = 256, N = 256, K = 256;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    printf("=== Tiled GEMM with %dx%d Tiles ===\n", TILE_SIZE, TILE_SIZE);
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Memory usage: A=%.1fMB, B=%.1fMB, C=%.1fMB\n", 
           size_A/1e6, size_B/1e6, size_C/1e6);
    
    // Host matrices
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_tiled = (float*)malloc(size_C);
    float *h_C_simple = (float*)malloc(size_C);
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Generate random matrices
    printf("\nGenerating random matrices...\n");
    srand(time(NULL));
    generateRandomMatrix(h_A, M, K, 1.0f, 10.0f);
    generateRandomMatrix(h_B, K, N, 1.0f, 10.0f);
    
    // Print sample of input matrices
    printMatrix(h_A, M, K, "Matrix A (sample)");
    printMatrix(h_B, K, N, "Matrix B (sample)");
    
    // Copy to device
    printf("\nCopying matrices to GPU...\n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // === TILED GEMM ===
    printf("\n=== Running Tiled GEMM ===\n");
    
    // Grid and block dimensions for tiled GEMM
    dim3 block_tiled(TILE_SIZE, TILE_SIZE);
    dim3 grid_tiled((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Tiled kernel config: Grid(%d,%d), Block(%d,%d)\n", 
           grid_tiled.x, grid_tiled.y, block_tiled.x, block_tiled.y);
    
    // Warm up
    tiled_gemm_kernel<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Time tiled GEMM
    cudaEventRecord(start);
    tiled_gemm_kernel<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // === SIMPLE GEMM (for comparison) ===
    printf("\n=== Running Simple GEMM for Comparison ===\n");
    
    // Grid and block dimensions for simple GEMM
    dim3 block_simple(32, 32);
    dim3 grid_simple((N + block_simple.x - 1) / block_simple.x, 
                     (M + block_simple.y - 1) / block_simple.y);
    
    printf("Simple kernel config: Grid(%d,%d), Block(%d,%d)\n", 
           grid_simple.x, grid_simple.y, block_simple.x, block_simple.y);
    
    // Warm up
    simple_gemm_kernel<<<grid_simple, block_simple>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Time simple GEMM
    cudaEventRecord(start);
    simple_gemm_kernel<<<grid_simple, block_simple>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float simple_time = 0;
    cudaEventElapsedTime(&simple_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C_simple, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // === RESULTS ===
    printf("\n=== Performance Results ===\n");
    
    long long ops = 2LL * M * N * K;
    double tiled_gflops = (ops / (tiled_time / 1000.0f)) / 1e9;
    double simple_gflops = (ops / (simple_time / 1000.0f)) / 1e9;
    
    printf("Tiled GEMM:  %.3f ms, %.2f GFLOPS\n", tiled_time, tiled_gflops);
    printf("Simple GEMM: %.3f ms, %.2f GFLOPS\n", simple_time, simple_gflops);
    printf("Speedup: %.2fx\n", simple_time / tiled_time);
    
    // Verify correctness
    printf("\n=== Verification ===\n");
    bool correct = verifyResults(h_C_tiled, h_C_simple, M, N);
    printf("Results match: %s\n", correct ? "✓ YES" : "✗ NO");
    
    // Print sample of output
    printMatrix(h_C_tiled, M, N, "Result Matrix C (sample)");
    
    // Memory bandwidth analysis
    double mem_ops = (size_A + size_B + size_C) / (1024.0 * 1024.0 * 1024.0); // GB
    double tiled_bandwidth = mem_ops / (tiled_time / 1000.0);
    double simple_bandwidth = mem_ops / (simple_time / 1000.0);
    
    printf("\n=== Memory Bandwidth ===\n");
    printf("Data transferred: %.2f GB\n", mem_ops);
    printf("Tiled bandwidth:  %.1f GB/s\n", tiled_bandwidth);
    printf("Simple bandwidth: %.1f GB/s\n", simple_bandwidth);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_tiled); free(h_C_simple);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== Program Complete ===\n");
    return 0;
}
