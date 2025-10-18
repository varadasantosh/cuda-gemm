
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>


__global__ void register_tiling(int* A, int* B, int* C, int M, int N, int K) {
    // Each thread computes one element of C
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2 ;


    float c_reg[2][2] = {0};

    for(int k=0;k<K;k++){

        float a_reg[2];
        float b_reg[2];

        // Load elements into registers
        for(int i=0;i<2;i++){
            a_reg[i] = A[(row + i) * K + k];
            b_reg[i] = B[k * N + (col + i)];
        }

        // Compute partial products and accumulate
        
        for(int i=0;i<2;i++){
            for(int j=0;j<2;j++){
                c_reg[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }


    for (int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            C[(row + i) * N + (col + j)] = c_reg[i][j];
        }
    }
}


int main(){

    int M=4, N=4,K=4;
    size_t size_A = M * K * sizeof(int);
    size_t size_B = K * N * sizeof(int);
    size_t size_C = M * N * sizeof(int);

    int *h_A = (int *)malloc(size_A);
    int *h_B = (int *)malloc(size_B);
    int *h_C = (int *)malloc(size_C);

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);


    // Initialize matrices
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = i * K + j + 1;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N + j] = i * N + j + 2;

    // Perform matrix multiplication

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);   
    dim3 blockDim(2, 2);
    dim3 gridDim((N + blockDim.x * 2 - 1) / (blockDim.x * 2), (M + blockDim.y * 2 - 1) / (blockDim.y * 2));
    register_tiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("\n-------------\n");
    printf("Matrix A:");
    printf("\n-------------\n");
    for (int i=0;i<M;i++){
        for (int j=0; j<K;j++){

            printf("%3d", h_A[i*K+j]);
        }
        printf("\n");
    }    

    printf("\n-------------\n");
    printf("Matrix B:");
    printf("\n-------------\n");

    for(int i=0;i<K;i++){
        for(int j=0; j<N;j++){

            printf("%3d",h_B[i*N+j]);
        }
        printf("\n");
    }
    
    printf("\n-------------\n");
    printf("Result Matrix:C ");
    printf("\n-------------\n");
    // Print result matrix C

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%3d ", h_C[i * N + j]);
        }
        printf("\n");
    }
   

    // Free allocated memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
