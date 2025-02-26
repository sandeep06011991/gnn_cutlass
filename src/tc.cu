#include<stdio.h>
#include "helper.h"

#include <mma.h>
using namespace nvcuda; 

#define M 16
#define N 16
#define K 16

__global__
void kernel(){
    __shared__ int A[4][4];
    __shared__ int B[4][4];
    __shared__ int C[4][4];

    int x = threadIdx.x /4;
    int y = threadIdx.x % 4;
    A[x][y] = threadIdx.x;
    B[x][y] = threadIdx.x + 16;
    C[x][y] = threadIdx.x + 32;
    

    // fragment<matrix_a, M, N, K, half, row_major> a_frag;
    
}

int main(){
    
    kernel<<<1,16>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}