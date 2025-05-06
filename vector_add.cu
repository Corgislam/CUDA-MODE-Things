#include "solve.h"
#include <cuda_runtime.h>
#include <float.h>
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = reinterpret_cast<float4*>(const_cast<float*>(&(A[idx])))[0];
        float4 reg_b = reinterpret_cast<float4*>(const_cast<float*>(&(B[idx])))[0];
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(C[idx]) = reg_c;
  }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
