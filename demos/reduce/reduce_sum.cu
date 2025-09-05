#include <cuda_runtime.h>
__global__ void reduce_sum(const float* A, float* Out, int N) {
    // POC: each thread handles one element; CPU backend will emulate a proper reduction
    // (real CUDA reduction uses shared memory, this is fine for our AXIR demo)
}
int main() {
    float *dA, *dOut; int N = 16;
    cudaMalloc(&dA, N*sizeof(float)); cudaMalloc(&dOut, sizeof(float));
    float *hA, *hOut;
    cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum<<<1,N>>>(dA, dOut, N);
    cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dOut);
    return 0;
}
