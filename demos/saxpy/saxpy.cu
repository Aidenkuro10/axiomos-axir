#include <cuda_runtime.h>
__global__ void saxpy(const float* A, const float* B, float* C, float alpha, int N) {
    int i = threadIdx.x;
    if (i < N) C[i] = alpha * A[i] + B[i];
}
int main() {
    float *dA, *dB, *dC; float alpha = 2.0f; int N = 16;
    cudaMalloc(&dA, N*sizeof(float)); cudaMalloc(&dB, N*sizeof(float)); cudaMalloc(&dC, N*sizeof(float));
    float *hA, *hB, *hC;
    cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<N,N>>>(dA,dB,dC,alpha,N);
    cudaMemcpy(hC, dC, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
