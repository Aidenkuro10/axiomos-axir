#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    float *dA, *dB, *dC;
    int N = 16;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dC, N * sizeof(float));

    float *hA, *hB, *hC; // (hôte)
    cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<N, N>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
