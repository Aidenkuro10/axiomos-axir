#include <hip/hip_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    float *dA, *dB, *dC;
    int N = 16;
    hipMalloc(&dA, N * sizeof(float));
    hipMalloc(&dB, N * sizeof(float));
    hipMalloc(&dC, N * sizeof(float));

    float *hA, *hB, *hC; // hôte
    hipMemcpy(dA, hA, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, N * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(vector_add, dim3(N,1,1), dim3(N,1,1), 0, 0, dA, dB, dC, N);

    hipMemcpy(hC, dC, N * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
