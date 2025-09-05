#include <hip/hip_runtime.h>

__global__ void saxpy(const float* A, const float* B, float* C, float alpha, int N) {
    int i = threadIdx.x;
    if (i < N) C[i] = alpha * A[i] + B[i];
}

int main() {
    float *dA, *dB, *dC;
    float alpha = 2.0f;
    int N = 16;

    hipMalloc(&dA, N*sizeof(float));
    hipMalloc(&dB, N*sizeof(float));
    hipMalloc(&dC, N*sizeof(float));

    float *hA, *hB, *hC;
    hipMemcpy(dA, hA, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, N*sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(saxpy, dim3(N,1,1), dim3(N,1,1), 0, 0, dA, dB, dC, alpha, N);

    hipMemcpy(hC, dC, N*sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return 0;
}
