#include <hip/hip_runtime.h>
__global__ void reduce_sum(const float* A, float* Out, int N) { /* see note above */ }
int main() {
    float *dA, *dOut; int N=16;
    hipMalloc(&dA, N*sizeof(float)); hipMalloc(&dOut, sizeof(float));
    float *hA, *hOut;
    hipMemcpy(dA, hA, N*sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(reduce_sum, dim3(1,1,1), dim3(N,1,1), 0, 0, dA, dOut, N);
    hipMemcpy(hOut, dOut, sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    hipFree(dA); hipFree(dOut);
    return 0;
}
