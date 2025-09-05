#include <cuda_runtime.h>
// AXIR -> CUDA (glue POC)

// DeviceSelect: auto
// kernel k -> vector_add
cudaMalloc(&dA, N * sizeof(float));
cudaMalloc(&dB, N * sizeof(float));
cudaMalloc(&dC, N * sizeof(float));
cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(dB, hB, N*sizeof(float), cudaMemcpyHostToDevice);
// setarg k[0]=dA
// setarg k[1]=dB
// setarg k[2]=dC
// setarg k[3]=N
vector_add<<<&global,&local,0,0>>>(dA, dB, dC, N);
cudaMemcpy(hC, dC, N*sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
cudaFree(dA);
cudaFree(dB);
cudaFree(dC);
