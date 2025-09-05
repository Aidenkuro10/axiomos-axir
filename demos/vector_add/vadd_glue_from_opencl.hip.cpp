#include <hip/hip_runtime.h>
// AXIR -> HIP (glue POC)

// DeviceSelect: auto
// kernel k -> vector_add
hipMalloc(&dA, N * sizeof(float));
hipMalloc(&dB, N * sizeof(float));
hipMalloc(&dC, N * sizeof(float));
hipMemcpy(dA, hA, N*sizeof(float), hipMemcpyHostToDevice);
hipMemcpy(dB, hB, N*sizeof(float), hipMemcpyHostToDevice);
// setarg k[0]=dA
// setarg k[1]=dB
// setarg k[2]=dC
// setarg k[3]=N
hipLaunchKernelGGL(vector_add, dim3(&global,1,1), dim3(&local,1,1), 0, 0, dA, dB, dC, N);
hipMemcpy(hC, dC, N*sizeof(float), hipMemcpyDeviceToHost);
hipDeviceSynchronize();
hipFree(dA);
hipFree(dB);
hipFree(dC);
