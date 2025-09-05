// OpenCL host pseudo-code (POC parsing)
cl_kernel k = clCreateKernel(program, "vector_add", &err);

cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);

clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, N*sizeof(float), hA, 0, NULL, NULL);
clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, N*sizeof(float), hB, 0, NULL, NULL);

clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
clSetKernelArg(k, 2, sizeof(cl_mem), &dC);
clSetKernelArg(k, 3, sizeof(int),    &N);

size_t global= (size_t)N;
size_t local = (size_t)N;
clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);

clEnqueueReadBuffer(q, dC, CL_TRUE, 0, N*sizeof(float), hC, 0, NULL, NULL);
clFinish(q);

clReleaseMemObject(dA);
clReleaseMemObject(dB);
clReleaseMemObject(dC);
