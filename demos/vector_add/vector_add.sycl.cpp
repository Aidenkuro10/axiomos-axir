#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    queue q;
    const int N = 16;

    float *hA = nullptr, *hB = nullptr, *hC = nullptr; // host placeholders (POC)
    float *dA = malloc_device<float>(N, q);
    float *dB = malloc_device<float>(N, q);
    float *dC = malloc_device<float>(N, q);

    // H2D
    q.memcpy(dA, hA, N * sizeof(float));
    q.memcpy(dB, hB, N * sizeof(float));

    // KERNEL: vector_add(dA,dB,dC,N)
    // (POC: the actual lambda is irrelevant for the AXIR flow demo)

    // D2H
    q.memcpy(hC, dC, N * sizeof(float));
    q.wait();

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}
