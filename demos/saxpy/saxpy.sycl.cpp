#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    queue q;
    const int N = 16;
    float alpha = 2.0f;

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    float *dA = malloc_device<float>(N, q);
    float *dB = malloc_device<float>(N, q);
    float *dC = malloc_device<float>(N, q);

    q.memcpy(dA, hA, N * sizeof(float));
    q.memcpy(dB, hB, N * sizeof(float));

    // KERNEL: saxpy(dA,dB,dC,alpha,N)

    q.memcpy(hC, dC, N * sizeof(float));
    q.wait();

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}
