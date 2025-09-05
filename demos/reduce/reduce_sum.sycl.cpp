#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    queue q;
    const int N = 16;

    float *hA = nullptr, *hOut = nullptr;
    float *dA  = malloc_device<float>(N, q);
    float *dOut = malloc_device<float>(1, q);

    q.memcpy(dA, hA, N * sizeof(float));

    // KERNEL: reduce_sum(dA,dOut,N)

    q.memcpy(hOut, dOut, sizeof(float));
    q.wait();

    sycl::free(dA, q);
    sycl::free(dOut, q);
    return 0;
}
