#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    printf("cuda runtime version check\n");
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("found %d cuda devices\n", deviceCount);
    return 0;
}
