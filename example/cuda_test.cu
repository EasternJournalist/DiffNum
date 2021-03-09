#include <stdio.h>
#include <iostream>
#include <DiffNum_cuda.h>
#include <cuda_test.h>
#include <vector>

using namespace DiffNum;

__global__ void kernel(ddouble_cuda<2> a, ddouble_cuda<2> b, ddouble_cuda<2>* c) {
    *c = a + Math_cuda<ddouble_cuda<2>>::Sin(b);
    return;
}


ddouble_cuda<2> cuda_test(ddouble_cuda<2> a, ddouble_cuda<2> b) {
    ddouble_cuda<2>* dev_c;
    a.setVar(0), b.setVar(1);
  
    cudaMalloc((void**)&dev_c, sizeof(ddouble_cuda<2>));

    kernel<<<1, 1>>>(a, b, dev_c);

    ddouble_cuda<2> c;
    cudaMemcpy(&c, dev_c, sizeof(ddouble_cuda<2>), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);

    return c;
}