#include <stdio.h>
#include <iostream>
#include <DiffNum_cuda.h>
#include <cuda_test.h>

using namespace DiffNum;

__global__ void kernel(ddouble_arr_cuda<2> a, ddouble_arr_cuda<2> b, ddouble_arr_cuda<2>* c) {
    *c = a + Math_cuda<ddouble_arr_cuda<2>>::Sin(b);
    return;
}


ddouble_arr_cuda<2> cuda_test(ddouble_arr_cuda<2> a, ddouble_arr_cuda<2> b) {
    ddouble_arr_cuda<2>* dev_c;
    a.SetVar(0), b.SetVar(1);
  
    cudaMalloc((void**)&dev_c, sizeof(ddouble_arr_cuda<2>));

    kernel<<<1, 1>>>(a, b, dev_c);

    ddouble_arr_cuda<2> c;
    cudaMemcpy(&c, dev_c, sizeof(ddouble_arr_cuda<2>), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);

    return c;
}