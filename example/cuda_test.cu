#include <stdio.h>
#include <iostream>
#include <DiffNum.h>
#include <vector>
#include <Vec.h>

using namespace DiffNum;
using namespace Common;

__global__ void kernel(ddouble<2> a, ddouble<2> b, ddouble<2>* c) {
    *c = a + Math<ddouble<2>>::Sin(b);
    return;
}


ddouble<2> cuda_test(ddouble<2> a, ddouble<2> b) {
    ddouble<2>* dev_c;
    a.setVar(0), b.setVar(1);
  
    cudaMalloc((void**)&dev_c, sizeof(ddouble<2>));

    kernel<<<1, 1>>>(a, b, dev_c);

    ddouble<2> c;
    cudaMemcpy(&c, dev_c, sizeof(ddouble<2>), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);

    return c;
}