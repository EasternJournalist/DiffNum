

// By defining this macro, DiffNum_cuda will be tested in main.cpp. Make sure you have installed CUDA toolkits.
// If CUDA is not installed on your device, undefine DIFFNUM_WITH_CUDA and define DIFFNUM_NO_CUDA
#define DIFFNUM_WITH_CUDA   

#include <iostream>
#include <DiffNum.h>
#include <Vec.h>
#include <Mat.h>
#include <Tensor.h>

#ifdef DIFFNUM_WITH_CUDA
#include <cuda_test.h>
#endif


using namespace DiffNum;
using namespace Common;


int main()
{

    array<float, 3> nowke;
    
    nowke[2] = 1.f;
    using dmath = Math<ddouble<0>>;
    // Example 1. a, b are variables. c = a+b; d
    ddouble<0> a = 2., b = 3.;
    // 2 total variables, a is the first, b is the second 
    a.setVar(2, 0); b.setVar(2, 1);

    auto c = a + b;
    auto d = dmath::Log(dmath::Max(dmath::Sin(a / c), b));

    std::cout << d << std::endl;
   
    tensor<float, 3, 4, 5> t1;
    t1[2][1][0] = 1.f;

    auto defq = t1.shape<2>;
    t1.fill(2.f);
    tensor<float, 3, 4, 5> t2;
    t2 = t1;
    // t1.einsum<1, 2, 3>(t1);
    // Example 2. Vec v1 v2. v1[2] is the variable. q = v1 dot v2.
    vec<ddouble<0>, 3> v1, v2; 

    v1[0] = 8.7;
    v1[1] = 4.3;
    v1[2] = 7.;

    v2[0] = -6.7;
    v2[1] = 4.1;
    v2[2] = 2.3;

    // Set v1[2] as the only variable.
    v1[2].setVar(1, 0);

    auto q = vec<ddouble<0>, 3>::dot(v1, v2);

    std::cout << q << std::endl;
    std::cout << v1.norm() << std::endl;
    std::cout << std::endl;

    // Example 4. Evaluating secondary derivative.
    using ddmath = Math<dddouble<2>>;
    

    dddouble<2> x = 2., y = 3.;
    
    x.setVar(0); y.setVar(1),

    std::cout << "x := 2, y := 3" << std::endl;
    std::cout << "x^3 + 2*y^2 = ";
    std::cout << ddmath::Pow(x, unsigned int(3)) + 2. * ddmath::Pow(y, unsigned int(2)) << std::endl;

    std::cout << "x + x^3*y + x*y + 2*y = ";
    std::cout <<  1. - x + ddmath::Pow(x, unsigned int(3)) * y + x * y + 2. * y << std::endl;
    
    // Example 5. Implement in CUDA kernal.
#ifdef DIFFNUM_WITH_CUDA
    std::cout << "\n *** Test on CUDA *** \n u := 1, v := Pi / 3" << std::endl;
    ddouble<2> u = 1., v = Pi<double> / 3.;
    u.setVar(0); v.setVar(1);
    std::cout << "u + sin(v) = " << cuda_test(u, v) << std::endl;
#endif
    return 0;
}
