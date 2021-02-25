﻿
#include <iostream>
#include <DiffNum.h>
#include <Vec.h>

using namespace DiffNum;

int main()
{
    using dmath = Math<ddouble>;
    // Example 1. a, b are variables. c = a+b; d
    ddouble a = 2., b = 3.;
    // 2 total variables, a is the first, b is the second 
    a.SetVar(2, 0); b.SetVar(2, 1);

    auto c = a + b;
    auto d = dmath::Log(dmath::Max(dmath::Sin(a / c), b));

    std::cout << d << std::endl;
   


    // Example 2. Vec v1 v2. v1[2] is the variable. q = v1 dot v2.
    Vec<ddouble, 3> v1, v2;

    v1[0] = 8.7;
    v1[1] = 4.3;
    v1[2] = 7.;

    v2[0] = -6.7;
    v2[1] = 4.1;
    v2[2] = 2.3;

    // Set v1[2] as the only variable.
    v1[2].SetVar(1, 0);

    auto q = Vec<ddouble, 3>::dot(v1, v2);

    std::cout << q << std::endl;

    // Example 3. Use the DiffManager to help you set up variables.
    DiffManager<float> manager;

    // Example 4. Evaluating secondary derivative.
    using ddouble_arr = DiffArrayVar<double, 2>;
    using dddouble_arr = DiffArrayVar<ddouble_arr, 2>;

    using ddmath_arr = Math<dddouble_arr>;
    
    dddouble_arr x = ddouble_arr(2.), y = ddouble_arr(3.);
    x.SetVar(2, 0); x.value.SetVar(2, 0);
    y.SetVar(2, 1), y.value.SetVar(2, 1);

    std::cout << "x := 2, y := 3" << std::endl;
    std::cout << "x^3 + y^2 = ";
    std::cout << ddmath_arr::Pow(x, 3) + ddmath_arr::Pow(y, 2) << std::endl;

    std::cout << "x + x^3*y + x*y + y = ";
    std::cout <<  x + ddmath_arr::Pow(x, 3) * y + x * y + y << std::endl;


    return 0;
}
