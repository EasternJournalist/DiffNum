
#include <iostream>
#include <DiffMath.h>
#include <Vec.h>
#include <DiffManager.h>

using namespace DiffNum;

int main()
{
    // Example 1. a, b are variables. c = a+b; d
    DiffDouble a = 2., b = 3.;
    // 2 total variables, a is the first, b is the second 
    a.SetVar(2, 0); b.SetVar(2, 1);

    auto c = a + b;
    auto d = Log(Max(Sin(a / c), b));

    std::cout << d << std::endl;
   


    // Example 2. Vec v1 v2. v1[2] is the variable. q = v1 dot v2.
    Vec<DiffDouble, 3> v1, v2;

    v1[0] = 8.7;
    v1[1] = 4.3;
    v1[2] = 7.;

    v2[0] = -6.7;
    v2[1] = 4.1;
    v2[2] = 2.3;

    // Set v1[2] as the only variable.
    v1[2].SetVar(1, 0);

    auto q = Vec<DiffDouble, 3>::dot(v1, v2);

    std::cout << q << std::endl;

    // Example 3. Use the DiffManager to help you set up variables.
    DiffManager<float> manager;

    return 0;
}
