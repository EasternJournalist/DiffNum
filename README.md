# DiffNum
 A simple head-only library for differentiable programming. Just replace `float` and `double` with `dfloat`, and `ddouble`, then it does Auto-Grad for you. Secondary derivatives and higher order derivatives are also supported. 

Easy to use. Primary math functions are supported in `Math<n_type>`

```c++
// Example 1. a, b are variables. c = a+b; d
ddouble a = 2., b = 3.;
// 2 total variables, a is the first, b is the second 
a.SetVar(2, 0); b.SetVar(2, 1);

// Equations (Computed in time)
auto c = a + b;
auto d = dmathd::Log(dmathd::Max(dmathd::Sin(a / c), b));

// Output the result and the gradient
std::cout << d << std::endl;
```



We also offer dense `Vec`  and `Mat` . Since `DiffVar` is so similar to `float` and `double`, they can be easily adopted into any advanced numerical structure. 

```c++
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

// Equations (Computed in time)
auto q = Vec<ddouble, 3>::dot(v1, v2);

// Output the result and the gradient
std::cout << q << std::endl;
```

