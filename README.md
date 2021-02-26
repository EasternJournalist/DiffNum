# DiffNum
 A light-weighted head-only c++ library for differentiable programming. It is implemented  simply with forward inference with chain rule, instead of computation graph, source code transformation or other high level auto-grad algorithms. Thus it takes few efforts to implement and apply.

## Features

**Advantages**

* **Extremely easy to use and flexible**.  Just replace `float`   `double` with `dfloat`  `ddouble`, and specify the independent variables via `SetVar()`. Then it does autograd for you. The gradients can be accessed at any stage of computation. 
* **Saving memory in long iterations.** It does not record the computation graph (like tape-based approach of Torch).  
* **Secondary derivatives and higher order derivatives supported naturally**.  Higher order derivatives are derived via nesting template class `DiffVar`. It might be written like `DiffVar<DiffVar<double>>`. It means doing auto-grad for the gradients, that is the secondary derivatives or Hessian matrices. 
* **CUDA supported**. DiffNum can be even used in CUDA kernel functions (array-like gradients only). Just replace `float` , `double` with `dfloat_arr_cuda` , and `ddouble_arr_cuda`. This data structure can be seamlessly applied in CUDA functions as local variables, parameters or other purposes. (Please note that the CUDA code auto-grad loops of a single variable are still sequential, not parallel. In large scale computations, the parallelism should be given to higher level of computations like matrix operations.) 
* **(Extra and in progress)** . Independent from DiffNum, we offer template classes `Vec`, `Matrix` and `Tensor` . They might be kind of useful. 

**Disadvantages**

* The time complexity is greatly many times of reversed differentiating algorithms (back propagation) when there is large number of independent variables. Thus it can be extremely inefficient when there are many variables!

## Examples

 Primary math functions are supported in `Math<n_type>`

```c++
// Example 1. a, b are variables. c = a+b; d
ddouble a = 2., b = 3.;
// 2 total variables, a is the first, b is the second 
a.SetVar(2, 0); b.SetVar(2, 1);

// Equations (Computed in time)
auto c = a + b;
auto d = dmathd::Log(dmathd::Max(dmathd::Sin(a / c), b));

// Output both the result and the gradient
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

// Output both the result and the gradient
std::cout << q << std::endl;
```

## Install & Build

This is a head-only library. Just clone this repository and include the headers in your codes.

```c++
#include <DiffNum.h>
```

And for CUDA applications

```
#include <DiffNum_cuda.h>
```

