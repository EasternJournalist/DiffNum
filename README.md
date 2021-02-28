# DiffNum
 A light-weighted head-only c++ library for differentiable programming. Unlike the popular TensorFlow and Torch, DiffNum is implemented  simply with forward inference with chain rule, instead of computation graph, source code transformation or other high level autograd algorithms. Thus it takes few efforts to implement and apply.

## Features

**Advantages**

* **Extremely easy to use and flexible**.  Just replace `float`   `double` with `dfloat`  `ddouble`, and specify the independent variables via `setVar()`. Then it does autograd for you. The gradients can be accessed at any stage of computation. And it has flexible indexing, because the differentiating is applied to single variables instead of vectors or tensors.
* **Saving memory in long iterations.** It does not record the computation graph (like tape-based approach of Torch).  So it is efficient when the computation involves a large number of iteration, especially self-accumulation iterations. 
* **Secondary derivatives and higher order derivatives supported**.  Higher order derivatives are derived via recursive template definition  `DiffVar`. It might be written like `DiffVar<DiffVar<double, 0>, 0>`. It means autograding the gradients, that is the secondary derivatives or Hessian matrices. 
* **CUDA supported**. DiffNum can be even used in CUDA kernel functions. Just replace `float` , `double` with `dfloat_arr_cuda` , and `ddouble_arr_cuda`. This data structure can be seamlessly applied in CUDA functions as local variables, parameters or other purposes. (Please note that the CUDA code auto-grad loops of a single variable are still sequential, not parallel. In large scale computations, the parallelism should be given to higher level of computations like matrix operations.) 
* **(Extra and in progress)** . Independent from DiffNum, we offer template classes `Vec`, `Matrix` and `Tensor` . They might be kind of useful. 

**Disadvantages**

* The time complexity is greatly many times of reversed differentiating algorithms (back propagation) when there is large number of independent variables. Thus it can be extremely inefficient when there are many variables! 

## Usage

* **Using the differentiable variable** 
  
  We offer the differentiable variable template classes: `DiffVar<d_type, size>`, where `d_type` can be `float` , `double`, or even differentiable variable type. The template parameter `size` is the number of independent variables, that is the length of gradient vector. If `size` is zero, the independent variables will be uncertain and can be dynamically changed. 
  
  To assign value to `DiffVar` variables is just like using `float` or `double` variables. The can be assigned values  simply using `operator=`. Real values can be directly assigned to `DiffVar` variables.
  
  For short, you can use`dfloat<size>` ,  `ddouble<size>`. They are the same to `DiffVar<size>` 
  
  Here is an example
  
  ```c++
  #include <DiffNum.h>
  #include <iostream>
  
  using namespace DiffNum;
  
  // Let's define a function: f(x, y, z) = 2*x^2 - 3*y*z + 1. And apply DiffVar to autograd.
  dfloat<3> f(float _x, float _y, float _z) {
      // We have 3 independent variables to study: x, y, z
      dfloat<3> x = _x, y = _y, z = _z;
      // The independent variables must be specified, otherwise they will be treated as constants. Here, let x be the 1st, y the 2nd, z the 3rd. Their indices are 0, 1, and 2 respectively.
      x.setVar(0); y.setVar(1); z.setVar(2);
  	// Then use them like using floats.
      return 2 * x * x - 3 * y * z + 1.0;
  }
  
  int main(void) {
      dfloat<3> u = f(3.7, 4.0, 4.3);
      std::cout << u << std::endl; 	// DiffNum can be directly outputted to ostream.
      //	Output: -23.22(14.8, -12.9, -12)
      // The first real number is the value of u. The following vector is the gradient to (x, y, z)
      return 0;
  }
  ```
  
  
  
* **Access the value and gradients**

  To access the value: `.getValue()`

  To access the gradient, use operator `[]`. With higher-order derivatives, use multiple `[]` to get the derivatives. For example `a[1], b[1][2]`.

* **Higher-order derivatives**

  Use `DiffVar` recursively. For example `DiffVar<DiffVar<double, 3>, 3>`. Currently, the size of gradients of each recursion must be the same. 

* **Primary mathematical functions**

  For `DiffVar`, we provide mathematical functions that performs autograding. They are in template class `Math<T>`, where `T` is the type of the `DiffVar` you are using.

* **CUDA supported DiffNum**

  To make DiffNum available in CUDA programs which are needed for many scientific computation tasks, we offer a CUDA version. Use `DiffVar_cuda` and `Math_cuda` in both host codes and device codes. 

  `DiffVar_cuda` can be parameters of `__global__` functions.

* **Directly memcpy DiffVar/DiffVar_cuda arrays ?**

  Except for `DiffVar<d_type, 0>` , which is dynamic, all other `DiffVar` arrays can be directly copied.

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

```c++
#include <DiffNum_cuda.h>
```



## By the way

Thanks to this project, I learned CUDA... 