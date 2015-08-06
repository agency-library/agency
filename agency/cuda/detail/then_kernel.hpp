#pragma once

#include <memory>

namespace agency
{
namespace cuda
{
namespace detail
{


// XXX WAR nvbug 1671566
struct my_nullptr_t
{
  inline __host__ __device__ my_nullptr_t(std::nullptr_t) {}
};


template<class Function>
inline __device__ void then_kernel_impl(my_nullptr_t, Function f, my_nullptr_t)
{
  f();
}

template<class T, class Function>
inline __device__ void then_kernel_impl(T* result_ptr, Function f, my_nullptr_t)
{
  *result_ptr = f();
}


template<class Function, class T>
inline __device__ void then_kernel_impl(my_nullptr_t, Function f, T* arg_ptr)
{
  f(*arg_ptr);
}

template<class T1, class Function, class T2>
inline __device__ void then_kernel_impl(T1* result_ptr, Function f, T2* arg_ptr)
{
  *result_ptr = f(*arg_ptr);
}

template<class Pointer1, class Function, class Pointer2>
__global__ void then_kernel(Pointer1 result_ptr, Function f, Pointer2 arg_ptr)
{
  agency::cuda::detail::then_kernel_impl(result_ptr, f, arg_ptr);
}


} // end detail
} // end cuda
} // end agency

