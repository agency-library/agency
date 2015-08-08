#pragma once

#include <memory>

namespace agency
{
namespace cuda
{
namespace detail
{


struct unit {};

struct unit_ptr
{
  __host__ __device__
  unit operator*() const
  {
    return unit{};
  }
};


template<class Function>
inline __device__ void then_kernel_impl(unit, Function f, unit)
{
  f();
}

template<class T, class Function>
inline __device__ void then_kernel_impl(T& result, Function f, unit)
{
  result = f();
}


template<class Function, class T>
inline __device__ void then_kernel_impl(unit, Function f, T& arg)
{
  f(arg);
}

template<class T1, class Function, class T2>
inline __device__ void then_kernel_impl(T1& result, Function f, T2& arg)
{
  result = f(arg);
}

template<class Pointer1, class Function, class Pointer2>
__global__ void then_kernel(Pointer1 result_ptr, Function f, Pointer2 arg_ptr)
{
  agency::cuda::detail::then_kernel_impl(*result_ptr, f, *arg_ptr);
}


} // end detail
} // end cuda
} // end agency

