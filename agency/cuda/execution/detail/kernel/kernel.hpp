#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function, class... Args>
__global__ void cuda_kernel(Function f, Args... args)
{
  f(args...);
}


template<class Function, class... Args>
__host__ __device__
auto make_cuda_kernel(const Function&, const Args&...) ->
  decltype(&cuda_kernel<Function,Args...>)
{
  return &cuda_kernel<Function,Args...>;
}


} // end detail
} // end cuda
} // end agency

