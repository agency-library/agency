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


} // end detail
} // end cuda
} // end agency

