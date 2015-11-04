#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function>
__global__ void cuda_kernel(Function f)
{
  f();
}


} // end detail
} // end cuda
} // end agency

