#pragma once

#include <agency/detail/config.hpp>
#include <agency/container/vector.hpp>
#include <agency/cuda/memory/allocator/allocator.hpp>

namespace agency
{
namespace cuda
{


// XXX for the moment, don't define an Allocator template parameter
//     to prevent use of allocators which are not compatible with CUDA
template<class T>
using vector = agency::vector<T, agency::cuda::allocator<T>>;


} // end cuda
} // end agency

