#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/heterogeneous_allocator.hpp>
#include <agency/detail/memory/caching_allocator.hpp>

namespace agency
{
namespace cuda
{


template<class T, class Alloc = managed_allocator<T>>
using allocator = agency::detail::caching_allocator<
  heterogeneous_allocator<T,Alloc>
>;


} // end cuda
} // end agency

