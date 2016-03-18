#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/split_allocator.hpp>
#include <agency/detail/memory/caching_allocator.hpp>

namespace agency
{
namespace cuda
{


template<class T, class Alloc = managed_allocator<T>>
using allocator = agency::detail::caching_allocator<
  split_allocator<T,Alloc>
>;


} // end cuda
} // end agency

