#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/memory/split_allocator.hpp>
#include <agency/detail/memory/caching_allocator.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class Alloc = managed_allocator<T>>
using allocator = agency::detail::caching_allocator<
  split_allocator<T,Alloc>
>;


} // end detail
} // end cuda
} // end agency

