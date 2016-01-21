#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/unique_ptr.hpp>
#include <agency/cuda/detail/memory/allocator.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
using unique_ptr = agency::detail::unique_ptr<T, agency::detail::deleter<allocator<T>>>;


template<class T, class... Args>
__host__ __device__
unique_ptr<T> make_unique(Args&&... args)
{
  return agency::detail::allocate_unique<T>(allocator<T>(),std::forward<Args>(args)...);
}


} // end detail
} // end cuda
} // end agency

