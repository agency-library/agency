#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/allocator/allocator.hpp>
#include <agency/detail/boxed_value.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
using boxed_value = agency::detail::boxed_value<T,allocator<T>>;


} // end detail
} // end cuda
} // end agency

