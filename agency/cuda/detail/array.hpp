#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/memory/allocator.hpp>
#include <agency/detail/array.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class Shape = size_t, class Alloc = allocator<T>, class Index = Shape>
using array = agency::detail::array<T,Shape,Alloc,Index>;


} // end detail
} // end cuda
} // end agency

