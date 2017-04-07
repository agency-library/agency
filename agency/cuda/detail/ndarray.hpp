#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/allocator.hpp>
#include <agency/detail/ndarray.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class Shape = size_t, class Alloc = allocator<T>, class Index = Shape>
using ndarray = agency::detail::ndarray<T,Shape,Alloc,Index>;


} // end detail
} // end cuda
} // end agency

