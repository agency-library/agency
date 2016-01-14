#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/asynchronous_state.hpp>
#include <agency/cuda/detail/allocator.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
using asynchronous_state = agency::detail::asynchronous_state<T,detail::allocator<T>>;

  
} // end detail
} // end cuda
} // end agency

