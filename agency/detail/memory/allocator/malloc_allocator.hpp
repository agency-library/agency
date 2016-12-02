#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/memory/allocator/allocator_adaptor.hpp>
#include <agency/detail/memory/resource/malloc_resource.hpp>

namespace agency
{
namespace detail
{


template<class T>
using malloc_allocator = allocator_adaptor<T,malloc_resource>;


} // end detail
} // end agency

