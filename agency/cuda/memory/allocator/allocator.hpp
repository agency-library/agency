#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/allocator/heterogeneous_allocator.hpp>
#include <agency/cuda/memory/resource/managed_resource.hpp>
#include <agency/memory/detail/resource/cached_resource.hpp>

namespace agency
{
namespace cuda
{

template<class T, class HostResource = managed_resource>
using allocator = heterogeneous_allocator<T,agency::detail::globally_cached_resource<HostResource>>;


} // end cuda
} // end agency

