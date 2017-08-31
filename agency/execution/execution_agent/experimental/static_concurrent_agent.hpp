#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/experimental/detail/basic_static_execution_agent.hpp>
#include <agency/execution/execution_agent/concurrent_agent.hpp>
#include <agency/memory/detail/resource/arena_resource.hpp>
#include <cstddef>

namespace agency
{
namespace experimental
{

// XXX consider moving this to a location where default_concurrent_resource can call it
__AGENCY_ANNOTATION
constexpr std::size_t default_pool_size(std::size_t group_size)
{
  return group_size * sizeof(int);
}

template<std::size_t group_size, std::size_t grain_size = 1, std::size_t pool_size = default_pool_size(group_size)>
using static_concurrent_agent = detail::basic_static_execution_agent<
  agency::detail::basic_concurrent_agent<
    std::size_t,
    agency::detail::default_barrier,
    agency::detail::arena_resource<pool_size>
  >,
  group_size,
  grain_size
>;


} // end experimental
} // end agency

