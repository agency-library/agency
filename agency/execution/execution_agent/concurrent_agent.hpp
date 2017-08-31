#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_concurrent_agent.hpp>
#include <agency/detail/concurrency/any_barrier.hpp>
#include <agency/memory/detail/resource/arena_resource.hpp>
#include <agency/memory/detail/resource/malloc_resource.hpp>
#include <agency/memory/detail/resource/tiered_resource.hpp>
#include <agency/coordinate/point.hpp>
#include <cstddef>


namespace agency
{
namespace detail
{


using default_barrier = any_barrier;
using default_concurrent_resource = tiered_resource<arena_resource<sizeof(int) * 128>, malloc_resource>;


} // end detail


// XXX consider introducing unique types for concurrent_agent & concurrent_agent_2d for the sake of better compiler error messages
using concurrent_agent = detail::basic_concurrent_agent<std::size_t, detail::default_barrier, detail::default_concurrent_resource>;
using concurrent_agent_1d = concurrent_agent;
using concurrent_agent_2d = detail::basic_concurrent_agent<size2, detail::default_barrier, detail::default_concurrent_resource>;


} // end agency

