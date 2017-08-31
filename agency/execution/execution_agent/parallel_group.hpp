#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/execution_group.hpp>
#include <agency/execution/execution_agent/parallel_agent.hpp>

namespace agency
{


template<class InnerExecutionAgent>
using parallel_group = detail::execution_group<parallel_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using parallel_group_1d = detail::execution_group<parallel_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using parallel_group_2d = detail::execution_group<parallel_agent_2d, InnerExecutionAgent>;


} // end agency

