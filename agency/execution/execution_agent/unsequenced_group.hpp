#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/execution_group.hpp>
#include <agency/execution/execution_agent/unsequenced_agent.hpp>

namespace agency
{


template<class InnerExecutionAgent>
using unsequenced_group = detail::execution_group<unsequenced_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using unsequenced_group_1d = detail::execution_group<unsequenced_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using unsequenced_group_2d = detail::execution_group<unsequenced_agent_2d, InnerExecutionAgent>;


} // end agency

