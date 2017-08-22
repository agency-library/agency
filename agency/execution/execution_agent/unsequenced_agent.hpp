#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/coordinate/point.hpp>

namespace agency
{


using unsequenced_agent = detail::basic_execution_agent<unsequenced_execution_tag>;
using unsequenced_agent_1d = unsequenced_agent;
using unsequenced_agent_2d = detail::basic_execution_agent<unsequenced_execution_tag, size2>;


} // end agency

