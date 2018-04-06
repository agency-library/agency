#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/coordinate/point.hpp>

namespace agency
{


using unsequenced_agent = detail::basic_execution_agent<bulk_guarantee_t::unsequenced_t>;
using unsequenced_agent_1d = unsequenced_agent;
using unsequenced_agent_2d = detail::basic_execution_agent<bulk_guarantee_t::unsequenced_t, size2>;


} // end agency

