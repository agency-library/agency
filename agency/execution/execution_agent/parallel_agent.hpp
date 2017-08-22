#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/coordinate/point.hpp>

namespace agency
{


using parallel_agent = detail::basic_execution_agent<parallel_execution_tag>;
using parallel_agent_1d = parallel_agent;
using parallel_agent_2d = detail::basic_execution_agent<parallel_execution_tag, size2>;


} // end agency

