#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/coordinate/point.hpp>

namespace agency
{


using parallel_agent = detail::basic_execution_agent<bulk_guarantee_t::parallel_t>;
using parallel_agent_1d = parallel_agent;
using parallel_agent_2d = detail::basic_execution_agent<bulk_guarantee_t::parallel_t, size2>;


} // end agency

