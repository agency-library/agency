#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/coordinate/point.hpp>

namespace agency
{


using sequenced_agent = detail::basic_execution_agent<bulk_guarantee_t::sequenced_t>;
using sequenced_agent_1d = sequenced_agent;
using sequenced_agent_2d = detail::basic_execution_agent<bulk_guarantee_t::sequenced_t, size2>;


} // end agency

