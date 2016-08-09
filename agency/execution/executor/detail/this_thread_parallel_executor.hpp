#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>

namespace agency
{
namespace this_thread
{


using parallel_executor = sequenced_executor;


} // end this_thread
} // end agency

