#pragma once

#include <agency/detail/config.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/nested_executor.hpp>
#include <agency/flattened_executor.hpp>
#include <agency/detail/this_thread_parallel_executor.hpp>

namespace agency
{


using parallel_executor = flattened_executor<nested_executor<concurrent_executor, this_thread::parallel_executor>>;


} // end agency

