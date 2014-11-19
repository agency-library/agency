#pragma once

#include <agency/concurrent_executor.hpp>
#include <agency/sequential_executor.hpp>
#include <agency/nested_executor.hpp>
#include <agency/flattened_executor.hpp>

namespace agency
{


using parallel_executor = flattened_executor<nested_executor<concurrent_executor, sequential_executor>>;


} // end agency

