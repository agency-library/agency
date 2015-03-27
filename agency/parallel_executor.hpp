#pragma once

#include <agency/concurrent_executor.hpp>
#include <agency/sequential_executor.hpp>
#include <agency/nested_executor.hpp>
#include <agency/flattened_executor.hpp>

namespace agency
{
namespace this_thread
{


class parallel_executor : public sequential_executor
{
  public:
    using execution_category = parallel_execution_tag;
};


} // end this_thread


using parallel_executor = flattened_executor<nested_executor<concurrent_executor, this_thread::parallel_executor>>;


} // end agency

