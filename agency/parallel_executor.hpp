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


class parallel_executor : public flattened_executor<nested_executor<concurrent_executor, this_thread::parallel_executor>>
{
  private:
    using super_t = flattened_executor<nested_executor<concurrent_executor, this_thread::parallel_executor>>;

  public:
    using super_t::super_t;

    parallel_executor()
      : super_t(2 * std::thread::hardware_concurrency())
    {}
};


} // end agency

