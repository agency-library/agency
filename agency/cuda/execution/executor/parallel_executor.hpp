#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/flattened_executor.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/memory/resource/pinned_resource.hpp>
#include <agency/experimental/ndarray/ndarray.hpp>
#include <agency/cuda/future.hpp>

namespace agency
{
namespace cuda
{
namespace this_thread
{


class parallel_executor
{
  public:
    using execution_category = parallel_execution_tag;

    template<class T>
    using allocator = cuda::allocator<T, pinned_resource>;

    template<class T>
    using future = cuda::future<T>;

    template<class Function, class ResultFactory, class SharedFactory>
    __AGENCY_ANNOTATION
    agency::detail::result_of_t<ResultFactory()>
      bulk_sync_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory)
    {
      auto result = result_factory();
      auto shared_arg = shared_factory();

      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_arg);
      }

      return std::move(result);
    }
};


} // end this_thread


using parallel_executor = flattened_executor<grid_executor>;


} // end cuda
} // end agency

