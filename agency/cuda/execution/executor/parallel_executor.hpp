#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/flattened_executor.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/memory/resource/pinned_resource.hpp>
#include <agency/experimental/ndarray/ndarray.hpp>
#include <agency/cuda/future.hpp>
#include <agency/execution/executor/properties/always_blocking.hpp>


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

    __AGENCY_ANNOTATION
    constexpr static bool query(always_blocking_t)
    {
      return true;
    }

    template<class Function, class ResultFactory, class SharedFactory>
    __AGENCY_ANNOTATION
    future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      auto result = result_factory();
      auto shared_arg = shared_factory();

      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_arg);
      }

      // XXX async_future is expensive to use here
      //     this function should return an always_ready_future,
      //     but cuda::future is not currently interoperable with it
      return cuda::make_ready_async_future(std::move(result));
    }
};


} // end this_thread


using parallel_executor = flattened_executor<grid_executor>;


} // end cuda
} // end agency

