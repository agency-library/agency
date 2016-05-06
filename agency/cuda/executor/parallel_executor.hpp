#pragma once

#include <agency/cuda/executor/grid_executor.hpp>
#include <agency/executor/flattened_executor.hpp>
#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/memory/pinned_allocator.hpp>
#include <agency/cuda/detail/array.hpp>
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
    using allocator = cuda::allocator<T, pinned_allocator<T>>;

    template<class T>
    using container = cuda::detail::array<T, size_t, allocator<T>>;

    template<class T>
    using future = cuda::future<T>;
};


} // end this_thread


using parallel_executor = flattened_executor<grid_executor>;


} // end cuda
} // end agency

