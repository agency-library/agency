#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/flattened_executor.hpp>
#include <agency/cuda/detail/memory/allocator.hpp>
#include <agency/cuda/detail/memory/pinned_allocator.hpp>
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
    using allocator = cuda::detail::allocator<T, cuda::detail::pinned_allocator<T>>;

    template<class T>
    using container = cuda::detail::array<T, size_t, allocator<T>>;

    template<class T>
    using future = cuda::future<T>;
};


} // end this_thread


using parallel_executor = agency::flattened_executor<grid_executor>;


} // end cuda
} // end agency

