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


class parallel_executor : public flattened_executor<grid_executor>
{
  private:
    using super_t = flattened_executor<grid_executor>;

    static constexpr size_t maximum_blocksize = 256;

  public:
    using super_t::super_t;

    parallel_executor(const grid_executor& exec = grid_executor(detail::current_device()))
      : super_t(exec,
                detail::maximum_grid_size_x(detail::current_device()),
                maximum_blocksize)
    {}
};


} // end cuda
} // end agency

