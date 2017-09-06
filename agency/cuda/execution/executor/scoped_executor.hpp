#pragma once

// XXX seems like this should not be a public header and instead be automatically
//     included by grid_executor.hpp, parallel_executor.hpp, and block_executor.hpp
//     to be sure we catch the specialization
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/concurrent_grid_executor.hpp>
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <agency/cuda/execution/executor/concurrent_executor.hpp>

namespace agency
{


template<>
class scoped_executor<cuda::parallel_executor, cuda::concurrent_executor>
  : public cuda::grid_executor
{
  private:
    using super_t = cuda::grid_executor;

  public:
    using outer_executor_type = cuda::parallel_executor;
    using inner_executor_type = cuda::concurrent_executor;

    scoped_executor() = default;

    scoped_executor(const outer_executor_type& outer_ex,
                    const inner_executor_type& inner_ex)
      : super_t(outer_ex.base_executor().device()),
        outer_ex_(outer_ex),
        inner_ex_(inner_ex)
    {}

    outer_executor_type& outer_executor()
    {
      return outer_ex_;
    }

    const outer_executor_type& outer_executor() const
    {
      return outer_ex_;
    }

    inner_executor_type& inner_executor()
    {
      return inner_ex_;
    }

    const inner_executor_type& inner_executor() const
    {
      return inner_ex_;
    }

  private:
    outer_executor_type outer_ex_;
    inner_executor_type inner_ex_;
}; // end scoped_executor


template<>
class scoped_executor<cuda::concurrent_executor, cuda::concurrent_executor>
  : public cuda::concurrent_grid_executor
{
  private:
    using super_t = cuda::concurrent_grid_executor;

  public:
    using outer_executor_type = cuda::concurrent_executor;
    using inner_executor_type = cuda::concurrent_executor;

    scoped_executor() = default;

    scoped_executor(const outer_executor_type& outer_ex,
                    const inner_executor_type& inner_ex)
      : super_t(outer_ex.device()),
        outer_ex_(outer_ex),
        inner_ex_(inner_ex)
    {}

    outer_executor_type& outer_executor()
    {
      return outer_ex_;
    }

    const outer_executor_type& outer_executor() const
    {
      return outer_ex_;
    }

    inner_executor_type& inner_executor()
    {
      return inner_ex_;
    }

    const inner_executor_type& inner_executor() const
    {
      return inner_ex_;
    }

  private:
    outer_executor_type outer_ex_;
    inner_executor_type inner_ex_;
}; // end scoped_executor


} // end agency

