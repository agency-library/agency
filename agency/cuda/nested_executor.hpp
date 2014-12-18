#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/parallel_executor.hpp>
#include <agency/cuda/block_executor.hpp>

namespace agency
{


template<>
class nested_executor<cuda::parallel_executor, cuda::block_executor>
  : public cuda::grid_executor
{
  public:
    // XXX might be worth adapting the underlying cuda::grid_executor in some way that would
    //     yield outer & inner executor types
    using outer_executor_type = cuda::parallel_executor;
    using inner_executor_type = cuda::block_executor;

    nested_executor() = default;

    nested_executor(const outer_executor_type& outer_ex,
                    const inner_executor_type& inner_ex)
      : outer_ex_(outer_ex),
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
}; // end nested_executor


} // end agency

