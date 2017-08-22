#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/concurrency/barrier.hpp>

namespace agency
{
namespace detail
{


// this the type of barrier used by basic_concurrent_agent's implementation
class concurrent_agent_barrier
{
  public:
    __AGENCY_ANNOTATION
    concurrent_agent_barrier(size_t num_threads)
#ifndef __CUDA_ARCH__
      : barrier_(num_threads)
#endif
    {}

    __AGENCY_ANNOTATION
    size_t count() const
    {
#ifndef __CUDA_ARCH__
       return barrier_.count();
#else
       return blockDim.x * blockDim.y * blockDim.z;
#endif
    }

    __AGENCY_ANNOTATION
    void arrive_and_wait()
    {
#ifndef __CUDA_ARCH__
      barrier_.arrive_and_wait();
#else
      __syncthreads();
#endif
    }

#ifndef __CUDA_ARCH__
  private:
    agency::detail::barrier barrier_;
#endif
};


} // end detail
} // end agency

