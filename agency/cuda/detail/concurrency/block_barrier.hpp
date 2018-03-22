#pragma once

#include <agency/detail/config.hpp>
#include <cassert>


namespace agency
{
namespace cuda
{
namespace detail
{


class block_barrier
{
  public:
    __AGENCY_ANNOTATION
    block_barrier(int count)
#ifndef __CUDA_ARCH__
      : count_(count)
#endif
    {}

    __AGENCY_ANNOTATION
    block_barrier(block_barrier&&) = delete;

    __AGENCY_ANNOTATION
    int count() const
    {
#ifndef __CUDA_ARCH__
      return count_;
#else
      return blockDim.x * blockDim.y * blockDim.z;
#endif
    }

    __AGENCY_ANNOTATION
    void arrive_and_wait()
    {
#ifdef __CUDA_ARCH__
      // since we sometimes need to call .arrive_and_wait() from textually-unaligned code locations,
      // use this barrier intrinsic instead of __syncthreads().
      // unlike __syncthreads(), __barrier_sync(0) does not assume textual alignment.
      __barrier_sync(0);
#else
      assert(0);
#endif
    }

  private:
#ifndef __CUDA_ARCH__
    int count_;
#endif
};


} // end detail
} // end cuda
} // end agency

