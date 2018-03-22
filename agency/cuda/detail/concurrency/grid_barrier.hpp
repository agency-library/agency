#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <cstddef>
#include <cassert>

#if __has_include(<cooperative_groups.h>)
#include <cooperative_groups.h>
#endif


namespace agency
{
namespace cuda
{
namespace detail
{


class grid_barrier
{
  public:
    __AGENCY_ANNOTATION
    inline grid_barrier(std::size_t count)
#ifndef __CUDA_ARCH__
      : count_(count)
#endif
    {}

    __AGENCY_ANNOTATION
    grid_barrier(grid_barrier&&) = delete;

    // use a deduced template non-type parameter here
    // so that the static_assert below is raised only if .count() is called
    template<bool deduced_false = false>
    __AGENCY_ANNOTATION
    inline int count() const
    {
#ifndef __CUDA_ARCH__
      // when called from __host__ code, just return count_
      return count_;
#else
      // when called from __device__ code...
  #if(__cuda_lib_has_cooperative_groups)
      // if we have cooperative_groups, use it
      return cooperative_groups::this_grid().size();
  #else
      // if we don't have cooperative_groups, calculate the size of the grid ourself
      return (blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z);
  #endif
#endif
    }

    // use a deduced template non-type parameter here
    // so that the static_assert below is raised only if .arrive_and_wait() is called
    template<bool deduced_false = false>
    __AGENCY_ANNOTATION
    inline void arrive_and_wait()
    {
#ifndef __CUDA_ARCH__
      // when called from __host__ code, create a runtime error
      assert(0);
#else
      // when called from __device__ code...
  #if(__cuda_lib_has_cooperative_groups)
      return cooperative_groups::this_grid().sync();
  #else
      // if we haven't compiled the code correctly, create a compile-time error
      static_assert(deduced_false, "Use of grid_barrier::arrive_and_wait() in __device__ code requires CUDA version >= 9, __CUDA_ARCH__ >= 600 and relocatable device code.");
  #endif
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


