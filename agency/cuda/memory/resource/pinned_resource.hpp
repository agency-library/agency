#pragma once

#include <agency/detail/config.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cuda_runtime.h>


namespace agency
{
namespace cuda
{


class pinned_resource
{
  public:
    inline void* allocate(size_t num_bytes)
    {
      void* result = nullptr;
  
      cudaError_t error = cudaHostAlloc(&result, num_bytes, cudaHostAllocPortable);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "pinned_resource::allocate(): cudaMallocManaged");
      }
  
      return result;
    }

    inline void deallocate(void* ptr, size_t)
    {
      cudaError_t error = cudaFreeHost(ptr);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "pinned_resource::deallocate(): cudaFree");
      }
    }

    inline bool is_equal(const pinned_resource&) const
    {
      return true;
    }
};


inline bool operator==(const pinned_resource& a, const pinned_resource& b)
{
  return a.is_equal(b);
}

inline bool operator!=(const pinned_resource& a, const pinned_resource& b)
{
  return !(a == b);
}

inline bool operator<(const pinned_resource&, const pinned_resource&)
{
  return false;
}


} // end cuda
} // end agency

