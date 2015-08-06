#pragma once

#include <agency/detail/config.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
class managed_allocator
{
  public:
    using value_type = T;
  
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;
  
      cudaError_t error = cudaMallocManaged(&result, n, cudaMemAttachGlobal);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
      }
  
      return result;
    }
  
    __host__ __device__
    void deallocate(value_type* ptr, size_t)
    {
      cudaError_t error = cudaFree(ptr);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
      }
    }
};


} // end detail
} // end cuda
} // end agency

