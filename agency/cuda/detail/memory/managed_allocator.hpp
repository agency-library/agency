#pragma once

#include <agency/detail/config.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cuda_runtime.h>

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
  
      cudaError_t error = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
      }
  
      return result;
    }
  
    void deallocate(value_type* ptr, size_t)
    {
      cudaError_t error = cudaFree(ptr);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
      }
    }

    template<class U, class... Args>
    void construct(U* ptr, Args&&... args)
    {
      cudaError_t error = cudaDeviceSynchronize();

      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaDeviceSynchronize");
      }

      new(ptr) T(std::forward<Args>(args)...);
    }
};


} // end detail
} // end cuda
} // end agency

