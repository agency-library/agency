#pragma once

#include <agency/detail/config.hpp>
#include <cuda_runtime.h>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
class pinned_allocator
{
  public:
    using value_type = T;

    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

      cudaError_t error = cudaHostAlloc(reinterpret_cast<void**>(&result), n * sizeof(value_type), cudaHostAllocPortable);

      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "pinned_allocator::allocate(): cudaHostAlloc");
      }

      return result;
    }

    void deallocate(value_type* ptr, size_t n)
    {
      cudaError_t error = cudaFreeHost(ptr);

      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "pinned_allocator::deallocate(): cudaFree");
      }
    }

    template<class U, class... Args>
    void construct(U* ptr, Args&&... args)
    {
      cudaError_t error = cudaDeviceSynchronize();

      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "pinned_allocator::allocate(): cudaDeviceSynchronize");
      }

      new(ptr) T(std::forward<Args>(args)...);
    }
};


} // end detail
} // end cuda
} // end agency

