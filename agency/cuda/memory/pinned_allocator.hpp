#pragma once

#include <agency/detail/config.hpp>
#include <cuda_runtime.h>

namespace agency
{
namespace cuda
{


template<class T>
class pinned_allocator
{
  public:
    using value_type = T;

    pinned_allocator() = default;

    pinned_allocator(const pinned_allocator&) = default;

    template<class U>
    pinned_allocator(const pinned_allocator<U>&) {}

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
};


} // end cuda
} // end agency

