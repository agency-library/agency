#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/managed_allocator.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
class allocator
{
  public:
    using value_type = T;

    __host__ __device__
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

#ifndef __CUDA_ARCH__
      managed_allocator<T> alloc;
      result = alloc.allocate(n);
#else
      result = reinterpret_cast<T*>(malloc(sizeof(T)));

      if(result == nullptr)
      {
        terminate_with_message("agency::cuda::detail::allocator::allocate(): malloc failed");
      }
#endif

      return result;
    }

    __host__ __device__
    void deallocate(value_type* ptr, size_t n)
    {
#ifndef __CUDA_ARCH__
      managed_allocator<T> alloc;
      alloc.deallocate(ptr, n);
#else
      agency::cuda::detail::workaround_unused_variable_warning(n);

      free(ptr);
#endif
    }
}; // end allocator


} // end detail
} // end cuda
} // end agency

