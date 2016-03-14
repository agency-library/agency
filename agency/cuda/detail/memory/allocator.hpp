#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/memory/managed_allocator.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <memory>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class Alloc = managed_allocator<T>>
class allocator
{
  private:
    using host_allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;

  public:
    using value_type = T;

    __host__ __device__
    allocator() = default;

    __host__ __device__
    allocator(const allocator&) = default;

    template<class U>
    __host__ __device__
    allocator(const allocator<U>&) {}

    __host__ __device__
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

#ifndef __CUDA_ARCH__
      host_allocator alloc;
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
      host_allocator alloc;
      alloc.deallocate(ptr, n);
#else
      agency::cuda::detail::workaround_unused_variable_warning(n);

      free(ptr);
#endif
    }

    __agency_hd_warning_disable__
    template<class U, class... Args>
    __host__ __device__
    void construct(U* ptr, Args&&... args)
    {
#ifndef __CUDA_ARCH__
      host_allocator alloc;
      alloc.template construct<U>(ptr, std::forward<Args>(args)...);
#else
      new(ptr) U(std::forward<Args>(args)...);
#endif
    }
}; // end allocator


} // end detail
} // end cuda
} // end agency

