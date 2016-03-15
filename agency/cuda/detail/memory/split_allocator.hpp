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


// split_allocator uses a different primitive allocator depending on whether
// its operations are called from __host__ or __device__ code.
// the Alloc template parameter corresponds to the type of allocator to use
// in __host__ code.
// XXX we might want to take two allocators instead of just one
template<class T, class Alloc = managed_allocator<T>>
class split_allocator
{
  private:
    using host_allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;

  public:
    using value_type = T;

    template<class U>
    struct rebind
    {
      using other = split_allocator<
        U,
        typename std::allocator_traits<host_allocator>::template rebind_alloc<U>
      >;
    };

    __host__ __device__
    split_allocator() = default;

    __host__ __device__
    split_allocator(const split_allocator&) = default;

    template<class U>
    __host__ __device__
    split_allocator(const split_allocator<U>&) {}

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
}; // end split_allocator


} // end detail
} // end cuda
} // end agency

