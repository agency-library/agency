#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/cuda/memory/managed_allocator.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/detail/memory/malloc_allocator.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <memory>

namespace agency
{
namespace cuda
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
    using device_allocator = agency::detail::malloc_allocator<T>;

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

    __agency_exec_check_disable__
    __host__ __device__
    split_allocator(const host_allocator& host_alloc, const device_allocator& device_alloc = device_allocator())
      : host_alloc_(host_alloc),
        device_alloc_(device_alloc)
    {}

    __host__ __device__
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

#ifndef __CUDA_ARCH__
      result = host_alloc_.allocate(n);
#else
      result = device_alloc_.allocate(n);
#endif

      return result;
    }

    __host__ __device__
    void deallocate(value_type* ptr, size_t n)
    {
#ifndef __CUDA_ARCH__
      host_alloc_.deallocate(ptr, n);
#else
      device_alloc_.deallocate(ptr, n);
#endif
    }

    template<class Iterator, class... Iterators>
    __host__ __device__
    agency::detail::tuple<Iterator,Iterators...> construct_n(Iterator first, size_t n, Iterators... iters)
    {
#ifndef __CUDA_ARCH__
      return agency::detail::allocator_traits<host_allocator>::construct_n(host_alloc_, first, n, iters...);
#else
      return agency::detail::allocator_traits<device_allocator>::construct_n(device_alloc_, first, n, iters...);
#endif
    }

    // XXX we might want to derive from these instead of make them members
    //     to get the empty base class optimization
    host_allocator host_alloc_;
    device_allocator device_alloc_;
}; // end split_allocator


} // end cuda
} // end agency

