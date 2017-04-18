#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_adaptor.hpp>
#include <agency/memory/detail/resource/cached_resource.hpp>
#include <agency/cuda/memory/resource/pinned_resource.hpp>
#include <cuda_runtime.h>

namespace agency
{
namespace cuda
{


template<class T>
class pinned_allocator : public agency::detail::allocator_adaptor<T,pinned_resource>
{
  private:
    using super_t = agency::detail::allocator_adaptor<T,pinned_resource>;

  public:
    using super_t::super_t;

    pinned_allocator() = default;

    pinned_allocator(const pinned_allocator&) = default;

    template<class U>
    pinned_allocator(const pinned_allocator<U>& other)
      : super_t(other)
    {}
};


} // end cuda
} // end agency

